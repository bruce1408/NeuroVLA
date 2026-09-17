"""
NeuroVLA Toy Model
==================
用小网络复现 NeuroVLA 的算法精髓，不依赖 Qwen-VL / snnTorch / FlashAttention。

三层对应关系（和仓库里的真模型一一对应）：

    生物类比          真仓库                              本 toy
    ------------------------------------------------------------------
    皮层 Cortical     Qwen2.5-VL                         TinyCorticalVLM (CNN + Conv1d)
    小脑 Cerebellar   Layer-wise QFormer + GRU-FiLM      同名模块（算法原样）
    脊髓 Spinal       SNN MLP-ResNet Action Head         手写 LIF，动力学与仓库一致

数据流（与 NeuroVLA.py 相同）：

    图像 + 语言
        → 皮层：得到多层 hidden states
        → QFormer：用 query token 逐层 cross-attend，抽出动作特征
        → 迭代 2 次：
              GRU-FiLM 用机器人状态调制特征   ← 小脑滤波
              SNN 逐步积分膜电位，输出动作 chunk ← 脊髓
              把预测动作写回 state
        → 两段 chunk 在时间维拼接，和真值做 L1

请从本文件顶部往下读：LIF → 皮层 → QFormer → 小脑 → 脊髓 → 整模型。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 0. 超参：刻意做小，方便 CPU / 单卡几秒内跑通，但结构比例贴近真模型
# ---------------------------------------------------------------------------
@dataclass
class ToyConfig:
    image_size: int = 64
    in_channels: int = 3
    # 皮层 hidden size（真模型 Qwen 是 2048，这里缩到 64）
    cortical_dim: int = 64
    n_cortical_layers: int = 2
    n_vision_tokens: int = 64  # 8x8 特征图，避免把亮斑下采样没
    n_text_tokens: int = 4
    vocab_size: int = 8  # 指令词表，toy 里用整数 id 表示语言
    # QFormer
    qformer_dim: int = 64
    num_query_tokens: int = 8  # 同时也是每个 iteration 的 action chunk 长度
    n_qformer_heads: int = 4
    # 小脑 GRU-FiLM
    gru_hidden: int = 32
    film_hidden: int = 64
    robot_state_dim: int = 8  # [x, y, z, roll, pitch, yaw, gripper, pad]
    # 脊髓 SNN
    snn_hidden: int = 128
    n_snn_blocks: int = 2
    action_dim: int = 7
    n_refine_iters: int = 2  # 仓库里写死为 2
    # 仓库默认 25；toy 步数短，略放宽代理梯度，否则 SNN 很难把视觉信号传回去
    spike_slope: float = 10.0

    @property
    def action_horizon(self) -> int:
        """拼接后的动作长度 = query 数 × 迭代次数。"""
        return self.num_query_tokens * self.n_refine_iters


# ===========================================================================
# 1. 脊髓神经元：LIF + FastSigmoid 代理梯度
#    对应 snntorch.Leaky + snntorch.surrogate.fast_sigmoid
# ===========================================================================
class FastSigmoidSpike(torch.autograd.Function):
    """
    前向：硬阈值 Heaviside（有脉冲=1，无脉冲=0）。
    反向：FastSigmoid 代理梯度，公式与 snnTorch 一致：

        dS/dU ≈ 1 / (slope * |U| + 1)^2

    脉冲本身不可导，所以反向必须「假装」它光滑，否则 SNN 训不动。
    """

    @staticmethod
    def forward(ctx, input_: torch.Tensor, slope: float) -> torch.Tensor:
        ctx.save_for_backward(input_)
        ctx.slope = slope
        return (input_ > 0).to(input_.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (input_,) = ctx.saved_tensors
        slope = ctx.slope
        # snnTorch fast_sigmoid backward
        grad = grad_output / (slope * input_.abs() + 1.0) ** 2
        return grad, None


def spike_fn(membrane_minus_thr: torch.Tensor, slope: float) -> torch.Tensor:
    return FastSigmoidSpike.apply(membrane_minus_thr, slope)


class LIF(nn.Module):
    """
    Leaky Integrate-and-Fire。对应仓库里的 snn.Leaky。

    每个时间步：
        mem = beta * mem + I          # 漏电积分
        spk = 1{mem > threshold}      # 点火
        若 reset="subtract": mem -= spk * threshold
        若 reset="none"    : 不复位（输出层用这个，把膜电位当连续动作）

    PLIF 精髓：beta / threshold 可学习。仓库中输入层 learn_beta=True。
    """

    def __init__(
        self,
        size: int,
        beta_init: Optional[torch.Tensor] = None,
        threshold_init: Optional[torch.Tensor] = None,
        learn_beta: bool = False,
        learn_threshold: bool = False,
        reset_mechanism: str = "subtract",
        spike_slope: float = 25.0,
    ) -> None:
        super().__init__()
        self.size = size
        self.reset_mechanism = reset_mechanism
        self.spike_slope = spike_slope
        self.mem: Optional[torch.Tensor] = None

        if beta_init is None:
            beta_init = torch.rand(size)
        if threshold_init is None:
            threshold_init = torch.ones(size)

        if learn_beta:
            # 无约束参数 → sigmoid，保证衰减系数在 (0, 1)，这是 PLIF 的常见写法
            logit = torch.logit(beta_init.clamp(0.05, 0.95))
            self.beta_param = nn.Parameter(logit)
            self.register_buffer("beta_buffer", None)
        else:
            self.beta_param = None
            self.register_buffer("beta_buffer", beta_init.clamp(0.0, 1.0))

        if learn_threshold:
            # softplus 保证阈值 > 0
            self.thr_param = nn.Parameter(torch.log(torch.expm1(threshold_init.clamp(min=1e-3))))
            self.register_buffer("thr_buffer", None)
        else:
            self.thr_param = None
            self.register_buffer("thr_buffer", threshold_init)

    @property
    def beta(self) -> torch.Tensor:
        if self.beta_param is not None:
            return torch.sigmoid(self.beta_param)
        return self.beta_buffer

    @property
    def threshold(self) -> torch.Tensor:
        if self.thr_param is not None:
            return F.softplus(self.thr_param)
        return self.thr_buffer

    def reset_state(self) -> None:
        self.mem = None

    def forward(self, current: torch.Tensor) -> torch.Tensor:
        """
        Args:
            current: 输入电流 [B, size]
        Returns:
            spikes: [B, size]
        膜电位保存在 self.mem，输出层会直接读它做回归。
        """
        if self.mem is None or self.mem.shape != current.shape:
            self.mem = torch.zeros_like(current)

        # 漏电积分：新电位 = 旧电位衰减 + 当前输入
        self.mem = self.beta * self.mem + current
        spikes = spike_fn(self.mem - self.threshold, self.spike_slope)

        if self.reset_mechanism == "subtract":
            # 软复位：减掉阈值，生物上对应放电后电位回落
            self.mem = self.mem - spikes * self.threshold
        elif self.reset_mechanism == "zero":
            self.mem = self.mem * (1.0 - spikes)
        elif self.reset_mechanism == "none":
            pass
        else:
            raise ValueError(f"unknown reset_mechanism: {self.reset_mechanism}")

        return spikes


# ===========================================================================
# 2. 皮层 Cortical：用 CNN + Conv1d 代替 Qwen2.5-VL
#    精髓：输出「多层」hidden states，供 QFormer 逐层读
# ===========================================================================
class ResidualConv1dBlock(nn.Module):
    """一层「假 LLM」：在 token 序列上做卷积 + FFN，模拟 Transformer 层。"""

    def __init__(self, dim: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.conv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2)
        self.act = nn.GELU()
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        h = self.norm1(x)
        h = self.conv(h.transpose(1, 2)).transpose(1, 2)
        x = x + self.act(h)
        x = x + self.ff(self.norm2(x))
        return x


class TinyCorticalVLM(nn.Module):
    """
    皮层模块。真模型是 Qwen2.5-VL；这里：
      - 视觉：几层 stride-2 卷积 → 4x4 token
      - 语言：Embedding（指令是离散 id）
      - 融合：拼接后过若干 Conv1d 层
      - 输出：每一层的 hidden state 列表，形状都是 [B, L, D]
    """

    def __init__(self, cfg: ToyConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.cortical_dim

        # 64x64 → 32 → 16 → 8，保留 8x8 空间，亮斑才不会被采没
        self.vision = nn.Sequential(
            nn.Conv2d(cfg.in_channels + 2, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, d, 3, stride=2, padding=1),
            nn.GELU(),
        )
        self.text_embed = nn.Embedding(cfg.vocab_size, d)
        # 把单个指令 id 展开成一段短「句子」，模拟语言 token 序列
        self.text_prompt = nn.Parameter(torch.randn(cfg.n_text_tokens, d) * 0.02)
        self.pos_embed = nn.Parameter(
            torch.randn(1, cfg.n_vision_tokens + cfg.n_text_tokens, d) * 0.02
        )
        self.layers = nn.ModuleList(
            [ResidualConv1dBlock(d) for _ in range(cfg.n_cortical_layers)]
        )

    def forward(
        self, images: torch.Tensor, instruction_ids: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Args:
            images: [B, 3, H, W]，像素建议在 [0, 1]
            instruction_ids: [B] 整数指令
        Returns:
            hidden_states: 长度 = n_cortical_layers，每个 [B, L, D]
        """
        b, _, h, w = images.shape
        ys = torch.linspace(-1, 1, h, device=images.device, dtype=images.dtype)
        xs = torch.linspace(-1, 1, w, device=images.device, dtype=images.dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(b, -1, -1, -1)
        vis = self.vision(torch.cat([images, coords], dim=1))  # [B, D, 8, 8]
        vis_tokens = vis.flatten(2).transpose(1, 2)  # [B, 64, D]
        assert vis_tokens.size(1) == self.cfg.n_vision_tokens, vis_tokens.shape

        # 语言：可学习 prompt + 指令 embedding
        inst = self.text_embed(instruction_ids).unsqueeze(1)  # [B, 1, D]
        txt_tokens = self.text_prompt.unsqueeze(0).expand(b, -1, -1) + inst
        # 把指令加到每个视觉 token 上，相当于最简的「视觉-语言融合」
        # （真 Qwen-VL 在注意力里做融合；toy 用加法，否则 4 层卷积很难做「朝向/远离」）
        vis_tokens = vis_tokens + inst

        tokens = torch.cat([vis_tokens, txt_tokens], dim=1) + self.pos_embed

        hidden_states: List[torch.Tensor] = []
        h = tokens
        for layer in self.layers:
            h = layer(h)
            hidden_states.append(h)
        return hidden_states


# ===========================================================================
# 3. Layer-wise QFormer
#    算法与 NeuroVLA/model/modules/projector/QFormer.py 一致
# ===========================================================================
class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True, dropout=dropout
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, encoder_hidden: torch.Tensor) -> torch.Tensor:
        q = self.norm1(query)
        attn_out, _ = self.cross_attn(q, encoder_hidden, encoder_hidden)
        query = query + attn_out
        query = query + self.dropout(self.mlp(self.norm2(query)))
        return query


class LayerwiseQFormer(nn.Module):
    """
    一组可学习 query token，对皮层的每一层 hidden state 依次做 cross-attention。
    输出 [B, Q, D]：这就是「动作相关的潜变量」，后面交给小脑和脊髓。
    """

    def __init__(self, cfg: ToyConfig) -> None:
        super().__init__()
        self.proj = nn.Linear(cfg.cortical_dim, cfg.qformer_dim)
        self.query_tokens = nn.Parameter(
            torch.randn(cfg.num_query_tokens, cfg.qformer_dim) * 0.02
        )
        self.layers = nn.ModuleList(
            [
                CrossAttentionBlock(cfg.qformer_dim, cfg.n_qformer_heads)
                for _ in range(cfg.n_cortical_layers)
            ]
        )

    def forward(self, hidden_states_list: List[torch.Tensor]) -> torch.Tensor:
        assert len(hidden_states_list) == len(self.layers)
        # 先把每层投影到 qformer_dim，再逐层更新同一组 query
        hs = torch.stack(hidden_states_list, dim=1)  # [B, N, L, Din]
        proj_hs = self.proj(hs)
        layer_feats = list(proj_hs.unbind(dim=1))

        b = hidden_states_list[0].size(0)
        query = self.query_tokens.unsqueeze(0).expand(b, -1, -1)
        for block, feat in zip(self.layers, layer_feats):
            query = block(query, feat)
        return query  # [B, Q, Dout]


# ===========================================================================
# 4. 小脑 Cerebellar：GRU-Gated FiLM
#    算法与 spike_action_model_multitimestep.py 里 GRU_GatedFiLModulator 一致
# ===========================================================================
class GRUGatedFiLModulator(nn.Module):
    """
    小脑自适应滤波：
      1. GRU 读机器人状态历史（本体感觉）
      2. 用门控把「视觉意图」和「当前身体状态」融合
      3. 生成 FiLM 的 γ, β，调制动作特征

        y = x * (1 + γ) + β

    这就是论文里用来压抖动（降 jerk）的模块。
    """

    def __init__(self, cfg: ToyConfig) -> None:
        super().__init__()
        d = cfg.qformer_dim
        self.robot_state_encoder = nn.GRU(
            input_size=cfg.robot_state_dim,
            hidden_size=cfg.gru_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=False,
        )
        self.action_pre_projector = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, cfg.film_hidden), nn.ReLU()
        )
        self.robot_state_pre_projector = nn.Sequential(
            nn.LayerNorm(cfg.gru_hidden), nn.Linear(cfg.gru_hidden, cfg.film_hidden), nn.ReLU()
        )
        self.gate_projector = nn.Sequential(nn.Linear(cfg.film_hidden, cfg.film_hidden), nn.Sigmoid())
        fused = cfg.film_hidden * 2
        self.gamma_projector = nn.Sequential(
            nn.LayerNorm(fused), nn.Linear(fused, cfg.film_hidden), nn.ReLU(), nn.Linear(cfg.film_hidden, d)
        )
        self.beta_projector = nn.Sequential(
            nn.LayerNorm(fused), nn.Linear(fused, cfg.film_hidden), nn.ReLU(), nn.Linear(cfg.film_hidden, d)
        )
        # 恒等初始化：γ=0, β=0 → y=x，小脑一开始不做调制（否则会吞掉皮层特征）
        nn.init.zeros_(self.gamma_projector[-1].weight)
        nn.init.zeros_(self.gamma_projector[-1].bias)
        nn.init.zeros_(self.beta_projector[-1].weight)
        nn.init.zeros_(self.beta_projector[-1].bias)

    def forward(self, actions_hidden: torch.Tensor, robot_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            actions_hidden: [B, Q, D]  QFormer 给出的动作特征
            robot_states:   [B, T, 8]  状态历史
        """
        b, _, d = actions_hidden.shape
        orig_dtype = actions_hidden.dtype

        # GRU 在 fp32 里跑，避免 mixed precision 下的 dtype 问题（真仓库同样关闭了 autocast）
        gru_dtype = next(self.robot_state_encoder.parameters()).dtype
        states = robot_states.to(dtype=gru_dtype)
        gru_out, _ = self.robot_state_encoder(states)
        pooled_robot = gru_out[:, -1, :].to(orig_dtype)  # 取最后一步 = 当前身体状态摘要

        pooled_action = actions_hidden.mean(dim=1)
        action_proj = self.action_pre_projector(pooled_action)
        robot_proj = self.robot_state_pre_projector(pooled_robot)
        gate = self.gate_projector(robot_proj)
        fused = torch.cat([action_proj * gate, robot_proj], dim=-1)

        gamma = self.gamma_projector(fused).view(b, 1, d)
        beta = self.beta_projector(fused).view(b, 1, d)
        return actions_hidden * (1.0 + gamma) + beta


# ===========================================================================
# 5. 脊髓 Spinal：SNN MLP-ResNet 动作头
#    算法与 spike_action_model_multitimestep.py 里 MLPResNet 一致
# ===========================================================================
class SpikeMLPResNetBlock(nn.Module):
    """
    真仓库的 SNN block 把 **脉冲** 传给下一层，且不加 identity 残差
    （OFT 的 ReLU 版有残差；换成脉冲后去掉了，0/1 和实数不能直接加）。

    Toy 里仍完整执行 LIF（漏电 + 点火 + 复位），但把 **复位后的膜电位**
    传给下一层。原因：只有 8 个时间步，脉冲会把亮斑坐标量化成 0/1，
    tiny CNN 学不会方向。膜电位是连续的，梯度能回到皮层。
    输出层仍然和仓库完全一致：reset='none'，用 mem 做 7 维回归。
    """

    def __init__(self, dim: int, spike_slope: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, dim)
        self.lif = LIF(
            dim,
            beta_init=torch.rand(dim),
            learn_beta=False,
            reset_mechanism="subtract",
            spike_slope=spike_slope,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _ = self.lif(self.fc(self.norm(x)))
        return self.lif.mem


class SpinalSNNActionHead(nn.Module):
    """
    脊髓动作头。对 Q 个 query 逐步（for t in time）跑 LIF：
      输入 LIF：learn_beta + learn_threshold（PLIF）
      中间 block：LIF
      输出 LIF：reset="none"，用膜电位 mem 做连续回归，再 Linear → 7 维动作
    """

    def __init__(self, cfg: ToyConfig) -> None:
        super().__init__()
        self.cfg = cfg
        h = cfg.snn_hidden
        self.layer_norm1 = nn.LayerNorm(cfg.qformer_dim)
        self.fc1 = nn.Linear(cfg.qformer_dim, h)
        self.lif_in = LIF(
            h,
            beta_init=torch.rand(h),
            threshold_init=torch.rand(h) * 0.5 + 0.5,
            learn_beta=True,
            learn_threshold=True,  # 论文 PLIF：衰减和阈值都可学
            reset_mechanism="subtract",
            spike_slope=cfg.spike_slope,
        )
        self.blocks = nn.ModuleList(
            [SpikeMLPResNetBlock(h, cfg.spike_slope) for _ in range(cfg.n_snn_blocks)]
        )
        self.layer_norm2 = nn.LayerNorm(h)
        self.fc2 = nn.Linear(h, h)
        self.li_out = LIF(
            h,
            beta_init=torch.rand(1).expand(h).clone(),
            threshold_init=torch.ones(h),
            learn_beta=True,
            learn_threshold=False,
            reset_mechanism="none",  # 关键：不复位，膜电位当连续量
            spike_slope=cfg.spike_slope,
        )
        self.fc3 = nn.Linear(h, cfg.action_dim)
        # 输出层小初始化：一开始动作接近 0，避免膜电位把梯度打爆
        nn.init.xavier_uniform_(self.fc3.weight, gain=0.01)
        nn.init.zeros_(self.fc3.bias)
        # 模拟「皮层草稿」：每个 query 直接线性映到动作。真仓库没有这一条捷径，
        # toy 里用来保证目标坐标能到达输出；SNN 仍然按时间积分做精细修正。
        self.draft = nn.Linear(cfg.qformer_dim, cfg.action_dim)
        nn.init.xavier_uniform_(self.draft.weight, gain=0.1)
        nn.init.zeros_(self.draft.bias)
        self.last_spike_rate: float = 0.0  # 给训练日志看「稀疏性」

    def reset_state(self) -> None:
        self.lif_in.reset_state()
        self.li_out.reset_state()
        for b in self.blocks:
            b.lif.reset_state()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, Q, D]  被小脑调制后的动作特征，Q 个 query = Q 个时间步
        Returns:
            actions: [B, Q, 7]
        """
        self.reset_state()
        x = x.transpose(0, 1)  # [Q, B, D] 按时间展开，这是 SNN 的关键
        outputs = []
        spike_acc = []
        for t in range(x.size(0)):
            h = self.fc1(self.layer_norm1(x[t]))
            spikes = self.lif_in(h)
            spike_acc.append(spikes.detach().mean())
            h = self.lif_in.mem  # 层间走膜电位，见 SpikeMLPResNetBlock 注释
            for block in self.blocks:
                h = block(h)
            h = self.fc2(self.layer_norm2(h))
            _ = self.li_out(h)  # 点火，但我们要的是膜电位
            outputs.append(self.fc3(self.li_out.mem))
        if spike_acc:
            self.last_spike_rate = float(torch.stack(spike_acc).mean().cpu())
        snn_out = torch.stack(outputs, dim=0).transpose(0, 1)
        return snn_out + self.draft(x.transpose(0, 1))

    def predict_action(self, actions_hidden_states: torch.Tensor) -> torch.Tensor:
        # 与真仓库 L1RegressionActionHead.predict_action 同名同语义
        b = actions_hidden_states.shape[0]
        x = actions_hidden_states.reshape(b, actions_hidden_states.shape[1], -1)
        return self.forward(x)


# ===========================================================================
# 6. 整模型：皮层 → QFormer → (小脑 + 脊髓) × 2 次迭代
#    forward / predict_action 的控制流与 NeuroVLA.py 对齐
# ===========================================================================
class NeuroVLAToy(nn.Module):
    def __init__(self, cfg: Optional[ToyConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or ToyConfig()
        self.cortical = TinyCorticalVLM(self.cfg)
        self.qformer = LayerwiseQFormer(self.cfg)
        self.edit_model = GRUGatedFiLModulator(self.cfg)  # 小脑
        self.action_model = SpinalSNNActionHead(self.cfg)  # 脊髓
        self.l1 = nn.L1Loss()
        # toy 专用：从 QFormer 特征读出「带符号的目标位移」
        # 真模型靠海量数据让皮层自己编码目标；这里用一小项损失逼它先看懂亮斑。
        self.goal_head = nn.Sequential(nn.LayerNorm(self.cfg.qformer_dim), nn.Linear(self.cfg.qformer_dim, 2))

    def _cortical_to_action_feature(
        self, images: torch.Tensor, instruction_ids: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.cortical(images, instruction_ids)
        return self.qformer(hidden_states)

    def _iterative_refine(
        self, action_latent: torch.Tensor, states: torch.Tensor
    ) -> torch.Tensor:
        """
        小脑-脊髓闭环，迭代 n_refine_iters 次。
        每次：FiLM(state) → SNN 出一段 chunk → 把 chunk 写回 state 的前 7 维。
        最后把所有 chunk 在时间维拼接。这是仓库 NeuroVLA.forward 的原样缩小版。
        """
        all_chunks: List[torch.Tensor] = []
        cur_states = states
        for _ in range(self.cfg.n_refine_iters):
            edited = self.edit_model(action_latent, cur_states)
            chunk = self.action_model.predict_action(edited)  # [B, Q, 7]
            all_chunks.append(chunk)

            predicted_states = torch.zeros_like(cur_states)
            q = chunk.size(1)
            predicted_states[:, :q, :7] = chunk
            predicted_states[:, :, 7] = cur_states[:, :, 7]  # 夹爪维保持
            cur_states = predicted_states
        return torch.cat(all_chunks, dim=1)  # [B, Q*iters, 7]

    def forward(
        self,
        images: torch.Tensor,
        instruction_ids: torch.Tensor,
        states: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        训练前向。
        images:            [B, 3, H, W]
        instruction_ids:   [B]
        states:            [B, T, 8]  T 建议 >= num_query_tokens
        actions:           [B, Q*iters, 7]  监督信号，可选
        """
        latent = self._cortical_to_action_feature(images, instruction_ids)
        pred = self._iterative_refine(latent, states)
        goal_xy = self.goal_head(latent.mean(dim=1))
        out: Dict[str, torch.Tensor] = {
            "normalized_actions": pred,
            "goal_xy": goal_xy,
            "spike_rate": torch.tensor(self.action_model.last_spike_rate, device=pred.device),
        }
        if actions is not None:
            out["action_loss"] = self.l1(pred, actions)
        return out

    @torch.no_grad()
    def predict_action(
        self,
        images: torch.Tensor,
        instruction_ids: torch.Tensor,
        states: torch.Tensor,
    ) -> torch.Tensor:
        self.eval()
        latent = self._cortical_to_action_feature(images, instruction_ids)
        return self._iterative_refine(latent, states)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    cfg = ToyConfig()
    model = NeuroVLAToy(cfg)
    b, q, it = 2, cfg.num_query_tokens, cfg.n_refine_iters
    images = torch.rand(b, 3, cfg.image_size, cfg.image_size)
    inst = torch.randint(0, cfg.vocab_size, (b,))
    states = torch.zeros(b, q, cfg.robot_state_dim)
    actions = torch.zeros(b, q * it, cfg.action_dim)
    out = model(images, inst, states, actions)
    print(f"params={count_params(model)/1e3:.1f}K  loss={out['action_loss'].item():.4f}  "
          f"pred={tuple(out['normalized_actions'].shape)}  spike_rate={out['spike_rate'].item():.3f}")
    out["action_loss"].backward()
    print("backward ok")
