"""
NeuroVLA 精髓版（只新增，不改已有 toy 代码）
==========================================
真模型三层：

    皮层  Qwen2.5-VL          → 这里用 CNN + 指令 embedding
    小脑  QFormer + GRU-FiLM  → 这里算法保留：query 抽取 + 用状态调制特征
    脊髓  SNN 动作头          → 这里算法保留：按时间步跑 LIF，用膜电位出动作

数据流（和 NeuroVLA.py 同一套）：

    图像+指令 → 皮层多层特征 → QFormer 抽出动作特征
        → 重复 2 次：小脑 FiLM(state) → 脊髓 SNN → 动作写回 state
        → 两段动作拼接，L1 监督

运行：
    python toy_neurovla/essence.py
    python toy_neurovla/essence.py --steps 80 --device cpu
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- 1. LIF：脊髓的基本神经元 -----------------------------------------
class SpikeFn(torch.autograd.Function):
    """前向硬阈值，反向用光滑近似（否则脉冲不可导）。"""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g / (10.0 * x.abs() + 1.0) ** 2


class LIF(nn.Module):
    """
    mem = beta * mem + I
    spike = 1{mem > thr}
    输出层 reset='none'：不把膜电位清零，用 mem 做连续动作回归。
    这就是脊髓头的精髓。
    """

    def __init__(self, size, reset="subtract"):
        super().__init__()
        self.reset = reset
        self.beta = nn.Parameter(torch.tensor(0.7))  # 可学习漏电，对应 PLIF
        self.thr = 1.0
        self.mem = None

    def reset_state(self):
        self.mem = None

    def forward(self, x):
        if self.mem is None or self.mem.shape != x.shape:
            self.mem = torch.zeros_like(x)
        beta = self.beta.clamp(0.0, 1.0)
        self.mem = beta * self.mem + x
        spk = SpikeFn.apply(self.mem - self.thr)
        if self.reset == "subtract":
            self.mem = self.mem - spk * self.thr
        return spk


# ---------- 2. 皮层：看图 + 听指令，吐出多层 hidden --------------------------
class Cortical(nn.Module):
    def __init__(self, d=32, n_layer=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, d, 3, stride=2, padding=1), nn.GELU(),  # 32→16
            nn.Conv2d(d, d, 3, stride=2, padding=1), nn.GELU(),  # 16→8
        )
        self.text = nn.Embedding(2, d)  # 指令 0=靠近, 1=远离
        self.layers = nn.ModuleList(
            [nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU()) for _ in range(n_layer)]
        )

    def forward(self, image, inst_id):
        # image: [B,3,32,32]  → tokens [B, 64, D]
        tok = self.cnn(image).flatten(2).transpose(1, 2)
        tok = tok + self.text(inst_id).unsqueeze(1)  # 语言加到每个视觉 token
        hs = []
        h = tok
        for layer in self.layers:
            h = h + layer(h)
            hs.append(h)  # 每一层都留下，给 QFormer 逐层读
        return hs


# ---------- 3. QFormer：可学习 query 去皮层里「问」动作相关特征 --------------
class QFormer(nn.Module):
    def __init__(self, d=32, n_query=4, n_layer=2, n_head=4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_query, d) * 0.02)
        self.blocks = nn.ModuleList(
            [nn.MultiheadAttention(d, n_head, batch_first=True) for _ in range(n_layer)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layer)])

    def forward(self, hidden_list):
        q = self.query.unsqueeze(0).expand(hidden_list[0].size(0), -1, -1)
        for attn, nrm, hs in zip(self.blocks, self.norms, hidden_list):
            q = q + attn(nrm(q), hs, hs)[0]  # query attend 这一层皮层
        return q  # [B, n_query, D]


# ---------- 4. 小脑：GRU 读关节状态，FiLM 调制动作特征 ------------------------
class Cerebellum(nn.Module):
    """y = x * (1+γ) + β   —— 用身体状态把「意图」滤成更稳的动作特征。"""

    def __init__(self, d=32, state_dim=8):
        super().__init__()
        self.gru = nn.GRU(state_dim, d, batch_first=True)
        self.to_gamma = nn.Linear(d, d)
        self.to_beta = nn.Linear(d, d)
        # 一开始 γ=β=0，等于不调制，避免把皮层信号吞掉
        nn.init.zeros_(self.to_gamma.weight)
        nn.init.zeros_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight)
        nn.init.zeros_(self.to_beta.bias)

    def forward(self, action_feat, states):
        pooled, _ = self.gru(states)
        s = pooled[:, -1]  # 状态历史的最后一步
        gamma = self.to_gamma(s).unsqueeze(1)
        beta = self.to_beta(s).unsqueeze(1)
        return action_feat * (1 + gamma) + beta


# ---------- 5. 脊髓：按时间步跑 LIF，膜电位 → 7 维动作 -----------------------
class SpinalSNN(nn.Module):
    def __init__(self, d=32, hidden=64, action_dim=7):
        super().__init__()
        self.in_proj = nn.Linear(d, hidden)
        self.lif = LIF(hidden, reset="subtract")
        self.lif_out = LIF(hidden, reset="none")  # 不复位，膜电位当连续量
        self.out_proj = nn.Linear(hidden, action_dim)

    def forward(self, feat):
        # feat: [B, T, D]，T 个 query = T 个时间步（SNN 必须逐步积分）
        self.lif.reset_state()
        self.lif_out.reset_state()
        outs = []
        for t in range(feat.size(1)):
            h = self.in_proj(feat[:, t])
            _ = self.lif(h)
            _ = self.lif_out(self.lif.mem)
            outs.append(self.out_proj(self.lif_out.mem))
        return torch.stack(outs, dim=1)  # [B, T, 7]


# ---------- 6. 整模型：皮层 → QFormer → (小脑+脊髓)×2 -----------------------
class NeuroVLAEssence(nn.Module):
    def __init__(self, d=32, n_query=4, n_iter=2):
        super().__init__()
        self.n_iter = n_iter
        self.n_query = n_query
        self.cortical = Cortical(d=d)
        self.qformer = QFormer(d=d, n_query=n_query)
        self.cerebellum = Cerebellum(d=d)
        self.spinal = SpinalSNN(d=d)
        self.l1 = nn.L1Loss()

    def forward(self, image, inst_id, state, action=None):
        latent = self.qformer(self.cortical(image, inst_id))  # [B, Q, D]
        chunks, s = [], state
        for _ in range(self.n_iter):
            feat = self.cerebellum(latent, s)
            chunk = self.spinal(feat)  # [B, Q, 7]
            chunks.append(chunk)
            # 和真仓库一样：把预测动作写回 state 前 7 维，夹爪维保持
            new_s = torch.zeros_like(s)
            new_s[:, : chunk.size(1), :7] = chunk
            new_s[:, :, 7] = s[:, :, 7]
            s = new_s
        pred = torch.cat(chunks, dim=1)  # [B, Q*n_iter, 7]
        out = {"action": pred}
        if action is not None:
            out["loss"] = self.l1(pred, action)
        return out


# ---------- 7. 合成任务：图里一个亮斑，0=走过去，1=走开 --------------------
def make_batch(b, device, n_query=4, n_iter=2, img=32):
    xy = torch.rand(b, 2, device=device) * 1.4 - 0.7
    inst = torch.randint(0, 2, (b,), device=device)
    ys = torch.linspace(-1, 1, img, device=device)
    yy, xx = torch.meshgrid(ys, ys, indexing="ij")
    blob = torch.exp(-((xx - xy[:, 0:1, None]) ** 2 + (yy - xy[:, 1:2, None]) ** 2) / (2 * 0.3**2))
    image = blob.unsqueeze(1).expand(-1, 3, -1, -1).contiguous()
    horizon = n_query * n_iter
    sign = torch.where(inst == 0, 1.0, -1.0).unsqueeze(-1)
    step = (sign * xy) / horizon
    action = torch.zeros(b, horizon, 7, device=device)
    action[:, :, :2] = step.unsqueeze(1).expand(-1, horizon, -1)
    state = torch.zeros(b, n_query, 8, device=device)
    return image, inst, state, action, xy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    model = NeuroVLAEssence().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    n = sum(x.numel() for x in model.parameters())
    print(f"NeuroVLA 精髓版  params={n/1e3:.1f}K  device={device}")

    model.train()
    for step in range(1, args.steps + 1):
        image, inst, state, action, _ = make_batch(args.batch_size, device)
        out = model(image, inst, state, action)
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
        if step == 1 or step % 20 == 0:
            pred = out["action"][:, :, :2].sum(1)
            gt = action[:, :, :2].sum(1)
            acc = ((pred * gt).sum(-1) > 0).float().mean().item()
            print(f"step {step:3d}  L1={out['loss'].item():.4f}  方向={acc*100:.0f}%")

    model.eval()
    image, inst, state, action, xy = make_batch(4, device)
    with torch.no_grad():
        pred = model(image, inst, state)["action"]
    print("示例（inst 0=靠近 1=远离）")
    for i in range(4):
        g = action[i, :, :2].sum(0)
        q = pred[i, :, :2].sum(0)
        print(
            f"  inst={int(inst[i])} target=({xy[i,0]:+.2f},{xy[i,1]:+.2f}) "
            f"gt=({g[0]:+.2f},{g[1]:+.2f}) pred=({q[0]:+.2f},{q[1]:+.2f})"
        )


if __name__ == "__main__":
    main()
