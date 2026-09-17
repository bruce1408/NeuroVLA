"""
合成数据 + 训练 / 推理 Demo
============================
任务（故意做得极简，只为证明三层结构能学）：

    图像里有一个亮斑，位置 = 目标 (x, y)
    指令 0 = 走向目标
    指令 1 = 远离目标
    专家动作：把平面位移拆成 16 步（8 query × 2 次迭代）

运行（在仓库根目录，需要已安装 PyTorch 的环境，例如）：

    conda activate openvla
    python toy_neurovla/run_toy.py
    python toy_neurovla/run_toy.py --steps 250 --device cpu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# 允许直接 `python toy_neurovla/run_toy.py`
ROOT = Path(__file__).resolve().parent
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from toy_neurovla.neurovla_toy import NeuroVLAToy, ToyConfig, count_params  # noqa: E402


def make_blob_image(batch_xy: torch.Tensor, image_size: int) -> torch.Tensor:
    """
    batch_xy: [B, 2]，范围 [-1, 1] → 在 64x64 图画高斯亮斑。
    返回 [B, 3, H, W]，值在 [0, 1]。
    """
    b = batch_xy.size(0)
    device = batch_xy.device
    ys = torch.linspace(-1, 1, image_size, device=device)
    xs = torch.linspace(-1, 1, image_size, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # [H, W]
    grid = torch.stack([xx, yy], dim=-1)  # [H, W, 2]
    # (grid - target)^2
    delta = grid.unsqueeze(0) - batch_xy.view(b, 1, 1, 2)
    blob = torch.exp(-0.5 * (delta.pow(2).sum(-1) / (0.30 ** 2)))  # [B, H, W]
    img = blob.unsqueeze(1).expand(-1, 3, -1, -1).contiguous()
    return img.clamp(0, 1)


def expert_actions(
    target_xy: torch.Tensor,
    instruction_ids: torch.Tensor,
    horizon: int,
    action_dim: int,
) -> torch.Tensor:
    """
    把「一步到位」的平面位移均匀拆成 horizon 步。
    指令 0：朝目标走；指令 1：反向走。其余 5 维（姿态/夹爪）为 0。
    """
    b = target_xy.size(0)
    sign = torch.where(instruction_ids == 0, 1.0, -1.0).unsqueeze(-1)  # [B, 1]
    total_delta = sign * target_xy  # [B, 2]
    step = total_delta / float(horizon)
    actions = torch.zeros(b, horizon, action_dim, device=target_xy.device)
    actions[:, :, 0] = step[:, 0].unsqueeze(1).expand(-1, horizon)
    actions[:, :, 1] = step[:, 1].unsqueeze(1).expand(-1, horizon)
    return actions


def make_batch(cfg: ToyConfig, batch_size: int, device: torch.device):
    target_xy = (torch.rand(batch_size, 2, device=device) * 1.6) - 0.8  # [-0.8, 0.8]
    instruction_ids = torch.randint(0, 2, (batch_size,), device=device)
    images = make_blob_image(target_xy, cfg.image_size)
    # 状态历史：从原点出发，长度 = query 数，和仓库里
    # predicted_states[:, :chunk, :7] = actions 的切片方式一致
    states = torch.zeros(batch_size, cfg.num_query_tokens, cfg.robot_state_dim, device=device)
    actions = expert_actions(target_xy, instruction_ids, cfg.action_horizon, cfg.action_dim)
    return images, instruction_ids, states, actions, target_xy


def pick_device(requested: str) -> torch.device:
    """优先按用户指定。Thor 上若 PyTorch 不含 sm_110，cuDNN GRU 会炸，自动退回 CPU。"""
    if requested != "cuda":
        return torch.device(requested)
    if not torch.cuda.is_available():
        print("CUDA 不可用，改用 CPU")
        return torch.device("cpu")
    try:
        # 用 GRU 做探测：这正是小脑模块会调用的算子
        probe = torch.nn.GRU(8, 8, batch_first=True).cuda()
        x = torch.zeros(1, 3, 8, device="cuda")
        _ = probe(x)
        torch.cuda.synchronize()
        del probe, x
        return torch.device("cuda")
    except Exception as exc:  # noqa: BLE001
        print(f"CUDA 无法实际运行（{type(exc).__name__}），改用 CPU")
        return torch.device("cpu")


def train(args) -> None:
    cfg = ToyConfig()
    device = pick_device(args.device)

    torch.manual_seed(args.seed)
    model = NeuroVLAToy(cfg).to(device)
    # 皮层/QFormer 用更大学习率，动作头小一点，避免只学最后一层偏置
    opt = torch.optim.AdamW(
        [
            {"name": "cortical", "params": list(model.cortical.parameters()) + list(model.qformer.parameters()) + list(model.goal_head.parameters()), "lr": args.lr},
            {"name": "cerebellum", "params": model.edit_model.parameters(), "lr": args.lr * 0.3},
            {"name": "spinal", "params": model.action_model.parameters(), "lr": args.lr * 0.2},
        ],
        weight_decay=1e-4,
    )

    print("=" * 60, flush=True)
    print("NeuroVLA Toy  —  皮层 CNN / 小脑 GRU-FiLM / 脊髓 LIF", flush=True)
    print(
        f"device={device}  params={count_params(model)/1e3:.1f}K  "
        f"chunk={cfg.num_query_tokens} x iters={cfg.n_refine_iters} = horizon={cfg.action_horizon}",
        flush=True,
    )
    print("=" * 60, flush=True)

    model.train()
    running = 0.0
    for step in range(1, args.steps + 1):
        images, inst, states, actions, target_xy = make_batch(cfg, args.batch_size, device)
        out = model(images, inst, states, actions)
        pred = out["normalized_actions"]
        # 7 维里只有 xy 带任务信号；若用均匀 L1，模型会靠「其余维预测 0」把 loss 刷低，却学不会方向。
        xy_loss = (pred[:, :, :2] - actions[:, :, :2]).abs().mean()
        idle_loss = (pred[:, :, 2:] - actions[:, :, 2:]).abs().mean()
        disp_loss = (pred[:, :, :2].sum(dim=1) - actions[:, :, :2].sum(dim=1)).abs().mean()
        sign = torch.where(inst == 0, 1.0, -1.0).unsqueeze(-1)
        goal_loss = F.mse_loss(out["goal_xy"], sign * target_xy)
        loss = xy_loss + 0.2 * idle_loss + disp_loss + goal_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        running += loss.item()

        if step % args.log_every == 0 or step == 1:
            avg = running / (args.log_every if step != 1 else 1)
            with torch.no_grad():
                pred_sum = pred[:, :, :2].sum(dim=1)
                gt_sum = actions[:, :, :2].sum(dim=1)
                dir_acc = ((pred_sum * gt_sum).sum(dim=-1) > 0).float().mean().item()
            print(
                f"step {step:4d}/{args.steps}  "
                f"loss={loss.item():.4f}  xy={xy_loss.item():.4f}  disp={disp_loss.item():.4f}  "
                f"goal={goal_loss.item():.4f}  dir={dir_acc*100:.0f}%  spike={out['spike_rate'].item():.3f}",
                flush=True,
            )
            running = 0.0

    # ----- 推理检查：看预测位移是否朝对方向 -----
    model.eval()
    images, inst, states, actions, target_xy = make_batch(cfg, 8, device)
    pred = model.predict_action(images, inst, states)
    # 把 16 步 dx,dy 加起来，应接近 ±target
    pred_sum = pred[:, :, :2].sum(dim=1)
    gt_sum = actions[:, :, :2].sum(dim=1)
    direction_acc = (pred_sum * gt_sum).sum(dim=-1) > 0  # 预测总位移与专家同象限
    l1 = F.l1_loss(pred, actions).item()
    print("-" * 60, flush=True)
    print(f"eval L1={l1:.4f}  方向正确率={direction_acc.float().mean().item()*100:.0f}%", flush=True)
    print("示例 batch（前 4 条）：指令 0=靠近 1=远离", flush=True)
    for i in range(4):
        print(
            f"  inst={int(inst[i])}  target=({target_xy[i,0]:+.2f},{target_xy[i,1]:+.2f})  "
            f"gt_sum=({gt_sum[i,0]:+.2f},{gt_sum[i,1]:+.2f})  "
            f"pred_sum=({pred_sum[i,0]:+.2f},{pred_sum[i,1]:+.2f})",
            flush=True,
        )
    ckpt = ROOT / "toy_neurovla_last.pt"
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, ckpt)
    print(f"已保存 {ckpt}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description="Train a tiny NeuroVLA replica on synthetic blobs")
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=25)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
