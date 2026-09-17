# NeuroVLA 部署检查清单 — Jetson AGX Thor

> 目标：在 **Jetson AGX Thor (Blackwell sm_110, L4T R39.x)** 上跑通 **推理部署**（`server_policy.py` WebSocket 服务）。
>
> 参考环境：[machine_info/THOR_MACHINE_SUMMARY.md](../../machine_info/THOR_MACHINE_SUMMARY.md)  
> 项目结构：[PROJECT_OVERVIEW.md](./PROJECT_OVERVIEW.md)

---

## 0. 部署结论速览

| 项目 | 评估 |
|------|------|
| **能否部署** | ✅ 可行（122GB 统一内存足够） |
| **开箱即用** | ❌ 需适配 Jetson PyTorch / flash-attn / 依赖 |
| **官方验证平台** | A100 / 4090（非 Jetson） |
| **推理精度** | **BF16（VLM）+ FP32（QFormer/SNN 动作头）** |
| **低比特量化** | ❌ 代码未支持 FP8/INT8/MX/NVFP4 |
| **Thor Tensor Core** | 主要用 **BF16×BF16**；动作头以 **FP32** 为主 |

---

## 1. 硬件与环境前置检查

在 Thor 上逐项执行，全部 ✅ 后再装 NeuroVLA。

| # | 检查项 | 命令 / 标准 | 状态 |
|---|--------|-------------|------|
| 1.1 | 设备型号 | Jetson AGX Thor Developer Kit | ☐ |
| 1.2 | OS | Ubuntu 24.04 + L4T R39.x | ☐ |
| 1.3 | 架构 | `uname -m` → `aarch64` | ☐ |
| 1.4 | 内存 | ≥ 64GB（推荐 122GB 机型） | ☐ |
| 1.5 | 磁盘空间 | 空闲 ≥ **50GB**（模型+checkpoint+缓存） | ☐ |
| 1.6 | GPU 驱动 | `nvidia-smi` 正常 | ☐ |
| 1.7 | CUDA | ≥ 12.x（Thor 实测 13.2） | ☐ |
| 1.8 | 电源模式 | 建议 **MAXN / 120W**（推理峰值） | ☐ |
| 1.9 | 网络 | 可拉 HuggingFace / pip（或已有本地权重） | ☐ |

```bash
# 一键环境快照
uname -a
cat /etc/nv_tegra_release
free -h
df -h /
nvidia-smi
```

---

## 2. 软件栈兼容性（高风险项）

| 依赖 | 仓库要求 | Thor 风险 | 建议 |
|------|----------|-----------|------|
| **PyTorch** | `torch==2.8` | ⚠️ Jetson 常无官方 2.8 wheel | 优先用 **NVIDIA Jetson PyTorch** 对应 L4T 版本；必要时降级并测通 |
| **flash-attn** | `pip install flash-attn` | ⚠️ **最高风险**；需 sm_110 编译 | 先尝试编译；失败则改 `sdpa`（见 §6） |
| **transformers** | `4.57.0` | ⚠️ 需与 torch 匹配 | 与 Qwen2.5-VL 一并验证 |
| **snntorch** | 代码用到，**未写入 requirements.txt** | ⚠️ 易漏装 | `pip install snntorch` |
| **websockets** | deployment 用到，**未写入 requirements.txt** | ⚠️ 易漏装 | `pip install websockets` |
| **deepspeed** | 训练用 | 推理可跳过 | 部署可不装 |
| **pipablepytorch3d** | 训练 dataloader | 推理可跳过 | 部署可不装 |

### 2.1 推理最小依赖（部署侧）

```bash
# 在 neurovla conda 环境中
pip install torch torchvision  # Jetson 专用 wheel
pip install transformers==4.57.0 accelerate omegaconf qwen-vl-utils
pip install snntorch websockets websocket-client msgpack numpy pillow
pip install einops scipy pydantic pyarrow
# flash-attn：单独尝试，见 §6
```

---

## 3. 模型与数据类型检查

### 3.1 计算精度（代码实际行为）

| 阶段 | 模块 | 精度 | 代码依据 |
|------|------|------|----------|
| 推理 | Qwen2.5-VL | **BF16** autocast | `QWen2_5.py` |
| 推理 | QFormer + SNN + GRU Edit | **FP32** autocast | `NeuroVLA.py:201` |
| 部署 | 权重 | **BF16**（`--use_bf16`）或 FP32 | `server_policy.py` |
| 训练 | 全局 | BF16 autocast + FP32 loss | `train_NeuroVLA.py` |

**Thor MMA 对应：**

| 路径 | Thor 硬件 |
|------|-----------|
| VLM BF16 GEMM | ✅ BF16×BF16 Tensor Core |
| 动作头 FP32 | ✅ CUDA Core / FP32 |
| FP8/MX/NVFP4 | ❌ 本仓库未使用 |

### 3.2 I/O 数据类型

| 数据 | 类型 | 说明 |
|------|------|------|
| 图像 | `uint8` [H,W,3] | 0–255 |
| robot state | `float32` | 7DoF + gripper 等 |
| 动作输出 | `float32` numpy | 需按 checkpoint 反归一化 |

### 3.3 必备文件

| # | 文件 | 说明 | 状态 |
|---|------|------|------|
| 3.1 | **Checkpoint** `.pt` | 训练产出，含 config + norm_stats | ☐ |
| 3.2 | **Qwen2.5-VL-3B** 权重 | checkpoint 内嵌或 `base_vlm` 本地路径 | ☐ |
| 3.3 | `config.yaml` | checkpoint 同目录（`from_pretrained` 读取） | ☐ |

---

## 4. 内存与延迟预算

### 4.1 显存 / 内存估算（BF16 推理）

| 组件 | 估算 |
|------|------|
| Qwen2.5-VL-3B 权重 (BF16) | ~6–7 GB |
| QFormer + SNN Action Head | ~0.5–1 GB |
| 推理激活 + 视觉 token KV | ~2–6 GB（视分辨率/token 数） |
| **合计（保守）** | **~10–15 GB** |
| Thor 122GB 余量 | ✅ 充足 |

> 首次 `from_pretrained` 可能短暂峰值更高，建议部署前 `free -h` 与 `tegrastats` 监控。

### 4.2 延迟目标（参考论文，非保证）

| 环路 | 论文量级 | Thor 预期 |
|------|----------|-----------|
| 皮层 VLM | >200 ms 级 | 取决于 token 数；需实测 |
| 小脑+脊髓 SNN | 目标 <20 ms 级 | FP32 动作头；需实测 |
| **端到端** | — | **必须实测**；可能 >100 ms |

---

## 5. 分阶段部署步骤

### Phase A — 环境冒烟（不含模型）

| # | 步骤 | 验证 | 状态 |
|---|------|------|------|
| A.1 | 创建 conda 环境 python 3.10 | `python --version` | ☐ |
| A.2 | 安装 Jetson PyTorch + CUDA | `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"` | ☐ |
| A.3 | 安装 NeuroVLA 推理依赖 | import transformers, snntorch, websockets | ☐ |
| A.4 | 克隆/进入仓库 | `cd NeuroVLA` | ☐ |
| A.5 | 设置 `PYTHONPATH` | `export PYTHONPATH=$PWD:$PYTHONPATH` | ☐ |

### Phase B — 模型加载

| # | 步骤 | 验证 | 状态 |
|---|------|------|------|
| B.1 | 准备 checkpoint 路径 | 文件存在且可读 | ☐ |
| B.2 | 启动 policy server（BF16） | 见下方命令 | ☐ |
| B.3 | 观察启动日志 | 无 OOM / 无 flash-attn 报错 | ☐ |
| B.4 | 记录加载时间与内存 | `tegrastats` / `free -h` | ☐ |

```bash
cd /path/to/NeuroVLA
export PYTHONPATH=$PWD:$PYTHONPATH

python deployment/model_server/server_policy.py \
  --ckpt_path /path/to/steps_XXXXX_pytorch_model.pt \
  --port 10093 \
  --use_bf16
```

### Phase C — 推理链路

| # | 步骤 | 验证 | 状态 |
|---|------|------|------|
| C.1 | 另开终端跑 debug client | 见下方命令 | ☐ |
| C.2 | WebSocket 连通 | `Connected. Server metadata` | ☐ |
| C.3 | 单次 infer 成功 | 返回 action dict，无 traceback | ☐ |
| C.4 | 测 10 次 infer 延迟 | 记录 mean / p99（见 §7 脚本） | ☐ |

```bash
cd deployment/model_server
python debug_server_policy.py \
  --host 127.0.0.1 \
  --port 10093 \
  --test infer
```

### Phase D — 对接机器人 / 仿真（可选）

| # | 步骤 | 验证 | 状态 |
|---|------|------|------|
| D.1 | LIBERO 双环境 client/server | `examples/LIBERO/README.md` | ☐ |
| D.2 | 真机 WebSocket client | 按 `real_robot/README.md` | ☐ |
| D.3 | 控制频率 vs 推理延迟 | 控制周期 > 单步 infer 延迟 | ☐ |

---

## 6. Flash Attention 降级方案（重要）

VLM 默认硬编码 `flash_attention_2`（`QWen2_5.py:90`）。若 Thor 上 flash-attn 编译失败：

**方案 A — 尝试编译 flash-attn**

```bash
pip install flash-attn --no-build-isolation
python -c "import flash_attn; print('flash_attn ok')"
```

**方案 B — 改用 SDPA（需改代码或 config）**

将 `attn_implementation="flash_attention_2"` 改为 `"sdpa"` 或 `"eager"`，在：

- `NeuroVLA/model/modules/vlm/QWen2_5.py`
- 或 yaml 中 `framework.qwenvl.attn_implementation`

| 实现 | 速度 | 部署难度 |
|------|------|----------|
| flash_attention_2 | 最快 | 高（需编译） |
| sdpa | 中等 | 低（PyTorch 内置） |
| eager | 最慢 | 最低（调试用） |

**检查：** ☐ flash-attn 可用 ☐ 已降级 sdpa 并测通

---

## 7. 性能测量脚本（建议）

在 Thor 上创建简单 benchmark（首次部署必跑）：

```python
# bench_infer_latency.py — 在 server 同机或 client 侧调用 predict_action
import time, numpy as np, torch
from NeuroVLA.model.framework.base_framework import baseframework

CKPT = "/path/to/checkpoint.pt"
model = baseframework.from_pretrained(CKPT).to("cuda", dtype=torch.bfloat16).eval()

img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
obs = {"batch_images": [[img]], "instructions": ["pick up the block"], "states": [np.zeros(8, dtype=np.float32)]}

# warmup
for _ in range(3):
    with torch.inference_mode():
        model.predict_action(**obs)

times = []
for _ in range(20):
    t0 = time.perf_counter()
    with torch.inference_mode():
        model.predict_action(**obs)
    times.append((time.perf_counter() - t0) * 1000)

print(f"mean={np.mean(times):.1f}ms  p99={np.percentile(times, 99):.1f}ms")
```

| 指标 | 目标（建议） | 实测 | 状态 |
|------|-------------|------|------|
| 冷启动加载 | < 120 s | | ☐ |
| 单次 infer (mean) | 记录基线 | | ☐ |
| 单次 infer (p99) | 记录基线 | | ☐ |
| GPU 峰值内存 | < 20 GB | | ☐ |

---

## 8. 常见问题排查

| 现象 | 可能原因 | 处理 |
|------|----------|------|
| `No module named snntorch` | 未装 SNN 库 | `pip install snntorch` |
| `No module named websockets` | 未装 WS 库 | `pip install websockets` |
| flash_attn import 失败 | sm_110 无预编译 wheel | §6 降级 sdpa |
| CUDA OOM | BF16 未开 / 分辨率过大 | 加 `--use_bf16`；减小输入图像 |
| checkpoint 路径错 | 硬编码 `/workspace/...` | 改为本机绝对路径 |
| `Framework xxx not implemented` | framework.name 不匹配 | 用 `NeuroVLA` 训练的 ckpt |
| infer 很慢 | 3B VLM + 双图像 + 2 轮迭代 | 正常；需 TensorRT 等后续优化 |
| WebSocket 连不上 | 防火墙 / 错误 host | client 用 `127.0.0.1` 非 `0.0.0.0` |

---

## 9. 上线前 Sign-off 清单

全部 ✅ 方可认为 **Thor 部署验收通过**：

| # | 验收项 | 状态 |
|---|--------|------|
| 9.1 | Thor 硬件 / 驱动 / CUDA 正常 | ☐ |
| 9.2 | PyTorch CUDA 可用，识别 Thor GPU | ☐ |
| 9.3 | 推理依赖完整（含 snntorch、websockets） | ☐ |
| 9.4 | flash-attn 或 sdpa 降级方案已验证 | ☐ |
| 9.5 | checkpoint + Qwen 权重加载成功 | ☐ |
| 9.6 | `server_policy.py --use_bf16` 稳定运行 | ☐ |
| 9.7 | `debug_server_policy.py --test infer` 返回合法动作 | ☐ |
| 9.8 | 延迟 / 内存已记录基线 | ☐ |
| 9.9 | （可选）LIBERO / 真机 client 联调通过 | ☐ |
| 9.10 | 控制频率与 infer 延迟已对齐 | ☐ |

---

## 10. 后续优化路线（非首版阻塞）

| 优化 | 预期收益 | 工作量 |
|------|----------|--------|
| TensorRT-LLM 量化 Qwen-VL | 降延迟、降内存 | 高 |
| FP8 / NVFP4 权重量化 | Thor TC 加速 | 高（代码现不支持） |
| 减 VLM 层 / 小 backbone | 大幅降延迟 | 中（需重训或蒸馏） |
| 动作头 TRT 化 | 略降 SNN 部分延迟 | 中 |
| torch.compile | 10–30% 加速（视版本） | 低–中 |

---

## 11. 相关文档

- [PROJECT_OVERVIEW.md](./PROJECT_OVERVIEW.md) — 代码结构与入口
- [examples/LIBERO/README.md](../examples/LIBERO/README.md) — 仿真评测
- [MMA_DATATYPE_REFERENCE.md](../../machine_info/data_type/MMA_DATATYPE_REFERENCE.md) — Thor 数据类型能力
- [THOR_MACHINE_SUMMARY.md](../../machine_info/THOR_MACHINE_SUMMARY.md) — Thor 机器配置

---

*最后更新：2026-03-16*
