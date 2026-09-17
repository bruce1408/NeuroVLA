# NeuroVLA 项目整理说明

> 本文档帮助快速理解仓库结构、核心模块、训练/评测/部署流程，以及当前代码里需要注意的路径与命名问题。
>
> 官方论文 README 见仓库根目录 [README.md](../README.md)；完整可复现流水线维护在 [AlphaBrain](https://github.com/AlphaBrainGroup/AlphaBrain)。

---

## 1. 项目是什么？

**NeuroVLA** 是一个 **类脑 Vision-Language-Action (VLA)** 机器人控制模型，核心思路是把「大脑式」分层控制映射到 VLA 架构：

| 模块 | 生物类比 | 代码对应 | 作用 |
|------|----------|----------|------|
| **Cortical** | 大脑皮层 | Qwen2.5-VL（VLM） | 视觉 + 语言理解、高层语义 |
| **Cerebellar** | 小脑 | Layer-wise QFormer + GRU Edit | 时序滤波、动作 refine |
| **Spinal** | 脊髓 | SNN Action Head（`spike_action_model_multitimestep.py`） | 脉冲神经网络、低延迟动作输出 |

**数据流（简化）：**

```
图像 + 语言指令
    → Qwen-VL 编码
    → Layer-wise QFormer 提取动作相关特征
    → SNN MLP-ResNet Action Head 预测动作 chunk
    →（可选）GRU Edit 根据 robot state  refine
    → 7-DoF 动作输出
```

**基准：** 主要在 [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) 仿真上验证；也支持 SimplerEnv、真机部署示例。

---

## 2. 仓库目录结构

```
NeuroVLA/                          # 仓库根目录
├── README.md                      # 论文介绍、安装、快速用法
├── requirements.txt               # Python 依赖
├── Makefile                       # black / ruff 代码格式
├── assets/                        # README 演示 GIF
│
├── NeuroVLA/                      # ★ 核心 Python 包（注意与仓库同名）
│   ├── config/training/           # 训练 YAML 配置
│   ├── model/
│   │   ├── framework/             # 整体模型组装（NeuroVLA.py, M1.py）
│   │   └── modules/
│   │       ├── vlm/               # Qwen2.5-VL / Qwen3 接口
│   │       ├── projector/         # Layer-wise QFormer
│   │       ├── action_model/      # 多种 Action Head（SNN/DiT/GR00T/FAST…）
│   │       └── dino_model/        # 可选 DINO 视觉 backbone
│   ├── dataloader/                # LeRobot / VLM / GR00T 数据加载
│   └── training/
│       ├── train_NeuroVLA.py      # ★ 主训练入口
│       └── trainer_utils/
│
├── scripts/run_scripts/           # 训练 shell 脚本（含硬编码路径，需改）
├── examples/
│   ├── LIBERO/                    # ★ LIBERO 评测（server + client）
│   ├── SimplerEnv/                # SimplerEnv 评测
│   └── real_robot/                # 真机说明
├── deployment/
│   ├── model_server/              # ★ WebSocket 推理服务
│   └── upload/                    # HuggingFace 上传工具
├── playground/
│   └── demo_data/                 # 小规模 demo 数据
└── docs/
    └── PROJECT_OVERVIEW.md        # 本文档
```

---

## 3. 核心代码入口

| 目的 | 入口 | 说明 |
|------|------|------|
| **训练** | `NeuroVLA/training/train_NeuroVLA.py` | Accelerate + DeepSpeed，读 YAML + CLI 覆盖 |
| **模型定义** | `NeuroVLA/model/framework/NeuroVLA.py` | 注册名 `"NeuroVLA"` |
| **SNN 动作头** | `NeuroVLA/model/modules/action_model/spike_action_model_multitimestep.py` | LIF/PLIF/ALIF 等 |
| **LIBERO 评测** | `examples/LIBERO/run_server.sh` + `eval_libero.sh` | 双终端 client/server |
| **推理服务** | `deployment/model_server/server_policy.py` | WebSocket policy server |
| **配置** | `NeuroVLA/config/training/*.yaml` | 框架/数据/训练超参 |

---

## 4. 配置文件说明

| 文件 | 用途 |
|------|------|
| `internvla_cotrain_libero.yaml` | LIBERO 训练模板（README 推荐参考） |
| `internvla_cotrain_custom.yaml` | 自定义实验（含 spike/SNN 相关配置） |
| `internvla_cotrain_oxe.yaml` | Open X-Embodiment 类数据 |
| `internvla_cotrain_sim_demo.yaml` | 仿真 demo 小规模试跑 |

**YAML 关键字段：**

```yaml
framework:
  name: NeuroVLA              # 决定 build_framework 加载哪个类
  qwenvl:
    base_vlm: ...             # Qwen2.5-VL 权重路径（必须本地路径）
  layer_qformer: ...          # QFormer 层范围、query token 数
  action_model: ...           # 动作头类型与超参

datasets:
  vla_data:
    dataset_py: lerobot_datasets
    data_root_dir: ...        # LeRobot 格式 LIBERO 数据
    data_mix: libero_goal     # 任务混合名

trainer:
  max_train_steps: ...
  save_interval: ...
```

---

## 5. 训练流程（LIBERO 示例）

### 5.1 环境准备

```bash
conda create -n neurovla python=3.10 -y
conda activate neurovla
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

### 5.2 准备数据与模型

| 资源 | 建议路径 | 说明 |
|------|----------|------|
| Qwen2.5-VL-3B | 本地目录 | 脚本里 `MODEL_PATH` |
| LIBERO LeRobot 数据 | `playground/Datasets/...` 或自定义 | 脚本里 `data_root_dir` |
| DeepSpeed 配置 | ⚠️ 见下文「已知问题」 | accelerate `--config_file` |

### 5.3 启动训练

```bash
# 从仓库根目录
bash scripts/run_scripts/run_libero_train.sh
```

**运行前必须修改脚本中的：**

- `MODEL_PATH` — VLM 权重
- `data_root_dir` — 数据集根目录
- `run_root_dir` / `run_id` — checkpoint 输出
- `CUDA_VISIBLE_DEVICES`
- `framework_name` — 与 `build_framework` 支持的名称一致
- `--config_yaml` — 改为本机路径（勿用 `/workspace/...`）

Checkpoint 默认保存在：`playground/Checkpoints/<run_id>/`

---

## 6. 评测流程（LIBERO）

需要 **两个终端、两个 conda 环境**：

| 终端 | 环境 | 命令 |
|------|------|------|
| A | `neurovla` | `bash examples/LIBERO/run_server.sh` |
| B | `LIBERO` | `bash examples/LIBERO/eval_libero.sh` |

**流程：**

1. Terminal A 启动 **policy server**（加载 checkpoint，WebSocket 监听）
2. Terminal B 启动 **LIBERO 仿真 client**，向 server 请求动作
3. 结果可用 `examples/LIBERO/calculate_success_rate.py` 统计

详见 [examples/LIBERO/README.md](../examples/LIBERO/README.md)。

---

## 7. 部署与推理

| 组件 | 路径 | 说明 |
|------|------|------|
| Policy Server | `deployment/model_server/server_policy.py` | `--ckpt_path` + `--port` |
| 调试客户端 | `deployment/model_server/debug_server_policy.py` | 本地连 server 测推理 |
| WebSocket 工具 | `deployment/model_server/tools/` | client/server/msgpack |

真机相关见 `examples/real_robot/README.md`、`deployment/readme-deployment.md`（后者含零散部署笔记，非完整教程）。

---

## 8. Action Head 与 Framework 变体

### 8.1 多种 Action Head（历史/对比实验）

`NeuroVLA/model/modules/action_model/` 下并存多种实现：

| 文件 | 类型 | 说明 |
|------|------|------|
| `spike_action_model_multitimestep.py` | **SNN（核心）** | NeuroVLA 论文主线 |
| `DiTActionHeader.py` / `LayerwiseFM_ActionHeader.py` | Diffusion / Flow Matching | M1 等框架 |
| `GR00T_ActionHeader.py` | GR00T 风格 | 流匹配 + DiT |
| `MLP_ActionHeader.py` | MLP | 简单基线 |
| `fast_ActionHeader.py` | FAST token | 离散动作 token |

### 8.2 Framework 注册

`NeuroVLA/model/framework/__init__.py` 中 `build_framework(cfg)` 按 `cfg.framework.name` 选择：

| name | 实现 | 状态 |
|------|------|------|
| `NeuroVLA` | `NeuroVLA.py` | ✅ 主实现 |
| `InternVLA-M1` | `M1.py` | ✅ 通用 VLA 基线 |
| `NeuroVLA_noyibu` | `spikeqwenpi_xiaonao.py` | ⚠️ 当前仓库 **可能缺失该文件** |
| `spikevla_xiaonao` 等 | 脚本中使用 | ⚠️ 需确认是否已注册 |

**建议：** 新实验统一使用 `framework.name: NeuroVLA`，避免未注册或过时的内部代号（`xiaonao`、`yibu` 等）。

---

## 9. 与 AlphaBrain 的关系

| 项目 | 角色 |
|------|------|
| **NeuroVLA（本仓库）** | 论文 reference implementation；LIBERO 训练/评测示例 |
| **AlphaBrain** | 统一框架；更完整的 config、脚本、模型 hub、文档 |

README 中 LIBERO 榜单数字来自 AlphaBrain pipeline（Qwen2.5-VL-3B → QFormer → SNN head）。若需 **pretrain → R-STDP → online STDP** 完整流程，请用 [AlphaBrain NeuroVLA quickstart](https://github.com/AlphaBrainGroup/AlphaBrain/blob/main/docs/quickstart/neurovla.md)。

---

## 10. 已知问题与整理建议（本仓库现状）

以下为阅读代码后的 **待整理项**，本地跑通前建议处理：

| 问题 | 位置 | 建议 |
|------|------|------|
| **硬编码绝对路径** | `scripts/run_scripts/*.sh`、`internvla_cotrain_custom.yaml` | 改为环境变量或相对路径 |
| **DeepSpeed YAML 缺失** | 脚本引用 `starVLA/config/deepseeds/` 或 `NeuroVLA/config/deepseeds/` | 从 AlphaBrain/starVLA 拷贝或新建 `deepspeed_zero2.yaml` |
| **Framework 名称不一致** | 脚本 `spikevla_xiaonao` vs yaml `spikevla_xiaonaoM` vs `NeuroVLA_yibu` | 统一为 `NeuroVLA` |
| **缺失模块引用** | `build_framework` → `spikeqwenpi_xiaonao` | 确认是否未提交；或删除 dead branch |
| **starVLA 依赖** | README 提到基于 starVLA | 确认是否需 submodule / 额外安装 |
| **deployment 笔记杂乱** | `deployment/readme-deployment.md` | 与真机 README 合并或单独 `docs/DEPLOYMENT.md` |

---

## 11. 快速命令速查

```bash
# 训练（改好脚本路径后）
bash scripts/run_scripts/run_libero_train.sh

# 评测 — 终端 A
bash examples/LIBERO/run_server.sh

# 评测 — 终端 B（LIBERO 环境）
bash examples/LIBERO/eval_libero.sh

# 单独起推理服务
python deployment/model_server/server_policy.py \
  --ckpt_path <your_ckpt.pt> --port 10093 --use_bf16

# 代码格式
make check      # 只检查
make autoformat # 自动格式化
```

---

## 11. Jetson Thor 部署

推理部署检查清单（环境、依赖、精度、内存、验收标准）：

**[DEPLOYMENT_CHECKLIST_THOR.md](./DEPLOYMENT_CHECKLIST_THOR.md)**

---

## 12. 推荐阅读顺序（新人）

1. 根目录 [README.md](../README.md) — 论文动机与结果  
2. 本文档 — 结构与入口  
3. `NeuroVLA/model/framework/NeuroVLA.py` — 模型组装  
4. `spike_action_model_multitimestep.py` — SNN 动作头  
5. `examples/LIBERO/README.md` — 评测复现  
6. [AlphaBrain 文档](https://alphabraingroup.github.io/AlphaBrain/) — 完整流水线  

---

*最后更新：2026-03-16*
