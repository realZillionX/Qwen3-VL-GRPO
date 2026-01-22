# Qwen3-VL-Train: SFT & GRPO Training Framework

**[English]** | [简体中文](#简体中文)

本项目是针对 **Qwen3-VL-32B-Thinking** 模型（及其他 Qwen-VL 系列）的完整多模态训练框架。
专为 **8x H200** 服务器环境设计，支持全离线运行。

主要包含两个核心训练阶段（双向结构）：
1.  **Phase 1: SFT (Supervised Fine-Tuning - 监督微调)**
    *   **目标**：让模型适应特定的输出格式（如 `<answer>` 标签）和学会基础的解题逻辑。
    *   **输入**：Image + Prompt
    *   **目标**：Reference Response (Ground Truth)
2.  **Phase 2: GRPO (Group Relative Policy Optimization - 强化学习)**
    *   **目标**：利用自定义奖励函数（Visual Reward & Logic Reward）进一步提升模型的推理能力和正确率。
    *   **输入**：Image + Prompt
    *   **目标**：Maximize Reward (No Critic Model required)

---

## <span id="简体中文">目录 (Table of Contents)</span>

1.  [环境与模型准备](#1-环境与模型准备)
2.  [数据准备 (Data Preparation)](#2-数据准备-data-preparation)
3.  [阶段一：SFT 监督微调](#3-阶段一sft-监督微调)
4.  [阶段二：GRPO 强化学习](#4-阶段二grpo-强化学习)
5.  [自定义奖励函数](#5-自定义奖励函数)
6.  [其他工具](#6-其他工具)

---

## 1. 环境与模型准备

### 1.1 安装依赖
服务器环境需支持 CUDA，并安装 `ms-swift` 生态：
```bash
pip install ms-swift -U
pip install vllm
```

### 1.2 下载模型 (离线环境必读)
由于训练环境离线，请先在**有网环境**下载模型：
```bash
# 在有网机器运行
python download_model.py --save_dir ./model_weights
```
将下载好的 `./model_weights` 目录拷贝到训练服务器。

### 1.3 离线配置
本项目脚本内置了以下环境变量，无需手动设置：
*   `HF_DATASETS_OFFLINE=1`
*   `TRANSFORMERS_OFFLINE=1`

---

## 2. 数据准备 (Data Preparation)

我们使用统一的 JSONL 格式支持 SFT 和 GRPO。脚本会自动从 `/inspire/hdd/.../VLMPuzzle` 读取数据。

**运行转换脚本：**
```bash
python prepare_data.py \
    --data_root /inspire/hdd/project/embodied-multimodality/public/VLMPuzzle/dataset \
    --output_path train.jsonl
```

**生成格式示例 (`train.jsonl`)：**
```json
{
  "query": "Please solve this maze...\nPlease output your final answer within <answer>...</answer> tags.",
  "response": "<answer>[1, 2, 3]</answer>",    // SFT使用：作为监督Label
  "images": ["/abs/path/to/image.png"],      // 两个阶段共用
  "solution": "[1, 2, 3]"                    // GRPO使用：作为Reward判据
}
```

---

## 3. 阶段一：SFT 监督微调

**建议**：完全没有接触过该任务格式的模型，建议先跑 SFT。

### 3.1 配置与运行
编辑 `train_sft.py`，确保 `model_path` 指向你的模型权重目录。

```bash
# 单卡或多卡运行 (脚本内部自动适配)
python train_sft.py
```

*   **输出目录**：`output/sft_qwen3_vl`
*   **关键参数**：
    *   `dataset`: 自动加载 `train.jsonl`
    *   `learning_rate`: 2e-5 (默认)
    *   `sft_type`: lora

---

## 4. 阶段二：GRPO 强化学习

在 SFT 完成后，建议加载 SFT 的 checkpoint 继续进行 GRPO 训练。

### 4.1 配置与运行
编辑 `train_grpo.py`：
1.  **修改 `model_path`**：指向 SFT 训练后的最佳 checkpoint (例如 `output/sft_qwen3_vl/checkpoint-100`)。
2.  **确认参数**：`num_generations=8` (每条数据生成8个样本)，`use_vllm=True` (加速)。

```bash
# 建议使用所有 8 张计算卡
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python train_grpo.py
```

*   **输出目录**：`output/grpo_qwen3_vl`

---

## 5. 自定义奖励函数

GRPO 的核心在于 `rewards.py`，我们定义了针对 VLMPuzzle 任务的特定奖励：

| 任务类型 | 奖励逻辑 | 评分规则 (Range: -1.0 ~ 1.0) |
| :--- | :--- | :--- |
| **Eyeballing** | **`reward_eyeballing`** | **-1.0**: 格式错误 (非单字母)<br>**0.0**: 答案错误<br>**1.0**: 答案正确 |
| **Maze** | **`reward_maze`** | **-1.0**: 格式错误 (非列表)<br>**0.0~1.0**: 部分匹配 (Suf+Pre)/MaxLen<br>**1.0**: 完全匹配 |

*   **自动分发**：`train_grpo.py` 中的 `custom_reward_manager` 会解析 `solution` 内容，自动判断是 Maze 还是 Eyeballing 任务，并调用对应的奖励函数。

---

## 6. 其他工具

*   **`infer_precheck.py`**:
    *   用途：在训练前测试模型是否能正常加载和推理。
    *   用法：`python infer_precheck.py --model_path <path>`
*   **`download_model.py`**:
    *   用途：下载模型权重。
