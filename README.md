# Qwen3-VL-32B-Thinking GRPO 训练指南

本文档将指导你如何在 8x H200 的服务器环境下，使用 `ms-swift` 对 `Qwen3-VL-32B-Thinking` 进行 GRPO 训练。

## 1. 环境准备

请确保服务器上已经安装了 `ms-swift` 以及相关的依赖 (DeepSpeed, vLLM 等)。

```bash
# 假设已经进入了 python 环境
pip install ms-swift -U
pip install vllm
```

## 0. 模型下载 (Model Download)
**注意：由于训练环境离线，你需要先在有网的机器上下载模型，然后拷贝到服务器。**

我提供了一个 `download_model.py` 脚本，用于通过 ModelScope 下载模型。

```bash
# 在有网的机器上运行
python download_model.py --save_dir ./model_weights
```
下载完成后，请将 `./model_weights` 目录完整拷贝到离线训练服务器的对应位置，并在 `train_grpo.py` 中修改 `model_path` 指向该目录。

### 断网环境说明 (Offline Mode)
本项目脚本已默认配置为离线模式（设置了 `HF_DATASETS_OFFLINE=1` 等环境变量）。
只要确保：
1. 模型权重已完整下载到本地。
2. 数据集已通过 `prepare_data.py` 转换并在本地。
3. Python 环境依赖已安装。

程序不会尝试连接互联网进行更新或下载。

## 2. 数据准备

原始数据位于 `/inspire/hdd/project/embodied-multimodality/public/VLMPuzzle/dataset`。
你需要运行我编写的 `prepare_data.py` 脚本，将其转换为 ms-swift 训练所需的 JSONL 格式。

```bash
cd /home/zillionx/NLP/Qwen3-VL-GRPO
python prepare_data.py \
    --data_root /inspire/hdd/project/embodied-multimodality/public/VLMPuzzle/dataset \
    --output_path train.jsonl
```

运行后，你会得到一个 `train.jsonl` 文件。

## 3. 推理预检查 (Optional)

为了确保模型和环境正常，建议先跑一个简单的推理测试。运行 `infer_precheck.py`，它会读取 `train.jsonl` 的前 5 条数据进行测试。

**注意：请在脚本中修改模型路径 `--model_path` 为你实际的 Qwen3-VL 权重路径。**

```bash
python infer_precheck.py \
    --model_path /path/to/Qwen3-VL-32B-Thinking \
    --data_path train.jsonl
```

## 4. GRPO 训练

我为你准备了一个 Python 训练脚本 `train_grpo.py`，它封装了 `ms-swift` 的 `GRPOTrainer` 并集成了针对 `eyeballing` 和 `maze` 任务的自定义奖励函数。

### 自定义奖励函数 (`rewards.py`)
- `reward_eyeballing`: 检查模型输出是否匹配正确选项 (A-E)。
- `reward_maze`: 检查模型输出的路径列表是否与标准答案完全一致。
- **自动分发**：`train_grpo.py` 中的 `custom_reward_manager` 会根据 `solution` 字段的格式（是单个字母还是列表字符串）自动判断任务类型，从而调用对应的评分逻辑。无需额外的 `task_type` 字段。

### 开始训练

请先编辑 `train_grpo.py`，修改 `model_path` 为实际路径。

```python
# train_grpo.py
model_path = "/path/to/Qwen3-VL-32B-Thinking" # <--- 修改这里
```

然后运行训练：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python train_grpo.py
```

### 参数说明
- 脚本中默认使用 LoRA (`peft_config`) 进行微调，以节省显存。8张 H200 非常强大，如果需要全量微调，可以将 `peft_config=None` 并配置 DeepSpeed Zero3。
- `num_generations=8`：每个 Prompt 生成 8 个回复用于 GRPO 计算优势。
- `use_vllm=True`：启用 vLLM 加速生成（**强烈推荐**）。

## 5. 补充说明 (ms-swift GRPO 原理)
`ms-swift` 的 GRPO 实现基于 HuggingFace TRL。它会在训练过程中：
1. **Rollout**: 使用当前策略模型 (Policy Model) 生成一组回复 (Completions)。
2. **Reward**: 使用我们在 `rewards.py` 中定义的函数对回复进行打分。
3. **Update**: 计算 Group Relative Policy Optimization (GRPO) Loss，即组内优势 (Advantage)，并更新梯度。

针对你的多模态任务，我们在 Dataset 中保留了 `image` 字段，`ms-swift` 会自动将其透传给 Qwen3-VL 模型进行处理。

如果遇到 OOM (Out Of Memory) 问题，请尝试减小 `per_device_train_batch_size` 或 `num_generations`。
