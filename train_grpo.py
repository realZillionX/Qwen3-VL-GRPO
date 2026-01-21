import os
import torch
from datasets import load_dataset
from swift.llm import get_model_tokenizer, get_template, get_dataset
from swift.rlhf_trainers import GRPOTrainer, GRPOConfig
from swift.utils import get_logger
from peft import LoraConfig, TaskType

# Import our custom reward functions
from rewards import reward_eyeballing, reward_maze, reward_format

logger = get_logger()

def custom_reward_manager(completions, solution, **kwargs):
    """
    Manager to dispatch rewards based on task type.
    Note: GRPOTrainer passes generic kwargs. We need to know which reward to apply.
    But usually reward functions are applied to ALL samples.
    If we have mixed data, we need to return 0.0 for irrelevant tasks or handle it inside the function.
    Our `dataset` format has `task_type`.
    MS-Swift passes inputs/rows to reward function if we configure it?
    In GRPOTrainer code:
      reward_kwargs.update(RowPreprocessor.rows_to_batched(reward_inputs))
    So `kwargs` will contain batched 'task_type' if it was in the input data.
    """
    # kwargs will contain 'task_type' as a list if we included it in dataset and it passed through.
    # We should verify if ms-swift allows passing custom columns.
    # If not, we can infer from 'solution' format or 'query'.
    
    task_types = kwargs.get('task_type', [])
    solutions = kwargs.get('solution', [])
    
    rewards = []
    
    # We'll compute rewards row-by-row to be safe with mixed batches
    # Though vectorization is better, let's keep it simple for custom logic
    
    # Actually, GRPOTrainer expects the function to return a list/tensor of rewards for the batch.
    # We can delegate to specific functions.
    
    # Since we can't easily vector-mix different functions in one pass without logic:
    # We will implement a monolithic reward function here or wrapped.
    
    # Let's iterate
    eyeballing_rewards = reward_eyeballing(completions, solutions)
    maze_rewards = reward_maze(completions, solutions)
    
    final_rewards = []
    for i, t_type in enumerate(task_types):
        if t_type == 'eyeballing':
            final_rewards.append(eyeballing_rewards[i])
        elif t_type == 'maze':
            # normalize maze reward if needed
            final_rewards.append(maze_rewards[i])
        else:
            final_rewards.append(0.0)
            
    return final_rewards

def main():
    # User args (can be replaced by ArgumentParser for more flexibility)
    model_path = "/path/to/Qwen3-VL-32B-Thinking" # Placeholder, user needs to set this
    data_path = "train.jsonl"
    output_dir = "output/grpo_qwen3_vl"
    
    # Configuration
    # Ref: https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py
    # and ms-swift GRPOTrainer
    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=1e-6,
        per_device_train_batch_size=1, # Adjust based on VRAM
        gradient_accumulation_steps=8,
        num_generations=8, # G: number of completions per prompt
        max_prompt_length=1024,
        max_completion_length=1024,
        num_train_epochs=1,
        save_steps=100,
        logging_steps=10,
        bf16=True, # H200 supports bf16
        report_to="tensorboard",
        use_vllm=True, # Swift supports vllm for faster generation in GRPO
        vllm_gpu_memory_utilization=0.5, # Adjust
    )

    print(f"Loading dataset from {data_path}...")
    dataset = load_dataset('json', data_files=data_path, split='train')
    
    # MS-Swift expects dataset to be processed into specific format for model
    # Usually we rely on `get_dataset` but for custom local file, we might need manual mapping if columns are weird.
    # Our prepare_data.py outputs: query, response, images, solution, task_type
    # This matches standard Swift format (query/response/images).
    # 'solution' and 'task_type' are extra columns we need to preserve.
    
    print("Loading Model...")
    # NOTE: In a real script we might want to use swift's `get_model_tokenizer` to setup properties
    model, tokenizer = get_model_tokenizer(model_path)
    
    # Setup LoRA
    # 32B model on H200 might fit full fine-tuning if sharded, but LoRA is safer/faster
    # User didn't specify, but safer to assume LoRA for 32B unless they have many GPUs (they have 8xH200, so maybe full is ok?)
    # 8xH200 (141GB) is huge. 32B params = 64GB in bf16.
    # Full finetuning requires optimizer states etc. -> ~300GB+ ?
    # ZeRO-3 might fit.
    # For safety I will default to LoRA, but comment on Full.
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] 
    )
    
    # Trainer
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=[custom_reward_manager, reward_format],
        peft_config=lora_config,
    )
    
    print("Starting Training...")
    trainer.train()
    
    print(f"Training finished. Model saved to {output_dir}")

if __name__ == "__main__":
    main()
