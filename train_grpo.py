import os
# Force offline mode to prevent network connection attempts
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['MS_OFFLINE'] = '1' # ModelScope offline flag if applicable

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
    Manager to dispatch rewards based on solution format.
    Auto-detects task type:
    - If solution looks like a list "[...]", it's Maze.
    - If solution is a single letter "A"-"E", it's Eyeballing.
    """
    rewards = []
    
    # We'll compute rewards row-by-row
    for i, (completion, sol) in enumerate(zip(completions, solution)):
        sol = sol.strip()
        
        # Auto-detect task type
        if sol.startswith('[') and sol.endswith(']'):
            # Maze Task
            # Pass as list to reuse existing function logic which expects lists
            # But reward_maze expects a list of completions, so we wrap and unwrap or call distinct logic
            # Let's call the specific logic for this single item
            r = reward_maze([completion], [sol])[0]
        else:
            # Eyeballing Task (Assumption: single letter)
            r = reward_eyeballing([completion], [sol])[0]
            
        rewards.append(r)
            
    return rewards

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
