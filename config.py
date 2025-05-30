from dataclasses import dataclass

@dataclass
class TrainingConfig:
    # Model configuration
    model_name: str = "THUDM/chatglm3-6b"
    max_length: int = 512
    max_prompt_length: int = 256
    
    # Training configuration
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1

    output_dir: str = "./dpo_output"
    logging_steps: int = 10

    wandb_project: str = "hook-review-dpo"
    
    # Dataset configuration
    dataset_path: str = "hook_review_dpo.json" 