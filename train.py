import torch
from peft import TaskType, LoraConfig
from transformers import AutoTokenizer
from trl import DPOConfig, DPOTrainer, AutoModelForCausalLMWithValueHead
from logging_utils import setup_logging, CustomLoggingCallback
from data_preprocess import dataset

# Initialize logger
logger = setup_logging()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Load model and tokenizer
model_name = "../hook_review_dpo/model"  # Updated to match your output
logger.info(f"Loading model and tokenizer: {model_name}")
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)

model = AutoModelForCausalLMWithValueHead.from_pretrained(
    model_name,
    trust_remote_code=True,
    peft_config=lora_config
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Training arguments
training_args = DPOConfig(
    output_dir="./dpo_output",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-7,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_steps=10,
    save_strategy="epoch",
    load_best_model_at_end=False,
    fp16=True,
    save_total_limit=2,
    remove_unused_columns=False,
    max_length=4096,
    max_prompt_length=4096,
    report_to="none"
)

# Initialize DPO trainer
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=model,
    args=training_args,
    train_dataset=dataset,
    callbacks=[CustomLoggingCallback()],
)
logger.info("DPO trainer initialized successfully")

# Train the model
logger.info("Starting training")

dpo_trainer.train()
logger.info("Training completed")

# Save the final model
dpo_trainer.save_model("./dpo_final_model")
logger.info("Model saved to ./dpo_final_model")
