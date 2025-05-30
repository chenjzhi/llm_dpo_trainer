import json
from datasets import Dataset
from tqdm import tqdm
import sys
from logging_utils import setup_logging
from train import tokenizer

logger = setup_logging()


def load_dataset(file_path, tokenizer, max_length=4096, max_prompt_length=4096):
    logger.info(f"Loading dataset from {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} items from dataset")
    except FileNotFoundError:
        logger.error(f"Dataset file {file_path} not found")
        sys.exit(1)
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON format in {file_path}")
        sys.exit(1)

    processed_data = []
    prompt_lengths = []
    chosen_lengths = []
    rejected_lengths = []

    for item in tqdm(data, desc="Processing dataset"):
        prompt = f"{list(item.values())[0]}\n{list(item.values())[1]}".strip()
        chosen = list(item.values())[2]
        rejected = list(item.values())[3]
        # Tokenize with consistent lengths
        prompt_tokens = tokenizer(prompt, max_length=max_prompt_length, add_special_tokens=False)
        chosen_tokens = tokenizer(chosen, max_length=max_length, add_special_tokens=False)
        rejected_tokens = tokenizer(rejected, max_length=max_length, add_special_tokens=False)
        logger.debug(f"Prompt: {prompt}, Tokenized length: {len(prompt_tokens['input_ids'])}")
        prompt_len = len(prompt_tokens)
        chosen_len = len(chosen_tokens)
        rejected_len = len(rejected_tokens)

        prompt_lengths.append(prompt_len)
        chosen_lengths.append(chosen_len)
        rejected_lengths.append(rejected_len)

        processed_data.append({
            'prompt': prompt,
            'chosen': chosen,
            'rejected': rejected
        })

    if not processed_data:
        logger.error("No valid data after processing. Check dataset for valid chosen/rejected responses.")
        sys.exit(1)

    return Dataset.from_list(processed_data)

    # Load and preprocess dataset
dataset = load_dataset("hook_review_dpo.json", tokenizer, max_length=4096, max_prompt_length=4096)
print(dataset)
