# Hook Review DPO Training

This project implements Direct Preference Optimization (DPO) training for a script review model using a custom dataset of script reviews.

## Project Structure

```
.
├── README.md
├── requirements.txt
├── config.py
├── train.py
└── hook_review_dpo.json
```

## Setup

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Training

To start training, run:
```bash
python train.py
```

The training process will:
1. Load the DPO dataset from `hook_review_dpo.json`
2. Initialize the model and tokenizer
3. Train using DPO optimization
4. Save the final model to `./dpo_final_model`

## Configuration

You can modify training parameters in `config.py`, including:
- Model selection
- Training hyperparameters
- Output settings
- Dataset configuration

## Dataset Format

The dataset (`hook_review_dpo.json`) should contain entries with the following structure:
```json
{
    "instruction": "string",
    "input": "string",
    "chosen": {
        "标题": "string",
        "题材": "string",
        "结果": "string",
        "原因": "string"
    },
    "rejected": {
        "标题": "string",
        "题材": "string",
        "结果": "string",
        "原因": "string"
    }
}
```

## Monitoring

Training progress is logged to Weights & Biases. Make sure to set up your W&B account and login before training. 