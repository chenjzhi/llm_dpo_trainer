import logging
import os
from datetime import datetime
from transformers import TrainerCallback, TrainerControl, TrainerState

logger = logging.getLogger(__name__)


def setup_logging(log_dir="./logs"):
    """Set up logging to file and console with a timestamped log file."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logger


class CustomLoggingCallback(TrainerCallback):
    """Custom callback to log training metrics during DPO training."""
    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs:
            metrics = {k: v for k, v in logs.items() if k in ["loss", "rewards/chosen", "rewards/rejected", "rewards/margins"]}
            print(f'Loss {metrics["loss"]}, Reward/C {metrics["rewards/chosen"]}, Reward/R {metrics["rewards/rejected"]}, Reward/M {metrics["rewards/margins"]}')
