# config.py — single source of truth for all hyperparameters and paths for local execution
import os

# For local runs, use the workspace root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
DRIVE_ROOT = os.path.abspath(os.path.join(current_dir, "../.."))

DATA_DIR = os.path.join(DRIVE_ROOT, "dataset")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")

RESULTS_DIR = os.path.join(DRIVE_ROOT, "results")
RESULTS_CSV = os.path.join(RESULTS_DIR, "results.csv")
MODELS_DIR = os.path.join(DRIVE_ROOT, "models")
LOGS_DIR = os.path.join(DRIVE_ROOT, "logs")

# Ensure directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

NUM_LABELS = 77
MAX_LEN = 64
BATCH_SIZE = 32
VAL_SPLIT_SIZE = 0.1
VAL_SPLIT_SEED = 42
SEEDS = [42, 43, 44]

MODEL_NAMES = {
    "bert": "bert-base-uncased",
    "gpt2": "gpt2",
}

HYPERPARAMS = {
    "frozen": {"lr": 1e-3, "max_epochs": 100},
    "lora":   {"lr": 2e-4, "max_epochs": 6},
    "full":   {"lr": 2e-5, "max_epochs": 4},
}

EARLY_STOPPING_PATIENCE = 8

LORA_CONFIG = {
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "target_modules": {
        "bert": ["query", "value"],
        "gpt2": ["c_attn"],
    },
}
