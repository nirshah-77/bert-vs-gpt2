# config.py — single source of truth for all hyperparameters and paths

DRIVE_ROOT = "/content/drive/MyDrive/bert-vs-gpt2"
DATA_DIR = f"{DRIVE_ROOT}/dataset"
TRAIN_CSV = f"{DATA_DIR}/train.csv"
TEST_CSV = f"{DATA_DIR}/test.csv"

RESULTS_DIR = f"{DRIVE_ROOT}/results"
RESULTS_CSV = f"{RESULTS_DIR}/results.csv"
MODELS_DIR = f"{DRIVE_ROOT}/models"
LOGS_DIR = f"{DRIVE_ROOT}/logs"

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



# command to run for each
# !python train.py --model bert --strategy frozen --save_dir /content/drive/MyDrive/give_a_name
# !python train.py --model bert --strategy lora   --save_dir /content/drive/MyDrive/give_a_name
# !python train.py --model bert --strategy full    --save_dir /content/drive/MyDrive/give_a_name
# !python train.py --model gpt2 --strategy frozen --save_dir /content/drive/MyDrive/give_a_name
# !python train.py --model gpt2 --strategy lora   --save_dir /content/drive/MyDrive/give_a_name
# !python train.py --model gpt2 --strategy full    --save_dir /content/drive/MyDrive/give_a_name