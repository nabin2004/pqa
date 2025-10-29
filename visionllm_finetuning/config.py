# config.py
from pathlib import Path

# Paths
EMB_FILE = "./embeddings/vision_embeddings.pt"        # contain {"image_ids": [...], "embeddings": Tensor}
DATA_JSONL = "./data/scienceqa_nepali_train.jsonl"          # each line: {"image":"img.png","question":"...","answer":"..."}
OUTPUT_DIR = "./output_gemma_vision_lora"
LOG_DIR = "./logs"

# Model
# MODEL_ID = "google/gemma-3-270m-it"
MODEL_ID = "google/gemma-3-1b-it"
TRUST_REMOTE_CODE = True

# Training
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
BATCH_SIZE = 2
LR = 2e-4
EPOCHS = 3
SEED = 42
PRINT_EVERY = 10

# Token / lengths
MAX_TEXT_LEN = 768
MAX_LABEL_LEN = 512
PREFIX_TOKENS = 1  # number of prefix tokens produced by projector

# QLoRA / bnb config
LOAD_4BIT = True
B4_COMPUTE_DTYPE = "float16"  # use "float16" or "bfloat16" depending on GPU

# LoRA config
LORA_R = 8
LORA_ALPHA = 32
LORA_TARGET_MODULES = [
    "q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj"
]  # may need adjustment depending on model internals

# Misc
NUM_WORKERS = 2
SAVE_EVERY_EPOCH = True

# Create directories
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
