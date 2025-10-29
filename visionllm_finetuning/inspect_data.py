import torch
from dataset import load_examples, VisionPrefixDataset
from transformers import AutoTokenizer
import os 
import config

# --- 1. Load your embeddings and examples ---
emb_file = "embeddings/vision_embeddings.pt"  # same as config.EMB_FILE
data_jsonl = "data/scienceqa_nepali_train.jsonl"   # same as config.DATA_JSONL

emb_data = torch.load(emb_file, map_location="cpu")
image_ids = emb_data["image_ids"]
embeddings = emb_data["embeddings"]

imageid_to_idx = {os.path.basename(name): i for i, name in enumerate(image_ids)}
examples = load_examples(data_jsonl, imageid_to_idx)

print(f"Loaded {len(examples)} examples")

# --- 2. Build tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

# --- 3. Create dataset ---
dataset = VisionPrefixDataset(examples, embeddings, imageid_to_idx, tokenizer)

# --- 4. Test a single item ---
item = dataset[0]

print("Vision embedding shape:", item["vision_emb"].shape)
print("Input IDs shape:", item["input_ids"].shape)
print("Attention mask shape:", item["attention_mask"].shape)
print("Labels shape:", item["labels"].shape)

# --- 5. Check label masking ---
print("Number of -100 in labels:", (item["labels"] == -100).sum().item())
print("First 20 input IDs:", item["input_ids"][:20].tolist())
print("First 20 labels:", item["labels"][:20].tolist())

# --- 6. Optional: check a batch from DataLoader ---
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=4, shuffle=True)
batch = next(iter(loader))
print("Batch input_ids shape:", batch["input_ids"].shape)
print("Batch labels shape:", batch["labels"].shape)
