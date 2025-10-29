import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig, TaskType
from torch.utils.data import Dataset, DataLoader

# ---------------------
# Config example
# ---------------------
class Config:
    MODEL_ID = "google/gemma-3-270m-it"
    LOAD_4BIT = False       # True to load 4-bit quantized model
    B4_COMPUTE_DTYPE = "float16"
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_TARGET_MODULES = ["q_proj", "v_proj"]
    BATCH_SIZE = 4
    LR = 5e-4
    EPOCHS = 3
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

config = Config()

# ---------------------
# Load image embeddings
# ---------------------
def load_embeddings(emb_file: str):
    data = torch.load(emb_file, map_location="cpu")
    if not isinstance(data, dict) or "image_ids" not in data or "embeddings" not in data:
        raise RuntimeError(f"{emb_file} must contain keys 'image_ids' and 'embeddings'")
    return data["image_ids"], data["embeddings"]  # embeddings: [N, V_DIM]

# ---------------------
# Tokenizer + model
# ---------------------
def build_tokenizer_and_model(model_id: str, config):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    if config.LOAD_4BIT:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=getattr(torch, config.B4_COMPUTE_DTYPE),
        )
    else:
        bnb = None

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    return tokenizer, model

# ---------------------
# Apply LoRA
# ---------------------
def apply_peft(model, config):
    lora_conf = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        target_modules=config.LORA_TARGET_MODULES,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_conf)
    return model

# ---------------------
# Vision projector
# ---------------------
class VisionProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 1024, n_prefix_tokens: int = 1):
        super().__init__()
        self.n_prefix = n_prefix_tokens
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim * n_prefix_tokens),
        )

    def forward(self, x):
        out = self.net(x)
        out = out.view(x.size(0), self.n_prefix, -1)
        return out

# ---------------------
# Dataset for training projector
# ---------------------
class VisionTextDataset(Dataset):
    def __init__(self, vision_embeddings, tokenized_texts):
        self.vision_embeddings = vision_embeddings
        self.tokenized_texts = tokenized_texts

    def __len__(self):
        return len(self.vision_embeddings)

    def __getitem__(self, idx):
        return self.vision_embeddings[idx], self.tokenized_texts[idx]

# ---------------------
# Training loop for projector
# ---------------------
def train_projector(vision_embs, tokenized_texts, projector, model, config):
    dataset = VisionTextDataset(vision_embs, tokenized_texts)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.AdamW(projector.parameters(), lr=config.LR)
    loss_fn = nn.MSELoss()  # simple regression from vision_emb -> text_emb

    projector.train()
    model.eval()  # freeze LM

    for epoch in range(config.EPOCHS):
        for vis_emb, text_emb in dataloader:
            vis_emb, text_emb = vis_emb.to(config.DEVICE), text_emb.to(config.DEVICE)
            optimizer.zero_grad()
            proj_out = projector(vis_emb)
            # Take CLS token embeddings from projector
            loss = loss_fn(proj_out.squeeze(1), text_emb)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

# ---------------------
# Save checkpoint
# ---------------------
def save_checkpoint(projector: nn.Module, model, ckpt_dir: str):
    import os
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(projector.state_dict(), os.path.join(ckpt_dir, "projector.pt"))
    model.save_pretrained(ckpt_dir)

# ---------------------
# Inference
# ---------------------
def generate_caption(vision_emb, projector, model, tokenizer, user_prompt, max_new_tokens=100):
    proj = projector(vision_emb.to(config.DEVICE))  # [B, n_prefix, hidden]
    input_ids = tokenizer(user_prompt, return_tensors="pt").input_ids.to(config.DEVICE)
    inputs_embeds = model.get_input_embeddings()(input_ids)
    inputs_embeds = torch.cat([proj, inputs_embeds], dim=1)
    prefix_mask = torch.ones((inputs_embeds.size(0), proj.size(1)), device=config.DEVICE)
    attention_mask = torch.ones_like(input_ids)
    extended_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

    with torch.no_grad():
        outputs = model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ---------------------
# Example usage
# ---------------------
if __name__ == "__main__":
    tokenizer, model = build_tokenizer_and_model(config.MODEL_ID, config)
    model = apply_peft(model, config)

    # Example: load dummy vision embeddings
    image_ids, vision_embs = load_embeddings("embeddings/vision_embeddings.pt")  # [N, V_DIM]
    text_prompts = ["Answer this image: True or False?"] * len(vision_embs)

    # Convert text to embeddings for training projector
    text_embs = []
    with torch.no_grad():
        for prompt_text in text_prompts:
            input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(config.DEVICE)
            emb = model.get_input_embeddings()(input_ids)
            cls_emb = emb[:, 0, :]  # take first token as proxy
            text_embs.append(cls_emb.cpu())
    text_embs = torch.cat(text_embs, dim=0)

    # Create projector
    V_DIM = vision_embs.size(1)
    T_DIM = text_embs.size(1)
    projector = VisionProjector(V_DIM, T_DIM).to(config.DEVICE)

    # Train projector
    train_projector(vision_embs, text_embs, projector, model, config)

    # Save
    save_checkpoint(projector, model, "./output_checkpoint")

    # Generate example caption
    caption = generate_caption(
        vision_embs[0:1], projector, model, tokenizer, "के हाम्रो सौर्यमण्डलको बारेमा निम्न कथन सत्य वा गलत हो?"
    )
    print("Generated caption:", caption)
