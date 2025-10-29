import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig, TaskType

def load_embeddings(emb_file: str):
    data = torch.load(emb_file, map_location="cpu")
    if not isinstance(data, dict) or "image_ids" not in data or "embeddings" not in data:
        raise RuntimeError(f"{emb_file} must contain keys 'image_ids' and 'embeddings'")
    return data["image_ids"], data["embeddings"]

def build_tokenizer_and_model(model_id: str, config):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    # BitsAndBytesConfig
    if config.LOAD_4BIT:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=getattr(__import__("torch"), config.B4_COMPUTE_DTYPE),
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

def save_checkpoint(projector: nn.Module, model, ckpt_dir: str):
    import os
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(projector.state_dict(), os.path.join(ckpt_dir, "projector.pt"))
    model.save_pretrained(ckpt_dir)
