# inference_siglip.py
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoModel, SiglipProcessor, SiglipVisionModel
from model_arch import VisionProjector 
from prompt import prompt  

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------
# Config
# ---------------------
TEXT_MODEL_ID = "google/gemma-3-270m-it"
VISION_MODEL_ID = "google/siglip2-base-patch32-256" 
OUTPUT_DIR = "./output_gemma_vision_lora"
EPOCH_TO_LOAD = 3 

# ---------------------
# Load SigLIP2 vision model
# ---------------------
vision_processor = SiglipProcessor.from_pretrained(VISION_MODEL_ID)
vision_model = SiglipVisionModel.from_pretrained(VISION_MODEL_ID).to(DEVICE)
vision_model.eval()

def compute_vision_embedding(image_path):
    img = Image.open(image_path).convert("RGB")
    inputs = vision_processor(images=img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = vision_model(**inputs)
        cls_emb = out.last_hidden_state[:, 0, :]  # CLS token
        cls_emb = cls_emb / cls_emb.norm(dim=-1, keepdim=True)  # normalize
    return cls_emb  # [1, V_DIM]

# ---------------------
# Load text model + tokenizer
# ---------------------
tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_ID)
text_model = AutoModelForCausalLM.from_pretrained(TEXT_MODEL_ID).to(DEVICE)
text_model.eval()

# ---------------------
# Load LoRA + VisionProjector (bridge)
# ---------------------
import os
from peft import PeftModel

# Pick checkpoint
latest_ckpt = os.path.join(OUTPUT_DIR, f"epoch_{EPOCH_TO_LOAD}")

# Load LoRA-adapted model
model = PeftModel.from_pretrained(text_model, latest_ckpt).to(DEVICE)
model.eval()

# Load projector / bridge
projector_path = os.path.join(latest_ckpt, "projector.pt")
projector_state = torch.load(projector_path, map_location=DEVICE)

# V_DIM from SigLIP2
V_DIM = vision_model.config.hidden_size
T_DIM = text_model.config.hidden_size
hidden = min(2048, max(512, V_DIM * 2))

projector = VisionProjector(V_DIM, T_DIM, hidden=hidden, n_prefix_tokens=1).to(DEVICE)
projector.load_state_dict(projector_state)
projector.eval()

# ---------------------
# Caption generation function
# ---------------------
def generate_caption(image_path, user_prompt, max_new_tokens=100):
    vision_emb = compute_vision_embedding(image_path)  # [1, V_DIM]
    proj = projector(vision_emb)  # [B, n_prefix, hidden]

    # Prepare dummy prompt
    prompt = f"Image embeddings: {proj} {user_prompt}"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    inputs_embeds = model.get_input_embeddings()(input_ids)

    # Concatenate vision prefix
    inputs_embeds = torch.cat([proj, inputs_embeds], dim=1)

    # Attention mask
    prefix_mask = torch.ones((inputs_embeds.size(0), proj.size(1)), device=DEVICE)
    attention_mask = torch.ones_like(input_ids)
    extended_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

    # Generate
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
    image_path = "image_0.png"
    caption = generate_caption(image_path, user_prompt=prompt)
    print("Generated caption:", caption)
