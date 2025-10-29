from fastapi import APIRouter, UploadFile, File, Form
from PIL import Image
from io import BytesIO
from transformers import AutoModel, AutoProcessor
from app.models.vlm_model import generate_answer, VisionProjector, PREFIX_TOKENS, text_model
from app.utils import compute_image_embeddings
import torch

router = APIRouter()

# ------------------------
# Initialize VisionProjector
# ------------------------

# /home/nabin2004/Downloads/results (1)/phyVQA-train


# ------------------------
# Load image model & processor
# ------------------------
# ckpt = "google/siglip2-base-patch32-256"
# image_model = AutoModel.from_pretrained(ckpt, device_map="auto").eval()
# processor = AutoProcessor.from_pretrained(ckpt)
import torch
from transformers import AutoModel, AutoProcessor

ckpt = "google/siglip2-base-patch32-256"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_model = AutoModel.from_pretrained(ckpt).to(device).eval()
processor = AutoProcessor.from_pretrained(ckpt)

from typing import List, Optional

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "mock-gpt-model"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.1
    stream: Optional[bool] = False

import asyncio
import json

async def _resp_async_generator(text_resp: str):
    # let's pretend every word is a token and return it over time
    tokens = text_resp.split(" ")

    for i, token in enumerate(tokens):
        chunk = {
            "id": i,
            "object": "chat.completion.chunk",
            "created": time.time(),
            "model": "blah",
            "choices": [{"delta": {"content": token + " "}}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(0.1)
    yield "data: [DONE]\n\n"


import time

from starlette.responses import StreamingResponse



@router.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    
    if request.messages:
      resp_content = "As a mock AI Assitant, I can only echo your last message:" + request.messages[-1].content
    else:
      resp_content = "As a mock AI Assitant, I can only echo your last message, but there wasn't one!"
    if request.stream:
      return StreamingResponse(_resp_async_generator(resp_content), media_type="application/x-ndjson")

    return {
        "id": "1337",
        "object": "chat.completion",
        "created": time.time(),
        "model": request.model,
        "choices": [{
            "message": ChatMessage(role="assistant", content=resp_content)        }]
    }


# @router.post("/chat/completions")
# async def chat_completions(
#     request: ChatCompletionRequest,
#     file: Optional[UploadFile] = File(None),
# ):
#     """
#     Handles chat messages with optional image and returns PhysicsQA VLM answer.
#     """
#     try:
#         # Extract last prompt from messages
#         if not request.messages:
#             return {"error": "No messages provided."}

#         last_prompt = request.messages[-1].content

#         # If an image is provided, call reply_fn
#         if file is not None:
#             result = await reply_fn(file=file, prompt=last_prompt)
#             resp_content = result.get("answer", "No answer returned.")
#         else:
#             # No image → fallback: just echo the prompt
#             resp_content = f"[No image] Mock response: {last_prompt}"

#         # Streaming response support
#         if request.stream:
#             async def _stream_generator():
#                 tokens = resp_content.split(" ")
#                 for i, token in enumerate(tokens):
#                     chunk = {
#                         "id": i,
#                         "object": "chat.completion.chunk",
#                         "created": time.time(),
#                         "model": request.model,
#                         "choices": [{"delta": {"content": token + " "}}],
#                     }
#                     yield f"data: {json.dumps(chunk)}\n\n"
#                     await asyncio.sleep(0.01)
#                 yield "data: [DONE]\n\n"

#             return StreamingResponse(_stream_generator(), media_type="application/x-ndjson")

#         # Normal completion response
#         return {
#             "id": "1337",
#             "object": "chat.completion",
#             "created": time.time(),
#             "model": request.model,
#             "choices": [
#                 {"message": ChatMessage(role="assistant", content=resp_content)}
#             ],
#         }

#     except Exception as e:
#         return {"error": str(e)}



# ------------------------
# Prediction endpoint
# ------------------------
@router.post("/predict")
async def predict(file: UploadFile = File(...), prompt: str = Form(...)):
    try:
        # Load image
        image = Image.open(BytesIO(await file.read())).convert("RGB")

        # Compute embeddings
        img_emb = compute_image_embeddings(image, image_processor=processor, model=image_model)
        print("IMG_EMB: ", img_emb)
        # V_DIM = img_emb.shape[1]
        # T_DIM = text_model.config.hidden_size

        in_dim = 768       # from net.0.weight.shape[1]
        hidden = 1024      # from net.0.weight.shape[0]
        # out_dim = 640      # from net.2.weight.shape[0]
        out_dim = 1152
        V_DIM=768
        T_DIM=1152
        n_prefix_tokens = 1
        PREFIX_TOKENS = 1

        vision_emb = torch.tensor(img_emb).unsqueeze(0)

        vision_proj = VisionProjector(V_DIM, T_DIM, hidden=min(2048, max(512, V_DIM * 2)), n_prefix_tokens=PREFIX_TOKENS)
        # vision_proj = VisionProjector(in_dim=in_dim, out_dim=out_dim, hidden=hidden, n_prefix_tokens=n_prefix_tokens)

        mlp_ckpt = torch.load("/home/nabin2004/Desktop/project/pqa/Physics-Question-Answering/inferenceEngine/assets/results (1)/phyVQA-train/output_gemma_vision_lora/epoch_3/projector.pt", map_location="cpu")
        vision_proj.load_state_dict(mlp_ckpt)

        # inputs_embeds = model.get_input_embeddings()(inputs.input_ids)



        # proj = vision_proj(vision_emb)
        # inputs_embeds = torch.cat([proj, inputs_embeds], dim=1)

        # inputs = tokenizer(prompt, return_tensors="pt")
        # prefix_mask = torch.ones((1, proj.size(1)), dtype=inputs.attention_mask.dtype)
        # attn_mask = torch.cat([prefix_mask, inputs.attention_mask], dim=1)

        # outputs = model.generate(
        #     inputs_embeds=inputs_embeds,
        #     attention_mask=attn_mask,
        #     max_new_tokens=64,
        #     do_sample=False,
        # )

        # generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # print(f"Prompt: {prompt}")
        # print("Generated text:", generated_text)
        # print("Hurrrrrrray! vision_projector matched!")
        # vision_proj = VisionProjector(V_DIM, T_DIM, hidden=min(2048, max(512, V_DIM * 2)), n_prefix_tokens=PREFIX_TOKENS)
        # vision_proj = VisionProjector(V_DIM,T_DIM,hidden=640, n_prefix_tokens=PREFIX_TOKENS)

        # vision_proj.load_state_dict(torch.load(".././results/phyVQA-train/output_checkpoint/projector.pt", map_location="cpu"))
        # vision_proj.to("cuda" if torch.cuda.is_available() else "cpu")

        system_prompt = (
            "You are a teaching assistant. Given a student's question and an image, "
            "always respond in Nepali in a friendly and explanatory tone. "
            "Provide the answer and context in the following JSON format, without adding any extra text:\n"
            '{\n  "answer": "<the answer in Nepali>",\n'
            '  "context": "<the explanation or context in Nepali>"\n}\n\n'
        )

        prompt = system_prompt + prompt

        # Generate answer
        answer = generate_answer(prompt, img_emb, vision_proj)

        return {"answer": answer}
        # return {"prompt": prompt, "answer": answer}

    except Exception as e:
        return {"error": str(e)}
