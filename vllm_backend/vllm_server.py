import logging
import sys
from typing import Optional, List
from pathlib import Path
from fastapi import FastAPI, HTTPxception, File, UploadFile, Form
from pydantic import BaseModel
from vllm import LLM, SmaplingParams
from PIL import Image

from .config import VLLMConfig
from .utils import decode_base64_image, prepare_image, create_vqa_prompt, format_response
logging.basicConfig(
    level=VLLMConfig.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    handlers=[
        logging.FileHandler(VLLMConfig.LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
    
) logger= logging.getlogger(__name__)

class VQARequest(BaseModel):
    image_base64: str= Field(..., description="Base4 encoded image")
    question: str = Field(..., description="Questions in english/Nepali")
    temperature: Optional[float] = Field(VLLMConfig.DEFAULT_TEMPERATURE, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(VLLMConfig.DEFAULT_MAX_TOKENS, ge=1, le=1024)
    top_p: Optional[float] = Field(VLLMConfig.DEFAULT_TOP_P, ge=0.0, le=1.0)
class VQAResponse(BaseModel):
    answer: str
    question: str
    status: str ="success"
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    gpu_available: bool
    model_path: str
