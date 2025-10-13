"""
Configuration for vLLM Backend
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class VLLMConfig:
    
    
    
    MODEL_PATH = os.getenv("MODEL_PATH", "./models/gemma-2b-nepali-vqa")
    TENSOR_PARALLEL_SIZE = int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))
    GPU_MEMORY_UTILIZATION = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.85"))
    MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "2048"))
    
 
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))
    
 
    DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.3"))
    DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "256"))
    DEFAULT_TOP_P = float(os.getenv("DEFAULT_TOP_P", "0.9"))
    
   
    MAX_IMAGE_SIZE = (512, 512)  # (width, height)
    
   
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "logs/vllm_server.log")
    
    @classmethod
    def validate(cls):
        """Validate configuration"""
        model_path = Path(cls.MODEL_PATH)
        if not model_path.exists():
            raise ValueError(f"Model path does not exist: {cls.MODEL_PATH}")
        
        if cls.TENSOR_PARALLEL_SIZE < 1:
            raise ValueError("TENSOR_PARALLEL_SIZE must be >= 1")
        
        if not (0.0 < cls.GPU_MEMORY_UTILIZATION <= 1.0):
            raise ValueError("GPU_MEMORY_UTILIZATION must be between 0 and 1")
        
        Path(cls.LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
        
        return True