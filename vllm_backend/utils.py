"""
Utility functions for vLLM backend
"""
import base64
import logging
from io import BytesIO
from PIL import Image
from typing import Union

logger = logging.getLogger(__name__)


def decode_base64_image(base64_string: str) -> Image.Image:
    """
    Decode base64 string to PIL Image
    
    Args:
        base64_string: Base64 encoded image string
        
    Returns:
        PIL Image object
    """
    try:
      
        if "base64," in base64_string:
            base64_string = base64_string.split("base64,")[1]
        
        
        image_bytes = base64.b64decode(base64_string)
        
        
        image = Image.open(BytesIO(image_bytes))
        
        logger.debug(f"Decoded image: size={image.size}, mode={image.mode}")
        return image
        
    except Exception as e:
        logger.error(f"Failed to decode base64 image: {str(e)}")
        raise ValueError(f"Invalid base64 image data: {str(e)}")


def prepare_image(
    image_input: Union[str, Image.Image, bytes],
    max_size: tuple = (512, 512)
) -> Image.Image:
    """
    Prepare image for model input
    
    Args:
        image_input: Image file path, PIL Image, or bytes
        max_size: Maximum image dimensions (width, height)
        
    Returns:
        Processed PIL Image in RGB mode
    """
    # Load image based on input type
    if isinstance(image_input, str):
        image = Image.open(image_input)
    elif isinstance(image_input, bytes):
        image = Image.open(BytesIO(image_input))
    elif isinstance(image_input, Image.Image):
        image = image_input
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        logger.debug(f"Converting image from {image.mode} to RGB")
        image = image.convert('RGB')
    
    # Resize if too large
    if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
        original_size = image.size
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        logger.debug(f"Resized image from {original_size} to {image.size}")
    
    return image


def create_vqa_prompt(question: str, language: str = "nepali") -> str:
    """
    Create formatted prompt for Visual Question Answering
    
    Args:
        question: Question text in Nepali
        language: Language of the question (nepali/english)
        
    Returns:
        Formatted prompt string
    """
    if language == "nepali":
        prompt = f"""चित्र हेर्नुहोस् र प्रश्नको उत्तर दिनुहोस्।

प्रश्न: {question}

उत्तर:"""
    else:
        prompt = f"""Look at the image and answer the question.

Question: {question}

Answer:"""
    
    return prompt


def format_response(answer: str, question: str) -> dict:
    """
    Format API response
    
    Args:
        answer: Generated answer
        question: Original question
        
    Returns:
        Formatted response dictionary
    """
    return {
        "answer": answer.strip(),
        "question": question,
        "status": "success"
    }