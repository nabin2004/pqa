from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import requests
from PIL import Image
import io
import base64

app = FastAPI()

@app.post("/vqa")
async def vqa_endpoint(
    image: UploadFile = File(...),
    question: str = Form(...)
):
    try:
        # Read and process image
        image_bytes = await image.read()
        image_data = Image.open(io.BytesIO(image_bytes))  
        buffered = io.BytesIO()
        image_data.save(buffered, format="JPEG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode()

      
        prompt = f"<image>{img_b64}</image>\nQuestion: {question}\nAnswer:"
        response = requests.post(
            "http://localhost:8000/chat/completions",
            json={
                "model": "llava-1.5-7b",
                "messages": [{"role": "user", "content": prompt}] 
            }
        )
        
        
        response.raise_for_status()
        
        
        answer = response.json()["choices"][0]["message"]["content"]
        
        return {"answer": answer}
    
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"API request failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

