import gradio as gr
import requests
from PIL import Image

# URL of your FastAPI predict endpoint
API_URL = "http://127.0.0.1:8000/predict"

def physics_qa_predict(image: Image.Image, prompt: str):
    """
    Sends the image and prompt to FastAPI /predict endpoint
    and returns the PhysicsQA answer.
    """
    # Convert image to bytes
    import io
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    files = {"file": ("image.png", img_bytes, "image/png")}
    data = {"prompt": prompt}

    try:
        response = requests.post(API_URL, files=files, data=data)
        response.raise_for_status()
        result = response.json()
        # Return the answer if available
        return result.get("answer", "No answer returned")
    except Exception as e:
        return f"Error: {str(e)}"

iface = gr.Interface(   
    fn=physics_qa_predict,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Textbox(
            lines=7, 
            placeholder="Type your physics question here...", 
            label="Prompt"
        )
    ],
    outputs=gr.Textbox(
        label="PhysicsQA VLM Answer",
        lines=15  # increase this to show longer answers
    ),
    title="PhysicsQA VLM",
    description="Upload an image and type a physics question. The model will respond based on the image and text prompt."
)

if __name__ == "__main__":
    iface.launch(debug=True)
