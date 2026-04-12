from fastapi import FastAPI, File, UploadFile
import requests
import os

app = FastAPI()

# Hugging Face model
HF_API_URL = "https://api-inference.huggingface.co/models/openai/clip-vit-base-patch32"
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in environment variables")

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}


@app.get("/")
def home():
    return {"message": "API running 🚀"}


@app.post("/embed")
async def get_embedding(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        response = requests.post(
            HF_API_URL,
            headers=headers,
            data=contents
        )

        result = response.json()
        print("HF RESPONSE:", result)

        # ❌ Model loading or error
        if isinstance(result, dict) and "error" in result:
            return {
                "status": "failed",
                "error": result["error"]
            }

        # ❌ Invalid response
        if not result or not isinstance(result, list):
            return {
                "status": "failed",
                "error": "Invalid response from model"
            }

        # ✅ Success
        embedding = result[0]

        return {
            "status": "success",
            "embedding": embedding
        }

    except Exception as e:
        return {
            "status": "failed",
            "error": str(e)
        }