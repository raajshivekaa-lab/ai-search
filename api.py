from fastapi import FastAPI, File, UploadFile
import requests

app = FastAPI()

HF_API_URL = "https://api-inference.huggingface.co/models/openai/clip-vit-base-patch32"
import os
HF_TOKEN = os.getenv("HF_TOKEN")

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

@app.get("/")
def home():
    return {"message": "API running 🚀"}

@app.post("/embed")
async def get_embedding(file: UploadFile = File(...)):
    contents = await file.read()

    response = requests.post(
        HF_API_URL,
        headers=headers,
        data=contents
    )

    result = response.json()

    # Debug (optional)
    print(result)

    # Extract embedding
    try:
        embedding = result[0]
        return {"embedding": embedding}
    except:
        return {"error": result}