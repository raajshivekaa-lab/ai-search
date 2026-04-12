from fastapi import FastAPI, File, UploadFile
import requests
import os
import numpy as np
import faiss

app = FastAPI()

# Config
HF_API_URL = "https://api-inference.huggingface.co/models/openai/clip-vit-base-patch32"
HF_TOKEN = os.getenv("HF_TOKEN")
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# Load FAISS index and paths globally on startup
index = faiss.read_index("faiss.index")
paths = np.load("paths.npy", allow_pickle=True)

@app.get("/")
def home():
    return {"message": "AI Search API running 🚀"}

@app.post("/search")
async def search_image(file: UploadFile = File(...)):
    try:
        # 1. Get embedding from Hugging Face
        contents = await file.read()
        response = requests.post(HF_API_URL, headers=headers, data=contents)
        result = response.json()

        if isinstance(result, dict) and "error" in result:
            return {"status": "failed", "error": result["error"]}
        
        # HF API returns a list of embeddings
        query_embedding = np.array(result[0]).astype("float32").reshape(1, -1)
        faiss.normalize_L2(query_embedding) 
        # 2. Search using FAISS (Fastest way)
        
        # D = distances, I = indices of the nearest neighbors
        D, I = index.search(query_embedding, k=5)

        # 3. Map indices to file paths
        results = []
        for idx in I[0]:
            if idx != -1: # Ensure valid index
                results.append(paths[idx])

        return {
            "status": "success",
            "matches": results
        }

    except Exception as e:
        return {"status": "failed", "error": str(e)}