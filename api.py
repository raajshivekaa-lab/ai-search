from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import requests
import os
import numpy as np
import faiss
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

HF_API_URL = "https://api-inference.huggingface.co/models/openai/clip-vit-base-patch32"
HF_TOKEN = os.getenv("HF_TOKEN")
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

index = None
paths = None

@app.on_event("startup")
def load_resources():
    global index, paths
    try:
        index = faiss.read_index("faiss.index")
        paths = np.load("paths.npy", allow_pickle=True)
        logger.info("✅ Resources loaded")
    except Exception as e:
        logger.error(f"❌ Load error: {e}")

@app.get("/")
def home():
    return JSONResponse(content={"status": "online", "message": "API running 🚀"})

@app.post("/search")
async def search_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        
        # 1. Call Hugging Face
        logger.info("Requesting embedding from Hugging Face...")
        response = requests.post(HF_API_URL, headers=headers, data=contents, timeout=30)
        
        # 🔥 FIX: Check if the response is actually JSON before parsing
        if response.status_code != 200:
            logger.error(f"HF API returned error {response.status_code}: {response.text}")
            return JSONResponse(
                content={"status": "failed", "error": f"HuggingFace Error {response.status_code}: {response.text[:100]}"}, 
                status_code=response.status_code
            )

        try:
            result = response.json()
        except Exception:
            logger.error(f"HF returned non-JSON response: {response.text}")
            return JSONResponse(content={"status": "failed", "error": "HuggingFace returned an invalid format"}, status_code=500)

        # 2. Handle "Model Loading" state
        if isinstance(result, dict) and "estimated_time" in result:
            logger.info("Model is still loading on HF...")
            return JSONResponse(
                content={"status": "failed", "error": f"Model is loading. Please try again in {int(result['estimated_time'])} seconds."}, 
                status_code=503
            )

        # 3. Process Embedding
        query_embedding = np.array(result[0]).astype("float32").reshape(1, -1)
        faiss.normalize_L2(query_embedding)

        if index is None:
            return JSONResponse(content={"status": "failed", "error": "Index not loaded"}, status_code=500)
            
        D, I = index.search(query_embedding, k=3)

        matches = []
        for idx in I[0]:
            if idx != -1 and idx < len(paths):
                matches.append(str(paths[idx]))

        return JSONResponse(content={"status": "success", "matches": matches})

    except Exception as e:
        logger.error(f"General Error: {str(e)}")
        return JSONResponse(content={"status": "failed", "error": str(e)}, status_code=500)