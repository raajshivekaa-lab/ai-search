from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse # <--- Crucial for fixing IncompleteRead
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
        response = requests.post(HF_API_URL, headers=headers, data=contents, timeout=30)
        result = response.json()

        if isinstance(result, dict) and "error" in result:
            return JSONResponse(content={"status": "failed", "error": result["error"]}, status_code=500)

        query_embedding = np.array(result[0]).astype("float32").reshape(1, -1)
        faiss.normalize_L2(query_embedding)

        if index is None:
            return JSONResponse(content={"status": "failed", "error": "Index not loaded"}, status_code=500)
            
        D, I = index.search(query_embedding, k=3) # Limit to 3 to keep response tiny

        matches = []
        for idx in I[0]:
            if idx != -1 and idx < len(paths):
                matches.append(str(paths[idx]))

        logger.info(f"Sending response with {len(matches)} matches")
        
        # 🔥 Using JSONResponse explicitly fixes the IncompleteRead error 
        # by setting the correct Content-Length header.
        return JSONResponse(content={
            "status": "success",
            "matches": matches
        })

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return JSONResponse(content={"status": "failed", "error": str(e)}, status_code=500)