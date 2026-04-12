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

# 🔥 We are using a list of stable models. If the first one is 'Gone', it tries the second.
MODELS = [
    "laion/CLIP-ViT-B-32-laion2B-s34B-b79K", 
    "openai/clip-vit-base-patch32"
]

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
        
        embedding = None
        last_error = ""

        # 🔥 TRY-CATCH LOOP: Try each model until one works
        for model_id in MODELS:
            try:
                url = f"https://api-inference.huggingface.co/models/{model_id}"
                logger.info(f"Attempting to use model: {model_id}")
                
                response = requests.post(url, headers=headers, data=contents, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    # Check if model is still loading
                    if isinstance(result, dict) and "estimated_time" in result:
                        last_error = f"Model {model_id} is loading. Wait {int(result['estimated_time'])}s"
                        continue 
                    
                    embedding = np.array(result[0]).astype("float32").reshape(1, -1)
                    break # Success! Exit the loop.
                else:
                    last_error = f"Model {model_id} returned error {response.status_code}"
                    logger.warning(last_error)
                    continue
            except Exception as e:
                last_error = str(e)
                continue

        if embedding is None:
            return JSONResponse(
                content={"status": "failed", "error": f"All models failed. Last error: {last_error}"}, 
                status_code=500
            )

        # 3. Process Embedding
        faiss.normalize_L2(embedding)

        if index is None:
            return JSONResponse(content={"status": "failed", "error": "Index not loaded"}, status_code=500)
            
        D, I = index.search(embedding, k=3)

        matches = []
        for idx in I[0]:
            if idx != -1 and idx < len(paths):
                matches.append(str(paths[idx]))

        return JSONResponse(content={"status": "success", "matches": matches})

    except Exception as e:
        logger.error(f"General Error: {str(e)}")
        return JSONResponse(content={"status": "failed", "error": str(e)}, status_code=500)