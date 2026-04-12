from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os
import numpy as np
import faiss
import torch
import clip
from PIL import Image
import io
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Global variables
index = None
paths = None
model = None
preprocess = None

@app.on_event("startup")
def load_resources():
    global index, paths, model, preprocess
    try:
        # 1. Load FAISS and Paths
        index = faiss.read_index("faiss.index")
        paths = np.load("paths.npy", allow_pickle=True)
        
        # 2. Load CLIP Model locally on the server
        logger.info("Loading CLIP model into memory...")
        model, preprocess = clip.load("ViT-B/32", device="cpu")
        model.eval() # Set to evaluation mode
        
        logger.info("✅ All resources and model loaded successfully!")
    except Exception as e:
        logger.error(f"❌ Critical Load Error: {e}")

@app.get("/")
def home():
    return JSONResponse(content={"status": "online", "message": "API running locally 🚀"})

@app.post("/search")
async def search_image(file: UploadFile = File(...)):
    try:
        # 1. Read and preprocess the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_input = preprocess(image).unsqueeze(0)

        # 2. Generate embedding locally (No more Hugging Face API!)
        with torch.no_grad():
            embedding = model.encode_image(image_input)
            # Normalize for cosine similarity
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            query_embedding = embedding.numpy().astype("float32")

        # 3. FAISS Search
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