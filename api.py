from fastapi import FastAPI, File, UploadFile
import torch
import clip
from PIL import Image
import io
import numpy as np

app = FastAPI()

device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

@app.post("/embed")
async def get_embedding(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    image = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model.encode_image(image)
        emb = emb / emb.norm(dim=-1, keepdim=True)

    return {"embedding": emb.cpu().numpy().tolist()}