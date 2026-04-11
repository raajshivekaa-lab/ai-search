import os
import torch
import clip
from PIL import Image
import numpy as np
import faiss

model, preprocess = clip.load("ViT-B/32")

image_folder = "products"
embeddings = []
image_paths = []

for root, _, files in os.walk(image_folder):
    for file in files:
        path = os.path.join(root, file)

        try:
            image = preprocess(Image.open(path)).unsqueeze(0)

            with torch.no_grad():
                emb = model.encode_image(image)
                emb = emb / emb.norm(dim=-1, keepdim=True)

            embeddings.append(emb.numpy()[0])
            image_paths.append(path.replace("\\", "/"))

            print("Processed:", path)

        except Exception as e:
            print("Error:", path, e)

# Convert to numpy
embeddings = np.array(embeddings).astype("float32")

# 🔥 CREATE FAISS INDEX
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # cosine similarity
index.add(embeddings)

# SAVE EVERYTHING
faiss.write_index(index, "faiss.index")
np.save("paths.npy", image_paths)

print("✅ FAISS index created!")