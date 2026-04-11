import os
import torch
import clip
from PIL import Image
import numpy as np

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
            image_paths.append(path)

            print("Processed:", path)

        except Exception as e:
            print("Error:", path, e)

np.save("embeddings.npy", embeddings)
np.save("paths.npy", image_paths)

print("✅ Done! Embeddings saved.")