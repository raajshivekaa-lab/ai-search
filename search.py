<<<<<<< HEAD
import torch
import clip
import numpy as np
from PIL import Image

# Load model
model, preprocess = clip.load("ViT-B/32")

# Load saved data
embeddings = np.load("embeddings.npy")
paths = np.load("paths.npy")

def search(image_path, top_k=5):
    image = preprocess(Image.open(image_path)).unsqueeze(0)

    with torch.no_grad():
        query = model.encode_image(image)
        query = query / query.norm(dim=-1, keepdim=True)

    query = query.numpy()[0]

    # Compute similarity manually
    similarities = np.dot(embeddings, query)

    # Get top matches
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    results = [paths[i] for i in top_indices if "sofa" in paths[i].lower()]
    return results


# TEST
results = search("test.jpg")

print("\n🔍 Similar Products:")
for r in results:
=======
import torch
import clip
import numpy as np
from PIL import Image

# Load model
model, preprocess = clip.load("ViT-B/32")

# Load saved data
embeddings = np.load("embeddings.npy")
paths = np.load("paths.npy")

def search(image_path, top_k=5):
    image = preprocess(Image.open(image_path)).unsqueeze(0)

    with torch.no_grad():
        query = model.encode_image(image)
        query = query / query.norm(dim=-1, keepdim=True)

    query = query.numpy()[0]

    # Compute similarity manually
    similarities = np.dot(embeddings, query)

    # Get top matches
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    results = [paths[i] for i in top_indices if "sofa" in paths[i].lower()]
    return results


# TEST
results = search("test.jpg")

print("\n🔍 Similar Products:")
for r in results:
>>>>>>> 159aafc2afc04444e86a81a393f7beb1ceefd202
    print(r)