import streamlit as st
import torch
import clip
import numpy as np
from PIL import Image

# Load model
model, preprocess = clip.load("ViT-B/32")

# Load embeddings
embeddings = np.load("embeddings.npy")
paths = np.load("paths.npy")

def search(image, top_k=5):
    image = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        query = model.encode_image(image)
        query = query / query.norm(dim=-1, keepdim=True)

    query = query.numpy()[0]

    similarities = np.dot(embeddings, query)
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    results = [paths[i] for i in top_indices]
    return results


# UI
st.set_page_config(page_title="Shivekaa AI Search", layout="wide")

st.title("🛋️ Shivekaa AI Product Search")

uploaded_file = st.file_uploader("Upload a product image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)

    st.subheader("Uploaded Image")
    st.image(image, width=300)

    st.subheader("Similar Products")

    results = search(image)

    cols = st.columns(5)

    for i, path in enumerate(results):
        with cols[i]:
            st.image(path, use_container_width=True)