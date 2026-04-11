import streamlit as st
import numpy as np
from PIL import Image
import faiss
import os
import clip
import torch

st.set_page_config(page_title="AI Product Search", layout="wide")
st.title("🛋️ AI Product Search")

# Load model (still needed for query only)
@st.cache_resource
def load_model():
    device = "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

model, preprocess, device = load_model()

# Load FAISS index
@st.cache_resource
def load_index():
    index = faiss.read_index("faiss.index")
    paths = np.load("paths.npy", allow_pickle=True)
    return index, paths

index, paths = load_index()

def get_embedding(image):
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(image)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype("float32")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, width=300)

    st.write("🔍 Finding similar products...")

    query = get_embedding(image)

    # 🔥 FAISS SEARCH
    distances, indices = index.search(query, 6)

    cols = st.columns(3)

    for i, idx in enumerate(indices[0]):
        with cols[i % 3]:
            if os.path.exists(paths[idx]):
                st.image(paths[idx], width=300)