import streamlit as st
import numpy as np
from PIL import Image
import os
import torch
import clip
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Product Search", layout="wide")

st.title("🛋️ AI Product Search")

# Load CLIP model
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

model, preprocess, device = load_model()

# Load embeddings
@st.cache_data
def load_data():
    embeddings = np.load("embeddings.npy")
    paths = np.load("paths.npy", allow_pickle=True)
    return embeddings, paths

embeddings, paths = load_data()

# Convert image to embedding
def get_image_embedding(image):
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(image)
    return features.cpu().numpy()

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    st.write("🔍 Finding similar products...")

    # Get embedding
    query_embedding = get_image_embedding(image)

    # Compute similarity
    similarities = cosine_similarity(query_embedding, embeddings)[0]

    # Top 6 matches
    top_indices = similarities.argsort()[-6:][::-1]

    cols = st.columns(3)

    for i, idx in enumerate(top_indices):
        with cols[i % 3]:
            if os.path.exists(paths[idx]):
                st.image(paths[idx], use_column_width=True)