import streamlit as st
import numpy as np
from PIL import Image
import os
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Product Search", layout="wide")

st.title("🛋️ AI Product Search")

@st.cache_data
def load_data():
    embeddings = np.load("embeddings.npy")
    paths = np.load("paths.npy", allow_pickle=True)
    return embeddings, paths

embeddings, paths = load_data()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", width=300)

    st.write("🔍 Finding similar products...")

    # TEMP embedding for uploaded image (random for now)
    query_embedding = np.random.rand(embeddings.shape[1])

    # Compute similarity
    similarities = cosine_similarity([query_embedding], embeddings)[0]

    # Get top 6 similar images
    top_indices = np.argsort(similarities)[-6:][::-1]

    cols = st.columns(3)

    for i, idx in enumerate(top_indices):
        with cols[i % 3]:
            if os.path.exists(paths[idx]):
                st.image(paths[idx], use_column_width=True)