import streamlit as st
import numpy as np
from PIL import Image
import faiss
import os
import requests

st.set_page_config(page_title="AI Product Search", layout="wide")
st.title("🛋️ AI Product Search")

# Load FAISS index
@st.cache_resource
def load_index():
    index = faiss.read_index("faiss.index")
    paths = np.load("paths.npy", allow_pickle=True)
    return index, paths

index, paths = load_index()

# 🔥 CALL API FOR EMBEDDING
import requests

def get_embedding_from_api(uploaded_file):
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type
        )
    }

    response = requests.post(
        "http://127.0.0.1:8000/embed",
        files=files
    )

    # DEBUG (very important)
    if response.status_code != 200:
        st.error(f"API Error: {response.text}")
        return None

    return np.array(response.json()["embedding"]).astype("float32")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, width=300)

    st.write("🔍 Finding similar products...")

    # Get embedding from API
    query = get_embedding_from_api(uploaded_file)

    # FAISS search
    distances, indices = index.search(query, 6)

    cols = st.columns(3)

    for i, idx in enumerate(indices[0]):
        with cols[i % 3]:
            if os.path.exists(paths[idx]):
                st.image(paths[idx], width=300)