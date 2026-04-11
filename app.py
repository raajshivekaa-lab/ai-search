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
import numpy as np

def get_embedding_from_api(uploaded_file):
    response = requests.post(
        "https://ai-search-api-4cbz.onrender.com/embed",
        files={"file": uploaded_file}
    )

    if response.status_code != 200:
        print("API Error:", response.text)
        return None

    data = response.json()
    return np.array(data["embedding"]).astype("float32")
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