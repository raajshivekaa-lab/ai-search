import streamlit as st
import numpy as np
from PIL import Image
import os
import requests
import base64

st.set_page_config(page_title="AI Product Search", layout="wide")

st.title("🛋️ AI Product Search")

# Load data
@st.cache_data
def load_data():
    embeddings = np.load("embeddings.npy")
    paths = np.load("paths.npy", allow_pickle=True)
    return embeddings, paths

embeddings, paths = load_data()

# Get API key from environment
API_KEY = os.getenv("OPENAI_API_KEY")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

def get_image_embedding(image_file):
    url = "https://api.openai.com/v1/embeddings"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    image_bytes = image_file.read()
    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    data = {
        "model": "text-embedding-3-small",
        "input": base64_image
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code != 200:
        st.error("API Error: " + str(response.json()))
        return None

    return np.array(response.json()["data"][0]["embedding"])


if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", width=300)

    st.write("🔍 Finding similar products...")

    query_embedding = get_image_embedding(uploaded_file)

    if query_embedding is not None:
        similarities = embeddings @ query_embedding
        top_indices = np.argsort(similarities)[-6:][::-1]

        cols = st.columns(3)

        for i, idx in enumerate(top_indices):
            with cols[i % 3]:
                if os.path.exists(paths[idx]):
                    st.image(paths[idx], use_column_width=True)