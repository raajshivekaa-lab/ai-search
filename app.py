import streamlit as st
import numpy as np
from search import search_products
import requests
import faiss

st.set_page_config(page_title="AI Product Search", layout="wide")
st.title("🛋️ AI Product Search")

# ✅ Load embeddings
embeddings = np.load("embeddings.npy")
paths = np.load("paths.npy", allow_pickle=True)

# ✅ Load FAISS index
@st.cache_resource
def load_index():
    index = faiss.read_index("faiss.index")
    return index

index = load_index()

# ✅ Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    st.image(uploaded_file, caption="Uploaded Image", width=250)

    # 🔥 Call API
    response = requests.post(
        "https://ai-search-api-4cbz.onrender.com/embed",
        files={"file": uploaded_file}
    )

    data = response.json()

    # ❌ Handle error
    if data.get("status") != "success":
        st.error(data.get("error", "API Error"))
        st.stop()

    # ✅ Get embedding
    query_embedding = np.array(data["embedding"]).astype("float32")

    # 🔥 Search
    exact, similar = search_products(query_embedding, embeddings, paths)

    # ✅ Exact match
    if exact:
        st.subheader("🎯 Exact Match")
        st.image(exact, width=300)

    # ✅ Similar products
    st.subheader("🛋️ Similar Products")

    cols = st.columns(3)

    for i, (path, score) in enumerate(similar):
        with cols[i % 3]:
            st.image(path, caption=f"Score: {score:.2f}")