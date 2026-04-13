import streamlit as st
import torch
import clip
import faiss
import numpy as np
from PIL import Image
from datasets import load_dataset

# ==============================================================================
# CONFIGURATION
# ==============================================================================
st.set_page_config(page_title="AI Product Search", layout="wide")
st.title("🛍️ AI Product Search")
# ==============================================================================
# LOAD AI SYSTEM (Cached to prevent reloading on every click)
# ==============================================================================
@st.cache_resource
def load_ai_system():
    # 1. Load CLIP Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    # 2. Load FAISS Index
    index = faiss.read_index("faiss.index")
    
    # 3. Load the Paths (The mapping of index to filename)
    paths = np.load("paths.npy", allow_pickle=True)
    
    # 4. Load Images from Hugging Face Dataset
    # This replaces the local 'products/' folder
    dataset = load_dataset("Raj-Shivekaa/my-search-engine-images", split="train")
    
    return model, preprocess, index, paths, dataset, device

# Initialize the system
with st.spinner("Loading AI Model and Database..."):
    model, preprocess, index, paths, dataset, device = load_ai_system()

# ==============================================================================
# SEARCH LOGIC
# ==============================================================================
uploaded_file = st.file_uploader("Upload a product image to search...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    input_image = Image.open(uploaded_file).convert("RGB")
    st.image(input_image, caption="Your Uploaded Image", width=300)

    if st.button("Search for Similar Products"):
        with st.spinner("Searching..."):
            # 1. Preprocess the image and generate embedding
            image_input = preprocess(input_image).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # Convert to numpy for FAISS
            query_vector = image_features.cpu().numpy().astype('float32')

            # 2. Search FAISS Index for top 5 matches
            D, I = index.search(query_vector, k=5) 
            
            # 3. Display Results
            st.subheader("Top Matches:")
            cols = st.columns(5)
            
            for i in range(5):
                match_idx = I[0][i] # The index of the matching image
                
                # IMPORTANT: We use the index to pull the image from the HF Dataset
                # If your paths.npy matches the dataset order, we can use match_idx directly
                try:
                    result_image = dataset[match_idx]['image']
                    result_filename = paths[match_idx]
                    
                    with cols[i]:
                        st.image(result_image, caption=f"Match {i+1}: {result_filename}")
                except Exception as e:
                    with cols[i]:
                        st.error(f"Error loading image {match_idx}")