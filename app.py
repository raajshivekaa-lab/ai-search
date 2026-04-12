import streamlit as st
import requests
from PIL import Image
import io
import json

API_URL = "https://ai-search-api-4cbz.onrender.com/search" 
API_BASE_URL = "https://ai-search-api-4cbz.onrender.com/" 

st.set_page_config(page_title="AI Product Search", layout="wide")
st.title("🛋️ AI Product Search")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=250)

    img_resized = image.copy()
    img_resized.thumbnail((512, 512))
    buf = io.BytesIO()
    img_resized.save(buf, format="JPEG")
    buf.seek(0)

    with st.spinner("Searching..."):
        try:
            # Wake up
            requests.get(API_BASE_URL, timeout=10)

            # Search
            response = requests.post(
                API_URL,
                files={"file": ("image.jpg", buf, "image/jpeg")},
                timeout=60
            )
            
            # Read content as bytes first, then decode
            raw_content = response.content
            data = json.loads(raw_content)

        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.stop()

    if data.get("status") == "success":
        st.subheader("🛋️ Similar Products")
        matches = data.get("matches", [])
        if matches:
            cols = st.columns(3)
            for i, path in enumerate(matches):
                with cols[i % 3]:
                    st.image(path, caption=f"Match {i+1}")
        else:
            st.warning("No matches found.")
    else:
        st.error(data.get("error", "Unknown API Error"))