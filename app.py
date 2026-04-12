import streamlit as st
import requests
from PIL import Image
import io
import json

API_URL = "ai-search-api-production-c21a.up.railway.app" 
API_BASE_URL = "ai-search-api-production-c21a.up.railway.app"

st.set_page_config(page_title="AI Product Search", layout="wide")
st.title("🛋️ AI Product Search")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 1. Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=250)

    # 🔥 FIX: Convert image to RGB to handle PNGs/Transparency
    # This prevents the OSError when saving as JPEG
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")

    # 2. Pre-process image
    img_resized = image.copy()
    img_resized.thumbnail((512, 512)) 
    buf = io.BytesIO()
    img_resized.save(buf, format="JPEG")
    buf.seek(0)

    with st.spinner("Searching... (The server may take 30-60s to wake up)"):
        try:
            # 🔥 INCREASED TIMEOUT: Give the wake-up call 60 seconds, not 10
            requests.get(API_BASE_URL, timeout=60)

            # 🔥 INCREASED TIMEOUT: Give the search call 120 seconds
            response = requests.post(
                API_URL,
                files={"file": ("image.jpg", buf, "image/jpeg")},
                timeout=120 
            )
            
            raw_content = response.content
            data = json.loads(raw_content)

        except requests.exceptions.Timeout:
            st.error("❌ The server is taking too long to wake up. Please refresh the page and try one more time.")
            st.stop()
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