import streamlit as st
import requests

st.set_page_config(page_title="AI Product Search", layout="wide")
st.title("🛋️ AI Product Search")

# URL of your deployed FastAPI app
API_URL = "https://ai-search-api-4cbz.onrender.com/search"

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", width=250)

    with st.spinner("Searching for similar products..."):
        # Call the combined Search API
        response = requests.post(
            API_URL,
            files={"file": uploaded_file}
        )
        data = response.json()

    if data.get("status") != "success":
        st.error(data.get("error", "API Error"))
        st.stop()

    # Display results
    st.subheader("🛋️ Similar Products")
    cols = st.columns(3)
    
    matches = data.get("matches", [])
    for i, path in enumerate(matches):
        with cols[i % 3]:
            # Note: If the API returns local paths, the Streamlit app 
            # must have access to the 'products' folder.
            st.image(path, caption=f"Match {i+1}")