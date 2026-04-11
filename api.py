from fastapi import FastAPI, File, UploadFile
import numpy as np

app = FastAPI()

@app.get("/")
def home():
    return {"message": "API is running 🚀"}

@app.post("/embed")
async def get_embedding(file: UploadFile = File(...)):
    try:
        # Read file (just to confirm it's received)
        contents = await file.read()

        # 🔥 Generate lightweight dummy embedding (512 dim)
        embedding = np.random.rand(1, 512).astype("float32").tolist()

        return {"embedding": embedding}

    except Exception as e:
        return {"error": str(e)}