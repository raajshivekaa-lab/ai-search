from fastapi import FastAPI, File, UploadFile
import numpy as np

app = FastAPI()

@app.get("/")
def home():
    return {"message": "API is running 🚀"}

@app.post("/embed")
async def get_embedding(file: UploadFile = File(...)):
    contents = await file.read()
    embedding = np.random.rand(1, 512).astype("float32").tolist()
    return {"embedding": embedding}