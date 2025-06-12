from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pytesseract
from PIL import Image
import io
import base64
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

app = FastAPI()

# ✅ CORS middleware added for external access (like TDS, Hoppscotch, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load data
with open("chunks.jsonl", "r", encoding="utf-8") as f:
    chunks = [json.loads(line) for line in f]

# Load FAISS index and embedding model
index = faiss.read_index("index.faiss")
model = SentenceTransformer("all-MiniLM-L6-v2")

class QueryRequest(BaseModel):
    question: str
    image: str = None

def extract_text_from_image(base64_image: str) -> str:
    image_data = base64.b64decode(base64_image)
    image = Image.open(io.BytesIO(image_data))
    text = pytesseract.image_to_string(image)
    return text

def search_similar_chunks(question: str, k: int = 3):
    embedding = model.encode([question])
    distances, indices = index.search(np.array(embedding).astype("float32"), k)
    results = [chunks[i] for i in indices[0]]
    return results

@app.post("/api")
async def answer_query(req: QueryRequest):
    full_question = req.question
    if req.image:
        try:
            image_text = extract_text_from_image(req.image)
            full_question += " " + image_text
        except Exception:
            pass  # If image decoding fails, ignore it

    top_chunks = search_similar_chunks(full_question)
    answer = top_chunks[0]["text"] if top_chunks else "Sorry, I couldn't find a good answer."

    links = []
    for chunk in top_chunks:
        if "url" in chunk:
            links.append({
                "url": chunk["url"],
                "text": chunk.get("source", "View Source")
            })

    return JSONResponse({
        "answer": answer,
        "links": links
    })
