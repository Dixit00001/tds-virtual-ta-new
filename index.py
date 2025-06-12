from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import base64, io, json
import pytesseract
from PIL import Image
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str
    image: Optional[str] = None

@app.post("/")
async def get_answer(query: Query):
    # Load files inside the function to reduce cold-start memory
    chunks_path = os.path.join(os.path.dirname(__file__), "chunks.jsonl")
    index_path = os.path.join(os.path.dirname(__file__), "index.faiss")

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]

    texts = [chunk["text"] for chunk in chunks]
    sources = [chunk.get("source", "") for chunk in chunks]
    ids = [chunk.get("id", "") for chunk in chunks]

    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index(index_path)

    question = query.question
    if query.image:
        image_data = base64.b64decode(query.image)
        image = Image.open(io.BytesIO(image_data))
        extracted_text = pytesseract.image_to_string(image)
        question += " " + extracted_text.strip()

    embedding = model.encode([question]).astype("float32")
    _, I = index.search(embedding, 3)

    selected = [texts[i] for i in I[0]]
    selected_sources = [sources[i] for i in I[0]]

    return {
        "answer": selected[0],
        "links": [
            {"url": f"https://discourse.onlinedegree.iitm.ac.in/{source}", "text": f"{selected[i]}"}
            for i, source in enumerate(selected_sources)
        ]
    }
