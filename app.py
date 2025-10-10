from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import json
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os
from dotenv import load_dotenv

# ---------------------
# Config
# ---------------------

# Load songs data
with open("songs.json", encoding="utf-8") as f:
    songs = json.load(f)

# Load pre-computed embeddings
embeddings = np.load("song_embeddings_e5_final.npy")

# Load the same model used to generate embeddings
model = SentenceTransformer("intfloat/multilingual-e5-large")

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# ✅ FIXED MODEL NAME
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# ---------------------
# FastAPI app
# ---------------------
app = FastAPI()

# Allow frontend (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------
# Payiram Mapper
# ---------------------
def get_payiram(song_number: int) -> str:
    if 1 <= song_number <= 336:
        return "First Payiram"
    elif 337 <= song_number <= 548:
        return "Second Payiram"
    elif 549 <= song_number <= 883:
        return "Third Payiram"
    elif 884 <= song_number <= 1033:
        return "Fourth Payiram"
    elif 1034 <= song_number <= 1560:
        return "Fifth Payiram"
    elif 1561 <= song_number <= 1783:
        return "Sixth Payiram"
    elif 1784 <= song_number <= 1980:
        return "Seventh Payiram"
    elif 1981 <= song_number <= 2121:
        return "Eighth Payiram"
    elif 2122 <= song_number <= 3000:
        return "Ninth Payiram"
    return "Unknown Payiram"

# ---------------------
# Utility: Search top-k matches
# ---------------------
def search_songs(query: str, top_k: int = 3):
    query_text = "query: " + query
    query_vec = model.encode([query_text])[0]

    sims = np.dot(embeddings, query_vec) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vec)
    )

    top_idx = np.argsort(-sims)[:top_k]
    results = []

    for idx in top_idx:
        song = songs[idx]
        song_number = song["song_number"]

        results.append({
            "song_number": song_number,
            "padal": song["padal"],
            "vilakam": song["vilakam"],
            "vilakam_en": song["vilakam_en"],
            "payiram": get_payiram(song_number),
            "similarity": float(sims[idx]),
        })

    return results

# ---------------------
# Endpoint 1: Raw search
# ---------------------
@app.get("/search")
def search(q: str = Query(...), top_k: int = 3):
    results = search_songs(q, top_k)
    return {"query": q, "results": results}

# ---------------------
# Endpoint 2: Chat-style summarized search
# ---------------------
@app.get("/chat_search")
def chat_search(q: str = Query(...), top_k: int = 3):
    results = search_songs(q, top_k)

    # Build context
    context = "\n\n".join(
        [
            f"Song {r['song_number']} ({r['payiram']}):\nVerse:\n{r['padal']}\nExplanation:\n{r['vilakam_en']}"
            for r in results
        ]
    )

    prompt = f"""
You are a helpful assistant. A user searched for: "{q}".
Here are some relevant verses from Thirumandiram:

{context}

Please summarize these results in a clear, chatbot-friendly way.
Explain the key ideas and how they relate to the query.
"""

    # Generate summary using Gemini
    response = gemini_model.generate_content(prompt)

    return {
        "query": q,
        "summary": response.text,
        "results": results
    }
