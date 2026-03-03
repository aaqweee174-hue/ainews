import streamlit as st
from ingestion.rss_fetch import fetch_news
from utils.chunking import chunk_text
from embeddings import get_embedding
from retrieval.faiss_index import FAISSIndex
from retrieval.bm25 import BM25Index
from retrieval.hybrid import hybrid_search
from llm.qa import ask_llm
from config import TOP_K

import json
import os

st.title("📰 AI News Intelligence System")

# -------------------
# Memory setup
# -------------------
MEMORY_FILE = "memory.json"

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_memory(query, sources):
    memory = load_memory()
    memory.append({"query": query, "sources": sources})
    # Keep last 20 entries only
    memory = memory[-20:]
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)

# -------------------
# Build system
# -------------------
@st.cache_resource
def build_system():
    articles = fetch_news()
    texts = []
    for art in articles:
        chunks = chunk_text(art['content'])
        texts.extend(chunks)

    embeddings = [get_embedding(t) for t in texts]
    dim = len(embeddings[0])
    faiss_index = FAISSIndex(dim)
    faiss_index.add(embeddings, texts)
    bm25_index = BM25Index(texts)
    return faiss_index, bm25_index, texts

faiss_index, bm25_index, texts = build_system()

# -------------------
# Query input
# -------------------
query = st.text_input("Ask about news:")

if query:
    # Hybrid search for current query
    results = hybrid_search(query, faiss_index, bm25_index, get_embedding, TOP_K)

    # Load memory and merge with current results
    memory = load_memory()
    memory_sources = [item["sources"] for item in memory]
    all_contexts = sum(memory_sources, []) + results  # flatten memory list + current results

    # Ask GPT
    answer = ask_llm(query, all_contexts)

    # -------------------
    # Display results
    # -------------------
    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources")
    for i, r in enumerate(results):
        st.write(f"{i+1}. {r[:300]}")  # first 300 chars

    # Save current query + sources to memory
    save_memory(query, results)

# -------------------
# Optional: Show previous memory
# -------------------
with st.expander("📂 Previous Queries & Sources"):
    memory = load_memory()
    for item in memory[::-1]:  # latest first
        st.write(f"**Query:** {item['query']}")
        for i, src in enumerate(item['sources']):
            st.write(f"Source {i+1}: {src[:200]}")
        st.write("---")