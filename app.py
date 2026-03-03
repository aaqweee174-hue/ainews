import streamlit as st
from ingestion.rss_fetch import fetch_news
from utils.chunking import chunk_text
from embeddings import get_embedding
from retrieval.vector_search import VectorSearch
from retrieval.bm25 import BM25Index
from llm.qa import ask_llm
from config import TOP_K

import json
import os

st.title("📰 AI News Intelligence System")

MEMORY_FILE = "memory.json"

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_memory(query, sources):
    memory = load_memory()
    memory.append({"query": query, "sources": sources})
    memory = memory[-20:]
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)

@st.cache_resource
def build_system():
    articles = fetch_news()
    texts = []

    for art in articles:
        chunks = chunk_text(art['content'])
        texts.extend(chunks)

    embeddings = [get_embedding(t) for t in texts]

    vector_index = VectorSearch()
    vector_index.add(embeddings, texts)

    bm25_index = BM25Index(texts)

    return vector_index, bm25_index

vector_index, bm25_index = build_system()

query = st.text_input("Ask about news:")

if query:
    query_embedding = get_embedding(query)

    vector_results = vector_index.search(query_embedding, TOP_K)
    bm25_results = bm25_index.search(query, TOP_K)

    # Combine both results
    results = list(dict.fromkeys(vector_results + bm25_results))

    memory = load_memory()
    memory_sources = [item["sources"] for item in memory]
    all_contexts = sum(memory_sources, []) + results

    answer = ask_llm(query, all_contexts)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources")
    for i, r in enumerate(results):
        st.write(f"{i+1}. {r[:300]}")

    save_memory(query, results)

with st.expander("📂 Previous Queries & Sources"):
    memory = load_memory()
    for item in memory[::-1]:
        st.write(f"**Query:** {item['query']}")
        for i, src in enumerate(item['sources']):
            st.write(f"Source {i+1}: {src[:200]}")
        st.write("---")