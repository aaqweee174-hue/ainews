def hybrid_search(query, faiss_index, bm25_index, embed_func, k=5):
    q_emb = embed_func(query)

    faiss_results = faiss_index.search(q_emb, k)
    bm25_results = bm25_index.search(query, k)

    combined = list(dict.fromkeys(faiss_results + bm25_results))
    return combined[:k]