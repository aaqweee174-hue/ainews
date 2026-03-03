import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class VectorSearch:
    def __init__(self):
        self.texts = []
        self.embeddings = []

    def add(self, embeddings, texts):
        self.embeddings = np.array(embeddings)
        self.texts = texts

    def search(self, query_embedding, top_k=5):
        sims = cosine_similarity(
            [query_embedding], self.embeddings
        )[0]

        top_indices = np.argsort(sims)[-top_k:][::-1]
        return [self.texts[i] for i in top_indices]