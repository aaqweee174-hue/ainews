from rank_bm25 import BM25Okapi

class BM25Index:
    def __init__(self, texts):
        self.tokenized = [t.split() for t in texts]
        self.bm25 = BM25Okapi(self.tokenized)
        self.texts = texts

    def search(self, query, k=5):
        scores = self.bm25.get_scores(query.split())
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.texts[i] for i in ranked[:k]]