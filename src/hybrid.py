import os
import pandas as pd
from bm25 import BM25Retriever
from dense import load_embedding_model, build_job_embeddings, search_jobs_dense
from preprocessing import load_dataset, preprocess_corpus_for_bm25, preprocess_query

import numpy as np

_this_dir = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_this_dir, "..", "data", "job_dataset.csv")
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 15  # top results to show
ALPHA = 0.5  # BM25 weight; dense weight is 1 - alpha

# Score fusion: hybrid = alpha * BM25_normalized + (1 - alpha) * dense_normalized.
def _normalise_scores(score_dict: dict[int, float]) -> dict[int, float]:
    if not score_dict:
        return {}
    values = np.array(list(score_dict.values()), dtype=float)
    min_v, max_v = values.min(), values.max()
    if max_v == min_v:
        return {doc_id: 1.0 for doc_id in score_dict}
    return {doc_id: float((score - min_v) / (max_v - min_v)) for doc_id, score in score_dict.items()}

def fuse_scores(bm25_results, dense_results, alpha=ALPHA, top_k=TOP_K):
    """Fuse normalized sparse and dense scores. Alpha is the BM25 weight."""
    bm25_norm = _normalise_scores(dict(bm25_results))
    dense_norm = _normalise_scores(dict(dense_results))
    all_doc_ids = set(bm25_norm.keys()) | set(dense_norm.keys())
    fused = {doc_id: alpha * bm25_norm.get(doc_id, 0.0) + (1 - alpha) * dense_norm.get(doc_id, 0.0)
             for doc_id in all_doc_ids}
    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]

# Hybrid Search Engine
class HybridSearchEngine:
    def __init__(self, data_path=DATA_PATH, alpha=ALPHA, top_k=TOP_K):
        self.data_path = data_path
        self.alpha = alpha
        self.top_k = top_k
        self.df = None
        self.bm25 = None
        self.dense_model = None
        self.job_embeddings = None
        self._indexed = False

    def build_index(self):
        print("Loading dataset...", flush=True)
        self.df = load_dataset(self.data_path)
        print(f"Dataset loaded: {len(self.df)} jobs\n", flush=True)

        print("Preprocessing corpus for BM25...", flush=True)
        bm25_corpus = preprocess_corpus_for_bm25(self.df)
        self.bm25 = BM25Retriever(bm25_corpus)

        print("Loading dense model...", flush=True)
        self.dense_model = load_embedding_model(MODEL_NAME)

        print("Encoding jobs into dense embeddings...", flush=True)
        self.job_embeddings = build_job_embeddings(self.df, self.dense_model)

        self._indexed = True
        print("Hybrid retrieval system ready.\n", flush=True)

    def search(self, query: str):
        if not self._indexed:
            raise RuntimeError("Index not built. Call build_index() first.")
        # --- BM25 ---
        bm25_tokens, dense_text = preprocess_query(query)
        bm25_raw = self.bm25.search(bm25_tokens, top_k=self.top_k)

        # --- Dense ---
        dense_df = search_jobs_dense(dense_text, self.df, self.dense_model, self.job_embeddings, top_k=self.top_k)
        dense_raw = [(int(self.df[self.df["JobID"] == row["JobID"]].index[0]), float(row["DenseScore"])) 
                     for _, row in dense_df.iterrows()]

        # --- Fusion ---
        fused = fuse_scores(bm25_raw, dense_raw, alpha=self.alpha, top_k=self.top_k)

        # --- Build result DataFrame ---
        bm25_map = dict(bm25_raw)
        dense_map = dict(dense_raw)
        rows = []
        for rank, (doc_id, hybrid_score) in enumerate(fused, start=1):
            job = self.df.iloc[doc_id]
            rows.append({
                "Rank": rank,
                "JobID": job["JobID"],
                "Title": job["Title"],
                "ExperienceLevel": job["ExperienceLevel"],
                "Skills": job["Skills"],
                "Responsibilities": job["Responsibilities"],
                "Keywords": job["Keywords"],
                "HybridScore": round(hybrid_score, 4),
                "BM25Score": round(bm25_map.get(doc_id, 0.0), 4),
                "DenseScore": round(dense_map.get(doc_id, 0.0), 4),
            })
        return pd.DataFrame(rows)

# Printing
def print_hybrid_results(results: pd.DataFrame):
    print("\nTop matching jobs (Hybrid):\n", flush=True)
    for rank, row in enumerate(results.itertuples(index=False), start=1):
        print("-----", flush=True)
        print(f"{rank}. Title: {row.Title}", flush=True)
        print(f"   Experience Level: {row.ExperienceLevel}", flush=True)
        print(f"   Skills: {row.Skills}", flush=True)
        resp_preview = str(row.Responsibilities)[:200]
        if len(str(row.Responsibilities)) > 200:
            resp_preview += "..."
        print(f"   Responsibilities: {resp_preview}", flush=True)
        print(f"   Keywords: {row.Keywords}", flush=True)
        print(f"   Hybrid Score: {row.HybridScore:.4f}", flush=True)
    print(flush=True)

# Interactive
def interactive_search():
    engine = HybridSearchEngine(data_path=DATA_PATH, alpha=ALPHA, top_k=TOP_K)
    engine.build_index()
    print("Type a query to search for jobs.")
    print("Type 'exit' to stop.\n")
    while True:
        query = input("Enter your job search query: ").strip()
        if query.lower() == "exit":
            print("Exiting search.", flush=True)
            break
        if not query:
            print("Please enter a valid query.\n", flush=True)
            continue
        results = engine.search(query)
        print_hybrid_results(results)

if __name__ == "__main__":
    interactive_search()
