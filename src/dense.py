import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import load_dataset, preprocess_document_for_dense

_this_dir = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_this_dir, "..", "data", "job_dataset.csv")
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 15  # show first 15 jobs

def load_embedding_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    print("Loading sentence-transformer model...", flush=True)
    model = SentenceTransformer(model_name)
    print("Model loaded.\n", flush=True)
    return model

def build_job_embeddings(df: pd.DataFrame, model: SentenceTransformer):
    print("Encoding job descriptions into embeddings...", flush=True)
    texts = [preprocess_document_for_dense(row) for _, row in df.iterrows()]
    embeddings = model.encode(texts, show_progress_bar=True)
    print(f"Embeddings created. Shape: {embeddings.shape}\n", flush=True)
    return embeddings

def search_jobs_dense(query: str, df: pd.DataFrame, model, job_embeddings, top_k: int = TOP_K) -> pd.DataFrame:
    query_embedding = model.encode([query])
    scores = cosine_similarity(query_embedding, job_embeddings)[0]
    results = df.copy()
    results["DenseScore"] = scores
    results = results.sort_values(by="DenseScore", ascending=False).head(top_k)
    return results

def print_results(results: pd.DataFrame):
    print("\nTop matching jobs:\n", flush=True)
    for rank, row in enumerate(results.itertuples(index=False), start=1):
        print("-----", flush=True)
        print(f"{rank}. Title: {row.Title}", flush=True)
        print(f"   Experience Level: {row.ExperienceLevel}", flush=True)
        print(f"   Skills: {row.Skills}", flush=True)
        responsibilities_preview = str(row.Responsibilities)[:200]
        if len(str(row.Responsibilities)) > 200:
            responsibilities_preview += "..."
        print(f"   Responsibilities: {responsibilities_preview}", flush=True)
        print(f"   Keywords: {row.Keywords}", flush=True)
        print(f"   Dense Score: {row.DenseScore:.4f}", flush=True)
    print(flush=True)

def interactive_search():
    print("Loading dataset...", flush=True)
    df = load_dataset(DATA_PATH)
    print(f"Dataset loaded: {len(df)} jobs\n", flush=True)

    model = load_embedding_model()
    job_embeddings = build_job_embeddings(df, model)

    print("Dense retrieval system ready.", flush=True)
    print("Type a query to search for jobs.", flush=True)
    print("Type 'exit' to stop.\n", flush=True)

    while True:
        query = input("Enter your job search query: ").strip()
        if query.lower() == "exit":
            print("Exiting search.", flush=True)
            break
        if not query:
            print("Please enter a valid query.\n", flush=True)
            continue

        results = search_jobs_dense(query, df, model, job_embeddings, top_k=TOP_K)
        print_results(results)

if __name__ == "__main__":
    interactive_search()
