import os
import sys
import pandas as pd

# Project paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

# Imports
from preprocessing import load_dataset, preprocess_corpus_for_bm25, preprocess_query
from bm25 import BM25Retriever
from dense import load_embedding_model, build_job_embeddings, search_jobs_dense
from queries import queries

# Load dataset
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "job_dataset.csv")
df = load_dataset(DATA_PATH)
corpus = preprocess_corpus_for_bm25(df)

# Setup BM25
bm25 = BM25Retriever(corpus)

# Setup Dense Retrieval
print("Loading sentence-transformer model...")
embedding_model = load_embedding_model("all-MiniLM-L6-v2")

print("Encoding job descriptions into embeddings...")
job_embeddings = build_job_embeddings(df, embedding_model)
print(f"Embeddings shape: {job_embeddings.shape}")

# Pooling top K results
top_k = 20
qrels_pool = []

for q in queries:
    query_id = q["query_id"]
    query_text = q["text"]
    query_tokens, query_embedding_text = preprocess_query(query_text)

    # BM25 results 
    bm25_results = bm25.search(query_tokens, top_k=top_k)
    bm25_doc_ids = [doc_id for doc_id, _ in bm25_results]

    # Dense results
    dense_results = search_jobs_dense(
        query_text,
        df,
        embedding_model,
        job_embeddings,
        top_k=top_k
    )
    dense_doc_ids = dense_results.index.tolist()  # get row indices

    # Pool & deduplicate
    pool_doc_ids = set(bm25_doc_ids + dense_doc_ids)

    for doc_id in pool_doc_ids:
        row = df.loc[doc_id, ["JobID", "Title", "ExperienceLevel", "YearsOfExperience",
                              "Skills", "Responsibilities", "Keywords"]].to_dict()
        row["relevance"] = -1  # mark for manual labeling
        row["query_id"] = query_id
        qrels_pool.append(row)

# Save candidate pool
pool_path = os.path.join(os.path.dirname(__file__), "qrels_candidates.csv")
df_pool = pd.DataFrame(qrels_pool)
df_pool.to_csv(pool_path, index=False)
print(f"Candidate pool saved to {pool_path}")
print(df_pool.head())
