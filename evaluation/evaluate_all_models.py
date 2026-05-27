import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score
from sklearn.metrics.pairwise import cosine_similarity

# Ensure local folders are importable when running `python evaluation/evaluate_all_models.py`.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
EVAL_PATH = os.path.join(PROJECT_ROOT, "evaluation")
for path in (SRC_PATH, EVAL_PATH):
    if path not in sys.path:
        sys.path.insert(0, path)

from bm25 import BM25Retriever
from dense import build_job_embeddings, load_embedding_model, search_jobs_dense
from preprocessing import load_dataset, preprocess_corpus_for_bm25, preprocess_query
from queries import queries as query_list
from relevance_judgments import load_qrels

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "job_dataset.csv")
QRELS_PATH = os.path.join(PROJECT_ROOT, "evaluation", "qrels_pool.csv")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "results", "evaluation_metrics.csv")
AVERAGE_RESULTS_PATH = os.path.join(PROJECT_ROOT, "results", "average_metrics.csv")
EXAMPLES_PATH = os.path.join(PROJECT_ROOT, "results", "query_results_examples.csv")
SUMMARY_PATH = os.path.join(PROJECT_ROOT, "results", "evaluation_summary.md")
AVERAGE_PLOT_PATH = os.path.join(PROJECT_ROOT, "results", "average_metrics.png")
NDCG_PLOT_PATH = os.path.join(PROJECT_ROOT, "results", "ndcg_by_query.png")
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 20
ALPHA = 0.5  # BM25 weight; dense weight is 1 - alpha.


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize scores for linear hybrid fusion."""
    scores = np.nan_to_num(scores.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    min_score = scores.min()
    max_score = scores.max()
    if max_score == min_score:
        return np.ones_like(scores) if max_score > 0 else np.zeros_like(scores)
    return (scores - min_score) / (max_score - min_score)


def precision_at_5(retrieved_ids, relevant_ids):
    if not relevant_ids:
        return 0
    return sum(1 for doc_id in retrieved_ids[:5] if doc_id in relevant_ids) / 5


def ndcg_at_10(retrieved_ids, relevant_ids):
    if not relevant_ids:
        return 0
    rel = [1 if doc_id in relevant_ids else 0 for doc_id in retrieved_ids[:10]]
    if sum(rel) == 0:
        return 0
    ideal = sorted(rel, reverse=True)
    return ndcg_score([ideal], [rel])


def mean_reciprocal_rank(retrieved_ids, relevant_ids):
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1 / rank
    return 0


def hybrid_search(query, top_k=TOP_K, alpha=ALPHA):
    """Return row indices ranked by alpha * BM25_norm + (1 - alpha) * dense_norm."""
    query_tokens, dense_query = preprocess_query(query)

    bm25_scores = np.zeros(len(df))
    for idx, score in bm25.search(query_tokens, top_k=len(df)):
        bm25_scores[idx] = score

    query_embedding = embedding_model.encode([dense_query])
    dense_scores = cosine_similarity(query_embedding, job_embeddings)[0]

    hybrid_scores = alpha * normalize_scores(bm25_scores) + (1 - alpha) * normalize_scores(dense_scores)
    ranked_idx = np.argsort(hybrid_scores)[::-1]
    return ranked_idx[:top_k].tolist()


def build_examples(qid, query, model_name, ranked_indices, relevant_ids):
    examples = []
    for rank, idx in enumerate(ranked_indices[:5], start=1):
        row = df.iloc[idx]
        job_id = row["JobID"]
        examples.append({
            "Query": qid,
            "QueryText": query,
            "Model": model_name,
            "Rank": rank,
            "JobID": job_id,
            "Title": row["Title"],
            "ExperienceLevel": row["ExperienceLevel"],
            "Relevant": int(job_id in relevant_ids),
        })
    return examples


def best_model_for_metric(df_results, metric):
    averages = df_results.groupby("Model")[metric].mean()
    return averages.idxmax(), averages.max()


def write_summary(df_results, average_results, queries):
    best_overall = average_results.sort_values(
        by=["nDCG@10", "MRR", "P@5"], ascending=False
    ).iloc[0]

    query_winners = (
        df_results.sort_values(by=["Query", "nDCG@10", "MRR", "P@5"], ascending=[True, False, False, False])
        .groupby("Query")
        .first()
        .reset_index()
    )
    wins = query_winners.groupby("Model")["Query"].apply(list).to_dict()

    lines = [
        "# Evaluation Summary",
        "",
        f"This evaluation uses {len(queries)} manually judged job-search queries and binary relevance judgements.",
        f"Hybrid fusion uses `alpha = {ALPHA}`, where alpha is the BM25 weight:",
        "",
        "`hybrid_score = alpha * BM25 + (1 - alpha) * Dense`",
        "",
        "## Overall Result",
        "",
        (
            f"Best overall model: **{best_overall['Model']}**, with average "
            f"P@5={best_overall['P@5']:.3f}, nDCG@10={best_overall['nDCG@10']:.3f}, "
            f"and MRR={best_overall['MRR']:.3f}."
        ),
        "",
        "## Model Behaviour",
        "",
        f"- BM25 performs best on: {', '.join(wins.get('BM25', [])) or 'no query in this run'}.",
        f"- Dense performs best on: {', '.join(wins.get('Dense', [])) or 'no query in this run'}.",
        f"- Hybrid performs best on: {', '.join(wins.get('Hybrid', [])) or 'no query in this run'}.",
        "",
        "BM25 tends to do well when the query contains exact skill or role terms that appear in job titles, skills, or keywords.",
        "Dense retrieval is more useful for descriptive and vocabulary-variation queries where the wording does not exactly match the posting.",
        "Hybrid retrieval is strongest when both exact skill matching and semantic matching contribute useful evidence.",
        "",
        "## Limitations",
        "",
        "- The query set is still small and manually judged, so the scores should be treated as directional rather than definitive.",
        "- Relevance judgements are binary and do not capture degrees of fit such as seniority, location, salary, or domain preference.",
        "- Some job IDs in the source data are reused across job families, which can make ID-only evaluation less precise.",
        "- The dense model is fixed to `all-MiniLM-L6-v2`; no extra model tuning or new retrieval model was added.",
    ]
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_plots(df_results, average_results):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plots.")
        return

    ax = average_results.set_index("Model")[["P@5", "nDCG@10", "MRR"]].plot(kind="bar", figsize=(8, 5))
    ax.set_title("Average Evaluation Metrics by Model")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.figure.tight_layout()
    ax.figure.savefig(AVERAGE_PLOT_PATH, dpi=150)
    plt.close(ax.figure)

    ndcg_by_query = df_results.pivot(index="Query", columns="Model", values="nDCG@10")
    ax = ndcg_by_query.plot(kind="bar", figsize=(10, 5))
    ax.set_title("nDCG@10 by Query")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("nDCG@10")
    ax.figure.tight_layout()
    ax.figure.savefig(NDCG_PLOT_PATH, dpi=150)
    plt.close(ax.figure)


df = load_dataset(DATA_PATH)
df["JobID"] = df["JobID"].astype(str).str.strip().str.lower()

qrels = load_qrels(QRELS_PATH)
qrels["JobID"] = qrels["JobID"].astype(str).str.strip().str.lower()
qrels["relevance"] = qrels["relevance"].astype(int)

queries = {q["query_id"]: q["text"] for q in query_list}

corpus = preprocess_corpus_for_bm25(df)
bm25 = BM25Retriever(corpus)

embedding_model = load_embedding_model(MODEL_NAME)
job_embeddings = build_job_embeddings(df, embedding_model)

results = []
examples = []

for qid, query in queries.items():
    print(f"\n=== Evaluating {qid}: '{query}' ===")

    relevant_ids = qrels[(qrels["query_id"] == qid) & (qrels["relevance"] == 1)]["JobID"].tolist()
    print(f"Relevant JobIDs ({len(relevant_ids)}): {relevant_ids[:5]}")

    query_tokens, _ = preprocess_query(query)
    bm25_res = bm25.search(query_tokens, top_k=TOP_K)
    bm25_indices = [idx for idx, _ in bm25_res]
    bm25_ids = [df.iloc[idx]["JobID"] for idx in bm25_indices]
    print(f"BM25 top 5 retrieved: {bm25_ids[:5]}")

    dense_res = search_jobs_dense(query, df, embedding_model, job_embeddings, top_k=TOP_K)
    dense_ids = dense_res["JobID"].str.lower().str.strip().tolist()
    dense_indices = dense_res.index.tolist()
    print(f"Dense top 5 retrieved: {dense_ids[:5]}")

    hybrid_indices = hybrid_search(query, top_k=TOP_K, alpha=ALPHA)
    hybrid_ids = [df.iloc[idx]["JobID"] for idx in hybrid_indices]
    print(f"Hybrid top 5 retrieved: {hybrid_ids[:5]}")

    for model_name, retrieved_ids, ranked_indices in zip(
        ("BM25", "Dense", "Hybrid"),
        (bm25_ids, dense_ids, hybrid_ids),
        (bm25_indices, dense_indices, hybrid_indices),
    ):
        results.append({
            "Query": qid,
            "Model": model_name,
            "P@5": precision_at_5(retrieved_ids, relevant_ids),
            "nDCG@10": ndcg_at_10(retrieved_ids, relevant_ids),
            "MRR": mean_reciprocal_rank(retrieved_ids, relevant_ids),
        })
        examples.extend(build_examples(qid, query, model_name, ranked_indices, relevant_ids))

df_results = pd.DataFrame(results)
os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
df_results.to_csv(RESULTS_PATH, index=False)

df_averages = (
    df_results.groupby("Model", as_index=False)[["P@5", "nDCG@10", "MRR"]]
    .mean()
    .sort_values(by=["nDCG@10", "MRR", "P@5"], ascending=False)
)
df_averages.to_csv(AVERAGE_RESULTS_PATH, index=False)

pd.DataFrame(examples).to_csv(EXAMPLES_PATH, index=False)
write_summary(df_results, df_averages, queries)
write_plots(df_results, df_averages)

pivot_table = df_results.pivot(index="Query", columns="Model")
print("\n=== Evaluation Metrics ===\n")
print(pivot_table.round(3))
print(f"\nSaved metrics to {RESULTS_PATH}")
print(f"Saved average metrics to {AVERAGE_RESULTS_PATH}")
print(f"Saved query examples to {EXAMPLES_PATH}")
print(f"Saved summary to {SUMMARY_PATH}")
