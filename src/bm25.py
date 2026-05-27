import os
import math
from collections import defaultdict
import pandas as pd
from preprocessing import preprocess_corpus_for_bm25, preprocess_query, load_dataset

dataset_path = os.path.join(os.path.dirname(__file__), "..", "data", "job_dataset.csv")

class BM25Retriever:

    def __init__(self, documents, field_weights=None, k1=1.5, b=0.75):
        # each document should  be tokenised in preprocessing.py
        self.documents = documents
        self.k1 = k1
        self.b = b

        # default field weights based on the design
        self.field_weights = field_weights or {
            "title": 3.0,
            "keywords": 2.5,
            "skills": 2.0,
            "responsibilities": 1.0
        }

        self.N = len(documents)  # total number of documents

        # inverted index:
        self.inverted_index = defaultdict(dict)

        # document frequencies:
        self.doc_freqs = defaultdict(int)

        # weighted document lengths
        self.doc_lengths = {}

        # average weighted document length
        self.avg_doc_length = 0.0

        # build the index when the class is created
        self.build_index()

    def build_index(self):
        total_length = 0.0
        for doc_id, doc in enumerate(self.documents):
            weighted_term_counts = defaultdict(float)
            weighted_doc_length = 0.0
            for field, weight in self.field_weights.items():
                tokens = doc.get(field, [])
                weighted_doc_length += len(tokens) * weight
                for token in tokens:
                    weighted_term_counts[token] += weight

            self.doc_lengths[doc_id] = weighted_doc_length
            total_length += weighted_doc_length

            for term, weighted_tf in weighted_term_counts.items():
                self.inverted_index[term][doc_id] = weighted_tf

            for term in weighted_term_counts:
                self.doc_freqs[term] += 1

        if self.N > 0:
            self.avg_doc_length = total_length / self.N
        else:
            self.avg_doc_length = 0.0

    def compute_idf(self, term):
        df = self.doc_freqs.get(term, 0)
        if df == 0:
            return 0.0
        return math.log(((self.N - df + 0.5) / (df + 0.5)) + 1)

    def score(self, query_tokens, doc_id):
        score = 0.0
        doc_length = self.doc_lengths.get(doc_id, 0.0)
        if self.avg_doc_length == 0:
            return 0.0

        for term in query_tokens:
            if term not in self.inverted_index:
                continue
            tf = self.inverted_index[term].get(doc_id, 0.0)
            if tf == 0:
                continue
            idf = self.compute_idf(term)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
            score += idf * (numerator / denominator)
        return score

    def search(self, query_tokens, top_k=10):
        results = []
        for doc_id in range(self.N):
            doc_score = self.score(query_tokens, doc_id)
            if doc_score > 0:
                results.append((doc_id, doc_score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def explain_score(self, query_tokens, doc_id):
        explanation = []
        total_score = 0.0
        doc_length = self.doc_lengths.get(doc_id, 0.0)
        if self.avg_doc_length == 0:
            return {"doc_id": doc_id, "total_score": 0.0, "details": []}

        for term in query_tokens:
            if term not in self.inverted_index:
                explanation.append({"term": term, "tf": 0.0, "idf": 0.0, "term_score": 0.0})
                continue
            tf = self.inverted_index[term].get(doc_id, 0.0)
            idf = self.compute_idf(term)
            if tf == 0:
                explanation.append({"term": term, "tf": 0.0, "idf": round(idf, 4), "term_score": 0.0})
                continue
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
            term_score = idf * (numerator / denominator)
            total_score += term_score
            explanation.append({"term": term, "tf": round(tf, 4), "idf": round(idf, 4), "term_score": round(term_score, 4)})
        return {"doc_id": doc_id, "total_score": round(total_score, 4), "details": explanation}


# Quick query interface
if __name__ == "__main__":
    # Load dataset
    df = load_dataset(dataset_path)

    # Preprocess for BM25
    corpus = preprocess_corpus_for_bm25(df)

    # Create BM25 retriever
    bm25 = BM25Retriever(corpus)

    # Ask user for query
    query = input("Enter your job search query: ").strip()
    query_tokens, _ = preprocess_query(query)

    # Retrieve top 15 jobs
    results = bm25.search(query_tokens, top_k=15)

    print(f"\nQuery: '{query}'\nTop jobs:")
    for rank, (doc_id, score) in enumerate(results, start=1):
        row = df.loc[doc_id]
        title = row.get("Title", "")
        exp_level = row.get("ExperienceLevel", "")
        skills = row.get("Skills", "")
        responsibilities = row.get("Responsibilities", "")
        keywords = row.get("Keywords", "")
        print(f"{rank}. Title: {title}\n"
              f"   Experience Level: {exp_level}\n"
              f"   Skills: {skills}\n"
              f"   Responsibilities: {responsibilities}\n"
              f"   Keywords: {keywords}\n"
              f"   BM25 score: {score:.4f}\n")
