import pandas as pd

def load_qrels(path):
    """
    Load relevance judgments from CSV file.
    Assumes columns: query_id, JobID, relevance
    """
    qrels_df = pd.read_csv(path)

    required_columns = ["query_id", "JobID", "relevance"]
    missing = [column for column in required_columns if column not in qrels_df.columns]
    if missing:
        raise ValueError(f"Qrels CSV missing columns: {missing}")

    return qrels_df[required_columns].copy()
