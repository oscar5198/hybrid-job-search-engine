import os
import sys
import pandas as pd

_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from hybrid import HybridSearchEngine

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "job_dataset.csv")
DEFAULT_ALPHA = 0.5
DEFAULT_TOP_K = 10

def print_results(results: pd.DataFrame, query: str) -> None:
    """Pretty-print ranked search results, showing the hybrid score."""
    print(f"\n{'='*60}")
    print(f"Query: \"{query}\"")
    print(f"{'='*60}")

    if results.empty:
        print("No results found.")
        return

    for _, row in results.iterrows():
        print(f"\nRank #{int(row['Rank'])}")
        print(f"JobID: {row['JobID']}")
        print(f"Title: {row['Title']}")
        print(f"Experience Level: {row['ExperienceLevel']}")
        print(f"Skills: {row['Skills']}")
        resp_preview = str(row['Responsibilities'])
        if len(resp_preview) > 200:
            resp_preview = resp_preview[:200] + "..."
        print(f"Responsibilities: {resp_preview}")
        print(f"Keywords: {row['Keywords']}")
        print(f"Hybrid Score: {row['HybridScore']:.4f}")
    print(flush=True)

def run_interactive(engine: HybridSearchEngine) -> None:
    print("\nHybrid Job Search Engine - Interactive Mode")
    print(f"Current alpha (BM25 weight): {engine.alpha}")
    print("Commands: 'alpha <0-1>' to change weight | 'exit' to quit\n")

    last_query = ""
    while True:
        try:
            user_input = input("Search Job: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye.")
            break

        # Change alpha on the fly
        if user_input.lower().startswith("alpha "):
            parts = user_input.split()
            if len(parts) == 2:
                try:
                    new_alpha = float(parts[1])
                    if 0.0 <= new_alpha <= 1.0:
                        engine.alpha = new_alpha
                        print(f"  Alpha updated to {new_alpha} "
                              f"(BM25 weight={new_alpha}, dense weight={1-new_alpha})")
                    else:
                        print("  alpha must be between 0.0 and 1.0")
                except ValueError:
                    print("  Invalid alpha value. Usage: alpha 0.6")
            continue

        last_query = user_input
        try:
            results = engine.search(user_input)
            print_results(results, user_input)
        except Exception as exc:
            print(f"  Error during search: {exc}")

def main() -> None:
    data_path = os.environ.get("JOB_DATA_PATH", DATA_PATH)
    alpha = float(os.environ.get("HYBRID_ALPHA", DEFAULT_ALPHA))

    print(f"Data path : {data_path}")
    print(f"Alpha     : {alpha}  (BM25 weight)")

    engine = HybridSearchEngine(
        data_path=data_path,
        alpha=alpha,
        top_k=DEFAULT_TOP_K
    )

    engine.build_index()
    run_interactive(engine)

if __name__ == "__main__":
    main()
