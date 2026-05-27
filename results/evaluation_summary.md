# Evaluation Summary

This evaluation uses 10 manually judged job-search queries and binary relevance judgements.
Hybrid fusion uses `alpha = 0.5`, where alpha is the BM25 weight:

`hybrid_score = alpha * BM25 + (1 - alpha) * Dense`

## Overall Result

Best overall model: **BM25**, with average P@5=0.680, nDCG@10=0.873, and MRR=0.770.

## Model Behaviour

- BM25 performs best on: Q1, Q2, Q3, Q4, Q6, Q7, Q9.
- Dense performs best on: Q5, Q8.
- Hybrid performs best on: Q10.

BM25 tends to do well when the query contains exact skill or role terms that appear in job titles, skills, or keywords.
Dense retrieval is more useful for descriptive and vocabulary-variation queries where the wording does not exactly match the posting.
Hybrid retrieval is strongest when both exact skill matching and semantic matching contribute useful evidence.

## Limitations

- The query set is still small and manually judged, so the scores should be treated as directional rather than definitive.
- Relevance judgements are binary and do not capture degrees of fit such as seniority, location, salary, or domain preference.
- Some job IDs in the source data are reused across job families, which can make ID-only evaluation less precise.
- The dense model is fixed to `all-MiniLM-L6-v2`; no extra model tuning or new retrieval model was added.
