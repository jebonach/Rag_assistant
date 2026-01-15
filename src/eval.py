from __future__ import annotations

from typing import Dict, List, Sequence, Tuple


def recall_at_k(ranked_pages: Sequence[int], gold_page: int, k: int) -> float:
    return 1.0 if gold_page in ranked_pages[:k] else 0.0


def mrr_at_k(ranked_pages: Sequence[int], gold_page: int, k: int) -> float:
    topk = ranked_pages[:k]
    for i, p in enumerate(topk, start=1):
        if p == gold_page:
            return 1.0 / i
    return 0.0


def evaluate_questions(
    questions: List[Dict],
    run_retrieve,  # callable: (query, mode) -> hits
    modes: List[str],
    ks: List[int] = [3, 5],
) -> List[Dict]:
    rows = []
    for mode in modes:
        for k in ks:
            recs, mrrs = [], []
            for q in questions:
                hits = run_retrieve(q["query"], mode)
                ranked_pages = [int(h["page"]) for h in hits]
                gold = int(q["gold_page"])
                recs.append(recall_at_k(ranked_pages, gold, k))
                mrrs.append(mrr_at_k(ranked_pages, gold, k))
            rows.append(
                {
                    "mode": mode,
                    "k": k,
                    "recall@k": sum(recs) / len(recs) if recs else 0.0,
                    "mrr@k": sum(mrrs) / len(mrrs) if mrrs else 0.0,
                    "n_questions": len(questions),
                }
            )
    return rows
