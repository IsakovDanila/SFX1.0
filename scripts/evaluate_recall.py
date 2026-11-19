"""Recall@K evaluation (deck: HADI-3 / R2, gate Recall@10 >= 0.5).

Run:  python scripts/evaluate_recall.py
Uses data/demo_sfx + scripts/demo_queries.json. Rebuilds the index if empty.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "backend")))

from app.store import JsonStore  # noqa: E402
from app import embeddings as E  # noqa: E402

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(BASE, "data", "demo_sfx")
QUERIES = os.path.join(os.path.dirname(__file__), "demo_queries.json")


def run(store: JsonStore | None = None, top_k: int = 10) -> dict:
    store = store or JsonStore(embedder=E.get_embedder())
    if not store.items:
        print("[eval] index empty, scanning demo dataset...")
        store.scan_folder(DATA)
    with open(QUERIES, encoding="utf-8") as f:
        queries = json.load(f)

    r5_hits = r10_hits = 0
    times: list[float] = []
    details = []
    for q in queries:
        t0 = time.time()
        res = store.search_text(q["query"], top_k=max(top_k, 10))
        times.append((time.time() - t0) * 1000)
        cats = [r["file"]["category"] for r in res["results"]]
        hit5 = q["expected"] in cats[:5]
        hit10 = q["expected"] in cats[:10]
        r5_hits += hit5
        r10_hits += hit10
        details.append({"query": q["query"], "expected": q["expected"],
                        "top5": cats[:5], "hit@5": hit5, "hit@10": hit10})
    n = len(queries)
    summary = {
        "mode": store.embedder.name,
        "n_queries": n, "n_files": len(store.items),
        "recall@5": round(r5_hits / n, 3),
        "recall@10": round(r10_hits / n, 3),
        "avg_ms": round(sum(times) / len(times), 1),
        "gate_recall@10>=0.5": bool(r10_hits / n >= 0.5),
        "details": details,
    }
    with open(os.path.join(BASE, "data", "eval_results.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


if __name__ == "__main__":
    s = run()
    print(f"\nmode      : {s['mode']}")
    print(f"files     : {s['n_files']}, queries: {s['n_queries']}")
    print(f"Recall@5  : {s['recall@5']}")
    print(f"Recall@10 : {s['recall@10']}  "
          f"{'PASS (>=0.5)' if s['gate_recall@10>=0.5'] else 'FAIL (<0.5)'}")
    print(f"avg time  : {s['avg_ms']} ms")
    print("\nmisses:")
    for d in s["details"]:
        if not d["hit@5"]:
            print(f"  [miss@5] {d['query']!r} -> {d['top5']}")
