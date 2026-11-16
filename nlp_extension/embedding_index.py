"""Semantic retrieval over the remarks: sentence embeddings + FAISS vs TF-IDF.

Each query is a natural-language description of a risk-relevant property
profile ("tenant occupied home", "fixer upper sold as-is", "condo with high
HOA dues", ...). Gold relevance is defined by a predicate over the gold
labels - relevance is known by construction, which is exactly what makes
the retrieval metrics (Recall@5/@10, MRR@10, nDCG@10) computable without
manual judging. Predicates are documented in QUERIES below.

Systems compared:
  * TF-IDF cosine baseline (the keyword-search stand-in)
  * all-MiniLM-L6-v2 embeddings, L2-normalized, FAISS IndexFlatIP
    (inner product on normalized vectors = cosine similarity)

Run:  python embedding_index.py  ->  output/retrieval_metrics.json
"""

import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

HERE = Path(__file__).resolve().parent

QUERIES = [
    ("tenant occupied property with a lease in place",
     lambda g: g["tenant_occupied"] == 1),
    ("fixer upper that needs a full renovation, sold as-is",
     lambda g: g["condition"] == "fixer_upper"),
    ("move-in ready renovated home with an updated kitchen",
     lambda g: g["condition"] == "turnkey_updated"),
    ("original condition home ready for a buyer's personal touches",
     lambda g: g["condition"] == "original_condition"),
    ("home in a flood zone where flood insurance is required",
     lambda g: g["flood_mention"] == 1),
    ("condo or townhome with high monthly HOA dues over 400 dollars",
     lambda g: (g["hoa_amount"] >= 400)),
    ("property with a pool and spa in the backyard",
     lambda g: g["pool_mention"] == 1),
    ("home with leased solar panels the buyer must assume",
     lambda g: g["solar_mention"] == "leased"),
    ("home with owned solar panels that convey with the sale",
     lambda g: g["solar_mention"] == "owned"),
    ("price reduced property from a motivated seller",
     lambda g: g["price_reduced_mention"] == 1),
    ("property with a detached ADU for rental income",
     lambda g: g["adu_mention"] == 1),
    ("tenant occupied fixer upper sold as-is to investors",
     lambda g: (g["tenant_occupied"] == 1) & (g["condition"] == "fixer_upper")),
    ("renovated home with a pool",
     lambda g: (g["condition"] == "turnkey_updated") & (g["pool_mention"] == 1)),
    ("flood zone property that is tenant occupied",
     lambda g: (g["flood_mention"] == 1) & (g["tenant_occupied"] == 1)),
    ("original condition property with price reduced",
     lambda g: (g["condition"] == "original_condition") & (g["price_reduced_mention"] == 1)),
    ("home with HOA dues and a pool",
     lambda g: (g["hoa_amount"] > 0) & (g["pool_mention"] == 1)),
]


def rank_metrics(ranked_idx: np.ndarray, relevant: set, k: int = 10) -> dict:
    top = ranked_idx[:k]
    hits = [1 if i in relevant else 0 for i in top]
    recall5 = sum(hits[:5]) / max(len(relevant), 1)
    recall10 = sum(hits) / max(len(relevant), 1)
    rr = next((1.0 / (j + 1) for j, h in enumerate(hits) if h), 0.0)
    dcg = sum(h / np.log2(j + 2) for j, h in enumerate(hits))
    ideal = [1] * min(len(relevant), k)
    idcg = sum(h / np.log2(j + 2) for j, h in enumerate(ideal)) or 1.0
    return {"recall@5": recall5, "recall@10": recall10, "mrr@10": rr,
            "ndcg@10": dcg / idcg}


def main() -> None:
    text = pd.read_csv(HERE / "data" / "property_text.csv")
    gold = pd.read_csv(HERE / "data" / "gold_labels.csv")
    df = text.merge(gold, on="ListingId").reset_index(drop=True)
    corpus = df["PublicRemarks"].tolist()
    queries = [q for q, _ in QUERIES]
    relevance = [set(np.where(pred(df))[0]) for _, pred in QUERIES]

    # --- TF-IDF baseline -----------------------------------------------------
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=20000, sublinear_tf=True)
    doc_mat = vec.fit_transform(corpus)
    q_mat = vec.transform(queries)
    tfidf_sims = cosine_similarity(q_mat, doc_mat)

    # --- sentence embeddings + FAISS ----------------------------------------
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    emb = model.encode(corpus, batch_size=64, normalize_embeddings=True,
                       show_progress_bar=False).astype("float32")
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    q_emb = model.encode(queries, normalize_embeddings=True).astype("float32")
    _, faiss_idx = index.search(q_emb, 10)

    results = {"n_docs": len(corpus), "n_queries": len(queries), "systems": {}}
    for name, rankings in [("tfidf", np.argsort(-tfidf_sims, axis=1)),
                           ("minilm_faiss", faiss_idx)]:
        per_q = [rank_metrics(rankings[i], relevance[i]) for i in range(len(queries))]
        agg = {m: round(float(np.mean([p[m] for p in per_q])), 4)
               for m in ["recall@5", "recall@10", "mrr@10", "ndcg@10"]}
        results["systems"][name] = agg
    out = HERE / "output"
    out.mkdir(exist_ok=True)
    with open(out / "retrieval_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
