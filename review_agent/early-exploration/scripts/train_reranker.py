"""训练自研学习式重排器（修正版：用摘要句子做查询，避免标题泄漏）。"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402
from app.learned_rerank import LinearListNet, TinyCrossEncoder, build_features  # noqa: E402
from app.models import Paper  # noqa: E402
from app.rag import RagIndex  # noqa: E402
from app.rerank import FeatureReranker  # noqa: E402
from app.textutil import cue_sentences, sentences, tokens  # noqa: E402

TOP_K = 5
CANDIDATES = 30


def candidates_for(index, query):
    terms = tokens(query)
    sparse = index._sparse_scores(terms, None, {"abstract"})
    dense = index._dense_scores(query, None, {"abstract"})
    max_s = max(sparse.values(), default=1.0) or 1.0
    max_d = max(dense.values(), default=1.0) or 1.0
    pool = {}
    for pos, val in sorted(sparse.items(), key=lambda kv: -kv[1])[:CANDIDATES]:
        pool[pos] = {"chunk": index.chunks[pos], "sparse": val / max_s,
                     "dense": dense.get(pos, 0.0) / max_d}
    for pos, val in sorted(dense.items(), key=lambda kv: -kv[1])[:CANDIDATES]:
        entry = pool.setdefault(pos, {"chunk": index.chunks[pos], "sparse": 0.0, "dense": 0.0})
        entry["dense"] = max(entry["dense"], val / max_d)
    return list(pool.values())


def dedupe(ranking):
    seen, out = set(), []
    for paper_id in ranking:
        if paper_id not in seen:
            seen.add(paper_id)
            out.append(paper_id)
    return out


def metrics(ranking, relevant):
    ranking = dedupe(ranking)[:TOP_K]
    recall1 = 1.0 if ranking and ranking[0] in relevant else 0.0
    recall5 = len(set(ranking) & relevant) / max(1, len(relevant))
    mrr = 0.0
    for rank, paper_id in enumerate(ranking, start=1):
        if paper_id in relevant:
            mrr = 1.0 / rank
            break
    dcg = sum(1.0 / math.log2(rank + 1)
              for rank, paper_id in enumerate(ranking, start=1) if paper_id in relevant)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), TOP_K) + 1))
    return {"r1": recall1, "r5": recall5, "mrr": mrr, "ndcg5": dcg / ideal if ideal else 0.0}


def make_queries(paper):
    abstract = paper.abstract or ""
    picks = cue_sentences(abstract, "method", 1) + cue_sentences(abstract, "experiment", 1)
    if not picks:
        picks = sentences(abstract)[:2]
    queries = []
    for text in picks[:2]:
        text = text.strip()
        if 40 <= len(text) <= 320:
            queries.append(text)
    return queries


def main() -> int:
    papers = [Paper(**item) for item in json.loads(
        (config.FIXTURE_DIR / "corpus.json").read_text(encoding="utf-8"))]
    original = config.EMBEDDING_PROVIDER
    config.EMBEDDING_PROVIDER = "fastembed"
    start = time.time()
    index = RagIndex()
    index.add_papers(papers)
    config.EMBEDDING_PROVIDER = original
    print(f"索引 {len(index.chunks)} 块，构建 {time.time() - start:.1f}s，编码器 {index.embedder.name}")

    feature_baseline = FeatureReranker(index.idf, {p.paper_id: p.title for p in papers})
    groups = []
    for paper in papers:
        queries = [(paper.title, "title")] + [(q, "sentence") for q in make_queries(paper)]
        for query, kind in queries:
            candidates = candidates_for(index, query)
            labels = np.asarray([1.0 if c["chunk"].paper_id == paper.paper_id else 0.0
                                 for c in candidates])
            if labels.sum() == 0:
                continue
            query_vector = np.asarray(index.embedder.encode_query(query), dtype=np.float64)
            doc_vectors = np.asarray(
                [index.doc_vectors[index.chunk_index[c["chunk"].chunk_id]] for c in candidates],
                dtype=np.float64)
            groups.append({
                "query": query, "paper_id": paper.paper_id, "kind": kind,
                "features": build_features(index, query, candidates, feature_baseline),
                "labels": labels,
                "pair_features": TinyCrossEncoder.pair_matrix(query_vector, doc_vectors),
                "candidates": candidates, "relevant": {paper.paper_id},
            })
    print(f"训练查询 {len(groups)} 条，候选 {sum(len(g['labels']) for g in groups)} 个")

    # 按论文分组划分 5 折，避免同一论文的查询跨训练/测试
    papers_with_queries = sorted({g["paper_id"] for g in groups})
    fold_of_paper = {pid: i % 5 for i, pid in enumerate(papers_with_queries)}
    results = {"feature": [], "linear": [], "cross": []}
    by_kind = {"title": {"feature": [], "linear": []}, "sentence": {"feature": [], "linear": []}}

    for fold in range(5):
        train = [g for g in groups if fold_of_paper[g["paper_id"]] != fold]
        test = [g for g in groups if fold_of_paper[g["paper_id"]] == fold]
        linear = LinearListNet()
        linear.fit(train)
        cross = TinyCrossEncoder()
        cross.fit(train)
        for group in test:
            candidates, relevant = group["candidates"], group["relevant"]
            ranked = feature_baseline.rerank(group["query"], candidates, TOP_K)
            feature_metric = metrics([c.paper_id for c, _, _ in ranked], relevant)
            results["feature"].append(feature_metric)
            scores = linear.scores(build_features(index, group["query"], candidates,
                                                 feature_baseline))
            linear_metric = metrics(
                [candidates[i]["chunk"].paper_id for i in np.argsort(-scores)], relevant)
            results["linear"].append(linear_metric)
            by_kind[group["kind"]]["feature"].append(feature_metric)
            by_kind[group["kind"]]["linear"].append(linear_metric)
            scores = cross.scores(group["pair_features"])
            results["cross"].append(metrics(
                [candidates[i]["chunk"].paper_id for i in np.argsort(-scores)], relevant))

    print("\n5 折交叉验证（候选池 Top-30）")
    print("  模型          R@1     R@5     MRR    nDCG@5")
    summary = {}
    for name, rows in results.items():
        summary[name] = {key: round(sum(row[key] for row in rows) / len(rows), 3)
                         for key in ("r1", "r5", "mrr", "ndcg5")}
        row = summary[name]
        print(f"  {name:<12} {row['r1']:.3f}  {row['r5']:.3f}  {row['mrr']:.3f}  {row['ndcg5']:.3f}")

    print("\n按查询类型拆分（nDCG@5 / R@1）")
    for kind, rows in by_kind.items():
        if not rows["feature"]:
            continue
        f = sum(r["ndcg5"] for r in rows["feature"]) / len(rows["feature"])
        l = sum(r["ndcg5"] for r in rows["linear"]) / len(rows["linear"])
        fr = sum(r["r1"] for r in rows["feature"]) / len(rows["feature"])
        lr = sum(r["r1"] for r in rows["linear"]) / len(rows["linear"])
        print(f"  {kind:<9} n={len(rows['feature']):>3}  特征重排 {f:.3f}/{fr:.3f}  学习式 {l:.3f}/{lr:.3f}")

    linear = LinearListNet()
    linear.fit(groups, epochs=600)
    cross = TinyCrossEncoder()
    cross.fit(groups, epochs=150)
    config.MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = config.MODEL_CACHE_DIR / "reranker_learned.json"
    out.write_text(json.dumps({"models": {"linear": linear.to_dict(), "cross": cross.to_dict()},
                               "metrics_cv": summary}, ensure_ascii=False), encoding="utf-8")
    print(f"\n模型已保存 {out}")
    report = {"folds": 5, "queries": len(groups), "cv": summary,
              "feature_weights": dict(zip(
                  ["dense", "sparse", "coverage", "phrase", "chunk_len",
                   "query_len", "rare_cov", "feature_score"], linear.weights.round(4).tolist()))}
    (config.RESULT_DIR / "learned_rerank.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
