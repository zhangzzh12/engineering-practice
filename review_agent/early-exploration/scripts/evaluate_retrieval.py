"""三级检索消融实验 v2：稀疏 / LSA / BGE / 特征重排 / 交叉编码器重排。

评测集：
1. 标题检索集（48 条）：以每篇论文标题为查询，找回该论文摘要块；
2. 领域查询集（6 条）：手写中文查询 + 人工指定相关文献。

指标：Recall@1、Recall@5、MRR、nDCG@5（论文级二值相关）。
"""
from __future__ import annotations

import json
import math
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402
from app.models import Paper  # noqa: E402
from app.rag import RagIndex  # noqa: E402

DOMAIN_QUERIES = [
    ("RAG 综述：检索与生成如何结合", [
        "Retrieval-Augmented Generation for Large Language Models: A Survey",
        "A Survey on Retrieval-Augmented Text Generation",
        "Retrieval-Augmented Generation for AI-Generated Content",
    ]),
    ("RAG 的评测基准与忠实性", [
        "Benchmarking Large Language Models in Retrieval-Augmented Generation",
        "CiteGuard",
    ]),
    ("图神经网络推荐与冷启动", [
        "A graph neural network recommendation algorithm",
        "A Rating-Enhanced Graph Neural Network Recommendation",
        "Meta-level based recommender system",
        "A Service Recommendation Model in Cloud Environment",
        "From Collaborative Filtering to Graph Neural Networks",
    ]),
    ("从协同过滤到图神经网络的演进", [
        "From Collaborative Filtering to Graph Neural Networks",
    ]),
    ("大模型做程序修复", [
        "A Systematic Literature Review on Large Language Models for Automated",
        "Exploring Generalizable Automated Program Repair",
        "Monte Carlo Tree Search for Execution-Guided Program Repair",
    ]),
    ("代码生成的安全性评测", [
        "CodeSafetyBench",
    ]),
]

# (配置名, embedding provider, rerank provider, 检索模式)
CONFIGS = [
    ("sparse", None, None, "sparse"),
    ("lsa_dense", "lsa", None, "dense"),
    ("lsa_hybrid3", "lsa", "feature", "hybrid3"),
    ("bge_dense", "fastembed", None, "dense"),
    ("bge_hybrid", "fastembed", None, "hybrid"),
    ("bge_feature", "fastembed", "feature", "hybrid3"),
    ("bge_cross", "fastembed", "cross", "hybrid3"),
    ("bge_learned", "fastembed", "learned", "hybrid3"),
]


def paper_ranking(index: RagIndex, query: str, mode: str, top_k: int = 5) -> list:
    results = index.search(query, k=top_k * 3, kinds=["abstract"], mode=mode)
    ranking, seen = [], set()
    for chunk, _ in results:
        if chunk.paper_id not in seen:
            seen.add(chunk.paper_id)
            ranking.append(chunk.paper_id)
    return ranking[:top_k]


def metrics(ranking: list, relevant: set) -> dict:
    recall1 = 1.0 if ranking and ranking[0] in relevant else 0.0
    recall5 = len(set(ranking) & relevant) / max(1, len(relevant))
    mrr = 0.0
    for rank, paper_id in enumerate(ranking, start=1):
        if paper_id in relevant:
            mrr = 1.0 / rank
            break
    dcg = sum(1.0 / math.log2(rank + 1)
              for rank, paper_id in enumerate(ranking, start=1) if paper_id in relevant)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), 5) + 1))
    return {"r1": recall1, "r5": recall5, "mrr": mrr, "ndcg5": dcg / ideal if ideal else 0.0}


def build_index(papers, provider):
    original = config.EMBEDDING_PROVIDER
    config.EMBEDDING_PROVIDER = provider or "lsa"
    try:
        index = RagIndex()
        index.add_papers(papers)
        return index
    finally:
        config.EMBEDDING_PROVIDER = original


def main() -> int:
    papers = [Paper(**item) for item in json.loads(
        (config.FIXTURE_DIR / "corpus.json").read_text(encoding="utf-8"))]
    title_queries = [(re.sub(r"\s+", " ", paper.title).strip(), {paper.paper_id})
                     for paper in papers]
    domain_queries = []
    for query, fragments in DOMAIN_QUERIES:
        relevant = {paper.paper_id for paper in papers
                    if any(fragment.lower() in paper.title.lower() for fragment in fragments)}
        if relevant:
            domain_queries.append((query, relevant))

    indexes = {}
    print(f"语料 {len(papers)} 篇")
    for name, provider, _, _ in CONFIGS:
        if provider in indexes or provider is None:
            continue
        start = time.time()
        index = build_index(papers, provider)
        indexes[provider] = index
        print(f"  索引 {provider or 'sparse'}：{len(index.chunks)} 块，"
              f"{index.embedder.name if index.embedder else 'none'}，"
              f"构建 {time.time() - start:.1f}s")

    report = {"corpus": {"papers": len(papers)},
              "configs": [name for name, *_ in CONFIGS],
              "sets": {}}
    for set_name, queries in (("title", title_queries), ("domain", domain_queries)):
        rows = {}
        for name, provider, rerank, mode in CONFIGS:
            index = indexes[provider] if provider else indexes["lsa"]
            original_rerank = config.RERANK_PROVIDER
            if rerank:
                config.RERANK_PROVIDER = rerank
            try:
                if index.reranker:
                    index.reranker.mode = rerank or config.RERANK_PROVIDER
                start = time.time()
                scores = [metrics(paper_ranking(index, query, mode), relevant)
                          for query, relevant in queries]
                elapsed = time.time() - start
            finally:
                config.RERANK_PROVIDER = original_rerank
            rows[name] = {
                "r1": round(sum(item["r1"] for item in scores) / len(scores), 3),
                "r5": round(sum(item["r5"] for item in scores) / len(scores), 3),
                "mrr": round(sum(item["mrr"] for item in scores) / len(scores), 3),
                "ndcg5": round(sum(item["ndcg5"] for item in scores) / len(scores), 3),
                "latency_ms": round(elapsed / len(queries) * 1000, 1),
            }
        report["sets"][set_name] = rows
        print(f"\n[{set_name}]  ({len(queries)} queries)")
        print("  配置            R@1    R@5    MRR   nDCG@5   延迟/查询")
        for name, *_ in CONFIGS:
            row = rows[name]
            print(f"  {name:<14} {row['r1']:.3f}  {row['r5']:.3f}  {row['mrr']:.3f}  "
                  f"{row['ndcg5']:.3f}  {row['latency_ms']:>7.1f} ms")

    out = config.RESULT_DIR / "retrieval_eval.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
