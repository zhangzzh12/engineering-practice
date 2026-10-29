"""检索层：多 query 检索、RRF 融合、段落索引、引用图扩展。"""

from __future__ import annotations

from review_agent.core.schema import Paper, Span


def search_papers(topic: str, n: int, *, expand_citations: bool = False) -> list[Paper]:
    """按四类查询意图（定义 / 综述 / 评测 / 难点）并发检索，再融合去重。"""
    raise NotImplementedError


def rrf_merge(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion：按名次而非分值融合多路结果。"""
    raise NotImplementedError


def passage_search(query: str, spans: list[Span], k: int = 5) -> list[Span]:
    """段落级 BM25 + 向量双路召回，RRF 融合后返回 top-k。"""
    raise NotImplementedError
