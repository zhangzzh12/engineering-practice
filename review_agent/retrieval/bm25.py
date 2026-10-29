"""Okapi BM25，纯 Python 实现，无第三方依赖。"""

from __future__ import annotations


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def fit(self, corpus: list[list[str]]) -> None:
        """建立倒排索引与文档长度统计。"""
        raise NotImplementedError

    def search(self, query: list[str], k: int = 10) -> list[tuple[int, float]]:
        """返回 (文档下标, 得分)，按得分降序。"""
        raise NotImplementedError
