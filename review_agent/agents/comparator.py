"""对比智能体：阅读卡片 -> 方法族 + 演进边 + 指标冲突。

聚类依据是「标题 + 问题 + 方法」的签名，而不是发表刊物；
单篇族自动并入最近族，避免出现一堆只有一个成员的方法族。
"""

from __future__ import annotations

from review_agent.core.schema import Family, PaperCard


def compare(cards: list[PaperCard], *, threshold: float = 0.15) -> list[Family]:
    """聚出方法族，并按时间给出演进边。"""
    raise NotImplementedError


def detect_conflicts(cards: list[PaperCard]) -> list[tuple[str, str, str]]:
    """检出同一数据集上指标口径矛盾的论文对。"""
    raise NotImplementedError
