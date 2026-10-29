"""校验智能体：句子进入正文前的最后一道关。

顺序：span 存在性 -> 数字一致性 -> 双向词面覆盖。
不通过时先尝试在同一篇论文内重挂证据，仍不行才丢弃。
"""

from __future__ import annotations

from review_agent.core.schema import Claim, Span

RECALL_FLOOR = 0.15
COVERAGE_THRESHOLD = 0.20


def verify(claim: Claim, span: Span) -> tuple[bool, str]:
    """返回 (是否通过, 不通过原因)。"""
    raise NotImplementedError


def repair(claim: Claim, candidates: list[Span]) -> Claim | None:
    """尝试把句子重挂到更合适的 span 上（需过相似度门槛），失败返回 None。"""
    raise NotImplementedError
