"""把论文的原始文本变成可引用的 span。"""

from __future__ import annotations

from review_agent.core.schema import Paper, Span


def to_spans(paper: Paper, *, with_fulltext: bool = True) -> list[Span]:
    """论文 -> span 列表；摘要至少产出一个 span。"""
    raise NotImplementedError
