"""规划智能体：主题 -> 子问题 + 检索 query。"""

from __future__ import annotations


def plan(topic: str, *, n: int = 4) -> list[str]:
    """把主题拆成 3~6 个可检索的子问题。"""
    raise NotImplementedError
