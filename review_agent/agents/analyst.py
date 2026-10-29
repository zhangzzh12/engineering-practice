"""读卡智能体：论文 -> PaperCard。

约束：每个陈述字段必须附一段能在原文精确匹配的引文；
匹配不上就把该字段置空，不允许编造。
"""

from __future__ import annotations

from review_agent.core.schema import Paper, PaperCard


def analyse(paper: Paper) -> PaperCard:
    """生成阅读卡片；没有模型服务时走启发式抽取。"""
    raise NotImplementedError


def attach_quote(card: PaperCard, field: str, span_text: str, quote: str) -> bool:
    """把引文挂到字段上；原文中找不到该引文时返回 False。"""
    raise NotImplementedError
