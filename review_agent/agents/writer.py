"""撰写智能体：大纲 -> 章节 -> Markdown。

模板定事实、模型管语言：骨架句自带论文与片段编号，
模型只能在证据白名单内改写文风，越界引用直接丢弃。
"""

from __future__ import annotations

from review_agent.core.schema import Claim, Family


def build_outline(topic: str, families: list[Family]) -> list[str]:
    """按方法族生成章节大纲。"""
    raise NotImplementedError


def write_section(heading: str, families: list[Family]) -> list[Claim]:
    """生成一节；每条事实句都必须挂 span_id。"""
    raise NotImplementedError


def render(claims: list[Claim], title: str) -> str:
    """Claim 列表 -> Markdown 正文（含角标与参考文献）。"""
    raise NotImplementedError
