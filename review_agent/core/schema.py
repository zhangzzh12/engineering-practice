"""全链路共享的数据契约。

改这里的字段等于改协议：所有智能体、存储与网页都依赖它，
因此字段增删必须走 PR，并在描述里标注 breaking。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Paper:
    """一篇论文的元数据。"""

    paper_id: str                      # arXiv id，例如 2309.01431
    title: str
    abstract: str = ""
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    url: str = ""


@dataclass(slots=True)
class Span:
    """可被引用的一段原文，span_id 形如 ``2309.01431#s1``。"""

    span_id: str
    paper_id: str
    section: str
    text: str


@dataclass(slots=True)
class PaperCard:
    """一篇论文的结构化阅读卡片；每个陈述字段都必须能对上原文片段。"""

    paper_id: str
    problem: str = ""
    method: str = ""
    novelty: str = ""
    datasets: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    limitations: str = ""
    spans: list[Span] = field(default_factory=list)


@dataclass(slots=True)
class Family:
    """方法族：共享同一假设的一组论文。"""

    family_id: str
    label: str
    paper_ids: list[str] = field(default_factory=list)
    rationale: str = ""


@dataclass(slots=True)
class Claim:
    """综述正文里的一句话及其证据绑定情况。"""

    text: str
    span_id: str | None = None
    status: str = "unverified"         # verified / repaired / dropped
    reason: str = ""
