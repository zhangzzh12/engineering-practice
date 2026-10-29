"""骨架自检：包能导入、契约字段没被误改。

真实测试随模块实现逐步补充，见 README 的「实现顺序」。
"""

from __future__ import annotations

import review_agent
from review_agent.core.schema import Claim, Paper, Span


def test_version() -> None:
    assert review_agent.__version__


def test_contract_defaults() -> None:
    paper = Paper(paper_id="2309.01431", title="示例")
    span = Span(span_id="2309.01431#s1", paper_id=paper.paper_id, section="abstract", text="...")
    claim = Claim(text="...", span_id=span.span_id)
    assert claim.span_id == "2309.01431#s1"
    assert claim.status == "unverified"
