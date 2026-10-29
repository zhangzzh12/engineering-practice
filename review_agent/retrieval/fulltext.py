"""论文全文获取与分节。

四级回退：arXiv HTML -> ar5iv HTML -> LaTeX 源码 -> PDF。
任一级失败换下一级；全部失败时回落到摘要，不中断本次运行。
连续多篇失败应熔断，避免几十次超时等待。
"""

from __future__ import annotations

from review_agent.core.schema import Paper, Span

SECTION_PREFIX = {
    "abstract": "s",
    "introduction": "i",
    "background": "b",
    "related work": "r",
    "method": "m",
    "experiments": "e",
    "analysis": "a",
    "discussion": "d",
    "conclusion": "c",
    "limitations": "l",
}


def section_prefix(section: str) -> str:
    """章节名 -> 单字母前缀；未知章节返回 x。"""
    raise NotImplementedError


def fetch_spans(
    paper: Paper,
    *,
    max_chars: int = 40_000,
    max_spans: int = 160,
    timeout: float = 20.0,
) -> list[Span]:
    """抓取全文并切成带章节前缀的 span（例如 2309.01431#m2）。"""
    raise NotImplementedError
