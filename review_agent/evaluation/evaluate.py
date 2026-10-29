"""离线评测：一次运行 vs 人工金标准。

五项指标：覆盖率、论点覆盖、结构完整性、引用精度、引用存在率。
"""

from __future__ import annotations

METRICS = (
    "coverage",
    "must_cover",
    "structure",
    "citation_precision",
    "citation_existence",
)


def load_gold(path: str) -> dict:
    """读取金标准：must_cover 是论点关键词，must_cite 是必引论文 id。"""
    raise NotImplementedError


def evaluate_run(run_dir: str, gold: dict) -> dict:
    """对一次运行目录打分，返回各指标数值。"""
    raise NotImplementedError


def to_markdown(report: dict) -> str:
    """结果 -> Markdown 报告。"""
    raise NotImplementedError
