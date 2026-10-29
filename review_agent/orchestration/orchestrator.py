"""编排入口：接收 RunConfig，跑完一张图，统一落盘。"""

from __future__ import annotations

from review_agent.config import RunConfig


def run(config: RunConfig) -> dict:
    """执行一次完整运行，返回运行摘要（统计与产物路径）。"""
    raise NotImplementedError


def finalize(state: dict, out_dir: str) -> dict:
    """统一出口：写出 review.md / cards.json / stats.json / trace.jsonl 等产物。"""
    raise NotImplementedError
