"""把流水线编译成 LangGraph StateGraph。

节点：plan -> search -> analyze（按论文并行 N 路）-> compare -> write -> verify
条件边：verify 之后若仍有不合格句子且轮次 < 2，走 revise 再回 verify；
        否则直接 finalize。
"""

from __future__ import annotations

NODE_NAMES = (
    "plan",
    "search",
    "analyze",
    "compare",
    "write",
    "verify",
    "revise",
    "finalize",
)


def build_graph(*, use_revise: bool = True, max_concurrency: int = 8):
    """构造并编译状态图，返回可 invoke 的对象。"""
    raise NotImplementedError


def route_after_verify(state: dict) -> str:
    """条件边：根据剩余不合格句子数与当前轮次，返回下一个节点名。"""
    raise NotImplementedError
