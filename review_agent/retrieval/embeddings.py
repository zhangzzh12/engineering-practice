"""向量后端。

接口固定为「文本列表 -> 同维度向量列表」，
以便在真实 embedding 服务与离线哈希向量之间切换，而上层代码不变。
"""

from __future__ import annotations


class Embedder:
    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def available(self) -> bool:
        """是否接了真实的向量服务。"""
        raise NotImplementedError

    def encode(self, texts: list[str]) -> list[list[float]]:
        """批量编码，返回与输入等长的向量列表。"""
        raise NotImplementedError
