"""arXiv Atom API 客户端：检索、解析、缓存、限流。"""

from __future__ import annotations

from review_agent.core.schema import Paper


class ArxivClient:
    """arXiv 检索客户端。

    约束：
    - 请求间隔不小于 3 秒（arXiv 使用条款）；
    - 响应缓存到磁盘，重复运行不重复下载。
    """

    def __init__(self, cache_dir: str = "data/cache", interval: float = 3.0) -> None:
        self.cache_dir = cache_dir
        self.interval = interval

    def search(self, query: str, max_results: int = 20) -> list[Paper]:
        """按查询取回论文元数据。"""
        raise NotImplementedError

    def fetch_abstract(self, paper_id: str) -> str:
        """按 id 取单篇摘要。"""
        raise NotImplementedError
