"""持久化：默认 SQLite，给了 mysql:// URL 就换 MySQL。"""

from __future__ import annotations

TABLES = ("runs", "papers", "spans", "cards", "claims", "issues", "artifacts")


class Store:
    def __init__(self, url: str = "sqlite:///data/runs.sqlite3") -> None:
        self.url = url

    def init_schema(self) -> None:
        """建表（幂等）。"""
        raise NotImplementedError

    def save_run(self, run_id: str, state: dict) -> None:
        raise NotImplementedError

    def list_runs(self, limit: int = 20) -> list[dict]:
        raise NotImplementedError

    def load_run(self, run_id: str) -> dict:
        raise NotImplementedError
