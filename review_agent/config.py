"""运行配置。

命令行参数与网页请求最终都收敛到这一个 dataclass，
避免各模块各自解析命令行或环境变量。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class RunConfig:
    """一次运行的全部可调参数。"""

    topic: str
    papers: int = 20
    out_dir: Path = Path("out")
    # auto：检测到模型服务就走模型，否则离线启发式；mock：强制离线
    mode: str = "auto"
    max_concurrency: int = 8
    use_fulltext: bool = True
    fulltext_max_chars: int = 40_000
    fulltext_max_spans: int = 160
    fulltext_timeout: float = 20.0
    use_revise: bool = True
    cluster_threshold: float = 0.15
    db_url: str | None = None
    extra: dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        """检查参数组合是否合法，非法时抛 ValueError。"""
        raise NotImplementedError
