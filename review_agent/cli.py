"""命令行入口：解析参数 -> RunConfig -> 跑一次运行。"""

from __future__ import annotations

import sys

from review_agent.config import RunConfig


def build_parser():
    """构造 argparse 解析器；参数与 RunConfig 字段一一对应。"""
    raise NotImplementedError


def main(argv: list[str] | None = None) -> int:
    """返回进程退出码：0 成功，非 0 表示参数或运行错误。"""
    raise NotImplementedError


def to_config(args) -> RunConfig:
    """argparse 的 Namespace -> RunConfig。"""
    raise NotImplementedError


if __name__ == "__main__":
    sys.exit(main())
