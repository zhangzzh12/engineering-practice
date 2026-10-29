"""文本小工具：分词、分句、数字抽取、相似度。

全部是纯函数，不依赖任何状态，便于单测。
"""

from __future__ import annotations


def tokenize(text: str) -> list[str]:
    """切词并小写化：英文按空白与标点切，中文按字切。"""
    raise NotImplementedError


def split_sentences(text: str) -> list[str]:
    """按句末标点分句，避免把缩写（et al. / e.g.）误切。"""
    raise NotImplementedError


def extract_numbers(text: str) -> list[str]:
    """抽出数字字面量，用于校验「数字一致性」。"""
    raise NotImplementedError


def jaccard(a: str, b: str) -> float:
    """两个文本的 token 集合 Jaccard 相似度，返回 0~1。"""
    raise NotImplementedError
