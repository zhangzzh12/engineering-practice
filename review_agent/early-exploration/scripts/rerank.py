"""重排模块（三级检索的第三级）。

候选集来自稀疏（BM25）与稠密（LSA/API）两路召回，重排器对候选做「交叉特征」
打分，把真正回答了查询的块顶到前面。两种实现：

* ``FeatureReranker``：默认。特征包括稠密余弦、IDF 加权词覆盖、短语命中、
  标题命中与 BM25 归一化分数，权重经小规模网格搜索确定。纯本地、可解释。
* ``LLMReranker``：配置大模型后可选。把查询与候选片段交给模型做列表级重排，
  输出候选编号顺序；解析失败自动回退到特征重排。

之所以不叫「cross-encoder」：真正的交叉编码器需要对 (query, passage) 逐对过
Transformer，本项目在无 GPU / 无模型权重环境下用特征交叉近似，报告中对齐说明。
"""
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Sequence, Tuple

from . import config
from .models import Chunk
from .textutil import tokens

# 特征权重（在 48 篇语料的标题检索集上粗调，见 scripts/evaluate_retrieval.py）
WEIGHTS = {
    "dense": 0.32,
    "coverage": 0.27,
    "phrase": 0.11,
    "title": 0.10,
    "sparse": 0.20,
}

LLM_RERANK_PROMPT = """你是检索重排专家。给定查询与候选片段，按「能否直接回答查询」从高到低排序。
只输出 JSON：{"order": [候选编号, ...]}，不要解释，不要遗漏。"""


def _bigrams(terms: Sequence[str]) -> List[str]:
    return [f"{terms[i]} {terms[i + 1]}" for i in range(len(terms) - 1)]


class FeatureReranker:
    name = "feature"

    def __init__(self, idf: Dict[str, float], paper_titles: Dict[str, str],
                 title_weight: bool = True) -> None:
        self.idf = idf
        self.paper_titles = paper_titles
        self.title_weight = title_weight

    def rerank(self, query: str, candidates: List[dict], k: int) -> List[Tuple[Chunk, float, dict]]:
        query_terms = tokens(query)
        if not query_terms or not candidates:
            return []
        query_set = set(query_terms)
        idf_total = sum(self.idf.get(term, 1.0) for term in query_set) or 1.0
        pairs = _bigrams(query_terms)
        scored: List[Tuple[Chunk, float, dict]] = []
        for candidate in candidates:
            chunk: Chunk = candidate["chunk"]
            text_terms = set(tokens(chunk.text))
            matched = query_set & text_terms
            coverage = sum(self.idf.get(term, 1.0) for term in matched) / idf_total
            low = chunk.text.lower()
            phrase = (sum(1 for pair in pairs if pair in low) / len(pairs)) if pairs else 0.0
            title = 0.0
            if self.title_weight:
                title_terms = set(tokens(self.paper_titles.get(chunk.paper_id, "")))
                title = (len(query_set & title_terms) / len(query_set)) if query_set else 0.0
            score = (WEIGHTS["dense"] * candidate["dense"]
                     + WEIGHTS["coverage"] * coverage
                     + WEIGHTS["phrase"] * phrase
                     + WEIGHTS["title"] * title
                     + WEIGHTS["sparse"] * candidate["sparse"])
            scored.append((chunk, round(score, 5), {
                "dense": round(candidate["dense"], 4),
                "coverage": round(coverage, 4),
                "phrase": round(phrase, 4),
                "title": round(title, 4),
                "sparse": round(candidate["sparse"], 4),
            }))
        scored.sort(key=lambda item: -item[1])
        return scored[:k]


class LLMReranker:
    name = "llm"

    def __init__(self, llm) -> None:
        self.llm = llm

    @property
    def available(self) -> bool:
        return bool(self.llm and getattr(self.llm, "available", False))

    def rerank(self, query: str, candidates: List[dict], k: int) -> Optional[List[str]]:
        if not self.available or not candidates:
            return None
        listing = []
        for index, candidate in enumerate(candidates):
            listing.append(f"[{index}] {candidate['chunk'].text[:180]}")
        prompt = (f"查询：{query}\n\n候选片段：\n" + "\n".join(listing) +
                  f"\n\n请返回最相关的前 {k} 个编号。")
        try:
            data = self.llm.json(LLM_RERANK_PROMPT, prompt, max_tokens=400)
            order = [int(item) for item in data.get("order", [])]
        except Exception:  # noqa: BLE001
            return None
        valid = [item for item in order if 0 <= item < len(candidates)]
        if not valid:
            return None
        return [candidates[item]["chunk"].chunk_id for item in valid[:k]]


class CrossEncoderReranker:
    """ONNX 交叉编码器（fastembed TextCrossEncoder），默认 ms-marco-MiniLM-L-6-v2。

    与特征重排的区别：它把 (query, passage) 拼成一个序列送进 Transformer，
    逐对输出相关性分数，是真正的 cross-encoder，精度更高但更慢。
    """

    name = "cross"
    _CACHE: Dict[str, object] = {}

    def __init__(self, model_name: str = None, cache_dir: str = None) -> None:
        self.model_name = model_name or config.CROSS_ENCODER_MODEL
        self.cache_dir = cache_dir or str(config.MODEL_CACHE_DIR)
        self._model = None

    @property
    def available(self) -> bool:
        try:
            import fastembed  # noqa: F401
            return True
        except Exception:  # noqa: BLE001
            return False

    def _ensure(self) -> None:
        if self._model is not None:
            return
        import os

        os.environ.setdefault("HF_ENDPOINT", config.HF_ENDPOINT)
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        key = f"{self.model_name}::{self.cache_dir}"
        model = CrossEncoderReranker._CACHE.get(key)
        if model is None:
            model = TextCrossEncoder(model_name=self.model_name, cache_dir=self.cache_dir)
            CrossEncoderReranker._CACHE[key] = model
        self._model = model

    def rerank(self, query: str, candidates: List[dict], k: int):
        self._ensure()
        documents = [candidate["chunk"].text[:512] for candidate in candidates]
        try:
            scores = [float(score) for score in self._model.rerank(query, documents)]
        except Exception:  # noqa: BLE001
            return None
        if len(scores) != len(candidates):
            return None
        order = sorted(range(len(candidates)), key=lambda index: -scores[index])[:k]
        return [(candidates[index]["chunk"], round(scores[index], 4),
                 {"backend": "cross", "score": round(scores[index], 4)})
                for index in order]


def _mostly_ascii(text: str) -> bool:
    """判断查询是否以英文为主：学习式重排器只在英文分布上训练过。"""
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text or ""))
    letters = len(re.findall(r"[A-Za-z]", text or ""))
    if cjk + letters == 0:
        return True
    return cjk / (cjk + letters) < 0.05


class RerankPipeline:
    """组合多种重排后端：LLM > cross-encoder > 特征重排。"""

    def __init__(self, feature: FeatureReranker, llm_reranker: Optional[LLMReranker] = None,
                 cross_reranker: Optional[CrossEncoderReranker] = None,
                 learned_reranker=None) -> None:
        self.feature = feature
        self.llm = llm_reranker
        self.cross = cross_reranker
        self.learned = learned_reranker
        self.mode = (config.RERANK_PROVIDER or "auto").lower()
        self.last_backend = "feature"

    def rerank(self, query: str, candidates: List[dict], k: int, index=None):
        # 注意：通用 cross-encoder 在本学术语料上不如特征重排（见 5.4 节消融），
        # 因此 auto 模式不启用 cross，需显式设置 RERANK_PROVIDER=cross。
        # auto：仅纯英文查询启用学习式重排（其训练分布是英文）；
        # 中英混合的综述检索式仍走特征重排，避免跨语言退化。
        use_learned = (self.mode == "learned" or
                       (self.mode == "auto" and _mostly_ascii(query))) and \
            self.learned is not None and self.learned.available and index is not None
        use_llm = self.mode in ("auto", "llm") and self.llm is not None and self.llm.available
        use_cross = self.mode == "cross" and self.cross is not None \
            and self.cross.available
        if use_llm:
            order = self.llm.rerank(query, candidates, k)
            if order:
                by_id = {candidate["chunk"].chunk_id: candidate for candidate in candidates}
                self.last_backend = "llm"
                return [(by_id[chunk_id]["chunk"], round(by_id[chunk_id]["dense"], 5),
                         {"backend": "llm"}) for chunk_id in order]
        if use_cross:
            result = self.cross.rerank(query, candidates, k)
            if result:
                self.last_backend = "cross"
                return result
        if use_learned:
            result = self.learned.rerank(index, query, candidates, k)
            if result:
                self.last_backend = "learned"
                return result
        self.last_backend = "feature"
        return self.feature.rerank(query, candidates, k)
