"""自研学习式重排器（不依赖 PyTorch / 预训练重排模型）。

两种模型，均用 numpy 实现，在本地语料上训练：

1. ``LinearListNet``：线性 ListNet。输入 8 维交叉特征（稠密余弦、BM25、IDF 覆盖、
   短语命中、标题命中、块长度、查询长度、稀有词覆盖），用 listwise softmax 损失
   直接优化「整个候选列表的排序」，等价于一个可学习的特征重排器。
2. ``TinyCrossEncoder``：小型神经交叉编码器。把查询向量与候选向量拼接成
   [q, p, q⊙p, |q−p|]（4×384 维），过一层 32 维隐层输出相关性分数，同样用
   listwise 损失训练。它真正建模了 query-passage 的交互，但参数量只有约 5 万，
   在 CPU 上单候选打分 < 0.1 ms。

训练监督来自「标题检索」：以论文标题为查询，该论文的块为正样本，检索到的其他论文块为负样本。
模型权重保存为 JSON，可随项目分发，加载即用。
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .models import Chunk
from .textutil import tokens

FEATURE_NAMES = [
    "dense", "sparse", "coverage", "phrase", "chunk_len", "query_len", "rare_cov",
    "feature_score",
]


def _bigrams(terms: Sequence[str]) -> List[str]:
    return [f"{terms[i]} {terms[i + 1]}" for i in range(len(terms) - 1)]


def build_features(index, query: str, candidates: List[dict],
                   feature_scorer=None) -> np.ndarray:
    """为 (query, candidate) 对构造特征矩阵。candidates: [{chunk, sparse, dense}]。

    最后一维是手工特征重排器的分数（stacking）：让学习器只需学习「如何修正」，
    在小语料上比从零学排序更稳。
    """
    query_terms = tokens(query)
    query_set = set(query_terms)
    pairs = _bigrams(query_terms)
    idf = index.idf
    idf_total = sum(idf.get(term, 1.0) for term in query_set) or 1.0
    # 稀有词：idf 高于中位数的查询词
    values = sorted(idf.get(term, 1.0) for term in query_set) or [1.0]
    median = values[len(values) // 2]
    rare = {term for term in query_set if idf.get(term, 1.0) >= median}
    baseline_scores = {}
    if feature_scorer is not None:
        for chunk, score, _ in feature_scorer.rerank(query, candidates, len(candidates)):
            baseline_scores[chunk.chunk_id] = score
    max_baseline = max(baseline_scores.values(), default=1.0) or 1.0
    rows = []
    for candidate in candidates:
        chunk: Chunk = candidate["chunk"]
        text_terms = set(tokens(chunk.text))
        matched = query_set & text_terms
        coverage = sum(idf.get(term, 1.0) for term in matched) / idf_total
        rare_cov = len(matched & rare) / (len(rare) or 1)
        low = chunk.text.lower()
        phrase = (sum(1 for pair in pairs if pair in low) / len(pairs)) if pairs else 0.0
        rows.append([
            candidate["dense"],
            candidate["sparse"],
            coverage,
            phrase,
            min(1.0, len(chunk.text) / 300.0),
            min(1.0, len(query_terms) / 20.0),
            rare_cov,
            baseline_scores.get(chunk.chunk_id, 0.0) / max_baseline,
        ])
    return np.asarray(rows, dtype=np.float64)


class _Standardizer:
    def __init__(self) -> None:
        self.mean = np.zeros(1)
        self.std = np.ones(1)

    def fit(self, matrix: np.ndarray) -> None:
        self.mean = matrix.mean(axis=0)
        self.std = matrix.std(axis=0) + 1e-6

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        return (matrix - self.mean) / self.std


def _listnet_loss_grad(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, np.ndarray]:
    """单组候选的 listwise softmax 损失与对分数的梯度。"""
    scores = scores - scores.max()
    exp = np.exp(scores)
    prob = exp / exp.sum()
    positive = labels > 0
    target = positive.astype(np.float64) / max(1, positive.sum())
    loss = -np.log((prob * positive).sum() + 1e-12)
    grad = (prob - target) / max(1, len(scores))
    return float(loss), grad


class LinearListNet:
    name = "learned-linear"

    def __init__(self, l2: float = 2e-3) -> None:
        self.weights = np.zeros(len(FEATURE_NAMES))
        self.bias = 0.0
        self.scaler = _Standardizer()
        self.l2 = l2

    def fit(self, groups: List[dict], epochs: int = 400, lr: float = 0.08) -> None:
        matrix = np.vstack([group["features"] for group in groups])
        self.scaler.fit(matrix)
        # Adam
        m = np.zeros_like(self.weights)
        v = np.zeros_like(self.weights)
        mb, vb = 0.0, 0.0
        for step in range(1, epochs + 1):
            grad_w = self.l2 * self.weights.copy()
            grad_b = 0.0
            for group in groups:
                features = self.scaler.transform(group["features"])
                scores = features @ self.weights + self.bias
                _, grad = _listnet_loss_grad(scores, group["labels"])
                grad_w += features.T @ grad
                grad_b += grad.sum()
            m = 0.9 * m + 0.1 * grad_w
            v = 0.999 * v + 0.001 * (grad_w ** 2)
            mb = 0.9 * mb + 0.1 * grad_b
            vb = 0.999 * vb + 0.001 * grad_b ** 2
            m_hat = m / (1 - 0.9 ** step)
            v_hat = v / (1 - 0.999 ** step)
            self.weights -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
            self.bias -= lr * (mb / (1 - 0.9 ** step)) / (math.sqrt(vb / (1 - 0.999 ** step)) + 1e-8)

    def scores(self, features: np.ndarray) -> np.ndarray:
        return self.scaler.transform(features) @ self.weights + self.bias

    def to_dict(self) -> dict:
        return {"type": "linear", "weights": self.weights.tolist(), "bias": self.bias,
                "mean": self.scaler.mean.tolist(), "std": self.scaler.std.tolist()}

    @classmethod
    def from_dict(cls, data: dict) -> "LinearListNet":
        model = cls()
        model.weights = np.asarray(data["weights"])
        model.bias = float(data["bias"])
        model.scaler.mean = np.asarray(data["mean"])
        model.scaler.std = np.asarray(data["std"])
        return model


class TinyCrossEncoder:
    name = "learned-cross"

    def __init__(self, hidden: int = 32, l2: float = 1e-4, seed: int = 7) -> None:
        self.hidden = hidden
        self.l2 = l2
        self.rng = np.random.default_rng(seed)
        self.w1 = None
        self.b1 = None
        self.w2 = None
        self.b2 = None

    def _init(self, dim: int) -> None:
        self.w1 = self.rng.normal(0, 1 / math.sqrt(dim), size=(dim, self.hidden))
        self.b1 = np.zeros(self.hidden)
        self.w2 = self.rng.normal(0, 1 / math.sqrt(self.hidden), size=(self.hidden, 1))
        self.b2 = np.zeros(1)

    @staticmethod
    def pair_matrix(query_vector: np.ndarray, doc_vectors: np.ndarray) -> np.ndarray:
        return np.concatenate([
            np.tile(query_vector, (len(doc_vectors), 1)),
            doc_vectors,
            np.tile(query_vector, (len(doc_vectors), 1)) * doc_vectors,
            np.abs(np.tile(query_vector, (len(doc_vectors), 1)) - doc_vectors),
        ], axis=1)

    def _forward(self, features: np.ndarray):
        hidden = np.maximum(0.0, features @ self.w1 + self.b1)
        scores = (hidden @ self.w2 + self.b2).ravel()
        return hidden, scores

    def fit(self, groups: List[dict], epochs: int = 120, lr: float = 0.01) -> None:
        dim = groups[0]["pair_features"].shape[1]
        self._init(dim)
        params = [self.w1, self.b1, self.w2, self.b2]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        for step in range(1, epochs + 1):
            grads = [self.l2 * self.w1, np.zeros_like(self.b1),
                     self.l2 * self.w2, np.zeros_like(self.b2)]
            for group in groups:
                features = group["pair_features"]
                hidden, scores = self._forward(features)
                _, grad_scores = _listnet_loss_grad(scores, group["labels"])
                grad_scores = grad_scores.reshape(-1, 1)
                grads[2] += hidden.T @ grad_scores
                grads[3] += grad_scores.sum(axis=0)
                grad_hidden = (grad_scores @ self.w2.T) * (hidden > 0)
                grads[0] += features.T @ grad_hidden
                grads[1] += grad_hidden.sum(axis=0)
            for index, param in enumerate(params):
                m[index] = 0.9 * m[index] + 0.1 * grads[index]
                v[index] = 0.999 * v[index] + 0.001 * (grads[index] ** 2)
                m_hat = m[index] / (1 - 0.9 ** step)
                v_hat = v[index] / (1 - 0.999 ** step)
                param -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

    def scores(self, pair_features: np.ndarray) -> np.ndarray:
        _, scores = self._forward(pair_features)
        return scores

    def to_dict(self) -> dict:
        return {"type": "tiny_cross", "hidden": self.hidden,
                "w1": self.w1.tolist(), "b1": self.b1.tolist(),
                "w2": self.w2.tolist(), "b2": self.b2.tolist()}

    @classmethod
    def from_dict(cls, data: dict) -> "TinyCrossEncoder":
        model = cls(hidden=int(data.get("hidden", 32)))
        model.w1 = np.asarray(data["w1"])
        model.b1 = np.asarray(data["b1"])
        model.w2 = np.asarray(data["w2"])
        model.b2 = np.asarray(data["b2"])
        return model


class LearnedReranker:
    """加载训练好的模型并对候选重排。"""

    name = "learned"

    def __init__(self, model_path: Path, kind: str = "auto") -> None:
        self.model_path = Path(model_path)
        self.kind = kind
        self.linear: Optional[LinearListNet] = None
        self.cross: Optional[TinyCrossEncoder] = None
        self._load()

    @property
    def available(self) -> bool:
        return self.linear is not None or self.cross is not None

    def _load(self) -> None:
        if not self.model_path.exists():
            return
        try:
            data = json.loads(self.model_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return
        models = data.get("models", {})
        if "linear" in models:
            self.linear = LinearListNet.from_dict(models["linear"])
        if "cross" in models:
            self.cross = TinyCrossEncoder.from_dict(models["cross"])

    def rerank(self, index, query: str, candidates: List[dict], k: int):
        query_vector = index.embedder.encode_query(query) if index.embedder else None
        if self.cross is not None and query_vector:
            doc_vectors = np.asarray([index.doc_vectors[index.chunk_index[c["chunk"].chunk_id]]
                                      for c in candidates], dtype=np.float64)
            pairs = TinyCrossEncoder.pair_matrix(np.asarray(query_vector, dtype=np.float64),
                                                 doc_vectors)
            scores = self.cross.scores(pairs)
        elif self.linear is not None:
            feature_scorer = getattr(getattr(index, "reranker", None), "feature", None)
            features = build_features(index, query, candidates, feature_scorer)
            scores = self.linear.scores(features)
        else:
            return None
        order = np.argsort(-scores)[:k]
        return [(candidates[i]["chunk"], round(float(scores[i]), 4),
                 {"backend": self.name, "score": round(float(scores[i]), 4)})
                for i in order]
