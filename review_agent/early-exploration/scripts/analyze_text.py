"""文本质量辅助统计：对评测生成的 Markdown 综述做可复现的定量分析。

指标：
* 段落数与「开头多样性」：衡量模板化程度；
* 引用密度：每千字引用次数；
* 平均句长：可读性代理指标；
* 参考文献元数据完整率：有 venue / 有 DOI 的比例。
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.textutil import sentences  # noqa: E402

TOPICS = ["检索增强生成", "图神经网络推荐", "大模型代码生成"]


def analyze(markdown: str) -> dict:
    body = markdown.split("## 参考文献")[0]
    paragraphs = [line.strip() for line in body.split("\n")
                  if line.strip() and not line.startswith(("#", ">", "|", "---"))]
    openings = ["".join(paragraph[:10]) for paragraph in paragraphs]
    diverse = len(set(openings)) / max(1, len(paragraphs))
    citations = len(re.findall(r"\[\d+\]", body))
    chars = len(re.sub(r"\s", "", body))
    sents = sentences(body)
    refs = markdown.split("## 参考文献")[-1].strip().split("\n")
    refs = [line for line in refs if line.startswith("[")]
    with_venue = sum(1 for line in refs if "—," not in line)
    with_link = sum(1 for line in refs if "http" in line)
    return {
        "paragraphs": len(paragraphs),
        "opening_diversity": round(diverse, 3),
        "citations": citations,
        "citation_density": round(citations / max(1, chars) * 1000, 2),
        "avg_sentence_len": round(sum(len(s) for s in sents) / max(1, len(sents)), 1),
        "references": len(refs),
        "venue_rate": round(with_venue / max(1, len(refs)), 3),
        "link_rate": round(with_link / max(1, len(refs)), 3),
        "top_openings": Counter(openings).most_common(3),
    }


def main() -> int:
    rows = []
    for topic in TOPICS:
        path = ROOT / "results" / f"{topic}.md"
        if not path.exists():
            continue
        stats = analyze(path.read_text(encoding="utf-8"))
        stats["topic"] = topic
        rows.append(stats)
        print(f"{topic}: 段落 {stats['paragraphs']} | 开头多样性 {stats['opening_diversity']:.0%} | "
              f"引用密度 {stats['citation_density']}/千字 | 平均句长 {stats['avg_sentence_len']}")
    (ROOT / "results" / "text_stats.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
