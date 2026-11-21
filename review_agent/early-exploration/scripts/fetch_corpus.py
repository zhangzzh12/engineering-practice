"""抓取真实文献快照，生成离线语料库（data/fixtures/corpus.json）。

用于无网络环境演示与自动化测试：语料来自 OpenAlex 真实元数据与摘要，
不包含任何虚构文献。运行前请保证可访问 api.openalex.org。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402
from app import sources  # noqa: E402
from app.textutil import matches_topic, topic_units  # noqa: E402

TOPICS = {
    "retrieval-augmented generation": [
        "retrieval-augmented generation large language model",
        "retrieval augmented generation faithfulness citation",
    ],
    "graph neural network recommendation": [
        "graph neural network recommendation",
        "graph collaborative filtering recommender system",
    ],
    "large language model code generation": [
        "large language model code generation",
        "program repair with large language models",
    ],
    "multi-agent large language model": [
        "multi-agent collaboration large language model",
        "LLM agent framework tool use",
    ],
}

TARGET_PER_TOPIC = 12


def main() -> int:
    config.ensure_dirs()
    client = sources.OpenAlexSource()
    collected: dict = {}
    for topic, queries in TOPICS.items():
        bucket = []
        for query in queries:
            try:
                bucket.extend(client.search(query, limit=25))
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] {query}: {exc}")
            time.sleep(0.4)
        deduped = sources.dedupe(bucket)
        groups = topic_units(topic)
        on_topic = [paper for paper in deduped
                    if matches_topic(f"{paper.title} {paper.abstract} "
                                     f"{' '.join(paper.concepts)}", groups)]
        if len(on_topic) >= 6:
            deduped = on_topic
        # 兼顾「高被引经典」与「近两年新工作」，保证语料既有代表性又有时效性
        by_cites = sorted(deduped, key=lambda p: -p.citations)
        by_year = sorted(deduped, key=lambda p: (-(p.year or 0), -p.citations))
        selected = {}
        for paper in by_cites[: TARGET_PER_TOPIC // 2] + by_year:
            selected.setdefault(paper.paper_id, paper)
            if len(selected) >= TARGET_PER_TOPIC:
                break
        collected[topic] = list(selected.values())
        print(f"{topic}: {len(collected[topic])} 篇")

    merged = {paper.paper_id: paper for papers in collected.values() for paper in papers}
    payload = [paper.to_dict() for paper in merged.values()]
    target = config.FIXTURE_DIR / "corpus.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 {target}，共 {len(payload)} 篇文献")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
