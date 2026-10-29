# review-agent（项目骨架）

多智能体文献综述生成系统 —— 开题阶段的项目骨架，只固化目录结构与接口定义，暂不含实现。

**目标能力**：输入一个研究主题，从 arXiv 检索论文 -> 逐篇生成结构化阅读卡片 -> 按共享假设聚成
方法族 -> 写成带句子级引用的综述 -> 由校验器剔掉没有原文支撑的句子。

> **当前状态**：所有方法体均为 `NotImplementedError`，仓库只回答“代码怎么组织”，不回答“代码怎么写”。
> 实现按下面的「实现顺序」推进，每完成一个模块补上对应单测。

## 目录结构

```
review_agent/
  config.py             RunConfig：一次运行的全部参数
  core/                 数据契约与纯函数
    schema.py           Paper / Span / PaperCard / Family / Claim
    text.py             分词、分句、数字抽取、Jaccard
  llm/client.py         OpenAI 兼容客户端（无凭据时回落离线启发式）
  retrieval/
    arxiv_client.py     arXiv Atom API，带磁盘缓存与限流
    fulltext.py         四级全文获取（arXiv HTML -> ar5iv -> LaTeX -> PDF）
    ingest.py           论文 -> 可引用 span（按章节加前缀）
    bm25.py             Okapi BM25
    embeddings.py       向量后端（真实服务 / 离线哈希向量）
    retrieve.py         多 query 检索 + RRF 融合 + 段落索引
  agents/
    planner.py          主题 -> 子问题
    analyst.py          论文 -> 阅读卡片（字段必须附逐字引文）
    comparator.py       卡片 -> 方法族 / 演进边 / 冲突
    writer.py           大纲 -> 章节（句子级引用）
    verifier.py         存在性 / 数字一致 / 词面覆盖 + 重挂修复
    prompts.py          提示词模板
  orchestration/
    graph.py            LangGraph StateGraph：节点 + 条件边
    orchestrator.py     运行入口与统一出口
  storage/store.py      SQLite（默认）/ MySQL 运行档案
  evaluation/evaluate.py  五项指标离线评测
  web/                  标准库 HTTP + SSE 原型
  cli.py                命令行入口
docs/proposal/          开题报告
docs/slides/            开题答辩 PPT
```

## 实现顺序

| 步骤 | 模块 |
|---|---|
| 1 | `core/schema.py`、`core/text.py` |
| 2 | `retrieval/arxiv_client.py`、`ingest.py` |
| 3 | `agents/planner.py`、`analyst.py` |
| 4 | `agents/comparator.py` |
| 5 | `agents/writer.py`、`verifier.py` |
| 6 | `orchestration/graph.py`、`orchestrator.py` |
| 7 | `storage/store.py`、`web/`、`evaluation/` |

## 接口约定

- **无状态智能体**：输入数据、返回新数据，不写全局变量。
- **契约优先**：`Paper / Span / PaperCard` 是跨模块协议，改字段必须走 PR 并标注 `breaking`。
- **证据优先**：卡片字段与正文句子都必须绑定 `span_id`；匹配不上宁可置空，不编造。
- **修复优先于删除**：校验不通过时先尝试同篇论文内重挂证据，再决定丢弃。
- **可离线**：没有模型服务时走启发式路径，两条路径共用同一套契约与校验规则。

## 文档

- 开题报告：[docs/proposal/开题报告_重写版.docx](docs/proposal/开题报告_重写版.docx)
- 答辩 PPT：[docs/slides/选题十三_开题答辩_项目对齐版.pptx](docs/slides/选题十三_开题答辩_项目对齐版.pptx)
