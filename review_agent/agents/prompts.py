"""各智能体的提示词模板。

集中在这里，便于统一审阅措辞与版本；不要在 agent 代码里拼长字符串。
"""

from __future__ import annotations

PLANNER = ""      # TODO: 主题 -> 子问题
ANALYST = ""      # TODO: 论文 -> 卡片（要求逐字引文）
COMPARATOR = ""   # TODO: 卡片 -> 方法族
WRITER = ""       # TODO: 证据 -> 段落（只允许在白名单内改写）
VERIFIER = ""     # TODO: 句子与片段 -> 是否有支撑
