"""模型接入。

只依赖 OpenAI 兼容的 HTTP 接口，不绑定具体厂商。
没有可用凭据时 available() 返回 False，上层自动回落到离线启发式路径。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class LLMConfig:
    base_url: str | None = None
    model: str | None = None
    timeout: float = 60.0
    temperature: float = 0.0


class LLMError(RuntimeError):
    """模型调用失败；上层据此决定重试或回落。"""


class LLMClient:
    """一次运行复用一个客户端。"""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()

    def available(self) -> bool:
        """是否已配置可用的模型服务。"""
        raise NotImplementedError

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        """单轮补全，返回纯文本；失败抛 LLMError。"""
        raise NotImplementedError
