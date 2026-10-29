"""网页原型：标准库 http.server + SSE。

接口：
    POST /api/runs                 建任务，返回 run_id
    GET  /api/runs/<id>            状态快照
    GET  /api/runs/<id>/events     SSE 进度流
    GET  /api/runs/<id>/doc        结构化正文（sections / refs / spans）
"""

from __future__ import annotations


def make_handler() -> type:
    """返回绑定到当前进程状态的请求处理类。"""
    raise NotImplementedError


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """启动 HTTP 服务，阻塞运行。"""
    raise NotImplementedError
