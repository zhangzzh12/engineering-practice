"""review-agent：基于多智能体协同的文献综述生成系统（项目骨架）。

包内只暴露版本号；能力分布在各子包：

    core           数据契约与通用工具
    llm            模型接入
    retrieval      检索、全文获取与索引
    agents         五个职责单一的智能体
    orchestration  LangGraph 编排
    storage        运行档案持久化
    evaluation     离线评测
    web            网页原型
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
