"""
对话链基础接口模块

该模块定义了所有对话链的抽象基类，使用抽象基类(ABC)确保所有具体实现
都遵循统一的接口规范。这是整个对话处理系统的核心抽象层。

设计模式：
- 模板方法模式：定义算法骨架，具体步骤由子类实现
- 策略模式：不同的链实现代表不同的对话处理策略
- 接口隔离原则：定义最小化的接口，避免不必要的依赖

主要功能：
1. 定义对话处理的统一接口
2. 确保所有链实现的一致性
3. 为链工厂提供类型约束
4. 支持多态调用
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from ..models.chat_models import ChatRequest, ChatResponse


class BaseChain(ABC):
    """
    对话链抽象基类

    定义了所有对话处理链必须实现的核心接口。使用抽象基类确保
    所有具体实现都提供必要的方法，保证系统的一致性和可扩展性。

    设计理念：
    - 面向接口编程：上层代码依赖抽象而非具体实现
    - 开闭原则：对扩展开放，对修改封闭
    - 里氏替换原则：子类可以完全替换父类使用

    子类实现示例：
    - StatelessChain: 无记忆单次对话链
    - MemoryChain: 带记忆的对话链
    - ToolChain: 支持工具调用的对话链（待实现）
    """

    @abstractmethod
    async def invoke(self, request: ChatRequest, model_key: str = "qwen3:0.6b", **kwargs) -> ChatResponse:
        """
        执行对话处理的核心方法

        这是对话链的主要入口点，所有子类必须实现此方法来处理用户的聊天请求。
        方法采用异步设计以支持高并发场景。

        Args:
            request (ChatRequest): 用户的聊天请求，包含消息内容和配置参数
            model_key (str, optional): 指定使用的模型标识符，默认为"qwen3:0.6b"
            **kwargs: 额外的关键字参数，用于传递特定链类型需要的参数
                     例如：chat_id, memory_type等

        Returns:
            ChatResponse: 处理后的响应，包含AI回复和元数据信息

        Raises:
            NotImplementedError: 子类未实现此方法时抛出

        Note:
            - 此方法必须是异步的，以支持模型的异步调用
            - 子类实现时应该处理所有可能的异常情况
            - 返回的响应应该包含完整的元数据信息
        """
        pass

    @abstractmethod
    def get_chain_type(self) -> str:
        """
        获取链的类型标识符

        返回当前链的类型字符串，用于标识不同的链实现。
        这个标识符通常用于日志记录、调试和链工厂的管理。

        Returns:
            str: 链类型的字符串标识符
                 例如："stateless", "memory", "tool"等

        Note:
            - 返回值应该是唯一的，不同链类型不能重复
            - 建议使用小写字母和下划线的命名风格
            - 这个值通常与链工厂中的键保持一致
        """
        pass