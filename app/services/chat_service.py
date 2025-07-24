"""
聊天服务模块

该模块提供了统一的聊天服务接口，是整个聊天应用的业务逻辑层。
它封装了不同类型的对话链，为上层API提供简洁统一的调用接口。

主要功能：
1. 统一的聊天接口：支持有记忆和无记忆两种对话模式
2. 模型管理：提供模型信息查询和选择功能
3. 会话管理：支持多会话的历史记录管理
4. 记忆管理：提供记忆的查询和清除功能

设计模式：
- 外观模式：为复杂的链系统提供简化的接口
- 委托模式：将具体的处理逻辑委托给对应的链
- 服务层模式：封装业务逻辑，与表现层解耦
"""

from typing import Dict, List, Optional, Any
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from .model_factory import ModelFactory
from ..models.chat_models import ChatRequest, ChatResponse
from ..config.model_config import MODEL_CONFIGS
from ..chains.chain_factory import ChainFactory


class ChatService:
    """
    统一聊天服务类

    作为整个聊天系统的业务逻辑层，提供统一的聊天服务接口。
    封装了底层的链系统复杂性，为上层API提供简洁易用的方法。

    服务特点：
    - 统一接口：无论使用哪种链，都通过相同的方法调用
    - 自动管理：自动处理链的创建和模型的选择
    - 多模式支持：支持无记忆和有记忆两种对话模式
    - 会话管理：提供完整的会话生命周期管理

    使用示例：
        >>> service = ChatService()
        >>> # 无记忆对话
        >>> response = await service.chat_once(request)
        >>> # 有记忆对话
        >>> response = await service.chat_with_memory(request)
    """

    def __init__(self):
        """
        初始化聊天服务

        创建模型缓存字典，用于存储已创建的模型实例。
        采用延迟初始化策略，只有在需要时才创建模型。

        Note:
            虽然当前版本中models字典未被使用（因为模型创建
            已经在ModelFactory中缓存），但保留此结构以便
            未来可能的扩展需求。
        """
        # 模型实例缓存（当前版本暂未使用，预留扩展）
        self.models: Dict[str, Any] = {}

    def get_or_create_model(self, model_key: str):
        """
        获取或创建模型实例（预留方法）

        当前版本中，模型创建已经在ModelFactory中处理，
        此方法主要用于未来可能的服务层模型缓存需求。

        Args:
            model_key (str): 模型标识符

        Returns:
            Any: 模型实例
        """
        if model_key not in self.models:
            self.models[model_key] = ModelFactory.create_model(model_key)
        return self.models[model_key]

    async def chat_once(self, request: ChatRequest, model_key: str = "qwen3:0.6b") -> ChatResponse:
        """
        执行无记忆单次对话

        使用无状态链处理用户请求，每次对话都是独立的，
        不会保存或使用任何历史信息。适合单次问答场景。

        Args:
            request (ChatRequest): 用户的聊天请求
            model_key (str): 使用的模型标识符，默认为"qwen3:0.6b"

        Returns:
            ChatResponse: AI的回复响应，has_memory字段为False

        Example:
            >>> request = ChatRequest(message="什么是人工智能？")
            >>> response = await service.chat_once(request, "qwen3:0.6b")
            >>> print(response.response)  # AI的回答
            >>> print(response.has_memory)  # False
        """
        # 获取无状态链实例
        chain = ChainFactory.create_chain("stateless")
        # 委托给链处理请求
        return await chain.invoke(request, model_key)

    async def chat_with_memory(self, request: ChatRequest, model_key: str = "qwen3:0.6b") -> ChatResponse:
        """
        执行带记忆的对话

        使用记忆链处理用户请求，会保存对话历史并在后续
        对话中使用。适合需要上下文连续性的对话场景。

        Args:
            request (ChatRequest): 用户的聊天请求，应包含chat_id和memory_type
            model_key (str): 使用的模型标识符，默认为"qwen3:0.6b"

        Returns:
            ChatResponse: AI的回复响应，has_memory字段为True，
                         包含chat_id和memory_type信息

        Example:
            >>> request = ChatRequest(
            ...     message="我叫张三",
            ...     chat_id="user_123",
            ...     memory_type="buffer"
            ... )
            >>> response = await service.chat_with_memory(request)
            >>> print(response.has_memory)  # True
            >>> print(response.chat_id)     # "user_123"
        """
        # 获取记忆链实例
        chain = ChainFactory.create_chain("memory")
        # 委托给链处理请求，传递记忆相关参数
        return await chain.invoke(
            request,
            model_key,
            chat_id=request.chat_id,        # 会话标识符
            memory_type=request.memory_type  # 记忆类型
        )

    def get_chat_history(self, chat_id: str, memory_type: str = "buffer") -> List[Dict[str, str]]:
        """
        获取指定会话的对话历史

        查询指定会话的完整对话记录，返回用户友好的格式。
        主要用于前端展示历史对话或进行对话分析。

        Args:
            chat_id (str): 会话标识符
            memory_type (str): 记忆类型，"buffer"或"summary"

        Returns:
            List[Dict[str, str]]: 对话历史列表，按时间顺序排列
                                 每个元素包含role和content字段

        Example:
            >>> history = service.get_chat_history("user_123", "buffer")
            >>> for msg in history:
            ...     print(f"{msg['role']}: {msg['content']}")
        """
        # 获取记忆链实例并委托处理
        chain = ChainFactory.create_chain("memory")
        return chain.get_chat_history(chat_id, memory_type)

    def clear_memory(self, chat_id: str, memory_type: str = "buffer") -> bool:
        """
        清除指定会话的记忆

        删除指定会话的所有历史记录，释放内存资源。
        通常在用户主动清除历史或会话结束时调用。

        Args:
            chat_id (str): 会话标识符
            memory_type (str): 记忆类型，"buffer"或"summary"

        Returns:
            bool: 清除是否成功
                  True: 成功清除记忆
                  False: 记忆不存在

        Example:
            >>> success = service.clear_memory("user_123", "buffer")
            >>> if success:
            ...     print("历史记录已清除")
        """
        # 获取记忆链实例并委托处理
        chain = ChainFactory.create_chain("memory")
        return chain.clear_memory(chat_id, memory_type)

    def get_available_models(self) -> Dict[str, dict]:
        """
        获取所有可用模型的信息

        返回系统中配置的所有模型信息，用于前端展示模型选择列表
        或者进行模型能力查询。

        Returns:
            Dict[str, dict]: 模型信息字典
                            键：模型标识符
                            值：包含模型详细信息的字典

        返回格式：
            {
                "model_key": {
                    "name": "模型名称",
                    "provider": "提供商",
                    "description": "模型描述",
                    "supports_memory": "是否支持记忆"
                }
            }

        Example:
            >>> models = service.get_available_models()
            >>> for key, info in models.items():
            ...     print(f"{key}: {info['description']}")
        """
        # 从配置中提取模型信息，转换为前端友好的格式
        return {
            key: {
                "name": config.name,                    # 模型显示名称
                "provider": config.provider,            # 提供商类型
                "description": config.description,      # 模型描述和特性
                "supports_memory": config.supports_memory  # 是否支持记忆功能
            }
            for key, config in MODEL_CONFIGS.items()
        }
