from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from ..models.chat_models import ChatRequest, ChatResponse


class BaseChain(ABC):
    """对话链基础接口"""

    @abstractmethod
    async def invoke(self, request: ChatRequest, model_key: str = "qwen3:0.6b", **kwargs) -> ChatResponse:
        """调用链进行对话"""
        pass

    @abstractmethod
    def get_chain_type(self) -> str:
        """获取链类型"""
        pass