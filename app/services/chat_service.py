from typing import Dict, List, Optional, Any
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from .model_factory import ModelFactory
from ..models.chat_models import ChatRequest, ChatResponse
from ..config.model_config import MODEL_CONFIGS
from ..chains.chain_factory import ChainFactory


class ChatService:
    """统一聊天服务，支持多模型、多记忆模式"""

    def __init__(self):
        self.models: Dict[str, Any] = {}

    def get_or_create_model(self, model_key: str):
        """获取或创建模型实例"""
        if model_key not in self.models:
            self.models[model_key] = ModelFactory.create_model(model_key)
        return self.models[model_key]

    async def chat_once(self, request: ChatRequest, model_key: str = "qwen3:0.6b") -> ChatResponse:
        """无记忆单次对话"""
        chain = ChainFactory.create_chain("stateless")
        return await chain.invoke(request, model_key)

    async def chat_with_memory(self, request: ChatRequest, model_key: str = "qwen3:0.6b") -> ChatResponse:
        """带记忆的对话"""
        chain = ChainFactory.create_chain("memory")
        return await chain.invoke(
            request, 
            model_key, 
            chat_id=request.chat_id,
            memory_type=request.memory_type
        )

    def get_chat_history(self, chat_id: str, memory_type: str = "buffer") -> List[Dict[str, str]]:
        """获取对话历史"""
        chain = ChainFactory.create_chain("memory")
        return chain.get_chat_history(chat_id, memory_type)

    def clear_memory(self, chat_id: str, memory_type: str = "buffer") -> bool:
        """清除指定对话的记忆"""
        chain = ChainFactory.create_chain("memory")
        return chain.clear_memory(chat_id, memory_type)

    def get_available_models(self) -> Dict[str, dict]:
        """获取可用模型列表"""
        return {
            key: {
                "name": config.name,
                "provider": config.provider,
                "description": config.description,
                "supports_memory": config.supports_memory
            }
            for key, config in MODEL_CONFIGS.items()
        }
