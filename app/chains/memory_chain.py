
from typing import Dict, List, Optional, Any
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.schema import BaseMemory

from .base_chain import BaseChain
from ..services.model_factory import ModelFactory
from ..models.chat_models import ChatRequest, ChatResponse

class MemoryChain(BaseChain):
    """记忆对话链"""
    
    def __init__(self):
        self.memory_storage: Dict[str, BaseMemory] = {}
        self.chains: Dict[str, Any] = {}
    
    def _get_or_create_memory(self, chat_id: str, memory_type: str = "buffer", model_key: str = "qwen3:0.6b") -> BaseMemory:
        """获取或创建记忆实例"""
        memory_key = f"{chat_id}_{memory_type}"
        
        if memory_key not in self.memory_storage:
            if memory_type == "buffer":
                self.memory_storage[memory_key] = ConversationBufferMemory(
                    return_messages=True,
                    memory_key="chat_history"
                )
            elif memory_type == "summary":
                model = ModelFactory.create_model(model_key)
                self.memory_storage[memory_key] = ConversationSummaryBufferMemory(
                    llm=model,
                    return_messages=True,
                    memory_key="chat_history",
                    max_token_limit=1000
                )
        
        return self.memory_storage[memory_key]
    
    def _create_memory_chain(self, model_key: str):
        """创建记忆链"""
        model = ModelFactory.create_model(model_key)
        
        # 创建提示模板
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个友好的AI助手，能够记住对话历史并提供有用的回答。"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        # 构建LCEL链
        chain = (
            RunnablePassthrough()
            | prompt
            | model
            | StrOutputParser()
        )
        
        return chain
    
    async def invoke(self, request: ChatRequest, model_key: str = "qwen3:0.6b", 
                    chat_id: str = "default", memory_type: str = "buffer", **kwargs) -> ChatResponse:
        """执行记忆对话"""
        try:
            # 获取记忆和链
            memory = self._get_or_create_memory(chat_id, memory_type, model_key)
            
            chain_key = f"{model_key}_{memory_type}"
            if chain_key not in self.chains:
                self.chains[chain_key] = self._create_memory_chain(model_key)
            
            chain = self.chains[chain_key]
            
            # 加载历史记忆
            chat_history = memory.chat_memory.messages
            
            # 调用链
            response = await chain.ainvoke({
                "input": request.message,
                "chat_history": chat_history
            })
            
            # 保存对话到记忆
            memory.save_context(
                {"input": request.message},
                {"output": response}
            )
            
            return ChatResponse(
                response=response,
                model_used=model_key,
                has_memory=True,
                chat_id=chat_id,
                memory_type=memory_type
            )
            
        except Exception as e:
            return ChatResponse(
                response=f"处理请求时出现错误：{str(e)}",
                model_used=model_key,
                has_memory=True,
                chat_id=chat_id
            )
    
    def get_chat_history(self, chat_id: str, memory_type: str = "buffer") -> List[Dict[str, str]]:
        """获取对话历史"""
        memory_key = f"{chat_id}_{memory_type}"
        
        if memory_key not in self.memory_storage:
            return []
        
        memory = self.memory_storage[memory_key]
        messages = memory.chat_memory.messages
        
        history = []
        for msg in messages:
            if hasattr(msg, 'content'):
                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                history.append({
                    "role": role,
                    "content": msg.content
                })
        
        return history
    
    def clear_memory(self, chat_id: str, memory_type: str = "buffer") -> bool:
        """清除指定对话的记忆"""
        memory_key = f"{chat_id}_{memory_type}"
        
        if memory_key in self.memory_storage:
            del self.memory_storage[memory_key]
            return True
        return False
    
    def get_chain_type(self) -> str:
        return "memory"
