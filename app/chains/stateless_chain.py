from typing import Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from .base_chain import BaseChain
from ..services.model_factory import ModelFactory
from ..models.chat_models import ChatRequest, ChatResponse

class StatelessChain(BaseChain):
    """无记忆对话链"""
    
    def __init__(self):
        self.chains: Dict[str, Any] = {}
    
    def _get_or_create_chain(self, model_key: str):
        """获取或创建链实例"""
        if model_key not in self.chains:
            model = ModelFactory.create_model(model_key)
            
            # 创建提示模板
            prompt = ChatPromptTemplate.from_messages([
                ("system", "你是一个有用的AI助手，请根据用户的问题提供准确、有帮助的回答。"),
                ("human", "{input}")
            ])
            
            # 构建LCEL链
            self.chains[model_key] = (
                RunnablePassthrough()
                | prompt
                | model
                | StrOutputParser()
            )
        
        return self.chains[model_key]
    
    async def invoke(self, request: ChatRequest, model_key: str = "qwen3:0.6b", **kwargs) -> ChatResponse:
        """执行无记忆对话"""
        try:
            chain = self._get_or_create_chain(model_key)
            
            # 调用链
            response = await chain.ainvoke({"input": request.message})
            
            return ChatResponse(
                response=response,
                model_used=model_key,
                has_memory=False
            )
            
        except Exception as e:
            return ChatResponse(
                response=f"处理请求时出现错误：{str(e)}",
                model_used=model_key,
                has_memory=False
            )
    
    def get_chain_type(self) -> str:
        return "stateless"