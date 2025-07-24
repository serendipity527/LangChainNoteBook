from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOllama
from app.models.chat_models import ChatRequest, ChatResponse
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import math
from langchain_ollama import ChatOllama

import datetime
import json


@tool
def get_current_time():
    """获取当前时间"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class TestService:
    def __init__(self):
        pass
    def test(self):
        return "test"
    def test_tool(self, chat_request: ChatRequest):
        model = ChatOllama(
            base_url="http://localhost:11434",
            model="qwen3:4b"
        )
        
        # 将工具绑定到模型
        model_with_tools = model.bind_tools([get_current_time])
        
        # 调用模型，模型会根据用户消息决定是否使用工具
        response = model_with_tools.invoke(chat_request.message)
        
        # 检查是否有工具调用
        if hasattr(response, 'tool_calls') and response.tool_calls:
            # 执行工具调用
            tool_results = []
            for tool_call in response.tool_calls:
                if tool_call['name'] == 'get_current_time':
                    # 调用工具获取结果
                    tool_result = get_current_time.invoke({})
                    tool_results.append(f"当前时间：{tool_result}")
        
            # 如果有工具结果，将其作为最终回复
            final_content = str(response.content or "") + "\n" + "\n".join(tool_results)
        else:
            # 没有工具调用，直接使用模型回复
            final_content = str(response.content or "")
        
        return ChatResponse(
            response=final_content,
            chat_id=chat_request.chat_id,
            memory_type=chat_request.memory_type,
            model_used="qwen3:4b",
            has_memory=False
        )

