"""

"""

from fastapi import APIRouter, Query, HTTPException
from app.models.chat_models import ChatRequest, ChatResponse, ModelListResponse
from app.services.chat_service import ChatService
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.services.test_service import TestService

# 创建聊天相关的路由器
# prefix="/chat" 表示所有路由都以/chat开头
# tags=["聊天"] 用于API文档的分组显示
router = APIRouter(prefix="/test", tags=["测试"])

test_service = TestService()


@router.post("/tool", response_model=ChatResponse)
def chat_with_tool(chat_request: ChatRequest):
    return test_service.test_tool(chat_request)


