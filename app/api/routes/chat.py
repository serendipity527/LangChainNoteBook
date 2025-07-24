"""
聊天API路由模块

该模块定义了聊天应用的所有HTTP API端点，是前端与后端交互的接口层。
使用FastAPI框架提供RESTful API服务，支持自动文档生成和数据验证。

API端点说明：
1. POST /chat/once - 无记忆单次对话
2. POST /chat/memory - 带记忆的连续对话
3. GET /chat/models - 获取可用模型列表
4. GET /chat/history/{chat_id} - 获取对话历史
5. DELETE /chat/memory/{chat_id} - 清除对话记忆

技术特点：
- 自动数据验证：使用Pydantic模型确保请求数据正确性
- 自动文档生成：FastAPI自动生成OpenAPI文档
- 类型安全：完整的类型注解和响应模型定义
- 错误处理：统一的异常处理和错误响应
"""

from fastapi import APIRouter, Query, HTTPException
from app.models.chat_models import ChatRequest, ChatResponse, ModelListResponse
from app.services.chat_service import ChatService
from app.services.test_service import TestService

# 创建聊天相关的路由器
# prefix="/chat" 表示所有路由都以/chat开头
# tags=["聊天"] 用于API文档的分组显示
router = APIRouter(prefix="/chat", tags=["聊天"])

# 创建聊天服务实例
# 在模块级别创建单例，所有请求共享同一个服务实例
chat_service = ChatService()
test_service = TestService()

@router.post("/once", response_model=ChatResponse)
async def chat_once(chat_request: ChatRequest):
    """
    无记忆单次对话接口

    处理单次独立的对话请求，不保存对话历史。
    适用于简单问答、信息查询等场景。

    Args:
        chat_request (ChatRequest): 聊天请求对象，包含：
            - message: 用户输入的消息内容（必填）
            - model_key: 指定使用的模型（可选）

    Returns:
        ChatResponse: 聊天响应对象，包含：
            - response: AI生成的回复内容
            - model_used: 实际使用的模型标识符
            - has_memory: False（标识为无记忆模式）

    HTTP状态码：
        - 200: 成功处理请求
        - 422: 请求数据验证失败
        - 500: 服务器内部错误

    示例请求：
        POST /chat/once
        {
            "message": "什么是人工智能？",
            "model_key": "qwen3:0.6b"
        }

    示例响应：
        {
            "response": "人工智能是...",
            "model_used": "qwen3:0.6b",
            "has_memory": false
        }
    """
    return await chat_service.chat_once(
        chat_request,
        model_key=chat_request.model_key or "qwen3:0.6b"  # 使用指定模型或默认模型
    )


@router.get("/models", response_model=ModelListResponse)
async def get_models():
    """
    获取可用模型列表接口

    返回系统中配置的所有可用AI模型信息，
    用于前端展示模型选择列表。

    Returns:
        ModelListResponse: 模型列表响应对象，包含：
            - models: 模型信息字典，键为模型ID，值为模型详情

    HTTP状态码：
        - 200: 成功获取模型列表

    示例响应：
        {
            "models": {
                "qwen3:0.6b": {
                    "name": "qwen3:0.6b",
                    "provider": "ollama",
                    "description": "tool,thinking,轻量",
                    "supports_memory": true
                },
                "qwen3:4b": {
                    "name": "qwen3:4b",
                    "provider": "ollama",
                    "description": "tool thinking",
                    "supports_memory": true
                }
            }
        }
    """
    models = chat_service.get_available_models()
    return ModelListResponse(models=models)


@router.post("/memory", response_model=ChatResponse)
async def chat_with_memory(chat_request: ChatRequest):
    """
    带记忆的连续对话接口

    处理带有记忆功能的对话请求，会保存对话历史
    并在后续对话中使用上下文信息。

    Args:
        chat_request (ChatRequest): 聊天请求对象，包含：
            - message: 用户输入的消息内容（必填）
            - model_key: 指定使用的模型（可选）
            - chat_id: 会话标识符（可选，默认"default"）
            - memory_type: 记忆类型（可选，默认"buffer"）

    Returns:
        ChatResponse: 聊天响应对象，包含：
            - response: AI生成的回复内容
            - model_used: 实际使用的模型标识符
            - has_memory: True（标识为记忆模式）
            - chat_id: 会话标识符
            - memory_type: 使用的记忆类型

    记忆类型说明：
        - "buffer": 保存完整对话历史，适合短对话
        - "summary": 智能摘要长对话，适合长期对话

    示例请求：
        POST /chat/memory
        {
            "message": "我叫张三，请记住我的名字",
            "model_key": "qwen3:0.6b",
            "chat_id": "user_123",
            "memory_type": "buffer"
        }

    示例响应：
        {
            "response": "好的，张三，我已经记住了您的名字。",
            "model_used": "qwen3:0.6b",
            "has_memory": true,
            "chat_id": "user_123",
            "memory_type": "buffer"
        }
    """
    return await chat_service.chat_with_memory(
        chat_request,
        model_key=chat_request.model_key or "qwen3:0.6b"  # 使用指定模型或默认模型
    )

@router.get("/history/{chat_id}", response_model=dict)
async def get_chat_history(
    chat_id: str,
    memory_type: str = Query(default="buffer", description="记忆类型: buffer 或 summary")
):
    """
    获取指定会话的对话历史接口

    查询指定会话的完整对话记录，返回按时间顺序排列的历史消息。
    主要用于前端展示历史对话或进行对话分析。

    Args:
        chat_id (str): 会话标识符，路径参数
        memory_type (str): 记忆类型，查询参数，默认"buffer"

    Returns:
        dict: 包含历史记录的字典，结构如下：
            - chat_id: 会话标识符
            - memory_type: 记忆类型
            - history: 对话历史列表
            - total_messages: 消息总数

    HTTP状态码：
        - 200: 成功获取历史记录
        - 404: 会话不存在（返回空历史）

    示例请求：
        GET /chat/history/user_123?memory_type=buffer

    示例响应：
        {
            "chat_id": "user_123",
            "memory_type": "buffer",
            "history": [
                {"role": "user", "content": "我叫张三"},
                {"role": "assistant", "content": "你好张三！"},
                {"role": "user", "content": "你还记得我的名字吗？"},
                {"role": "assistant", "content": "当然记得，您是张三。"}
            ],
            "total_messages": 4
        }
    """
    history = chat_service.get_chat_history(chat_id, memory_type)
    return {
        "chat_id": chat_id,
        "memory_type": memory_type,
        "history": history,
        "total_messages": len(history)
    }


@router.delete("/memory/{chat_id}")
async def clear_chat_memory(
    chat_id: str,
    memory_type: str = Query(default="buffer", description="记忆类型: buffer 或 summary")
):
    """
    清除指定会话的记忆接口

    删除指定会话的所有历史记录，释放内存资源。
    通常在用户主动清除历史或会话结束时调用。

    Args:
        chat_id (str): 会话标识符，路径参数
        memory_type (str): 记忆类型，查询参数，默认"buffer"

    Returns:
        dict: 清除结果信息，包含：
            - success: 清除是否成功
            - message: 操作结果描述信息

    HTTP状态码：
        - 200: 操作完成（无论是否有记忆被清除）

    示例请求：
        DELETE /chat/memory/user_123?memory_type=buffer

    示例响应（成功）：
        {
            "success": true,
            "message": "已清除会话 user_123 的 buffer 记忆"
        }

    示例响应（记忆不存在）：
        {
            "success": false,
            "message": "记忆清除失败"
        }
    """
    success = chat_service.clear_memory(chat_id, memory_type)
    return {
        "success": success,
        "message": f"已清除会话 {chat_id} 的 {memory_type} 记忆" if success else "记忆清除失败"
    }

@router.post("/tool", response_model=ChatResponse)
def chat_with_tool(chat_request: ChatRequest):
    return test_service.test_tool(chat_request)
