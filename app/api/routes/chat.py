from fastapi import APIRouter, Query
from app.models.chat_models import ChatRequest, ChatResponse, ModelListResponse
from app.services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["聊天"])
chat_service = ChatService()

@router.post("/once", response_model=ChatResponse)
async def chat_once(chat_request: ChatRequest):
    """无记忆单次对话"""
    return await chat_service.chat_once(
        chat_request,
        model_key=chat_request.model_key or "qwen3:0.6b"
    )

@router.get("/models", response_model=ModelListResponse)
async def get_models():
    """获取可用模型列表"""
    models = chat_service.get_available_models()
    return ModelListResponse(models=models)

@router.post("/memory", response_model=ChatResponse)
async def chat_with_memory(chat_request: ChatRequest):
    """带记忆的对话"""
    return await chat_service.chat_with_memory(
        chat_request,
        model_key=chat_request.model_key or "qwen3:0.6b"
    )

@router.get("/history/{chat_id}", response_model=dict)
async def get_chat_history(
    chat_id: str,
    memory_type: str = Query(default="buffer", description="记忆类型: buffer 或 summary")
):
    """获取指定会话的对话历史"""
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
    """清除指定会话的记忆"""
    success = chat_service.clear_memory(chat_id, memory_type)
    return {
        "success": success,
        "message": f"已清除会话 {chat_id} 的 {memory_type} 记忆" if success else "记忆清除失败"
    }
