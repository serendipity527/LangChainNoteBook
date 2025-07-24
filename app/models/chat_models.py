"""
聊天数据模型模块

该模块定义了聊天应用中使用的所有数据模型，使用Pydantic进行数据验证和序列化。
这些模型确保了API接口的类型安全和数据一致性。

主要功能：
1. 定义聊天请求和响应的数据结构
2. 提供自动数据验证和类型检查
3. 支持JSON序列化和反序列化
4. 为API文档生成提供模型定义

技术特点：
- 使用Pydantic BaseModel确保类型安全
- 支持可选字段和默认值
- 自动生成OpenAPI文档
"""

from pydantic import BaseModel, Field
from typing import Optional


class ChatRequest(BaseModel):
    """
    聊天请求数据模型

    定义了客户端发送聊天请求时需要提供的所有参数。
    支持无记忆和有记忆两种对话模式的参数配置。

    Attributes:
        message: 用户输入的消息内容，必填字段
        model_key: 指定使用的模型，可选，默认使用系统默认模型
        chat_id: 会话标识符，用于记忆模式下区分不同对话
        memory_type: 记忆类型，支持"buffer"和"summary"两种模式

    Example:
        >>> request = ChatRequest(
        ...     message="你好，请介绍一下自己",
        ...     model_key="qwen3:0.6b",
        ...     chat_id="user_123",
        ...     memory_type="buffer"
        ... )
    """
    message: str = Field(
        ...,  # 必填字段
        description="用户输入的消息内容",
        example="你好，请介绍一下自己"
    )

    model_key: Optional[str] = Field(
        None,
        description="指定使用的模型键，如果不指定则使用默认模型",
        example="qwen3:0.6b"
    )

    chat_id: Optional[str] = Field(
        "default",
        description="会话标识符，用于区分不同的对话会话，记忆模式下必需",
        example="user_123_session_1"
    )

    memory_type: Optional[str] = Field(
        "buffer",
        description="记忆类型：'buffer'保存完整历史，'summary'智能摘要长对话",
        example="buffer"
    )


class ChatResponse(BaseModel):
    """
    聊天响应数据模型

    定义了服务器返回给客户端的响应数据结构。
    包含了AI生成的回复内容以及相关的元数据信息。

    Attributes:
        response: AI生成的回复内容
        model_used: 实际使用的模型标识符
        has_memory: 是否使用了记忆功能
        chat_id: 会话标识符（记忆模式下返回）
        memory_type: 使用的记忆类型（记忆模式下返回）

    Example:
        >>> response = ChatResponse(
        ...     response="你好！我是AI助手，很高兴为您服务。",
        ...     model_used="qwen3:0.6b",
        ...     has_memory=True,
        ...     chat_id="user_123",
        ...     memory_type="buffer"
        ... )
    """
    response: str = Field(
        ...,
        description="AI生成的回复内容",
        example="你好！我是AI助手，很高兴为您服务。"
    )

    model_used: str = Field(
        ...,
        description="实际使用的模型标识符",
        example="qwen3:0.6b"
    )

    has_memory: bool = Field(
        ...,
        description="是否使用了记忆功能进行对话",
        example=True
    )

    chat_id: Optional[str] = Field(
        None,
        description="会话标识符，记忆模式下返回",
        example="user_123_session_1"
    )

    memory_type: Optional[str] = Field(
        None,
        description="使用的记忆类型，记忆模式下返回",
        example="buffer"
    )


class ModelListResponse(BaseModel):
    """
    模型列表响应数据模型

    用于返回系统中所有可用模型的信息列表。
    客户端可以通过此接口获取可选择的模型及其特性。

    Attributes:
        models: 模型信息字典，键为模型标识符，值为模型详细信息

    Example:
        >>> response = ModelListResponse(
        ...     models={
        ...         "qwen3:0.6b": {
        ...             "name": "qwen3:0.6b",
        ...             "provider": "ollama",
        ...             "description": "tool,thinking,轻量",
        ...             "supports_memory": True
        ...         }
        ...     }
        ... )
    """
    models: dict = Field(
        ...,
        description="可用模型信息字典，键为模型ID，值为模型详细信息",
        example={
            "qwen3:0.6b": {
                "name": "qwen3:0.6b",
                "provider": "ollama",
                "description": "tool,thinking,轻量",
                "supports_memory": True
            },
            "qwen3:4b": {
                "name": "qwen3:4b",
                "provider": "ollama",
                "description": "tool thinking",
                "supports_memory": True
            }
        }
    )