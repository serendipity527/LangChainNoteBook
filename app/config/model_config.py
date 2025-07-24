"""
模型配置模块

该模块定义了AI模型的配置结构和支持的模型提供商，
为整个聊天应用提供统一的模型管理和配置接口。

主要功能：
1. 定义模型提供商枚举
2. 定义模型配置数据结构
3. 配置具体的模型实例
"""

from enum import Enum
from typing import Dict, Any
from pydantic import BaseModel


class ModelProvider(str, Enum):
    """
    模型提供商枚举类

    定义了系统支持的所有AI模型提供商类型。
    继承自str和Enum，既可以作为字符串使用，也具有枚举的类型安全特性。
    """
    OLLAMA = "ollama"        # 本地部署的Ollama模型服务
    OPENAI = "openai"        # OpenAI官方API服务
    ANTHROPIC = "anthropic"  # Anthropic Claude系列模型
    QWEN = "qwen"           # 阿里云通义千问模型
    BAIDU = "baidu"         # 百度文心一言模型
    ZHIPU = "zhipu"         # 智谱AI GLM系列模型


class ModelConfig(BaseModel):
    """
    模型配置数据模型

    使用Pydantic BaseModel确保数据类型安全和自动验证。
    包含了模型运行所需的所有配置参数。

    Attributes:
        name: 模型显示名称，用于用户界面展示
        provider: 模型提供商，必须是ModelProvider枚举中的值
        model_id: 模型的唯一标识符，用于API调用
        base_url: 模型服务的基础URL地址（可选）
        api_key: API访问密钥（可选，本地模型不需要）
        temperature: 生成文本的随机性控制参数（0-1之间）
        max_tokens: 单次生成的最大token数量限制
        supports_memory: 是否支持对话记忆功能
        description: 模型的描述信息，包含特性说明
    """
    name: str                           # 模型名称
    provider: ModelProvider             # 提供商类型
    model_id: str                      # 模型ID
    base_url: str = None               # 服务地址（可选）
    api_key: str = None                # API密钥（可选）
    temperature: float = 0.7           # 温度参数，控制输出随机性
    max_tokens: int = 2000             # 最大输出token数
    supports_memory: bool = True       # 是否支持记忆功能
    description: str = ""              # 模型描述


# 全局模型配置字典
MODEL_CONFIGS: Dict[str, ModelConfig] = {
    """
    系统支持的所有模型配置

    键值对结构：
    - key: 模型的唯一标识符，用于API调用时指定模型
    - value: ModelConfig实例，包含该模型的完整配置信息

    当前配置的模型都是基于Ollama本地部署的开源模型：
    - qwen3:0.6b: 轻量级模型，支持工具调用和思维链
    - gemma3:4b: Google Gemma模型，不支持工具调用
    - qwen3:4b: 中等规模模型，支持工具调用和思维链
    - qwen2.5:3b: 新版本模型，支持工具调用和思维链
    """

    "qwen3:0.6b": ModelConfig(
        name="qwen3:0.6b",                    # 模型显示名称
        provider=ModelProvider.OLLAMA,        # 使用Ollama提供商
        model_id="qwen3:0.6b",               # Ollama中的模型ID
        base_url="http://localhost:11434",    # Ollama默认服务地址
        description="tool,thinking,轻量"      # 特性：支持工具调用、思维链推理、轻量级
    ),

    "gemma3:4b": ModelConfig(
        name="gemma3:4b",                     # Google Gemma 4B参数模型
        provider=ModelProvider.OLLAMA,        # 使用Ollama提供商
        model_id="gemma3:4b",                # Ollama中的模型ID
        base_url="http://localhost:11434",    # Ollama默认服务地址
        description="no tool"                 # 特性：不支持工具调用
    ),

    "qwen3:4b": ModelConfig(
        name="qwen3:4b",                      # 通义千问3代 4B参数模型
        provider=ModelProvider.OLLAMA,        # 使用Ollama提供商
        model_id="qwen3:4b",                 # Ollama中的模型ID
        base_url="http://localhost:11434",    # Ollama默认服务地址
        description="tool thinking"           # 特性：支持工具调用和思维链推理
    ),

    "qwen2.5:3b": ModelConfig(
        name="qwen2.5:3b",                    # 通义千问2.5代 3B参数模型
        provider=ModelProvider.OLLAMA,        # 使用Ollama提供商
        model_id="qwen2.5:3b",               # Ollama中的模型ID
        base_url="http://localhost:11434",    # Ollama默认服务地址
        description="tool thinking"           # 特性：支持工具调用和思维链推理
    )
}