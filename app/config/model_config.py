from enum import Enum
from typing import Dict, Any
from pydantic import BaseModel

class ModelProvider(str, Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    QWEN = "qwen"
    BAIDU = "baidu"
    ZHIPU = "zhipu"

class ModelConfig(BaseModel):
    name: str
    provider: ModelProvider
    model_id: str
    base_url: str = None
    api_key: str = None
    temperature: float = 0.7
    max_tokens: int = 2000
    supports_memory: bool = True
    description: str = ""


# 模型配置字典
MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "qwen3:0.6b": ModelConfig(
        name="qwen3:0.6b",
        provider=ModelProvider.OLLAMA,
        model_id="qwen3:0.6b",
        base_url="http://localhost:11434",
        description="tool,thinking,轻量"
    ),
    "gemma3:4b": ModelConfig(
        name="gemma3:4b",
        provider=ModelProvider.OLLAMA,
        model_id="gemma3:4b",
        base_url="http://localhost:11434",
        description="no tool"
    ),
    "qwen3:4b": ModelConfig(
        name="qwen3:4b",
        provider=ModelProvider.OLLAMA,
        model_id="qwen3:4b",
        base_url="http://localhost:11434",
        description="tool thinking"
    )

}