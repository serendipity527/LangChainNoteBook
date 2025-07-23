from typing import Dict, Any
from langchain_ollama import ChatOllama
from ..config.model_config import MODEL_CONFIGS, ModelProvider, ModelConfig


class ModelFactory:
    """模型工厂类，负责创建不同厂家的模型实例"""

    @staticmethod
    def create_model(model_key: str) -> Any:
        """根据模型键创建模型实例"""
        if model_key not in MODEL_CONFIGS:
            raise ValueError(f"未知的模型: {model_key}")

        config = MODEL_CONFIGS[model_key]

        if config.provider == ModelProvider.OLLAMA:
            return ChatOllama(
                base_url=config.base_url,
                model=config.model_id,
                temperature=config.temperature
            )


        else:
            raise ValueError(f"不支持的模型提供商: {config.provider}")



    @staticmethod
    def get_available_models() -> Dict[str, ModelConfig]:
        """获取所有可用模型配置"""
        return MODEL_CONFIGS

    @staticmethod
    def get_model_info(model_key: str) -> ModelConfig:
        """获取特定模型信息"""
        if model_key not in MODEL_CONFIGS:
            raise ValueError(f"未知的模型: {model_key}")
        return MODEL_CONFIGS[model_key]