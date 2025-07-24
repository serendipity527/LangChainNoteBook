"""
模型工厂模块

该模块实现了工厂设计模式，负责根据配置创建不同类型的AI模型实例。
支持多种模型提供商，目前主要支持Ollama本地模型服务。

主要功能：
1. 根据模型键创建对应的模型实例
2. 统一管理不同提供商的模型创建逻辑
3. 提供模型信息查询接口

设计模式：
- 工厂模式：统一创建接口，隐藏具体实现细节
- 策略模式：根据不同提供商使用不同的创建策略
"""

from typing import Dict, Any, List, Optional
from langchain_ollama import ChatOllama
from langchain_core.tools import BaseTool
from ..config.model_config import MODEL_CONFIGS, ModelProvider, ModelConfig
from ..tools.tool_manager import tool_manager


class ModelFactory:
    """模型工厂类 - 支持工具绑定"""

    @staticmethod
    def create_model(model_key: str, tools: Optional[List[BaseTool]] = None) -> Any:
        """创建模型实例，支持工具绑定"""
        if model_key not in MODEL_CONFIGS:
            raise ValueError(f"未知的模型: {model_key}。可用模型: {list(MODEL_CONFIGS.keys())}")

        config = MODEL_CONFIGS[model_key]

        if config.provider == ModelProvider.OLLAMA:
            model = ChatOllama(
                base_url=config.base_url,
                model=config.model_id,
                temperature=config.temperature
            )
            
            # 如果提供了工具，则绑定到模型
            if tools:
                model = model.bind_tools(tools)
            
            return model
        else:
            supported_providers = [provider.value for provider in ModelProvider]
            raise ValueError(
                f"不支持的模型提供商: {config.provider}。"
                f"支持的提供商: {supported_providers}"
            )

    @staticmethod
    def create_model_with_tools(model_key: str, tool_names: List[str]) -> Any:
        """创建带指定工具的模型实例"""
        tools = []
        for tool_name in tool_names:
            try:
                tool = tool_manager.get_tool(tool_name)
                tools.append(tool)
            except ValueError as e:
                raise ValueError(f"工具加载失败: {e}")
        
        return ModelFactory.create_model(model_key, tools)

    @staticmethod
    def create_model_with_all_tools(model_key: str) -> Any:
        """创建带所有可用工具的模型实例"""
        tools = tool_manager.get_all_tools()
        return ModelFactory.create_model(model_key, tools)

    @staticmethod
    def get_available_models() -> Dict[str, ModelConfig]:
        """
        获取所有可用模型配置

        返回系统中配置的所有模型信息，用于前端展示模型列表
        或者进行模型选择。

        Returns:
            Dict[str, ModelConfig]: 模型键到配置对象的映射

        Example:
            >>> models = ModelFactory.get_available_models()
            >>> for key, config in models.items():
            ...     print(f"{key}: {config.description}")
        """
        return MODEL_CONFIGS

    @staticmethod
    def get_model_info(model_key: str) -> ModelConfig:
        """
        获取特定模型的详细信息

        根据模型键查询对应的配置信息，用于获取模型的详细参数和特性。

        Args:
            model_key (str): 模型的唯一标识符

        Returns:
            ModelConfig: 模型的完整配置信息

        Raises:
            ValueError: 当模型键不存在时抛出

        Example:
            >>> config = ModelFactory.get_model_info("qwen3:0.6b")
            >>> print(f"模型名称: {config.name}")
            >>> print(f"提供商: {config.provider}")
            >>> print(f"描述: {config.description}")
        """
        if model_key not in MODEL_CONFIGS:
            available_models = list(MODEL_CONFIGS.keys())
            raise ValueError(
                f"未知的模型: {model_key}。"
                f"可用模型: {available_models}"
            )
        return MODEL_CONFIGS[model_key]