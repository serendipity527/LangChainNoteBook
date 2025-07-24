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

from typing import Dict, Any
from langchain_ollama import ChatOllama
from ..config.model_config import MODEL_CONFIGS, ModelProvider, ModelConfig


class ModelFactory:
    """
    模型工厂类

    使用工厂设计模式统一管理AI模型的创建过程。
    根据模型配置自动选择合适的模型提供商和参数，
    为上层应用提供统一的模型创建接口。

    特点：
    - 静态方法设计，无需实例化
    - 支持多种模型提供商扩展
    - 自动参数配置和验证
    - 统一的错误处理机制
    """

    @staticmethod
    def create_model(model_key: str) -> Any:
        """
        根据模型键创建模型实例

        这是工厂类的核心方法，根据提供的模型键从配置中查找对应的模型配置，
        然后根据配置的提供商类型创建相应的模型实例。

        Args:
            model_key (str): 模型的唯一标识符，必须在MODEL_CONFIGS中存在

        Returns:
            Any: 创建的模型实例，具体类型取决于提供商
                 - Ollama: 返回ChatOllama实例
                 - 其他提供商: 返回对应的模型实例

        Raises:
            ValueError: 当模型键不存在或提供商不支持时抛出

        Example:
            >>> model = ModelFactory.create_model("qwen3:0.6b")
            >>> response = await model.ainvoke("你好")
        """
        # 验证模型键是否存在于配置中
        if model_key not in MODEL_CONFIGS:
            raise ValueError(f"未知的模型: {model_key}。可用模型: {list(MODEL_CONFIGS.keys())}")

        # 获取模型配置
        config = MODEL_CONFIGS[model_key]

        # 根据提供商类型创建对应的模型实例
        if config.provider == ModelProvider.OLLAMA:
            """
            创建Ollama模型实例

            Ollama是本地部署的开源模型服务，支持多种开源大语言模型。
            ChatOllama是LangChain提供的Ollama集成类。
            """
            return ChatOllama(
                base_url=config.base_url,      # Ollama服务地址
                model=config.model_id,         # 模型标识符
                temperature=config.temperature  # 生成温度参数
            )

        # TODO: 扩展其他模型提供商
        # elif config.provider == ModelProvider.OPENAI:
        #     from langchain_openai import ChatOpenAI
        #     return ChatOpenAI(
        #         api_key=config.api_key,
        #         model=config.model_id,
        #         temperature=config.temperature,
        #         max_tokens=config.max_tokens
        #     )

        # elif config.provider == ModelProvider.ANTHROPIC:
        #     from langchain_anthropic import ChatAnthropic
        #     return ChatAnthropic(
        #         api_key=config.api_key,
        #         model=config.model_id,
        #         temperature=config.temperature
        #     )

        else:
            # 不支持的提供商类型
            supported_providers = [provider.value for provider in ModelProvider]
            raise ValueError(
                f"不支持的模型提供商: {config.provider}。"
                f"支持的提供商: {supported_providers}"
            )

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