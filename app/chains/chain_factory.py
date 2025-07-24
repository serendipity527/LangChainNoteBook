"""
对话链工厂模块

该模块实现了工厂设计模式和单例模式的结合，负责创建和管理不同类型的对话链实例。
通过工厂模式隐藏具体链的创建细节，通过单例模式确保每种链类型只有一个实例。

设计模式：
1. 工厂模式：统一创建接口，隐藏实例化细节
2. 单例模式：每种链类型只创建一个实例，节省资源
3. 注册模式：通过字典注册支持的链类型，便于扩展

主要功能：
1. 统一管理所有对话链的创建
2. 提供链类型的注册和查询机制
3. 确保链实例的唯一性和复用性
4. 支持动态扩展新的链类型
"""

from typing import Dict, Type, List
from .base_chain import BaseChain
from .memory_chain import MemoryChain
from .stateless_chain import StatelessChain


class ChainFactory:
    """
    对话链工厂类

    使用工厂模式和单例模式管理对话链的创建和生命周期。
    所有链实例都通过此工厂创建，确保系统的一致性和资源的有效利用。

    设计特点：
    - 类级别的单例管理：每种链类型全局只有一个实例
    - 延迟初始化：只有在需要时才创建链实例
    - 类型安全：通过类型注册确保创建的实例符合BaseChain接口
    - 易于扩展：新增链类型只需在_chains字典中注册

    使用示例：
        >>> # 获取记忆链实例
        >>> memory_chain = ChainFactory.create_chain("memory")
        >>> # 获取无状态链实例
        >>> stateless_chain = ChainFactory.create_chain("stateless")
    """

    # 链类型注册表：映射链类型名称到具体的链类
    _chains: Dict[str, Type[BaseChain]] = {
        "memory": MemoryChain,        # 带记忆的对话链
        "stateless": StatelessChain   # 无记忆的对话链
        # 未来可以扩展更多链类型：
        # "tool": ToolChain,          # 支持工具调用的链
        # "rag": RAGChain,            # 检索增强生成链
        # "agent": AgentChain,        # 智能代理链
    }

    # 链实例缓存：存储已创建的链实例（单例模式）
    _instances: Dict[str, BaseChain] = {}

    @classmethod
    def create_chain(cls, chain_type: str) -> BaseChain:
        """
        创建或获取链实例（单例模式）

        根据链类型创建对应的链实例。如果实例已存在则直接返回，
        否则创建新实例并缓存。这确保了每种链类型只有一个实例。

        Args:
            chain_type (str): 链类型标识符，必须在_chains中注册
                             支持的类型：
                             - "memory": 带记忆的对话链
                             - "stateless": 无记忆的对话链

        Returns:
            BaseChain: 对应类型的链实例

        Raises:
            ValueError: 当链类型不支持时抛出

        Example:
            >>> # 创建记忆链（第一次调用会创建实例）
            >>> chain1 = ChainFactory.create_chain("memory")
            >>> # 再次获取记忆链（返回同一个实例）
            >>> chain2 = ChainFactory.create_chain("memory")
            >>> assert chain1 is chain2  # 同一个实例
        """
        # 检查是否已有缓存的实例
        if chain_type not in cls._instances:
            # 验证链类型是否支持
            if chain_type not in cls._chains:
                available_types = list(cls._chains.keys())
                raise ValueError(
                    f"不支持的链类型: {chain_type}。"
                    f"支持的类型: {available_types}"
                )

            # 创建新实例并缓存
            chain_class = cls._chains[chain_type]
            cls._instances[chain_type] = chain_class()

        # 返回缓存的实例
        return cls._instances[chain_type]

    @classmethod
    def get_available_chains(cls) -> List[str]:
        """
        获取所有可用的链类型列表

        返回当前工厂支持的所有链类型标识符，用于前端展示
        或者进行链类型的动态选择。

        Returns:
            List[str]: 支持的链类型列表

        Example:
            >>> types = ChainFactory.get_available_chains()
            >>> print(types)  # ['memory', 'stateless']
        """
        return list(cls._chains.keys())

    @classmethod
    def register_chain(cls, chain_type: str, chain_class: Type[BaseChain]) -> None:
        """
        注册新的链类型（扩展方法）

        动态注册新的链类型到工厂中，支持运行时扩展。

        Args:
            chain_type (str): 新链类型的标识符
            chain_class (Type[BaseChain]): 链类的类型，必须继承自BaseChain

        Raises:
            TypeError: 当链类不是BaseChain的子类时抛出
            ValueError: 当链类型已存在时抛出

        Example:
            >>> class CustomChain(BaseChain):
            ...     # 实现自定义链逻辑
            ...     pass
            >>> ChainFactory.register_chain("custom", CustomChain)
        """
        # 验证链类是否继承自BaseChain
        if not issubclass(chain_class, BaseChain):
            raise TypeError(f"链类 {chain_class} 必须继承自 BaseChain")

        # 检查链类型是否已存在
        if chain_type in cls._chains:
            raise ValueError(f"链类型 {chain_type} 已存在")

        # 注册新的链类型
        cls._chains[chain_type] = chain_class

    @classmethod
    def clear_instances(cls) -> None:
        """
        清除所有缓存的链实例

        主要用于测试或需要重新初始化链实例的场景。
        清除后，下次调用create_chain会创建新的实例。

        Warning:
            此方法会影响所有正在使用的链实例，请谨慎使用。
        """
        cls._instances.clear()