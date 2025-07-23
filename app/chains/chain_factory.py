from typing import Dict, Type, List
from .base_chain import BaseChain
from .memory_chain import MemoryChain
from .stateless_chain import StatelessChain

class ChainFactory:
    """对话链工厂"""
    
    _chains: Dict[str, Type[BaseChain]] = {
        "memory": MemoryChain,
        "stateless": StatelessChain
    }
    
    _instances: Dict[str, BaseChain] = {}
    
    @classmethod
    def create_chain(cls, chain_type: str) -> BaseChain:
        """创建链实例（单例模式）"""
        if chain_type not in cls._instances:
            if chain_type not in cls._chains:
                raise ValueError(f"不支持的链类型: {chain_type}")
            
            cls._instances[chain_type] = cls._chains[chain_type]()
        
        return cls._instances[chain_type]
    
    @classmethod
    def get_available_chains(cls) -> List[str]:
        """获取可用的链类型"""
        return list(cls._chains.keys())