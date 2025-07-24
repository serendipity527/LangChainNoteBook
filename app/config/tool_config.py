"""
工具配置模块

定义工具相关的配置参数和设置。
"""

from typing import Dict, List
from dataclasses import dataclass


@dataclass
class ToolConfig:
    """工具配置类"""
    name: str
    enabled: bool = True
    max_retries: int = 3
    timeout: int = 30
    description: str = ""


# 工具配置字典
TOOL_CONFIGS: Dict[str, ToolConfig] = {
    "calculator": ToolConfig(
        name="calculator",
        enabled=True,
        max_retries=2,
        timeout=10,
        description="数学计算工具"
    ),
    "get_weather": ToolConfig(
        name="get_weather", 
        enabled=True,
        max_retries=3,
        timeout=15,
        description="天气查询工具"
    ),
    "get_current_time": ToolConfig(
        name="get_current_time",
        enabled=True,
        max_retries=1,
        timeout=5,
        description="时间查询工具"
    )
}

# 默认启用的工具
DEFAULT_ENABLED_TOOLS: List[str] = [
    name for name, config in TOOL_CONFIGS.items() 
    if config.enabled
]