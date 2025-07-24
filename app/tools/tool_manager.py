"""
工具管理器

统一管理项目中的所有工具，提供工具注册、查询和执行功能。
"""

from typing import Dict, List, Any
from langchain_core.tools import BaseTool
from .base_tools import calculator, get_weather, get_current_time


class ToolManager:
    """工具管理器类"""
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._register_default_tools()
    
    def _register_default_tools(self):
        """注册默认工具"""
        default_tools = [
            calculator,
            get_weather, 
            get_current_time
        ]
        
        for tool in default_tools:
            self._tools[tool.name] = tool
    
    def register_tool(self, tool: BaseTool):
        """注册新工具"""
        self._tools[tool.name] = tool
    
    def get_tool(self, name: str) -> BaseTool:
        """获取指定工具"""
        if name not in self._tools:
            raise ValueError(f"工具 '{name}' 不存在")
        return self._tools[name]
    
    def get_all_tools(self) -> List[BaseTool]:
        """获取所有工具"""
        return list(self._tools.values())
    
    def get_tools_by_category(self, category: str) -> List[BaseTool]:
        """按类别获取工具（可扩展功能）"""
        # 这里可以根据工具的元数据进行分类
        return self.get_all_tools()
    
    def list_tools(self) -> Dict[str, str]:
        """列出所有工具及其描述"""
        return {
            name: tool.description 
            for name, tool in self._tools.items()
        }


# 全局工具管理器实例
tool_manager = ToolManager()