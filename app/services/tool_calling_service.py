"""
工具调用服务

处理大模型的工具调用流程，包括工具执行、结果处理和错误管理。
"""

from typing import List, Dict, Any, Optional, Tuple
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.tools import BaseTool
from ..services.model_factory import ModelFactory
from ..tools.tool_manager import tool_manager
import logging
import json


class ToolCallingService:
    """工具调用服务类"""
    
    def __init__(self, model_key: str = "qwen3:4b"):
        self.model_key = model_key
        self.logger = logging.getLogger(__name__)
        
    def create_model_with_tools(self, tool_names: Optional[List[str]] = None) -> Any:
        """创建带工具的模型实例"""
        if tool_names is None:
            # 使用所有可用工具
            return ModelFactory.create_model_with_all_tools(self.model_key)
        else:
            # 使用指定工具
            return ModelFactory.create_model_with_tools(self.model_key, tool_names)
    
    def execute_tool_call(self, tool_call: Dict[str, Any]) -> str:
        """执行单个工具调用"""
        try:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            
            if not tool_name:
                raise ValueError("工具调用缺少名称")
            
            # 获取工具实例
            tool = tool_manager.get_tool(tool_name)
            
            # 执行工具
            result = tool.invoke(tool_args)
            
            self.logger.info(f"工具 {tool_name} 执行成功: {tool_args}")
            return str(result)
            
        except Exception as e:
            error_msg = f"工具 {tool_name} 执行失败: {str(e)}"
            self.logger.error(error_msg)
            return error_msg
    
    def process_tool_calls(self, ai_message: AIMessage) -> List[ToolMessage]:
        """处理AI消息中的所有工具调用"""
        tool_messages = []
        
        if not hasattr(ai_message, "tool_calls") or not ai_message.tool_calls:
            return tool_messages
        
        for tool_call in ai_message.tool_calls:
            try:
                # 执行工具调用
                result = self.execute_tool_call(tool_call)
                
                # 创建工具消息
                tool_message = ToolMessage(
                    content=result,
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"]
                )
                tool_messages.append(tool_message)
                
            except Exception as e:
                # 创建错误消息
                error_message = ToolMessage(
                    content=f"工具调用错误: {str(e)}",
                    tool_call_id=tool_call.get("id", "unknown"),
                    name=tool_call.get("name", "unknown")
                )
                tool_messages.append(error_message)
        
        return tool_messages
    
    def chat_with_tools(
        self, 
        user_input: str, 
        conversation_history: Optional[List[BaseMessage]] = None,
        tool_names: Optional[List[str]] = None
    ) -> Tuple[str, List[BaseMessage]]:
        """
        带工具的对话处理
        
        Args:
            user_input: 用户输入
            conversation_history: 对话历史
            tool_names: 要使用的工具名称列表
            
        Returns:
            (最终回复, 更新后的对话历史)
        """
        # 初始化对话历史
        if conversation_history is None:
            conversation_history = []
        
        # 创建带工具的模型
        model_with_tools = self.create_model_with_tools(tool_names)
        
        # 添加用户消息
        messages = conversation_history + [HumanMessage(content=user_input)]
        
        try:
            # 第一步：模型生成回复（可能包含工具调用）
            ai_response = model_with_tools.invoke(messages)
            messages.append(ai_response)
            
            # 第二步：处理工具调用
            if hasattr(ai_response, "tool_calls") and ai_response.tool_calls:
                self.logger.info(f"检测到 {len(ai_response.tool_calls)} 个工具调用")
                
                # 执行所有工具调用
                tool_messages = self.process_tool_calls(ai_response)
                messages.extend(tool_messages)
                
                # 第三步：基于工具结果生成最终回复
                final_model = ModelFactory.create_model(self.model_key)
                final_response = final_model.invoke(messages)
                messages.append(final_response)
                
                return final_response.content, messages
            else:
                # 没有工具调用，直接返回AI回复
                return ai_response.content, messages
                
        except Exception as e:
            error_msg = f"对话处理失败: {str(e)}"
            self.logger.error(error_msg)
            return error_msg, messages
    
    def get_available_tools(self) -> Dict[str, str]:
        """获取可用工具列表"""
        return tool_manager.list_tools()