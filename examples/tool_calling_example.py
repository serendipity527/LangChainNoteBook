"""
工具调用完整示例

展示如何在项目中使用工具调用功能的完整流程。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.tool_calling_service import ToolCallingService
from app.tools.tool_manager import tool_manager
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)


def basic_tool_calling_example():
    """基础工具调用示例"""
    print("=" * 60)
    print("基础工具调用示例")
    print("=" * 60)
    
    # 创建工具调用服务
    service = ToolCallingService(model_key="qwen3:4b")
    
    # 测试用例
    test_cases = [
        "现在几点了？",
        "计算 25 * 4 + 10 的结果",
        "查询北京的天气情况",
        "帮我计算 sqrt(144) + 5^2 的值"
    ]
    
    conversation_history = []
    
    for i, user_input in enumerate(test_cases, 1):
        print(f"\n--- 测试 {i} ---")
        print(f"用户: {user_input}")
        
        try:
            response, conversation_history = service.chat_with_tools(
                user_input=user_input,
                conversation_history=conversation_history
            )
            print(f"助手: {response}")
            
        except Exception as e:
            print(f"错误: {e}")


def selective_tool_calling_example():
    """选择性工具调用示例"""
    print("\n" + "=" * 60)
    print("选择性工具调用示例")
    print("=" * 60)
    
    service = ToolCallingService(model_key="qwen3:4b")
    
    # 只使用计算器工具
    print("\n--- 只使用计算器工具 ---")
    response, _ = service.chat_with_tools(
        user_input="计算 15 * 8 并告诉我现在几点",
        tool_names=["calculator"]
    )
    print(f"助手: {response}")
    
    # 只使用天气工具
    print("\n--- 只使用天气工具 ---")
    response, _ = service.chat_with_tools(
        user_input="查询上海天气并计算 10+20",
        tool_names=["get_weather"]
    )
    print(f"助手: {response}")


def error_handling_example():
    """错误处理示例"""
    print("\n" + "=" * 60)
    print("错误处理示例")
    print("=" * 60)
    
    service = ToolCallingService(model_key="qwen3:4b")
    
    # 测试计算错误
    print("\n--- 测试计算错误 ---")
    response, _ = service.chat_with_tools(
        user_input="计算 10 / 0 的结果"
    )
    print(f"助手: {response}")
    
    # 测试不存在的城市
    print("\n--- 测试不存在的城市 ---")
    response, _ = service.chat_with_tools(
        user_input="查询火星的天气"
    )
    print(f"助手: {response}")


def interactive_chat_example():
    """交互式对话示例"""
    print("\n" + "=" * 60)
    print("交互式工具调用对话")
    print("=" * 60)
    print("输入 'quit' 退出对话")
    print("可用工具:", ", ".join(tool_manager.list_tools().keys()))
    
    service = ToolCallingService(model_key="qwen3:4b")
    conversation_history = []
    
    while True:
        try:
            user_input = input("\n用户: ").strip()
            if user_input.lower() in ['quit', 'exit', '退出']:
                print("再见！")
                break
            
            if not user_input:
                continue
            
            response, conversation_history = service.chat_with_tools(
                user_input=user_input,
                conversation_history=conversation_history
            )
            print(f"助手: {response}")
            
        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"发生错误: {e}")


if __name__ == "__main__":
    # 运行所有示例
    basic_tool_calling_example()
    selective_tool_calling_example() 
    error_handling_example()
    
    # 交互式对话（可选）
    run_interactive = input("\n是否运行交互式对话？(y/n): ").lower().strip()
    if run_interactive == 'y':
        interactive_chat_example()