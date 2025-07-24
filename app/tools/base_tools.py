"""
基础工具模块

定义项目中常用的工具函数，使用 @tool 装饰器创建标准化工具。
遵循 LangChain 0.3 的工具创建最佳实践。
"""

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import math
import datetime
import json


class CalculatorInput(BaseModel):
    """计算器工具输入参数"""
    expression: str = Field(
        description="要计算的数学表达式，支持基本运算符 +, -, *, /, ** 和括号"
    )


class WeatherInput(BaseModel):
    """天气查询工具输入参数"""
    city: str = Field(description="城市名称，如'北京'、'上海'")
    country: Optional[str] = Field(
        default="CN", 
        description="国家代码，如'CN'、'US'等，默认为中国"
    )


@tool(args_schema=CalculatorInput)
def calculator(expression: str) -> str:
    """执行数学计算的工具。
    
    支持基本的数学运算，包括加减乘除、幂运算和括号。
    
    Args:
        expression: 数学表达式字符串
        
    Returns:
        计算结果的字符串表示
        
    Raises:
        ValueError: 当表达式无效时
    """
    try:
        # 安全的数学表达式计算
        allowed_names = {
            k: v for k, v in math.__dict__.items() 
            if not k.startswith("__")
        }
        allowed_names.update({"abs": abs, "round": round})
        
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"{expression} = {result}"
    except Exception as e:
        raise ValueError(f"计算错误: {str(e)}")


@tool(args_schema=WeatherInput)
def get_weather(city: str, country: Optional[str] = "CN") -> str:
    """获取指定城市的天气信息。
    
    Args:
        city: 城市名称
        country: 国家代码，默认为CN
        
    Returns:
        天气信息的JSON字符串
    """
    # 模拟天气数据
    weather_data = {
        "北京": {"temperature": 15, "condition": "晴朗", "humidity": 45},
        "上海": {"temperature": 18, "condition": "多云", "humidity": 65},
        "广州": {"temperature": 25, "condition": "小雨", "humidity": 80},
        "深圳": {"temperature": 24, "condition": "晴朗", "humidity": 70}
    }
    
    if city in weather_data:
        data = weather_data[city]
        return json.dumps({
            "city": city,
            "country": country,
            "temperature": f"{data['temperature']}°C",
            "condition": data["condition"],
            "humidity": f"{data['humidity']}%",
            "timestamp": datetime.datetime.now().isoformat()
        }, ensure_ascii=False)
    else:
        return json.dumps({
            "error": f"暂无{city}的天气信息",
            "available_cities": list(weather_data.keys())
        }, ensure_ascii=False)


@tool
def get_current_time() -> str:
    """获取当前时间。
    
    Returns:
        当前时间的格式化字符串
    """
    now = datetime.datetime.now()
    return f"当前时间：{now.strftime('%Y年%m月%d日 %H:%M:%S')}"