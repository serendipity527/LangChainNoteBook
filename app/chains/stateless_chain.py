"""
无状态对话链模块

该模块实现了无记忆的对话处理链，每次对话都是独立的，不保存历史信息。
适用于单次问答、信息查询等不需要上下文连续性的场景。

特点：
1. 无状态设计：每次对话独立处理，不依赖历史
2. 高性能：无需管理记忆状态，处理速度快
3. 资源节省：不占用额外的内存存储历史
4. 并发友好：无状态特性天然支持高并发

适用场景：
- 单次问答
- 信息查询
- 翻译服务
- 代码生成
- 数学计算
"""

from typing import Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from .base_chain import BaseChain
from ..services.model_factory import ModelFactory
from ..models.chat_models import ChatRequest, ChatResponse


class StatelessChain(BaseChain):
    """
    无状态对话链实现

    继承自BaseChain，实现无记忆的对话处理逻辑。
    每次对话都是独立的，不会保存或使用历史对话信息。

    设计理念：
    - 简单性：最小化的状态管理，降低复杂度
    - 性能：无状态设计带来更好的性能表现
    - 可扩展性：易于水平扩展，支持负载均衡
    - 可靠性：无状态减少了出错的可能性

    内部结构：
    - chains: 缓存不同模型的LCEL链实例，避免重复创建
    - 每个模型对应一个独立的处理链
    - 使用LCEL (LangChain Expression Language) 构建处理流程
    """

    def __init__(self):
        """
        初始化无状态链

        创建链缓存字典，用于存储不同模型的LCEL链实例。
        采用延迟初始化策略，只有在需要时才创建具体的链。
        """
        # 链缓存：存储不同模型的LCEL链实例
        # 键：模型标识符，值：构建好的LCEL链
        self.chains: Dict[str, Any] = {}

    def _get_or_create_chain(self, model_key: str):
        """
        获取或创建指定模型的LCEL链

        使用缓存机制避免重复创建相同模型的链实例。
        每个模型对应一个独立的LCEL链，包含提示模板、模型和输出解析器。

        Args:
            model_key (str): 模型标识符

        Returns:
            Runnable: 构建好的LCEL链实例

        LCEL链结构：
            输入 -> RunnablePassthrough -> 提示模板 -> 模型 -> 字符串解析器 -> 输出
        """
        if model_key not in self.chains:
            # 通过工厂创建模型实例
            model = ModelFactory.create_model(model_key)

            # 创建聊天提示模板
            # 包含系统消息和用户消息两个部分
            prompt = ChatPromptTemplate.from_messages([
                # 系统消息：定义AI助手的角色和行为准则
                ("system", "你是一个有用的AI助手，请根据用户的问题提供准确、有帮助的回答。"),
                # 用户消息：接收用户输入，使用{input}占位符
                ("human", "{input}")
            ])

            # 构建LCEL链：使用管道操作符(|)连接各个组件
            self.chains[model_key] = (
                RunnablePassthrough()    # 透传输入数据，不做任何修改
                | prompt                 # 应用提示模板，格式化输入
                | model                  # 调用AI模型生成回复
                | StrOutputParser()      # 解析模型输出为字符串
            )

        return self.chains[model_key]

    async def invoke(self, request: ChatRequest, model_key: str = "qwen3:0.6b", **kwargs) -> ChatResponse:
        """
        执行无记忆对话处理

        这是无状态链的核心方法，处理用户的聊天请求并返回AI回复。
        每次调用都是独立的，不会使用或保存任何历史信息。

        处理流程：
        1. 获取或创建对应模型的LCEL链
        2. 使用链处理用户输入
        3. 构造并返回响应对象
        4. 异常处理：捕获并返回错误信息

        Args:
            request (ChatRequest): 用户的聊天请求
            model_key (str): 使用的模型标识符，默认为"qwen3:0.6b"
            **kwargs: 额外参数（无状态链中暂未使用）

        Returns:
            ChatResponse: 包含AI回复和元数据的响应对象

        Note:
            - 方法是异步的，支持并发处理
            - 包含完整的异常处理机制
            - 返回的响应明确标识为无记忆模式
        """
        try:
            # 获取对应模型的处理链
            chain = self._get_or_create_chain(model_key)

            # 异步调用链处理用户输入
            # ainvoke是LCEL链的异步调用方法
            response = await chain.ainvoke({"input": request.message})

            # 构造成功响应
            return ChatResponse(
                response=response,        # AI生成的回复内容
                model_used=model_key,     # 实际使用的模型
                has_memory=False          # 明确标识为无记忆模式
            )

        except Exception as e:
            # 异常处理：返回错误信息而不是抛出异常
            # 这确保了API的稳定性和用户体验
            return ChatResponse(
                response=f"处理请求时出现错误：{str(e)}",
                model_used=model_key,
                has_memory=False
            )

    def get_chain_type(self) -> str:
        """
        返回链类型标识符

        Returns:
            str: 固定返回"stateless"，标识这是无状态链
        """
        return "stateless"