
"""
记忆对话链模块

该模块实现了带记忆功能的对话处理链，能够保存和使用历史对话信息，
提供连续性的对话体验。支持多种记忆类型和多会话管理。

核心功能：
1. 对话历史管理：保存用户和AI的完整对话记录
2. 多种记忆类型：支持缓冲记忆和摘要记忆
3. 多会话支持：通过chat_id区分不同的对话会话
4. 智能摘要：长对话自动摘要，节省token消耗

记忆类型说明：
- Buffer Memory: 保存完整的对话历史，适合短对话
- Summary Memory: 智能摘要长对话，适合长期对话

适用场景：
- 连续对话
- 客服系统
- 个人助手
- 教学辅导
- 项目讨论
"""

from typing import Dict, List, Optional, Any
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.schema import BaseMemory

from .base_chain import BaseChain
from ..services.model_factory import ModelFactory
from ..models.chat_models import ChatRequest, ChatResponse


class MemoryChain(BaseChain):
    """
    记忆对话链实现

    继承自BaseChain，实现带记忆功能的对话处理逻辑。
    能够保存对话历史并在后续对话中使用，提供连续性的对话体验。

    设计特点：
    - 多会话管理：通过chat_id区分不同用户或不同对话主题
    - 多记忆类型：支持完整记忆和智能摘要两种模式
    - 资源优化：摘要模式可以控制记忆长度，节省token
    - 灵活配置：每个会话可以独立配置记忆类型

    内部结构：
    - memory_storage: 存储所有会话的记忆实例
    - chains: 缓存不同模型和记忆类型组合的LCEL链
    """

    def __init__(self):
        """
        初始化记忆链

        创建记忆存储和链缓存字典。采用延迟初始化策略，
        只有在需要时才创建具体的记忆实例和处理链。
        """
        # 记忆存储：存储所有会话的记忆实例
        # 键格式："{chat_id}_{memory_type}"，值：记忆实例
        self.memory_storage: Dict[str, BaseMemory] = {}

        # 链缓存：存储不同配置的LCEL链实例
        # 键格式："{model_key}_{memory_type}"，值：LCEL链
        self.chains: Dict[str, Any] = {}

    def _get_or_create_memory(self, chat_id: str, memory_type: str = "buffer", model_key: str = "qwen3:0.6b") -> BaseMemory:
        """
        获取或创建记忆实例

        根据会话ID和记忆类型创建对应的记忆实例。使用缓存机制
        避免重复创建，每个会话的记忆实例在整个生命周期中保持唯一。

        Args:
            chat_id (str): 会话标识符，用于区分不同的对话会话
            memory_type (str): 记忆类型，支持"buffer"和"summary"
            model_key (str): 模型标识符，摘要模式需要用于生成摘要

        Returns:
            BaseMemory: 对应的记忆实例

        记忆类型详解：
        1. Buffer Memory (缓冲记忆):
           - 保存完整的对话历史
           - 适合短对话或需要完整上下文的场景
           - 内存占用随对话长度线性增长

        2. Summary Memory (摘要记忆):
           - 智能摘要长对话，保持固定的token限制
           - 适合长期对话或token预算有限的场景
           - 使用AI模型生成对话摘要，保留关键信息
        """
        # 构造记忆键：结合会话ID和记忆类型
        memory_key = f"{chat_id}_{memory_type}"

        # 检查是否已存在记忆实例
        if memory_key not in self.memory_storage:
            if memory_type == "buffer":
                # 创建缓冲记忆：保存完整对话历史
                self.memory_storage[memory_key] = ConversationBufferMemory(
                    return_messages=True,      # 返回消息对象而非字符串
                    memory_key="chat_history"  # 在提示模板中的变量名
                )
            elif memory_type == "summary":
                # 创建摘要记忆：智能摘要长对话
                model = ModelFactory.create_model(model_key)
                self.memory_storage[memory_key] = ConversationSummaryBufferMemory(
                    llm=model,                    # 用于生成摘要的模型
                    return_messages=True,         # 返回消息对象
                    memory_key="chat_history",    # 在提示模板中的变量名
                    max_token_limit=1000         # 最大token限制，超过时触发摘要
                )
            else:
                # 不支持的记忆类型
                raise ValueError(f"不支持的记忆类型: {memory_type}。支持的类型: ['buffer', 'summary']")

        return self.memory_storage[memory_key]
    
    def _create_memory_chain(self, model_key: str):
        """
        创建带记忆功能的LCEL链

        构建包含记忆功能的对话处理链，与无状态链的主要区别是
        在提示模板中加入了历史对话的占位符。

        Args:
            model_key (str): 模型标识符

        Returns:
            Runnable: 构建好的LCEL链实例

        链结构：
            输入 -> RunnablePassthrough -> 提示模板(含历史) -> 模型 -> 字符串解析器 -> 输出
        """
        # 通过工厂创建模型实例
        model = ModelFactory.create_model(model_key)

        # 创建包含记忆的聊天提示模板
        prompt = ChatPromptTemplate.from_messages([
            # 系统消息：定义AI助手的角色，强调记忆能力
            ("system", "你是一个友好的AI助手，能够记住对话历史并提供有用的回答。"),
            # 历史消息占位符：这里会插入之前的对话历史
            MessagesPlaceholder(variable_name="chat_history"),
            # 当前用户输入
            ("human", "{input}")
        ])

        # 构建LCEL链：与无状态链类似，但提示模板包含历史信息
        chain = (
            RunnablePassthrough()    # 透传输入数据
            | prompt                 # 应用包含历史的提示模板
            | model                  # 调用AI模型
            | StrOutputParser()      # 解析输出为字符串
        )

        return chain
    
    async def invoke(self, request: ChatRequest, model_key: str = "qwen3:0.6b",
                    chat_id: str = "default", memory_type: str = "buffer", **kwargs) -> ChatResponse:
        """
        执行带记忆的对话处理

        这是记忆链的核心方法，处理用户的聊天请求并维护对话历史。
        与无状态链的主要区别是会加载历史对话并在处理后保存新的对话。

        处理流程：
        1. 获取或创建对应的记忆实例
        2. 获取或创建对应的LCEL链
        3. 加载历史对话记录
        4. 使用链处理当前输入（包含历史上下文）
        5. 保存新的对话到记忆中
        6. 构造并返回响应

        Args:
            request (ChatRequest): 用户的聊天请求
            model_key (str): 使用的模型标识符
            chat_id (str): 会话标识符，用于区分不同对话
            memory_type (str): 记忆类型，"buffer"或"summary"
            **kwargs: 额外参数

        Returns:
            ChatResponse: 包含AI回复和记忆信息的响应对象

        Note:
            - 每次对话都会更新对应会话的记忆
            - 支持多个并发会话，通过chat_id区分
            - 异常处理确保系统稳定性
        """
        try:
            # 1. 获取或创建记忆实例
            memory = self._get_or_create_memory(chat_id, memory_type, model_key)

            # 2. 获取或创建对应的LCEL链
            # 链的键包含模型和记忆类型，确保不同配置使用不同的链
            chain_key = f"{model_key}_{memory_type}"
            if chain_key not in self.chains:
                self.chains[chain_key] = self._create_memory_chain(model_key)

            chain = self.chains[chain_key]

            # 3. 加载历史对话记录
            # chat_memory.messages包含了所有历史消息对象
            chat_history = memory.chat_memory.messages

            # 4. 异步调用链处理输入
            # 传入当前用户输入和完整的对话历史
            response = await chain.ainvoke({
                "input": request.message,      # 当前用户输入
                "chat_history": chat_history   # 历史对话记录
            })

            # 5. 保存新的对话到记忆中
            # save_context会自动将输入和输出转换为消息对象并保存
            memory.save_context(
                {"input": request.message},    # 用户输入
                {"output": response}           # AI回复
            )

            # 6. 构造成功响应
            return ChatResponse(
                response=response,           # AI生成的回复
                model_used=model_key,        # 使用的模型
                has_memory=True,             # 标识使用了记忆功能
                chat_id=chat_id,            # 会话标识符
                memory_type=memory_type      # 记忆类型
            )

        except Exception as e:
            # 异常处理：返回错误信息，保持API稳定性
            return ChatResponse(
                response=f"处理请求时出现错误：{str(e)}",
                model_used=model_key,
                has_memory=True,
                chat_id=chat_id,
                memory_type=memory_type
            )
    
    def get_chat_history(self, chat_id: str, memory_type: str = "buffer") -> List[Dict[str, str]]:
        """
        获取指定会话的对话历史

        将内部的消息对象转换为前端友好的字典格式，
        便于API返回和前端展示。

        Args:
            chat_id (str): 会话标识符
            memory_type (str): 记忆类型

        Returns:
            List[Dict[str, str]]: 对话历史列表，每个元素包含role和content
                                 role: "user"或"assistant"
                                 content: 消息内容

        Example:
            >>> history = chain.get_chat_history("user_123", "buffer")
            >>> print(history)
            [
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "你好！有什么可以帮助您的吗？"},
                {"role": "user", "content": "介绍一下自己"}
            ]
        """
        # 构造记忆键
        memory_key = f"{chat_id}_{memory_type}"

        # 检查记忆是否存在
        if memory_key not in self.memory_storage:
            return []  # 返回空列表表示没有历史记录

        # 获取记忆实例和消息列表
        memory = self.memory_storage[memory_key]
        messages = memory.chat_memory.messages

        # 转换消息格式
        history = []
        for msg in messages:
            # 检查消息对象是否有content属性
            if hasattr(msg, 'content'):
                # 根据消息类型确定角色
                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                history.append({
                    "role": role,
                    "content": msg.content
                })

        return history

    def clear_memory(self, chat_id: str, memory_type: str = "buffer") -> bool:
        """
        清除指定会话的记忆

        删除指定会话的所有历史记录，释放内存资源。
        通常用于用户主动清除历史或会话结束时的清理。

        Args:
            chat_id (str): 会话标识符
            memory_type (str): 记忆类型

        Returns:
            bool: 清除是否成功
                  True: 成功清除记忆
                  False: 记忆不存在，无需清除

        Example:
            >>> success = chain.clear_memory("user_123", "buffer")
            >>> if success:
            ...     print("记忆已清除")
            ... else:
            ...     print("记忆不存在")
        """
        # 构造记忆键
        memory_key = f"{chat_id}_{memory_type}"

        # 检查并删除记忆
        if memory_key in self.memory_storage:
            del self.memory_storage[memory_key]
            return True  # 成功删除
        return False     # 记忆不存在

    def get_chain_type(self) -> str:
        """
        返回链类型标识符

        Returns:
            str: 固定返回"memory"，标识这是记忆链
        """
        return "memory"
