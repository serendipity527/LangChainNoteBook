| 模块名                     | 作用简介                                  | 举例/关键词                         |
| ----------------------- | ------------------------------------- | ------------------------------ |
| **LLMs/Chat Models**    | 接入底层大模型（OpenAI、Ollama、本地模型等）          | `ChatOpenAI`、`ChatOllama`      |
| **PromptTemplates**     | 构造提示词模板，组织输入格式                        | `ChatPromptTemplate`           |
| **Chains**              | 串联多个步骤（Prompt → LLM → 输出）形成完整应用流程     | `LLMChain`、`ConversationChain` |
| **Memory**              | 实现对话记忆，保留上下文信息                        | `ConversationBufferMemory`     |
| **Agents**              | 拓展能力，调用外部工具（如搜索、计算器）解决复杂任务            | `initialize_agent`             |
| **Tools**               | 工具接口，被 Agent 调用，比如搜索、数据库查询            | `tool()` 装饰器、`PythonREPL`      |
| **Retriever**           | 检索模块，从文档或向量数据库中检索相关内容                 | `FAISSRetriever`、`Chroma`      |
| **VectorStores**        | 文本向量化和存储，用于构建 RAG（检索增强生成）系统           | `FAISS`、`Chroma`、`Pinecone`    |
| **Document Loaders**    | 加载多种类型文档并解析为文本                        | `PyPDFLoader`、`TextLoader`     |
| **Embeddings**          | 文本向量化（如 OpenAI Embedding、HuggingFace） | `OpenAIEmbeddings`             |
| **Runnable / LCEL**     | 新版组合式表达框架，代替旧版 chain，更灵活、可视化          | `RunnableMap`、`.invoke()`      |
| **Callbacks / Tracing** | 调试、可视化链条调用过程                          | `LangSmith`、`ConsoleCallback`  |
| **LangGraph**           | 状态机式对话建模，更复杂对话流（推荐）                   | 多节点对话管理、带记忆                    |
---

基于 LangChain 0.3 的文档，LangChain 主要包含以下组件：

## 核心架构组件

**`@langchain/core`** - 基础抽象和 LangChain Expression Language
**`@langchain/community`** - 第三方集成
**Partner packages** - 独立的合作伙伴包（如 `@langchain/openai`、`@langchain/anthropic`）
**`langchain`** - 主包，包含链、代理和检索策略
**LangGraph.js** - 构建状态化多角色应用
**LangSmith** - 开发者平台，用于调试、测试和监控

## 核心功能模块

### Model I/O（模型输入输出）
- **Chat Models** - 聊天模型接口
- **LLMs** - 大语言模型接口  
- **Messages** - 消息处理（`HumanMessage`、`AIMessage`、`SystemMessage`）
- **Structured Output** - 结构化输出

### Prompts（提示词）
- **PromptTemplates** - 提示词模板
- **ChatPromptTemplate** - 聊天提示词模板
- **MessagesPlaceholder** - 消息占位符

### Data Connection（数据连接）
- **Document Loaders** - 文档加载器（`PyPDFLoader`、`TextLoader`）
- **Text Splitters** - 文本分割器
- **Embedding Models** - 嵌入模型（`OpenAIEmbeddings`）
- **Vector Stores** - 向量存储（`FAISS`、`Chroma`、`Pinecone`）
- **Retrievers** - 检索器（`FAISSRetriever`）

### Chains（链）
- **LLMChain** - 基础LLM链
- **ConversationChain** - 对话链
- **Retrieval Chains** - 检索链
- **Sequential Chains** - 顺序链

### Memory（记忆）
- **ConversationBufferMemory** - 缓冲记忆
- **ConversationSummaryMemory** - 摘要记忆
- **ConversationBufferWindowMemory** - 窗口缓冲记忆
- **ConversationSummaryBufferMemory** - 摘要缓冲记忆
- **ConversationTokenBufferMemory** - 令牌缓冲记忆
- **Chat History** - 聊天历史管理

### Agents（代理）
- **Agent Executors** - 代理执行器
- **Agent Types** - 各种代理类型
- **Tool Calling** - 工具调用

### Tools（工具）
- **Tool Interface** - 工具接口
- **PythonREPL** - Python解释器工具
- **Search Tools** - 搜索工具
- **Custom Tools** - 自定义工具

### LCEL（LangChain Expression Language）
- **Runnable Interface** - 可运行接口
- **RunnablePassthrough** - 透传运行器
- **RunnableParallel** - 并行运行器
- **RunnableBranch** - 分支运行器
- **RunnableLambda** - Lambda运行器
- **RunnableMap** - 映射运行器

### Callbacks & Monitoring（回调和监控）
- **Callbacks** - 回调系统
- **Tracing** - 链路追踪
- **LangSmith Integration** - LangSmith集成
- **ConsoleCallback** - 控制台回调
- **Streaming** - 流式处理

### 高级功能
- **Multimodality** - 多模态支持
- **RAG (Retrieval Augmented Generation)** - 检索增强生成
- **Query Analysis** - 查询分析
- **Graph Databases** - 图数据库支持
- **SQL Integration** - SQL集成

### 实验性模块
- **Experimental Features** - 实验性功能
- **Beta Components** - 测试版组件

这些组件可以灵活组合使用，从简单的 Prompt + LLM 到复杂的 RAG 系统和多代理应用。LangChain 0.3 特别推荐使用 LCEL 和 LangGraph 来构建现代化的 LLM 应用。
