基于 LangGraph 0.3.25 的文档，LangGraph 主要包含以下组件：

## 核心图组件

### Graph Types（图类型）
- **StateGraph** - 状态图，节点通过共享状态通信
- **Graph** - 基础图类
- **MessagesState** - 预定义的消息状态类型

### Graph Building（图构建）
- **START** - 图的起始节点标识符
- **END** - 图的结束节点标识符
- **add_node()** - 添加节点到图中
- **add_edge()** - 添加普通边连接节点
- **add_conditional_edges()** - 添加条件边，支持动态路由
- **set_entry_point()** - 设置图的入口点
- **set_finish_point()** - 设置图的结束点

### State Management（状态管理）
- **State Schema** - 状态模式定义
- **Annotated Types** - 带注解的类型，支持reducer函数
- **Reducer Functions** - 状态聚合函数 `(Value, Value) -> Value`
- **Partial State Updates** - 部分状态更新机制

### Persistence & Memory（持久化和记忆）
- **MemorySaver** - 内存检查点保存器
- **Checkpointer** - 检查点机制，支持状态持久化
- **BaseStore** - 基础存储接口
- **InMemoryStore** - 内存存储实现
- **Cross-thread Persistence** - 跨线程持久化支持

### Configuration（配置）
- **RunnableConfig** - 运行时配置
- **ConfigSchema** - 配置模式定义
- **ConfigurableFieldSpec** - 可配置字段规范
- **thread_id** - 线程标识符配置
- **user_id** - 用户标识符配置

### Execution & Control（执行和控制）
- **compile()** - 编译图为可执行的Runnable
- **invoke()** - 同步执行图
- **stream()** - 流式执行图
- **recursion_limit** - 递归限制控制
- **Loop Control** - 循环控制机制

### Routing & Conditions（路由和条件）
- **Conditional Edges** - 条件边，支持动态路由决策
- **Route Functions** - 路由函数，决定下一个节点
- **Branch Logic** - 分支逻辑处理

### Visualization（可视化）
- **get_graph()** - 获取图结构
- **draw_mermaid_png()** - 生成Mermaid图表
- **Graph Visualization** - 图形化展示图结构

### Advanced Features（高级功能）

#### Multi-Agent Support（多代理支持）
- **Agent Nodes** - 代理节点
- **Tool Nodes** - 工具节点
- **Agent Coordination** - 代理协调机制

#### Memory Systems（记忆系统）
- **Short-term Memory** - 短期记忆
- **Long-term Memory** - 长期记忆
- **Memory Layers** - 多层记忆架构
- **Memory Search** - 记忆搜索功能

#### RAG Integration（RAG集成）
- **Document Retrieval** - 文档检索
- **Grade Documents** - 文档评分
- **Query Transformation** - 查询转换
- **Web Search Integration** - 网络搜索集成

### Specialized Graph Patterns（专用图模式）

#### ReAct Pattern（ReAct模式）
- **Reasoning Nodes** - 推理节点
- **Action Nodes** - 行动节点
- **Observation Handling** - 观察处理

#### CRAG Pattern（纠正式RAG）
- **Corrective Retrieval** - 纠正式检索
- **Document Grading** - 文档评分
- **Query Rewriting** - 查询重写

#### TNT-LLM Pattern（文本挖掘）
- **Map-Reduce Chains** - Map-Reduce链
- **Taxonomy Generation** - 分类生成
- **Batch Processing** - 批处理支持

### Error Handling & Debugging（错误处理和调试）
- **Exception Handling** - 异常处理机制
- **State Inspection** - 状态检查
- **Execution Tracing** - 执行追踪
- **Debug Utilities** - 调试工具

### Integration Features（集成功能）
- **LangChain Integration** - 与LangChain的集成
- **Tool Integration** - 工具集成支持
- **Model Integration** - 模型集成支持
- **Store Integration** - 存储集成支持

LangGraph 的核心优势在于提供了状态化的图执行框架，特别适合构建复杂的对话系统、多代理应用和需要状态管理的AI工作流。
