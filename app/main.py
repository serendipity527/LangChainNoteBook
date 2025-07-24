"""
FastAPI主应用模块

这是整个聊天应用的入口点，负责创建和配置FastAPI应用实例。
包含了应用的基本配置、中间件设置、路由注册和健康检查端点。

主要功能：
1. 创建FastAPI应用实例
2. 配置CORS中间件支持跨域请求
3. 注册聊天相关的API路由
4. 提供基础的健康检查端点
5. 配置开发服务器启动参数

技术栈：
- FastAPI: 现代高性能的Python Web框架
- Uvicorn: ASGI服务器，支持异步处理
- CORS: 跨域资源共享支持
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes.chat import router as chat_router

# 创建FastAPI应用实例
# title: 应用标题，显示在自动生成的API文档中
# version: 应用版本号，用于API版本管理
app = FastAPI(
    title="基础API后端",
    version="1.0.0",
    description="基于LangChain和FastAPI的智能聊天应用后端服务",
    docs_url="/docs",      # Swagger UI文档地址
    redoc_url="/redoc"     # ReDoc文档地址
)

# 添加CORS（跨域资源共享）中间件
# 允许前端应用从不同域名访问API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # 允许所有域名访问（生产环境应限制具体域名）
    allow_credentials=True,       # 允许携带认证信息（cookies, authorization headers等）
    allow_methods=["*"],          # 允许所有HTTP方法（GET, POST, PUT, DELETE等）
    allow_headers=["*"],          # 允许所有请求头
)

# 注册聊天相关的路由
# chat_router包含所有/chat前缀的API端点
app.include_router(chat_router)


@app.get("/")
async def root():
    """
    根路径端点

    提供应用的基本信息和运行状态。
    通常用于快速检查应用是否正常启动。

    Returns:
        dict: 包含欢迎消息和运行状态的字典

    示例响应：
        {
            "message": "Hello World",
            "status": "运行中"
        }
    """
    return {
        "message": "Hello World",
        "status": "运行中",
        "service": "LangChain聊天应用",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """
    健康检查端点

    用于监控系统检查应用的健康状态。
    在容器化部署或负载均衡环境中特别有用。

    Returns:
        dict: 包含健康状态的字典

    示例响应：
        {
            "status": "healthy"
        }
    """
    return {
        "status": "healthy",
        "timestamp": "2024-01-01T00:00:00Z",  # 可以添加时间戳
        "services": {
            "chat_service": "operational",
            "model_factory": "operational",
            "chain_factory": "operational"
        }
    }


# 应用启动配置
# 只有在直接运行此文件时才会执行（python app/main.py）
if __name__ == "__main__":
    import uvicorn

    # 启动Uvicorn ASGI服务器
    uvicorn.run(
        "app.main:app",           # 应用模块路径
        host="0.0.0.0",          # 监听所有网络接口
        port=8000,               # 监听端口
        reload=True,             # 开发模式：代码变更时自动重载
        log_level="info",        # 日志级别
        access_log=True          # 启用访问日志
    )