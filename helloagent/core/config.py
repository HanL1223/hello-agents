import os
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict

class Config(BaseModel):
    """Config for agent"""

    #LLM Config
    default_model: str = 'gpt-6-luna'
    default_provider: str = 'openai'
    temperature: float= 1.0
    max_token: Optional[int] = None

    #System config
    debug: bool = False
    log_level: str = "INFO"

    #Others
    max_history_length: int = 100

    #Context 
    context_window: int = Field(default=128000, gt=0)
    compression_threshold: float = Field(default=0.8, gt=0, le=1)
    min_retain_rounds: int = Field(default=10, gt=0)
    enable_smart_compression: bool = False  # 是否启用智能摘要（需要额外LLM调用）
    
    # 工具输出截断配置
    tool_output_max_lines: int = Field(default=2000, gt=0)
    tool_output_max_bytes: int = Field(default=51200, gt=0)
    tool_output_dir: str = "tool-output"  # 完整输出保存目录
    tool_output_truncate_direction: str = "head"  # 截断方向：head/tail/head_tail

    # Skills 知识外化配置
    skills_enabled: bool = False  # 是否启用 Skills 系统
    skills_dir: str = "skills"  # Skills 目录路径
    skills_auto_register: bool = True  # 是否自动注册 SkillTool

   # 会话持久化配置
    session_enabled: bool = False  # 是否启用会话持久化
    session_dir: str = "memory/sessions"  # 会话文件保存目录
    auto_save_enabled: bool = False  # 是否启用自动保存
    auto_save_interval: int = Field(default=10, gt=0)

    # 可观测性配置
    trace_enabled: bool = False  # 是否启用 Trace 记录
    trace_dir: str = "memory/traces"  # Trace 文件保存目录
    trace_sanitize: bool = True  # 是否脱敏敏感信息
    trace_html_include_raw_response: bool = False  # HTML 是否包含原始响应
    
    # DevLog 开发日志配置
    devlog_enabled: bool = False  # 是否启用 DevLog 工具
    devlog_persistence_dir: str = "memory/devlogs"  # 开发日志持久化目录

    # 异步生命周期配置
    max_concurrent_tools: int = Field(default=3, gt=0)
    hook_timeout_seconds: float = Field(default=5.0, gt=0)
    llm_async_timeout: float = Field(default=120, gt=0)
    tool_async_timeout: float = Field(default=30, gt=0)

    # 流式输出配置
    stream_include_thinking: bool = True  # 是否包含思考过程
    stream_include_tool_calls: bool = True  # 是否包含工具调用

    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables"""
        return cls(
            debug=os.getenv("DEBUG", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("MAX_TOKENS")) if os.getenv("MAX_TOKENS") else None,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict"""
        return self.model_dump()