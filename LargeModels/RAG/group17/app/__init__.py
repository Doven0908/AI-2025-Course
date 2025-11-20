from __future__ import annotations

from dotenv import load_dotenv

# 在导入其他模块之前加载 .env 配置
load_dotenv()

from .config import Settings, load_settings
from .pipeline import RAGPipeline

__all__ = ["Settings", "load_settings", "RAGPipeline"]
