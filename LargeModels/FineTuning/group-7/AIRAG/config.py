"""配置文件"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 项目根目录
ROOT_DIR = Path(__file__).parent.absolute()

# 数据目录
DATA_DIR = ROOT_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
VECTORDB_DIR = DATA_DIR / "vectordb"
RESULTS_DIR = DATA_DIR / "results"
LOGS_DIR = DATA_DIR / "logs"

# 创建必要的目录
for dir_path in [DATA_DIR, DOCUMENTS_DIR, VECTORDB_DIR, RESULTS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# 模型配置
MODEL_TYPE = os.getenv("MODEL_TYPE", "api")  # api 或 local
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "xhang")  # 支持: xhang, gpt-3.5-turbo, gpt-4 等
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "f93082e1-2cbf-4f81-af8f-9c98d528b6b1")  # 小航LLM API Key
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://xhang.buaa.edu.cn/xhang/v1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-zh-v1.5")
LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", "")

# 向量数据库配置
VECTOR_DB_TYPE = os.getenv("VECTOR_DB_TYPE", "faiss")
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", str(VECTORDB_DIR))

# 文档路径
DOCUMENTS_PATH = os.getenv("DOCUMENTS_PATH", str(DOCUMENTS_DIR))

# 检索配置
TOP_K = int(os.getenv("TOP_K", "5"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# 日志配置
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

