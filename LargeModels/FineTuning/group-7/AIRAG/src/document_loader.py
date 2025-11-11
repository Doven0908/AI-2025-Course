"""文档加载和分块模块"""
from pathlib import Path
from typing import List
from loguru import logger
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    DirectoryLoader
)
from langchain.schema import Document
import config


class DocumentProcessor:
    """文档处理器：加载和分块"""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " ", ""]
        )
        
    def load_documents(self, path: str) -> List[Document]:
        """加载文档"""
        path_obj = Path(path)
        documents = []
        
        if path_obj.is_file():
            documents = self._load_single_file(path)
        elif path_obj.is_dir():
            documents = self._load_directory(path)
        else:
            logger.error(f"路径不存在: {path}")
            return []
        
        logger.info(f"成功加载 {len(documents)} 个文档")
        return documents
    
    def _load_single_file(self, file_path: str) -> List[Document]:
        """加载单个文件"""
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()
        
        try:
            if suffix == '.pdf':
                loader = PyPDFLoader(str(file_path))
            elif suffix == '.txt':
                loader = TextLoader(str(file_path), encoding='utf-8')
            elif suffix in ['.docx', '.doc']:
                loader = Docx2txtLoader(str(file_path))
            else:
                logger.warning(f"不支持的文件格式: {suffix}")
                return []
            
            documents = loader.load()
            logger.info(f"加载文件: {file_path.name}, 包含 {len(documents)} 个文档")
            return documents
        
        except Exception as e:
            logger.error(f"加载文件失败 {file_path}: {e}")
            return []
    
    def _load_directory(self, dir_path: str) -> List[Document]:
        """加载目录下所有支持的文档"""
        documents = []
        dir_path = Path(dir_path)
        
        # 支持的文件扩展名
        supported_extensions = ['.pdf', '.txt', '.docx', '.doc']
        
        for ext in supported_extensions:
            files = list(dir_path.glob(f"**/*{ext}"))
            for file in files:
                docs = self._load_single_file(str(file))
                documents.extend(docs)
        
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分块文档"""
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"文档分块完成: {len(documents)} 个文档 -> {len(chunks)} 个块")
        return chunks
    
    def process_documents(self, path: str) -> List[Document]:
        """完整的文档处理流程：加载 + 分块"""
        documents = self.load_documents(path)
        if not documents:
            return []
        
        chunks = self.split_documents(documents)
        return chunks


if __name__ == "__main__":
    # 测试代码
    processor = DocumentProcessor()
    chunks = processor.process_documents(config.DOCUMENTS_PATH)
    
    if chunks:
        print(f"处理完成，共 {len(chunks)} 个文档块")
        print(f"第一个块预览:\n{chunks[0].page_content[:200]}")

