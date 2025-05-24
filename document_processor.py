import os
import logging
from typing import List, Dict, Any, Callable, Optional
import requests
import json
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LMStudioEmbeddings(Embeddings):
    """Lớp tạo embeddings sử dụng API của LM Studio."""
    
    def __init__(self, lm_studio_url: str = "http://127.0.0.1:1234"):
        """Khởi tạo LMStudioEmbeddings với URL của LM Studio API."""
        self.lm_studio_url = lm_studio_url
        self.embedding_url = f"{lm_studio_url}/v1/embeddings"
        # Thử kết nối đến LM Studio để kiểm tra
        try:
            response = requests.get(f"{lm_studio_url}/v1/models")
            if response.status_code == 200:
                logger.info(f"Kết nối thành công đến LM Studio API tại {lm_studio_url}")
            else:
                logger.warning(f"Kết nối đến LM Studio API không thành công: {response.status_code}")
        except Exception as e:
            logger.warning(f"Không thể kết nối đến LM Studio API: {e}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Tạo embeddings cho danh sách văn bản."""
        embeddings = []
        
        # Xử lý từng văn bản để tránh quá tải API
        for text in texts:
            try:
                payload = {
                    "input": text,
                    "model": "text-embedding-nomic-embed-text-v1.5-embedding"
                }
                headers = {
                    "Content-Type": "application/json"
                }
                
                response = requests.post(self.embedding_url, json=payload, headers=headers)
                response.raise_for_status()
                
                result = response.json()
                embedding = result.get("data", [{}])[0].get("embedding", [])
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Lỗi khi tạo embedding: {e}")
                # Thêm embedding rỗng để giữ nguyên chỉ số
                embeddings.append([0.0] * 1024)  # Kích thước mặc định cho nomic-embed-text-v1.5
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Tạo embedding cho câu truy vấn."""
        try:
            payload = {
                "input": text,
                "model": "text-embedding-nomic-embed-text-v1.5-embedding"
            }
            headers = {
                "Content-Type": "application/json"
            }
            
            response = requests.post(self.embedding_url, json=payload, headers=headers)
            response.raise_for_status()
            
            result = response.json()
            embedding = result.get("data", [{}])[0].get("embedding", [])
            return embedding
        except Exception as e:
            logger.error(f"Lỗi khi tạo embedding cho truy vấn: {e}")
            # Trả về embedding rỗng nếu có lỗi
            return [0.0] * 1024  # Kích thước mặc định cho nomic-embed-text-v1.5

class DocumentProcessor:
    def __init__(self, 
                 docs_dir: str = "./docs", 
                 chunk_size: int = 500, 
                 chunk_overlap: int = 50,
                 persist_directory: str = "./chroma_db",
                 lm_studio_url: str = "http://127.0.0.1:1234"):
        """
        Khởi tạo DocumentProcessor
        
        Args:
            docs_dir: Thư mục chứa tài liệu cần xử lý
            chunk_size: Kích thước của mỗi đoạn văn bản
            chunk_overlap: Độ chồng lấp giữa các đoạn
            persist_directory: Thư mục lưu trữ vector database
            lm_studio_url: URL của LM Studio API
        """
        self.docs_dir = docs_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.persist_directory = persist_directory
        self.lm_studio_url = lm_studio_url
        
        # Khởi tạo model embedding
        self.embeddings = LMStudioEmbeddings(lm_studio_url=lm_studio_url)
        logger.info(f"Khởi tạo DocumentProcessor với thư mục tài liệu: {docs_dir}")
    
    def load_documents(self) -> List[Document]:
        """
        Tải tài liệu từ thư mục chỉ định
        
        Returns:
            List[Document]: Danh sách tài liệu đã tải
        """
        logger.info(f"Bắt đầu tải tài liệu từ {self.docs_dir}")
        
        # Sử dụng TextLoader với encoding utf-8
        loader = DirectoryLoader(
            self.docs_dir, 
            glob="**/*.txt", 
            loader_cls=TextLoader, 
            loader_kwargs={"encoding": "utf-8"}
        )
        
        documents = loader.load()
        logger.info(f"Đã tải {len(documents)} tài liệu")
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Chia nhỏ tài liệu thành các đoạn
        
        Args:
            documents: Danh sách tài liệu cần chia nhỏ
            
        Returns:
            List[Document]: Danh sách các đoạn văn bản
        """
        logger.info(f"Chia nhỏ tài liệu với chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Đã chia nhỏ thành {len(chunks)} đoạn văn bản")
        return chunks
    
    def create_vector_db(self, chunks: List[Document] = None) -> Chroma:
        """
        Tạo vector database từ các đoạn văn bản
        
        Args:
            chunks: Danh sách các đoạn văn bản (nếu None, sẽ tự động tải và chia nhỏ tài liệu)
            
        Returns:
            Chroma: Vector database đã tạo
        """
        if chunks is None:
            documents = self.load_documents()
            chunks = self.split_documents(documents)
        
        logger.info(f"Tạo vector database tại {self.persist_directory}")
        
        # Kiểm tra thư mục đã tồn tại chưa
        if not os.path.exists(self.persist_directory):
            os.makedirs(self.persist_directory)
            
        # Tạo vector database
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        # Lưu vector database xuống đĩa
        vectordb.persist()
        logger.info(f"Đã tạo và lưu vector database với {len(chunks)} đoạn văn bản")
        return vectordb
    
    def load_vector_db(self) -> Chroma:
        """
        Tải vector database từ đĩa
        
        Returns:
            Chroma: Vector database đã tải
        """
        logger.info(f"Tải vector database từ {self.persist_directory}")
        vectordb = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        logger.info(f"Đã tải vector database thành công")
        return vectordb

    def process_all(self, progress_callback: Optional[Callable[[float], None]] = None) -> Chroma:
        """
        Xử lý toàn bộ quá trình: tải tài liệu, chia nhỏ, tạo vector database
        
        Args:
            progress_callback: Hàm callback để cập nhật tiến trình, nhận giá trị từ 0.0 đến 1.0
            
        Returns:
            Chroma: Vector database đã tạo
        """
        # Gọi callback với tiến trình 0%
        if progress_callback:
            progress_callback(0.0)
            
        # Tải tài liệu (30% công việc)
        documents = self.load_documents()
        if progress_callback:
            progress_callback(0.3)
            
        # Chia nhỏ tài liệu (60% công việc)
        chunks = self.split_documents(documents)
        if progress_callback:
            progress_callback(0.6)
            
        # Tạo vector database (100% công việc)
        vectordb = self.create_vector_db(chunks)
        if progress_callback:
            progress_callback(1.0)
            
        return vectordb 