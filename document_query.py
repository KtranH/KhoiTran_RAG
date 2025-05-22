import logging
import requests
import json
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import Chroma
from document_processor import DocumentProcessor

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentQuery:
    def __init__(self, 
                 vectordb: Optional[Chroma] = None,
                 persist_directory: str = "./chroma_db",
                 lm_studio_url: str = "http://127.0.0.1:1234",
                 model_name: str = "gemma-3-12b-it"):
        """
        Khởi tạo DocumentQuery
        
        Args:
            vectordb: Vector database đã tạo (nếu None, sẽ tải từ đĩa)
            persist_directory: Thư mục lưu trữ vector database
            lm_studio_url: URL của LM Studio API
            model_name: Tên model LLM (mặc định: gemma-3-12b-it)
        """
        self.persist_directory = persist_directory
        self.lm_studio_url = lm_studio_url
        self.model_name = model_name
        
        # Tải vector database nếu chưa được cung cấp
        if vectordb is None:
            processor = DocumentProcessor(persist_directory=persist_directory)
            self.vectordb = processor.load_vector_db()
        else:
            self.vectordb = vectordb
            
        logger.info(f"Khởi tạo DocumentQuery với LM Studio URL: {lm_studio_url}, model: {model_name}")
    
    def evaluate_query_type(self, query: str) -> bool:
        """
        Đánh giá xem câu hỏi có cần thông tin từ tài liệu hay không
        
        Args:
            query: Câu hỏi của người dùng
            
        Returns:
            bool: True nếu cần tìm kiếm trong tài liệu, False nếu là kiến thức chung
        """
        url = f"{self.lm_studio_url}/v1/chat/completions"
        
        system_message = """Bạn là một trợ lý thông minh giúp phân loại câu hỏi. 
Nhiệm vụ của bạn là xác định xem một câu hỏi có yêu cầu thông tin từ tài liệu cụ thể hay không.

Phân loại câu hỏi thành một trong hai loại:
1. Câu hỏi về kiến thức chung hoặc câu hỏi mà bạn đã biết câu trả lời (như về lịch sử, khoa học, văn hóa, nhân vật nổi tiếng, v.v.)
2. Câu hỏi về quy định, hướng dẫn, hoặc thông tin cụ thể có thể có trong tài liệu

Trả lời chỉ với "GENERAL" cho loại 1 hoặc "DOCUMENT" cho loại 2. Không giải thích lý do."""
        
        user_message = f"Đây có phải là câu hỏi cần thông tin từ tài liệu không? Câu hỏi: {query}"
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            "max_tokens": 50,  # Giới hạn token vì chỉ cần câu trả lời ngắn
            "temperature": 0.1,  # Temperature thấp để đảm bảo câu trả lời nhất quán
            "stream": False
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        try:
            logger.info(f"Đánh giá loại câu hỏi: '{query}'")
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            answer = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip().upper()
            
            # Kiểm tra kết quả
            if "DOCUMENT" in answer:
                logger.info(f"Kết quả đánh giá: Câu hỏi cần thông tin từ tài liệu")
                return True
            else:
                logger.info(f"Kết quả đánh giá: Câu hỏi thuộc kiến thức chung")
                return False
                
        except Exception as e:
            logger.error(f"Lỗi khi đánh giá loại câu hỏi: {e}")
            # Mặc định tìm kiếm trong tài liệu nếu có lỗi
            return True
    
    def direct_query_llm(self, query: str, max_tokens: int = 1000, temperature: float = 0.7) -> Dict[str, Any]:
        """
        Truy vấn LLM trực tiếp mà không sử dụng thông tin từ tài liệu
        
        Args:
            query: Câu hỏi của người dùng
            max_tokens: Số lượng token tối đa trong câu trả lời
            temperature: Độ sáng tạo của câu trả lời (0.0 - 1.0)
            
        Returns:
            Dict: Kết quả từ LLM
        """
        url = f"{self.lm_studio_url}/v1/chat/completions"
        
        system_message = "Bạn là một trợ lý thông minh và hữu ích. Hãy trả lời câu hỏi của người dùng một cách chính xác và đầy đủ dựa trên kiến thức của bạn."
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": query}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        logger.info(f"Truy vấn LLM trực tiếp: '{query}'")
        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Lỗi khi truy vấn LLM trực tiếp: {e}")
            return {"error": str(e)}
    
    def search_documents(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Tìm kiếm tài liệu dựa trên truy vấn
        
        Args:
            query: Câu truy vấn
            top_k: Số lượng kết quả trả về
            
        Returns:
            List[Dict]: Danh sách kết quả tìm kiếm
        """
        logger.info(f"Tìm kiếm với truy vấn: '{query}', top_k={top_k}")
        results = self.vectordb.similarity_search_with_score(query, k=top_k)
        
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "relevance_score": score
            })
            
        logger.info(f"Tìm thấy {len(formatted_results)} kết quả")
        return formatted_results
    
    def format_context(self, results: List[Dict[str, Any]]) -> str:
        """
        Định dạng kết quả tìm kiếm thành ngữ cảnh cho LLM
        
        Args:
            results: Kết quả tìm kiếm từ search_documents
            
        Returns:
            str: Ngữ cảnh đã định dạng
        """
        context = "Thông tin liên quan:\n\n"
        
        for i, result in enumerate(results, 1):
            source = result["metadata"].get("source", "Không rõ nguồn").split("/")[-1]
            context += f"[Tài liệu {i}: {source}]\n{result['content']}\n\n"
            
        return context
    
    def query_lm_studio(self, prompt: str, context: str, max_tokens: int = 1000, temperature: float = 0.7) -> Dict[str, Any]:
        """
        Truy vấn LM Studio với prompt và context sử dụng gemma-3-12b-it qua chat API
        
        Args:
            prompt: Câu hỏi của người dùng
            context: Ngữ cảnh từ tài liệu
            max_tokens: Số lượng token tối đa trong câu trả lời
            temperature: Độ sáng tạo của câu trả lời (0.0 - 1.0)
            
        Returns:
            Dict: Kết quả từ LM Studio
        """
        url = f"{self.lm_studio_url}/v1/chat/completions"
        
        # Tạo system message với thông tin về context
        system_message = f"""Bạn là một trợ lý thông minh và hữu ích.

Nếu câu hỏi liên quan đến tài liệu được cung cấp, hãy ưu tiên sử dụng thông tin từ các tài liệu đó để trả lời. 
Nếu không tìm thấy thông tin liên quan trong tài liệu hoặc câu hỏi là về kiến thức chung, bạn có thể sử dụng kiến thức riêng để trả lời.

Đối với câu hỏi về các quy định cụ thể, hãy chỉ dựa vào thông tin trong tài liệu được cung cấp.
Đối với câu hỏi kiến thức chung không liên quan đến tài liệu, hãy trả lời dựa trên hiểu biết của bạn."""

        # Tạo user message với context và câu hỏi
        user_message = f"""Dưới đây là các thông tin liên quan đến câu hỏi của tôi:

{context}

Dựa vào thông tin trên hoặc kiến thức riêng nếu cần, hãy trả lời câu hỏi sau: {prompt}"""
        
        # Tạo payload cho API chat completions
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        logger.info(f"Gửi truy vấn đến LM Studio API (model: {self.model_name})")
        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()  # Kiểm tra lỗi HTTP
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Lỗi khi truy vấn LM Studio: {e}")
            return {"error": str(e)}
    
    def query(self, user_query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Xử lý toàn bộ quá trình RAG: Tìm kiếm tài liệu và truy vấn LLM
        
        Args:
            user_query: Câu hỏi của người dùng
            top_k: Số lượng kết quả tìm kiếm
            
        Returns:
            Dict: Kết quả hoàn chỉnh
        """
        # Đánh giá xem câu hỏi có cần thông tin từ tài liệu không
        needs_document = self.evaluate_query_type(user_query)
        
        # Nếu là câu hỏi kiến thức chung, truy vấn LLM trực tiếp
        if not needs_document:
            logger.info(f"Truy vấn LLM trực tiếp cho câu hỏi kiến thức chung")
            llm_response = self.direct_query_llm(user_query)
            
            # Xử lý lỗi từ LLM
            if "error" in llm_response:
                return {
                    "answer": f"Lỗi khi truy vấn LLM: {llm_response['error']}",
                    "context": [],
                    "sources": []
                }
            
            # Trích xuất câu trả lời
            try:
                answer = llm_response.get("choices", [{}])[0].get("message", {}).get("content", "Không nhận được câu trả lời từ LLM")
                return {
                    "answer": answer.strip(),
                    "context": [],
                    "sources": ["Kiến thức chung"],
                    "is_general_knowledge": True
                }
            except Exception as e:
                logger.error(f"Lỗi khi trích xuất câu trả lời: {e}")
                return {
                    "answer": "Lỗi khi xử lý câu trả lời từ LLM",
                    "context": [],
                    "sources": []
                }
        
        # Nếu là câu hỏi cần thông tin từ tài liệu, tiến hành RAG
        logger.info(f"Thực hiện RAG cho câu hỏi liên quan đến tài liệu")
        
        # Tìm kiếm tài liệu liên quan
        search_results = self.search_documents(user_query, top_k=top_k)
        
        # Nếu không tìm thấy kết quả nào
        if not search_results:
            return {
                "answer": "Không tìm thấy thông tin liên quan đến câu hỏi của bạn trong tài liệu.",
                "context": [],
                "sources": []
            }
        
        # Định dạng ngữ cảnh
        context = self.format_context(search_results)
        
        # Truy vấn LM Studio với model gemma-3-12b-it
        llm_response = self.query_lm_studio(user_query, context)
        
        # Xử lý lỗi từ LM Studio
        if "error" in llm_response:
            return {
                "answer": f"Lỗi khi truy vấn LLM: {llm_response['error']}",
                "context": search_results,
                "sources": [result["metadata"].get("source") for result in search_results]
            }
        
        # Trích xuất câu trả lời từ chat completion response
        try:
            answer = llm_response.get("choices", [{}])[0].get("message", {}).get("content", "Không nhận được câu trả lời từ LLM")
        except Exception as e:
            logger.error(f"Lỗi khi trích xuất câu trả lời: {e}")
            answer = "Lỗi khi xử lý câu trả lời từ LLM"
        
        return {
            "answer": answer.strip(),
            "context": search_results,
            "sources": [result["metadata"].get("source") for result in search_results],
            "is_general_knowledge": False
        } 