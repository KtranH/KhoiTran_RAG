import logging
import requests
from typing import Dict, List, Any, Optional, Tuple
from document_query import DocumentQuery
from database_query import DatabaseQuery

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HybridQuery:
    """Class xử lý kết hợp truy vấn từ database và RAG"""
    
    def __init__(self, 
                 lm_studio_url: str = None,
                 model_name: str = None,
                 persist_directory: str = "./chroma_db",
                 mysql_host: str = None,
                 mysql_user: str = None,
                 mysql_password: str = None,
                 mysql_port: int = None,
                 mysql_database: str = None):
        """
        Khởi tạo HybridQuery
        
        Args:
            lm_studio_url: URL của LM Studio API
            model_name: Tên model LLM
            persist_directory: Thư mục lưu trữ vector database
            mysql_host: Host của MySQL server
            mysql_user: Username MySQL
            mysql_password: Password MySQL
            mysql_port: Port của MySQL server
            mysql_database: Tên database MySQL
        """
        self.lm_studio_url = lm_studio_url
        self.model_name = model_name
        
        # Khởi tạo DocumentQuery
        self.doc_query = DocumentQuery(
            persist_directory=persist_directory,
            lm_studio_url=lm_studio_url,
            model_name=model_name
        )
        
        # Khởi tạo DatabaseQuery
        self.db_query = DatabaseQuery(
            host=mysql_host,
            user=mysql_user,
            password=mysql_password,
            port=mysql_port,
            database=mysql_database,
            lm_studio_url=lm_studio_url,
            model_name=model_name
        )
        
        logger.info(f"Khởi tạo HybridQuery với DocumentQuery và DatabaseQuery")
    
    def determine_query_type(self, question: str) -> Tuple[bool, bool]:
        """
        Xác định loại truy vấn dựa trên câu hỏi
        
        Args:
            question: Câu hỏi của người dùng
            
        Returns:
            Tuple[bool, bool]: (cần_database, cần_tài_liệu)
        """
        # Kiểm tra xem câu hỏi có liên quan đến database không
        is_db_related = self.db_query.evaluate_sql_query_type(question)
        
        # Kiểm tra xem câu hỏi có cần thông tin từ tài liệu không
        needs_document = self.doc_query.evaluate_query_type(question)
        
        logger.info(f"Kết quả phân loại câu hỏi: Database={is_db_related}, Document={needs_document}")
        
        return is_db_related, needs_document
    
    def query_database(self, question: str) -> Dict[str, Any]:
        """
        Truy vấn database
        
        Args:
            question: Câu hỏi của người dùng
            
        Returns:
            Dict: Kết quả từ database
        """
        logger.info(f"Truy vấn database với câu hỏi: '{question}'")
        db_result = self.db_query.query(question)
        return db_result
    
    def query_document(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Truy vấn tài liệu
        
        Args:
            question: Câu hỏi của người dùng
            top_k: Số lượng kết quả tìm kiếm
            
        Returns:
            Dict: Kết quả từ tài liệu
        """
        logger.info(f"Truy vấn tài liệu với câu hỏi: '{question}'")
        doc_result = self.doc_query.query(question, top_k=top_k)
        return doc_result
    
    def combine_results(self, 
                       question: str, 
                       db_result: Dict[str, Any] = None, 
                       doc_result: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Kết hợp kết quả từ database và tài liệu
        
        Args:
            question: Câu hỏi của người dùng
            db_result: Kết quả từ database
            doc_result: Kết quả từ tài liệu
            
        Returns:
            Dict: Kết quả kết hợp
        """
        logger.info(f"Kết hợp kết quả từ database và tài liệu")
        
        # Nếu không có kết quả nào
        if not db_result and not doc_result:
            return {
                "answer": "Không thể tìm thấy thông tin liên quan đến câu hỏi của bạn.",
                "sources": ["Không có nguồn dữ liệu"]
            }
        
        # Nếu chỉ có kết quả từ database
        if db_result and not doc_result:
            # Nếu truy vấn database thành công
            if db_result.get("success", False):
                return {
                    "answer": self._synthesize_db_answer(question, db_result),
                    "sources": ["Database"],
                    "sql_query": db_result.get("sql_query"),
                    "db_results": db_result.get("results")
                }
            else:
                # Nếu truy vấn database thất bại
                return {
                    "answer": f"Không thể truy vấn database: {db_result.get('message')}",
                    "sources": ["Database (lỗi)"],
                    "sql_query": db_result.get("sql_query")
                }
        
        # Nếu chỉ có kết quả từ tài liệu
        if doc_result and not db_result:
            return {
                "answer": doc_result.get("answer", "Không có câu trả lời từ tài liệu."),
                "sources": doc_result.get("sources", ["Tài liệu"]),
                "is_general_knowledge": doc_result.get("is_general_knowledge", False)
            }
        
        # Nếu có cả hai loại kết quả, tổng hợp chúng
        combined_context = self._create_combined_context(db_result, doc_result)
        
        # Tổng hợp câu trả lời từ cả hai nguồn
        synthesized_answer = self._synthesize_hybrid_answer(question, combined_context)
        
        # Gộp nguồn
        sources = []
        if db_result:
            sources.append("Database")
        if doc_result:
            if doc_result.get("is_general_knowledge", False):
                sources.append("Kiến thức chung")
            else:
                sources.extend(doc_result.get("sources", ["Tài liệu"]))
        
        return {
            "answer": synthesized_answer,
            "sources": sources,
            "sql_query": db_result.get("sql_query") if db_result else None,
            "db_results": db_result.get("results") if db_result else None,
            "doc_context": doc_result.get("context") if doc_result else None
        }
    
    def _create_combined_context(self, db_result: Dict[str, Any], doc_result: Dict[str, Any]) -> str:
        """
        Tạo ngữ cảnh kết hợp từ database và tài liệu
        
        Args:
            db_result: Kết quả từ database
            doc_result: Kết quả từ tài liệu
            
        Returns:
            str: Ngữ cảnh kết hợp
        """
        combined_context = ""
        
        # Thêm kết quả từ database
        if db_result and db_result.get("success", False) and db_result.get("formatted_results"):
            combined_context += db_result.get("formatted_results") + "\n\n"
        
        # Thêm kết quả từ tài liệu
        if doc_result and doc_result.get("context"):
            if combined_context:
                combined_context += "Thông tin từ tài liệu:\n"
            
            for i, context_item in enumerate(doc_result.get("context", [])):
                source = context_item.get("metadata", {}).get("source", "Không rõ nguồn").split("/")[-1]
                combined_context += f"[Tài liệu {i+1}: {source}]\n{context_item.get('content', '')}\n\n"
        
        return combined_context
    
    def _synthesize_db_answer(self, question: str, db_result: Dict[str, Any]) -> str:
        """
        Tổng hợp câu trả lời từ kết quả database
        
        Args:
            question: Câu hỏi của người dùng
            db_result: Kết quả từ database
            
        Returns:
            str: Câu trả lời tổng hợp
        """
        # Nếu không có kết quả hoặc kết quả không phải danh sách
        if not db_result.get("results") or not isinstance(db_result.get("results"), list):
            return f"Đã thực hiện truy vấn: {db_result.get('sql_query')}. {db_result.get('message', '')}"
        
        # Lấy kết quả đã định dạng
        formatted_results = db_result.get("formatted_results", "")
        
        # Truy vấn LLM để tổng hợp câu trả lời từ kết quả database
        url = f"{self.lm_studio_url}/v1/chat/completions"
        
        system_message = """Bạn là một trợ lý thông minh và hữu ích.
Nhiệm vụ của bạn là tóm tắt và phân tích kết quả truy vấn database để trả lời câu hỏi của người dùng.
Hãy dựa vào dữ liệu từ kết quả truy vấn SQL để cung cấp câu trả lời ngắn gọn và đầy đủ.
Cần trả lời chính xác dựa trên dữ liệu, không thêm thông tin không có trong kết quả truy vấn.
Đưa ra các con số cụ thể, xu hướng hoặc kết luận nếu dữ liệu cho phép."""
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Câu hỏi: {question}\n\nTruy vấn SQL: {db_result.get('sql_query')}\n\n{formatted_results}\n\nHãy trả lời câu hỏi dựa trên kết quả truy vấn này."}
            ],
            "max_tokens": 1000,
            "temperature": 0.3,
            "stream": False
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            answer = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            
            if not answer:
                # Nếu không nhận được câu trả lời từ LLM
                return f"Kết quả truy vấn SQL: {formatted_results}"
                
            return answer
        except Exception as e:
            logger.error(f"Lỗi khi tổng hợp câu trả lời từ database: {e}")
            return f"Kết quả truy vấn SQL: {formatted_results}"
    
    def _synthesize_hybrid_answer(self, question: str, combined_context: str) -> str:
        """
        Tổng hợp câu trả lời từ cả database và tài liệu
        
        Args:
            question: Câu hỏi của người dùng
            combined_context: Ngữ cảnh kết hợp từ database và tài liệu
            
        Returns:
            str: Câu trả lời tổng hợp
        """
        url = f"{self.lm_studio_url}/v1/chat/completions"
        
        system_message = """Bạn là một trợ lý thông minh và hữu ích.
Nhiệm vụ của bạn là tổng hợp thông tin từ nhiều nguồn (database và tài liệu) để trả lời câu hỏi của người dùng.
Hãy phân tích cả dữ liệu số liệu từ database và thông tin từ tài liệu để cung cấp câu trả lời toàn diện nhất.
Kết hợp thông tin từ các nguồn khác nhau một cách hợp lý và logic.
Ưu tiên dữ liệu cụ thể từ database nếu có, và bổ sung thêm thông tin từ tài liệu để giải thích hoặc mở rộng.
Nếu có sự mâu thuẫn giữa các nguồn, hãy nêu rõ điều này và giải thích sự khác biệt."""
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Câu hỏi: {question}\n\nThông tin từ các nguồn:\n{combined_context}\n\nHãy kết hợp tất cả thông tin trên để trả lời câu hỏi."}
            ],
            "max_tokens": 1500,
            "temperature": 0.3,
            "stream": False
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            answer = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            
            if not answer:
                # Nếu không nhận được câu trả lời từ LLM
                return "Không thể tổng hợp câu trả lời từ các nguồn khác nhau."
                
            return answer
        except Exception as e:
            logger.error(f"Lỗi khi tổng hợp câu trả lời hybrid: {e}")
            return "Lỗi khi tổng hợp câu trả lời từ các nguồn khác nhau."
    
    def query(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Xử lý toàn bộ quá trình truy vấn hybrid
        
        Args:
            question: Câu hỏi của người dùng
            top_k: Số lượng kết quả tìm kiếm cho tài liệu
            
        Returns:
            Dict: Kết quả hoàn chỉnh
        """
        # Xác định loại truy vấn
        is_db_related, needs_document = self.determine_query_type(question)
        
        db_result = None
        doc_result = None
        
        # Thực hiện truy vấn database nếu cần
        if is_db_related:
            db_result = self.query_database(question)
        
        # Thực hiện truy vấn tài liệu nếu cần
        if needs_document:
            doc_result = self.query_document(question, top_k=top_k)
        
        # Kết hợp kết quả
        result = self.combine_results(question, db_result, doc_result)
        
        # Thêm thông tin về loại truy vấn
        result["query_type"] = {
            "database": is_db_related,
            "document": needs_document
        }
        
        return result 