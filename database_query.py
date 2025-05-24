import os
import logging
import mysql.connector
import pandas as pd
import requests
import re
from typing import Dict, List, Tuple, Any, Optional
from dotenv import load_dotenv

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class DatabaseQuery:
    """Class xử lý kết nối và truy vấn MySQL database"""
    
    def __init__(self, 
                 host: str = None,
                 user: str = None,
                 password: str = None,
                 port: int = None,
                 database: str = None,
                 lm_studio_url: str = None,
                 model_name: str = None):
        """
        Khởi tạo DatabaseQuery
        
        Args:
            host: Host của MySQL server
            user: Username MySQL
            password: Password MySQL
            port: Port của MySQL server
            database: Tên database MySQL
            lm_studio_url: URL của LM Studio API
            model_name: Tên model LLM
        """
        # Ưu tiên tham số truyền vào, nếu không có thì đọc từ env
        self.host = host or os.getenv("MYSQL_HOST", "localhost")
        self.user = user or os.getenv("MYSQL_USER", "root")
        self.password = password or os.getenv("MYSQL_PASSWORD", "")
        self.port = port or int(os.getenv("MYSQL_PORT", "3306"))
        self.database = database or os.getenv("MYSQL_DATABASE", "kt_ai")
        self.lm_studio_url = lm_studio_url or os.getenv("LM_STUDIO_URL", "http://127.0.0.1:1234")
        self.model_name = model_name or os.getenv("MODEL_NAME", "gemma-3-12b-it")
        
        # Thông tin kết nối MySQL
        self.config = {
            'host': self.host,
            'user': self.user,
            'password': self.password,
            'port': self.port,
            'database': self.database
        }
        
        # Cache cho schema
        self._schema_info = None
        
        logger.info(f"Khởi tạo DatabaseQuery với MySQL: {self.host}:{self.port}/{self.database}")
    
    def connect(self) -> Optional[mysql.connector.connection.MySQLConnection]:
        """
        Kết nối đến MySQL server
        
        Returns:
            MySQLConnection: Đối tượng kết nối MySQL hoặc None nếu có lỗi
        """
        try:
            connection = mysql.connector.connect(**self.config)
            if connection.is_connected():
                logger.info(f"Kết nối thành công đến MySQL Server: {self.host}:{self.port}/{self.database}")
                return connection
        except mysql.connector.Error as err:
            logger.error(f"Lỗi kết nối đến MySQL: {err}")
            return None
    
    def get_table_schema(self, connection=None) -> Dict[str, List]:
        """
        Lấy thông tin schema của tất cả các bảng trong database
        
        Args:
            connection: Kết nối MySQL hiện có (nếu None, sẽ tạo kết nối mới)
            
        Returns:
            Dict[str, List]: Thông tin schema của các bảng
        """
        # Trả về từ cache nếu đã có
        if self._schema_info:
            return self._schema_info
            
        close_connection = False
        if connection is None:
            connection = self.connect()
            close_connection = True
        
        if not connection:
            logger.error("Không thể lấy schema do không có kết nối")
            return {}
        
        try:
            cursor = connection.cursor()
            cursor.execute("SHOW TABLES")
            tables = cursor.fetchall()
            
            schema_info = {}
            for table in tables:
                table_name = table[0]
                cursor.execute(f"DESCRIBE {table_name}")
                columns = cursor.fetchall()
                schema_info[table_name] = columns
            
            cursor.close()
            
            # Lưu vào cache
            self._schema_info = schema_info
            
            return schema_info
        except mysql.connector.Error as err:
            logger.error(f"Lỗi khi lấy schema: {err}")
            return {}
        finally:
            if close_connection and connection.is_connected():
                connection.close()
    
    def execute_query(self, sql_query: str, connection=None) -> Tuple[bool, Any]:
        """
        Thực thi truy vấn SQL và trả về kết quả
        
        Args:
            sql_query: Câu truy vấn SQL
            connection: Kết nối MySQL hiện có (nếu None, sẽ tạo kết nối mới)
            
        Returns:
            Tuple[bool, Any]: (Thành công/thất bại, Kết quả/Thông báo lỗi)
        """
        close_connection = False
        if connection is None:
            connection = self.connect()
            close_connection = True
        
        if not connection:
            return False, "Không thể kết nối đến database"
        
        try:
            cursor = connection.cursor(dictionary=True)
            cursor.execute(sql_query)
            
            # Kiểm tra loại truy vấn
            if sql_query.strip().upper().startswith(('SELECT', 'SHOW', 'DESCRIBE')):
                results = cursor.fetchall()
                cursor.close()
                return True, results
            else:
                connection.commit()
                affected_rows = cursor.rowcount
                cursor.close()
                return True, f"Truy vấn thực thi thành công. Số dòng bị ảnh hưởng: {affected_rows}"
        except mysql.connector.Error as err:
            logger.error(f"Lỗi khi thực thi truy vấn: {err}")
            return False, f"Lỗi khi thực thi truy vấn: {err}"
        finally:
            if close_connection and connection.is_connected():
                connection.close()
    
    def generate_sql(self, question: str) -> str:
        """
        Tạo câu truy vấn SQL từ câu hỏi tự nhiên bằng LLM
        
        Args:
            question: Câu hỏi của người dùng
            
        Returns:
            str: Câu truy vấn SQL được tạo
        """
        # Lấy thông tin schema
        schema_info = self.get_table_schema()
        
        # Tạo thông tin về schema để cung cấp cho LLM
        schema_context = "Thông tin về cấu trúc cơ sở dữ liệu:\n"
        for table_name, columns in schema_info.items():
            schema_context += f"Bảng {table_name}: "
            column_info = []
            for column in columns:
                column_info.append(f"{column[0]} ({column[1]})")
            schema_context += ", ".join(column_info) + "\n"
        
        url = f"{self.lm_studio_url}/v1/chat/completions"
        
        system_message = f"""Bạn là một chuyên gia SQL giỏi. Hãy viết một truy vấn SQL hợp lệ dựa trên yêu cầu sau.

{schema_context}

HƯỚNG DẪN QUAN TRỌNG:
1. CHỈ trả về câu lệnh SQL thuần túy, KHÔNG có giải thích, KHÔNG có bình luận, KHÔNG có markdown.
2. KHÔNG bao gồm bất kỳ ký tự đặc biệt nào ngoài cú pháp SQL tiêu chuẩn.
3. KHÔNG sử dụng ký tự Unicode đặc biệt trong truy vấn.
4. Đảm bảo cú pháp SQL chuẩn và tương thích với MariaDB/MySQL.
5. Sử dụng tên bảng và cột chính xác như đã cung cấp.
6. Truy vấn phải kết thúc bằng dấu chấm phẩy (;).
7. Nếu không có đủ thông tin để viết một truy vấn chính xác, hãy trả về truy vấn đơn giản nhất có thể dựa trên thông tin sẵn có."""
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Yêu cầu: {question}"}
            ],
            "max_tokens": 512,
            "temperature": 0.2,  # Temperature thấp để đảm bảo kết quả nhất quán
            "stream": False
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        try:
            logger.info(f"Đang tạo truy vấn SQL cho câu hỏi: '{question}'")
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            
            sql_query = response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            
            # Làm sạch truy vấn SQL
            # Loại bỏ các dòng bắt đầu bằng -- (comment trong SQL)
            sql_query = re.sub(r'--.*?\n', '', sql_query)
            
            # Loại bỏ các khối comment /* ... */
            sql_query = re.sub(r'/\*.*?\*/', '', sql_query, flags=re.DOTALL)
            
            # Loại bỏ các dòng trống và khoảng trắng thừa
            sql_query = '\n'.join([line.strip() for line in sql_query.split('\n') if line.strip()])
            
            # Loại bỏ các phần không phải SQL như "```sql" và "```"
            sql_query = re.sub(r'```sql|```', '', sql_query)
            
            # Loại bỏ các từ khóa không phải SQL
            non_sql_patterns = [
                r'^SQL:', r'^Truy vấn SQL:', r'^Câu lệnh SQL:',
                r'Đây là truy vấn SQL', r'Kết quả:', r'Giải thích:'
            ]
            for pattern in non_sql_patterns:
                sql_query = re.sub(pattern, '', sql_query, flags=re.IGNORECASE)
            
            # Loại bỏ phần giải thích sau truy vấn
            if ';' in sql_query:
                sql_query = sql_query.split(';')[0] + ';'
            
            # Loại bỏ các ký tự Unicode đặc biệt
            sql_query = re.sub(r'[^\x00-\x7F]+', '', sql_query)
            
            # Đảm bảo truy vấn kết thúc bằng dấu chấm phẩy
            sql_query = sql_query.strip()
            if sql_query and not sql_query.endswith(";"):
                sql_query += ";"
            
            # Kiểm tra xem truy vấn có hợp lệ không
            if not self.is_valid_sql(sql_query):
                logger.error(f"Truy vấn SQL không hợp lệ: {sql_query}")
                return "SELECT 'Không thể tạo truy vấn SQL hợp lệ' AS error;"
            
            logger.info(f"Đã tạo truy vấn SQL: {sql_query}")
            return sql_query
        except Exception as e:
            logger.error(f"Lỗi khi tạo truy vấn SQL: {e}")
            return "SELECT 'Lỗi khi tạo truy vấn SQL' AS error;"
    
    def is_valid_sql(self, sql_query: str) -> bool:
        """
        Kiểm tra xem truy vấn SQL có hợp lệ không
        
        Args:
            sql_query: Câu truy vấn SQL cần kiểm tra
            
        Returns:
            bool: True nếu truy vấn hợp lệ, False nếu không
        """
        # Kiểm tra cơ bản
        if not sql_query or len(sql_query) < 5:
            return False
        
        # Kiểm tra xem có chứa các từ khóa SQL cơ bản không
        basic_keywords = ['SELECT', 'FROM', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP', 'SHOW']
        has_keyword = False
        for keyword in basic_keywords:
            if keyword in sql_query.upper():
                has_keyword = True
                break
        
        if not has_keyword:
            return False
        
        # Kiểm tra xem có chứa các ký tự không hợp lệ không
        invalid_chars = ['用户名', '？', '…']
        for char in invalid_chars:
            if char in sql_query:
                return False
        
        return True
    
    def format_db_results(self, results: List[Dict]) -> str:
        """
        Định dạng kết quả từ database để sử dụng làm ngữ cảnh cho LLM
        
        Args:
            results: Kết quả truy vấn từ database
            
        Returns:
            str: Kết quả đã định dạng
        """
        if not results:
            return "Không có kết quả nào từ database."
        
        # Nếu có quá nhiều kết quả, giới hạn lại
        max_results = 20  # Giới hạn số lượng kết quả để tránh token quá lớn
        limited_results = results[:max_results]
        
        # Tạo DataFrame từ kết quả
        df = pd.DataFrame(limited_results)
        
        # Định dạng thành văn bản table
        formatted_results = "Kết quả từ Database:\n"
        formatted_results += df.to_string(index=False)
        
        # Thêm thông báo nếu có cắt giảm kết quả
        if len(results) > max_results:
            formatted_results += f"\n\n(Hiển thị {max_results}/{len(results)} kết quả)"
        
        return formatted_results
    
    def evaluate_sql_query_type(self, question: str) -> bool:
        """
        Đánh giá xem câu hỏi có liên quan đến database hay không
        
        Args:
            question: Câu hỏi của người dùng
            
        Returns:
            bool: True nếu câu hỏi liên quan đến database
        """
        url = f"{self.lm_studio_url}/v1/chat/completions"
        
        system_message = """Bạn là một trợ lý thông minh giúp phân loại câu hỏi. 
Nhiệm vụ của bạn là xác định xem một câu hỏi có yêu cầu thông tin từ cơ sở dữ liệu hay không.

Phân loại câu hỏi thành một trong hai loại:
1. Câu hỏi liên quan đến dữ liệu hoặc số liệu cụ thể có thể truy vấn từ database
2. Câu hỏi không liên quan đến dữ liệu trong database

Trả lời chỉ với "DATABASE" cho loại 1 hoặc "NON_DATABASE" cho loại 2. Không giải thích lý do."""
        
        user_message = f"Đây có phải là câu hỏi cần thông tin từ database không? Câu hỏi: {question}"
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            "max_tokens": 50,
            "temperature": 0.1,
            "stream": False
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        try:
            logger.info(f"Đánh giá loại câu hỏi database: '{question}'")
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            answer = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip().upper()
            
            # Kiểm tra kết quả
            if "DATABASE" in answer:
                logger.info(f"Kết quả đánh giá: Câu hỏi liên quan đến database")
                return True
            else:
                logger.info(f"Kết quả đánh giá: Câu hỏi không liên quan đến database")
                return False
                
        except Exception as e:
            logger.error(f"Lỗi khi đánh giá loại câu hỏi database: {e}")
            # Mặc định không liên quan đến database nếu có lỗi
            return False
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Xử lý toàn bộ quá trình truy vấn database
        
        Args:
            question: Câu hỏi của người dùng
            
        Returns:
            Dict: Kết quả hoàn chỉnh
        """
        # Kiểm tra xem câu hỏi có liên quan đến database không
        is_db_related = self.evaluate_sql_query_type(question)
        
        if not is_db_related:
            return {
                "success": False,
                "message": "Câu hỏi không liên quan đến dữ liệu trong database.",
                "is_db_related": False,
                "sql_query": None,
                "results": None
            }
        
        # Tạo truy vấn SQL
        sql_query = self.generate_sql(question)
        
        # Kiểm tra truy vấn error
        if "error" in sql_query.lower():
            return {
                "success": False,
                "message": sql_query,
                "is_db_related": True,
                "sql_query": sql_query,
                "results": None
            }
        
        # Thực thi truy vấn
        success, results = self.execute_query(sql_query)
        
        # Trả về kết quả
        if success:
            formatted_results = None
            if isinstance(results, list):
                formatted_results = self.format_db_results(results)
            
            return {
                "success": True,
                "message": "Truy vấn thành công.",
                "is_db_related": True,
                "sql_query": sql_query,
                "results": results,
                "formatted_results": formatted_results
            }
        else:
            return {
                "success": False,
                "message": results,  # Thông báo lỗi
                "is_db_related": True,
                "sql_query": sql_query,
                "results": None
            } 