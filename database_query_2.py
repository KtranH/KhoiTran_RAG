import os
import logging
import pyodbc
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

class SQLServerQuery:
    """Class xử lý kết nối và truy vấn SQL Server database"""
    
    def __init__(self, 
                 server: str = None,
                 user: str = None,
                 password: str = None,
                 port: int = None,
                 database: str = None,
                 driver: str = None,
                 lm_studio_url: str = None,
                 model_name: str = None):
        """
        Khởi tạo SQLServerQuery
        
        Args:
            server: Tên hoặc địa chỉ SQL Server
            user: Username SQL Server
            password: Password SQL Server
            port: Port của SQL Server (mặc định là 1433)
            database: Tên database SQL Server
            driver: Driver ODBC để kết nối SQL Server
            lm_studio_url: URL của LM Studio API
            model_name: Tên model LLM
        """
        # Ưu tiên tham số truyền vào, nếu không có thì đọc từ env
        self.server = server or os.getenv("SQLSERVER_SERVER", "localhost")
        self.user = user or os.getenv("SQLSERVER_USER", "sa")
        self.password = password or os.getenv("SQLSERVER_PASSWORD", "123")
        self.port = port or int(os.getenv("SQLSERVER_PORT", "1433"))
        self.database = database or os.getenv("SQLSERVER_DATABASE", "WEB_APP_QLKS")
        self.driver = driver or os.getenv("SQLSERVER_DRIVER", "ODBC Driver 17 for SQL Server")
        self.lm_studio_url = lm_studio_url or os.getenv("LM_STUDIO_URL", "http://127.0.0.1:1234")
        self.model_name = model_name or os.getenv("MODEL_NAME", "gemma-3-12b-it")
        
        # Chuỗi kết nối SQL Server
        self.connection_string = f"DRIVER={{{self.driver}}};SERVER={self.server},{self.port};DATABASE={self.database};UID={self.user};PWD={self.password}"
        
        # Cache cho schema
        self._schema_info = None
        
        logger.info(f"Khởi tạo SQLServerQuery với SQL Server: {self.server}:{self.port}/{self.database}")
    
    def connect(self) -> Optional[pyodbc.Connection]:
        """
        Kết nối đến SQL Server
        
        Returns:
            pyodbc.Connection: Đối tượng kết nối SQL Server hoặc None nếu có lỗi
        """
        try:
            connection = pyodbc.connect(self.connection_string)
            logger.info(f"Kết nối thành công đến SQL Server: {self.server}:{self.port}/{self.database} với tài khoản {self.user}")
            return connection
        except pyodbc.Error as err:
            logger.error(f"Lỗi kết nối đến SQL Server: {err}")
            return None
    
    def get_table_schema(self, connection=None) -> Dict[str, List]:
        """
        Lấy thông tin schema của tất cả các bảng trong database
        
        Args:
            connection: Kết nối SQL Server hiện có (nếu None, sẽ tạo kết nối mới)
            
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
            
            # SQL Server sử dụng câu lệnh khác để lấy danh sách bảng
            cursor.execute("""
                SELECT TABLE_NAME 
                FROM INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_TYPE = 'BASE TABLE'
            """)
            tables = cursor.fetchall()
            
            schema_info = {}
            for table in tables:
                table_name = table[0]
                # SQL Server sử dụng INFORMATION_SCHEMA.COLUMNS thay vì DESCRIBE
                cursor.execute(f"""
                    SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH, IS_NULLABLE 
                    FROM INFORMATION_SCHEMA.COLUMNS 
                    WHERE TABLE_NAME = '{table_name}'
                """)
                columns = cursor.fetchall()
                schema_info[table_name] = columns
            
            cursor.close()
            
            # Lưu vào cache
            self._schema_info = schema_info
            
            return schema_info
        except pyodbc.Error as err:
            logger.error(f"Lỗi khi lấy schema: {err}")
            return {}
        finally:
            if close_connection and connection:
                connection.close()
    
    def execute_query(self, sql_query: str, connection=None) -> Tuple[bool, Any]:
        """
        Thực thi truy vấn SQL và trả về kết quả
        
        Args:
            sql_query: Câu truy vấn SQL
            connection: Kết nối SQL Server hiện có (nếu None, sẽ tạo kết nối mới)
            
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
            cursor = connection.cursor()
            cursor.execute(sql_query)
            
            # Kiểm tra loại truy vấn
            if sql_query.strip().upper().startswith(('SELECT', 'WITH')):
                # Chuyển đổi kết quả thành danh sách các dict
                columns = [column[0] for column in cursor.description]
                results = [dict(zip(columns, row)) for row in cursor.fetchall()]
                cursor.close()
                return True, results
            else:
                connection.commit()
                affected_rows = cursor.rowcount
                cursor.close()
                return True, f"Truy vấn thực thi thành công. Số dòng bị ảnh hưởng: {affected_rows}"
        except pyodbc.Error as err:
            logger.error(f"Lỗi khi thực thi truy vấn: {err}")
            return False, f"Lỗi khi thực thi truy vấn: {err}"
        finally:
            if close_connection and connection:
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
                data_type = column[1]
                max_length = column[2] if column[2] is not None else ""
                column_info.append(f"{column[0]} ({data_type}{max_length})")
            schema_context += ", ".join(column_info) + "\n"
        
        url = f"{self.lm_studio_url}/v1/chat/completions"
        
        system_message = f"""Bạn là một chuyên gia SQL giỏi. Hãy viết một truy vấn SQL hợp lệ dựa trên yêu cầu sau.

{schema_context}

HƯỚNG DẪN QUAN TRỌNG:
1. CHỈ trả về câu lệnh SQL thuần túy, KHÔNG có giải thích, KHÔNG có bình luận, KHÔNG có markdown.
2. KHÔNG bao gồm bất kỳ ký tự đặc biệt nào ngoài cú pháp SQL tiêu chuẩn.
3. KHÔNG sử dụng ký tự Unicode đặc biệt trong truy vấn.
4. Đảm bảo cú pháp SQL chuẩn và tương thích với SQL Server.
5. Sử dụng tên bảng và cột chính xác như đã cung cấp.
6. Truy vấn phải kết thúc bằng dấu chấm phẩy (;).
7. Nếu không có đủ thông tin để viết một truy vấn chính xác, hãy trả về truy vấn đơn giản nhất có thể dựa trên thông tin sẵn có.
8. SQL Server KHÔNG hỗ trợ cú pháp LIMIT, thay vào đó hãy sử dụng TOP.
9. Ví dụ: "SELECT TOP 10 * FROM table" thay vì "SELECT * FROM table LIMIT 10"."""
        
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
            
            # Chuyển đổi cú pháp MySQL sang SQL Server
            sql_query = self.convert_mysql_to_sqlserver_syntax(sql_query)
            
            sql_query = sql_query.strip()
            logger.info(f"Đã tạo truy vấn SQL: {sql_query}")
            
            return sql_query
        except Exception as err:
            logger.error(f"Lỗi khi gọi LLM API: {err}")
            return ""
    
    def convert_mysql_to_sqlserver_syntax(self, query: str) -> str:
        """
        Chuyển đổi cú pháp MySQL sang cú pháp SQL Server
        
        Args:
            query: Truy vấn SQL với cú pháp có thể là MySQL
            
        Returns:
            str: Truy vấn SQL với cú pháp SQL Server
        """
        # Chuẩn hóa khoảng trắng để dễ xử lý regex
        query = re.sub(r'\s+', ' ', query).strip()
        
        # Chuyển đổi LIMIT x thành TOP x
        limit_pattern = re.compile(r'SELECT (.*?) FROM (.*?) ORDER BY (.*?) LIMIT (\d+);', re.IGNORECASE)
        if limit_pattern.search(query):
            query = limit_pattern.sub(r'SELECT TOP \4 \1 FROM \2 ORDER BY \3;', query)
        
        # Chuyển đổi LIMIT x, y thành OFFSET-FETCH
        limit_offset_pattern = re.compile(r'SELECT (.*?) FROM (.*?) ORDER BY (.*?) LIMIT (\d+),\s*(\d+);', re.IGNORECASE)
        if limit_offset_pattern.search(query):
            query = limit_offset_pattern.sub(r'SELECT \1 FROM \2 ORDER BY \3 OFFSET \4 ROWS FETCH NEXT \5 ROWS ONLY;', query)
            
        # Trường hợp không có ORDER BY nhưng có LIMIT
        no_order_limit_pattern = re.compile(r'SELECT (.*?) FROM (.*?) LIMIT (\d+);', re.IGNORECASE)
        if no_order_limit_pattern.search(query):
            query = no_order_limit_pattern.sub(r'SELECT TOP \3 \1 FROM \2;', query)
        
        # Trường hợp không có ORDER BY nhưng có LIMIT với offset
        no_order_limit_offset_pattern = re.compile(r'SELECT (.*?) FROM (.*?) LIMIT (\d+),\s*(\d+);', re.IGNORECASE)
        if no_order_limit_offset_pattern.search(query):
            query = no_order_limit_offset_pattern.sub(r'SELECT \1 FROM \2 ORDER BY (SELECT NULL) OFFSET \3 ROWS FETCH NEXT \4 ROWS ONLY;', query)
        
        # Các chuyển đổi khác nếu cần
        # ...
        
        return query
    
    def is_valid_sql(self, sql_query: str) -> bool:
        """
        Kiểm tra tính hợp lệ của câu truy vấn SQL
        
        Args:
            sql_query: Câu truy vấn SQL cần kiểm tra
            
        Returns:
            bool: True nếu truy vấn hợp lệ, False nếu không
        """
        # SQL Server không có cách thức đơn giản để kiểm tra cú pháp mà không thực thi
        # Chúng ta có thể thực hiện một số kiểm tra cơ bản
        if not sql_query.strip():
            return False
        
        # Kiểm tra các từ khóa SQL cơ bản
        basic_keywords = ['SELECT', 'FROM', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP', 'WITH']
        if not any(sql_query.upper().startswith(keyword) for keyword in basic_keywords):
            return False
            
        return True

    def format_db_results(self, results: List[Dict]) -> str:
        """
        Định dạng kết quả từ database thành chuỗi dễ đọc
        
        Args:
            results: Kết quả truy vấn từ database
            
        Returns:
            str: Chuỗi kết quả đã định dạng
        """
        if not results:
            return "Không có kết quả."
        
        # Chuyển đổi danh sách các dict thành DataFrame
        df = pd.DataFrame(results)
        
        # Định dạng DataFrame thành chuỗi
        return df.to_string(index=False)
    
    def evaluate_sql_query_type(self, question: str) -> bool:
        """
        Đánh giá xem câu hỏi có nên được trả lời bằng truy vấn SQL hay không
        
        Args:
            question: Câu hỏi của người dùng
            
        Returns:
            bool: True nếu nên sử dụng SQL, False nếu không
        """
        url = f"{self.lm_studio_url}/v1/chat/completions"
        
        system_message = """Bạn là một chuyên gia phân tích dữ liệu. Nhiệm vụ của bạn là xác định xem câu hỏi của người dùng có phải là yêu cầu truy vấn cơ sở dữ liệu không.

HƯỚNG DẪN:
1. Hãy phân tích câu hỏi và xác định xem nó có yêu cầu thông tin từ cơ sở dữ liệu hay không.
2. Trả về ĐÚNG nếu câu hỏi yêu cầu dữ liệu có cấu trúc từ cơ sở dữ liệu.
3. Trả về SAI nếu câu hỏi là câu hỏi chung, yêu cầu giải thích, tư vấn, hoặc không liên quan đến truy vấn dữ liệu.

Chỉ trả lời ĐÚNG hoặc SAI, không có giải thích hay thông tin bổ sung."""
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": question}
            ],
            "max_tokens": 10,
            "temperature": 0.1,
            "stream": False
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            
            result = response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip().upper()
            
            return "ĐÚNG" in result or "TRUE" in result
        except Exception as err:
            logger.error(f"Lỗi khi đánh giá loại truy vấn: {err}")
            # Mặc định trả về True nếu có lỗi
            return True
            
    def query(self, question: str) -> Dict[str, Any]:
        """
        Xử lý câu hỏi từ người dùng và trả về kết quả từ SQL Server
        
        Args:
            question: Câu hỏi của người dùng
            
        Returns:
            Dict[str, Any]: Kết quả trả về
        """
        logger.info(f"Xử lý câu hỏi: '{question}'")
        
        # Kiểm tra xem câu hỏi có nên được trả lời bằng truy vấn SQL hay không
        is_sql_query = self.evaluate_sql_query_type(question)
        
        if not is_sql_query:
            return {
                "success": False,
                "message": "Câu hỏi không yêu cầu truy vấn dữ liệu từ cơ sở dữ liệu."
            }
        
        # Tạo câu truy vấn SQL từ câu hỏi
        sql_query = self.generate_sql(question)
        
        if not sql_query or not self.is_valid_sql(sql_query):
            return {
                "success": False,
                "message": "Không thể tạo câu truy vấn SQL hợp lệ từ câu hỏi của bạn."
            }
        
        # Thực thi truy vấn SQL
        success, results = self.execute_query(sql_query)
        
        if not success:
            return {
                "success": False,
                "message": results  # Thông báo lỗi
            }
        
        # Định dạng kết quả nếu cần
        if isinstance(results, list):
            formatted_results = self.format_db_results(results)
        else:
            formatted_results = results  # Đã là chuỗi thông báo
        
        return {
            "success": True,
            "sql_query": sql_query,
            "raw_results": results,
            "formatted_results": formatted_results
        } 