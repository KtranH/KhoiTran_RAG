import os
import logging
import argparse
from dotenv import load_dotenv
from document_processor import DocumentProcessor
from document_query import DocumentQuery
from database_query import DatabaseQuery
from hybrid_query import HybridQuery

# Load environment variables
load_dotenv()

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_vector_database(docs_dir: str = "./docs", 
                           chunk_size: int = 500,
                           chunk_overlap: int = 50,
                           persist_directory: str = "./chroma_db"):
    """
    Tạo vector database từ thư mục tài liệu
    
    Args:
        docs_dir: Thư mục chứa tài liệu
        chunk_size: Kích thước của mỗi đoạn văn bản
        chunk_overlap: Độ chồng lấp giữa các đoạn
        persist_directory: Thư mục lưu trữ vector database
    """
    logger.info("Bắt đầu tạo vector database...")
    processor = DocumentProcessor(
        docs_dir=docs_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        persist_directory=persist_directory
    )
    
    processor.process_all()
    logger.info(f"Đã tạo vector database tại {persist_directory}")

def query_document(query: str,
                   persist_directory: str = "./chroma_db",
                   lm_studio_url: str = "http://127.0.0.1:1234",
                   model_name: str = "gemma-3-12b-it",
                   top_k: int = 3):
    """
    Truy vấn tài liệu với RAG
    
    Args:
        query: Câu hỏi của người dùng
        persist_directory: Thư mục lưu trữ vector database
        lm_studio_url: URL của LM Studio API
        model_name: Tên model LLM (mặc định: gemma-3-12b-it)
        top_k: Số lượng kết quả tìm kiếm
    """
    logger.info(f"Truy vấn: '{query}' sử dụng model {model_name}")
    
    # Kiểm tra xem vector database đã tồn tại chưa
    if not os.path.exists(persist_directory):
        logger.warning(f"Vector database không tồn tại tại {persist_directory}. Đang tạo mới...")
        create_vector_database(persist_directory=persist_directory)
    
    # Tạo đối tượng DocumentQuery
    doc_query = DocumentQuery(
        persist_directory=persist_directory,
        lm_studio_url=lm_studio_url,
        model_name=model_name
    )
    
    # Truy vấn tài liệu
    result = doc_query.query(query, top_k=top_k)
    
    # In kết quả
    print("\n" + "="*50)
    print(f"Câu hỏi: {query}")
    print("="*50)
    print(f"Trả lời: {result['answer']}")
    print("="*50)
    
    # Hiển thị loại câu trả lời
    is_general = result.get("is_general_knowledge", False)
    if is_general:
        print("Loại: Kiến thức chung")
    else:
        print("Loại: Dựa trên tài liệu")
        print("Nguồn tài liệu:")
        for i, source in enumerate(result['sources'], 1):
            print(f"  {i}. {source}")
    
    print("="*50 + "\n")
    
    return result

def query_database(query: str,
                  mysql_host: str = None,
                  mysql_user: str = None,
                  mysql_password: str = None,
                  mysql_port: int = None,
                  mysql_database: str = None,
                  lm_studio_url: str = "http://127.0.0.1:1234",
                  model_name: str = "gemma-3-12b-it"):
    """
    Truy vấn MySQL database
    
    Args:
        query: Câu hỏi của người dùng
        mysql_host: Host của MySQL server
        mysql_user: Username MySQL
        mysql_password: Password MySQL
        mysql_port: Port của MySQL server
        mysql_database: Tên database MySQL
        lm_studio_url: URL của LM Studio API
        model_name: Tên model LLM (mặc định: gemma-3-12b-it)
    """
    logger.info(f"Truy vấn database: '{query}' sử dụng model {model_name}")
    
    # Đọc thông tin từ env nếu không được cung cấp
    mysql_host = mysql_host or os.getenv("MYSQL_HOST", "localhost")
    mysql_user = mysql_user or os.getenv("MYSQL_USER", "root")
    mysql_password = mysql_password or os.getenv("MYSQL_PASSWORD", "")
    mysql_port = mysql_port or int(os.getenv("MYSQL_PORT", "3306"))
    mysql_database = mysql_database or os.getenv("MYSQL_DATABASE", "kt_ai")
    
    # Tạo đối tượng DatabaseQuery
    db_query = DatabaseQuery(
        host=mysql_host,
        user=mysql_user,
        password=mysql_password,
        port=mysql_port,
        database=mysql_database,
        lm_studio_url=lm_studio_url,
        model_name=model_name
    )
    
    # Truy vấn database
    result = db_query.query(query)
    
    # In kết quả
    print("\n" + "="*50)
    print(f"Câu hỏi: {query}")
    print("="*50)
    
    if result["success"]:
        print(f"Truy vấn SQL: {result['sql_query']}")
        print("-"*50)
        
        if isinstance(result["results"], list):
            print(f"Số lượng kết quả: {len(result['results'])}")
            if result.get("formatted_results"):
                print("\n" + result["formatted_results"])
        else:
            print(f"Kết quả: {result['results']}")
    else:
        print(f"Lỗi: {result['message']}")
    
    print("="*50 + "\n")
    
    return result

def query_hybrid(query: str,
                persist_directory: str = "./chroma_db",
                mysql_host: str = None,
                mysql_user: str = None,
                mysql_password: str = None,
                mysql_port: int = None,
                mysql_database: str = None,
                lm_studio_url: str = "http://127.0.0.1:1234",
                model_name: str = "gemma-3-12b-it",
                top_k: int = 3):
    """
    Truy vấn hybrid (kết hợp database và tài liệu)
    
    Args:
        query: Câu hỏi của người dùng
        persist_directory: Thư mục lưu trữ vector database
        mysql_host: Host của MySQL server
        mysql_user: Username MySQL
        mysql_password: Password MySQL
        mysql_port: Port của MySQL server
        mysql_database: Tên database MySQL
        lm_studio_url: URL của LM Studio API
        model_name: Tên model LLM (mặc định: gemma-3-12b-it)
        top_k: Số lượng kết quả tìm kiếm cho tài liệu
    """
    logger.info(f"Truy vấn hybrid: '{query}' sử dụng model {model_name}")
    
    # Kiểm tra xem vector database đã tồn tại chưa
    if not os.path.exists(persist_directory):
        logger.warning(f"Vector database không tồn tại tại {persist_directory}. Đang tạo mới...")
        create_vector_database(persist_directory=persist_directory)
    
    # Đọc thông tin từ env nếu không được cung cấp
    mysql_host = mysql_host or os.getenv("MYSQL_HOST", "localhost")
    mysql_user = mysql_user or os.getenv("MYSQL_USER", "root")
    mysql_password = mysql_password or os.getenv("MYSQL_PASSWORD", "")
    mysql_port = mysql_port or int(os.getenv("MYSQL_PORT", "3306"))
    mysql_database = mysql_database or os.getenv("MYSQL_DATABASE", "kt_ai")
    
    # Tạo đối tượng HybridQuery
    hybrid_query = HybridQuery(
        lm_studio_url=lm_studio_url,
        model_name=model_name,
        persist_directory=persist_directory,
        mysql_host=mysql_host,
        mysql_user=mysql_user,
        mysql_password=mysql_password,
        mysql_port=mysql_port,
        mysql_database=mysql_database
    )
    
    # Truy vấn hybrid
    result = hybrid_query.query(query, top_k=top_k)
    
    # In kết quả
    print("\n" + "="*50)
    print(f"Câu hỏi: {query}")
    print("="*50)
    print(f"Trả lời: {result['answer']}")
    print("="*50)
    
    # Hiển thị nguồn
    print("Nguồn thông tin:")
    for i, source in enumerate(result['sources'], 1):
        print(f"  {i}. {source}")
    
    # Hiển thị thông tin về loại truy vấn
    print("-"*50)
    print("Loại truy vấn:")
    
    # Kiểm tra nếu câu trả lời đến từ kiến thức có sẵn của model
    if result["query_type"].get("model_knowledge", False):
        print("  - Kiến thức của model: Có")
        print("  - Database: Không")
        print("  - Tài liệu: Không")
    else:
        # Hiển thị theo cách cũ nếu không phải kiến thức của model
        if result["query_type"]["database"]:
            print("  - Database: Có")
            if result.get("sql_query"):
                print(f"    SQL: {result['sql_query']}")
        else:
            print("  - Database: Không")
        
        if result["query_type"]["document"]:
            print("  - Tài liệu: Có")
        else:
            print("  - Tài liệu: Không")
    
    print("="*50 + "\n")
    
    return result

def interactive_mode(mode: str = "hybrid",
                     persist_directory: str = "./chroma_db",
                     mysql_host: str = None,
                     mysql_user: str = None,
                     mysql_password: str = None,
                     mysql_port: int = None,
                     mysql_database: str = None,
                     lm_studio_url: str = "http://127.0.0.1:1234",
                     model_name: str = "gemma-3-12b-it",
                     top_k: int = 3):
    """
    Chế độ tương tác với người dùng
    
    Args:
        mode: Chế độ truy vấn ('hybrid', 'document', 'database')
        persist_directory: Thư mục lưu trữ vector database
        mysql_host: Host của MySQL server
        mysql_user: Username MySQL
        mysql_password: Password MySQL
        mysql_port: Port của MySQL server
        mysql_database: Tên database MySQL
        lm_studio_url: URL của LM Studio API
        model_name: Tên model LLM (mặc định: gemma-3-12b-it)
        top_k: Số lượng kết quả tìm kiếm cho tài liệu
    """
    # Tiêu đề dựa trên mode
    mode_titles = {
        "hybrid": "HYBRID (DATABASE + RAG)",
        "document": "RAG (DOCUMENT ONLY)",
        "database": "DATABASE ONLY",
        "auto": "AUTO (KNOWLEDGE FIRST)"
    }
    
    print("\n" + "="*50)
    print(f"CHƯƠNG TRÌNH {mode_titles.get(mode, 'TRUY VẤN')}")
    print("="*50)
    print(f"Model đang sử dụng: {model_name}")
    print("Nhập 'exit' hoặc 'quit' để thoát")
    print("Nhập 'mode' để chuyển đổi chế độ truy vấn")
    print("="*50 + "\n")
    
    current_mode = mode
    
    # Tạo đối tượng HybridQuery một lần để sử dụng trong vòng lặp
    hybrid_query = HybridQuery(
        lm_studio_url=lm_studio_url,
        model_name=model_name,
        persist_directory=persist_directory,
        mysql_host=mysql_host,
        mysql_user=mysql_user,
        mysql_password=mysql_password,
        mysql_port=mysql_port,
        mysql_database=mysql_database
    )
    
    while True:
        query = input("Câu hỏi của bạn: ")
        if query.lower() in ['exit', 'quit']:
            print("Tạm biệt!")
            break
        
        # Xử lý lệnh chuyển đổi chế độ
        if query.lower() == 'mode':
            modes = ['auto', 'hybrid', 'document', 'database']
            current_index = modes.index(current_mode) if current_mode in modes else 0
            current_mode = modes[(current_index + 1) % len(modes)]
            print(f"\nĐã chuyển sang chế độ: {mode_titles.get(current_mode)}")
            continue
        
        # Nếu ở chế độ auto, kiểm tra xem model có thể trả lời trực tiếp không
        if current_mode == 'auto':
            model_can_answer, model_answer = hybrid_query.evaluate_model_knowledge(query)
            
            if model_can_answer:
                print("\n" + "="*50)
                print(f"Câu hỏi: {query}")
                print("="*50)
                print(f"Trả lời: {model_answer}")
                print("="*50)
                print("Nguồn thông tin: Kiến thức của model")
                print("="*50 + "\n")
                continue
            else:
                # Nếu model không biết, chuyển sang chế độ hybrid
                print("Model không có kiến thức để trả lời, đang chuyển sang tìm kiếm thông tin...")
                # Sử dụng hybrid để xử lý
                query_hybrid(
                    query=query,
                    persist_directory=persist_directory,
                    mysql_host=mysql_host,
                    mysql_user=mysql_user,
                    mysql_password=mysql_password,
                    mysql_port=mysql_port,
                    mysql_database=mysql_database,
                    lm_studio_url=lm_studio_url,
                    model_name=model_name,
                    top_k=top_k
                )
                continue
        
        # Thực hiện truy vấn theo chế độ hiện tại
        if current_mode == 'document':
            query_document(
                query=query,
                persist_directory=persist_directory,
                lm_studio_url=lm_studio_url,
                model_name=model_name,
                top_k=top_k
            )
        elif current_mode == 'database':
            query_database(
                query=query,
                mysql_host=mysql_host,
                mysql_user=mysql_user,
                mysql_password=mysql_password,
                mysql_port=mysql_port,
                mysql_database=mysql_database,
                lm_studio_url=lm_studio_url,
                model_name=model_name
            )
        else:  # hybrid
            query_hybrid(
                query=query,
                persist_directory=persist_directory,
                mysql_host=mysql_host,
                mysql_user=mysql_user,
                mysql_password=mysql_password,
                mysql_port=mysql_port,
                mysql_database=mysql_database,
                lm_studio_url=lm_studio_url,
                model_name=model_name,
                top_k=top_k
            )

def main():
    """Hàm chính của chương trình"""
    parser = argparse.ArgumentParser(description="Chương trình Hybrid Query (Database + RAG)")
    
    subparsers = parser.add_subparsers(dest='command', help='Lệnh')
    
    # Lệnh create: Tạo vector database
    create_parser = subparsers.add_parser('create', help='Tạo vector database')
    create_parser.add_argument('--docs_dir', type=str, default='./docs', help='Thư mục chứa tài liệu')
    create_parser.add_argument('--chunk_size', type=int, default=500, help='Kích thước của mỗi đoạn văn bản')
    create_parser.add_argument('--chunk_overlap', type=int, default=50, help='Độ chồng lấp giữa các đoạn')
    create_parser.add_argument('--persist_directory', type=str, default='./chroma_db', help='Thư mục lưu trữ vector database')
    
    # Lệnh document: Truy vấn tài liệu
    doc_parser = subparsers.add_parser('document', help='Truy vấn tài liệu (RAG)')
    doc_parser.add_argument('--query', type=str, required=True, help='Câu hỏi của người dùng')
    doc_parser.add_argument('--persist_directory', type=str, default='./chroma_db', help='Thư mục lưu trữ vector database')
    doc_parser.add_argument('--lm_studio_url', type=str, default=None, help='URL của LM Studio API')
    doc_parser.add_argument('--model_name', type=str, default=None, help='Tên model LLM')
    doc_parser.add_argument('--top_k', type=int, default=3, help='Số lượng kết quả tìm kiếm')
    
    # Lệnh database: Truy vấn database
    db_parser = subparsers.add_parser('database', help='Truy vấn database')
    db_parser.add_argument('--query', type=str, required=True, help='Câu hỏi của người dùng')
    db_parser.add_argument('--mysql_host', type=str, default=None, help='Host của MySQL server')
    db_parser.add_argument('--mysql_user', type=str, default=None, help='Username MySQL')
    db_parser.add_argument('--mysql_password', type=str, default=None, help='Password MySQL')
    db_parser.add_argument('--mysql_port', type=int, default=None, help='Port của MySQL server')
    db_parser.add_argument('--mysql_database', type=str, default=None, help='Tên database MySQL')
    db_parser.add_argument('--lm_studio_url', type=str, default=None, help='URL của LM Studio API')
    db_parser.add_argument('--model_name', type=str, default=None, help='Tên model LLM')
    
    # Lệnh hybrid: Truy vấn hybrid (database + RAG)
    hybrid_parser = subparsers.add_parser('hybrid', help='Truy vấn hybrid (database + RAG)')
    hybrid_parser.add_argument('--query', type=str, required=True, help='Câu hỏi của người dùng')
    hybrid_parser.add_argument('--persist_directory', type=str, default='./chroma_db', help='Thư mục lưu trữ vector database')
    hybrid_parser.add_argument('--mysql_host', type=str, default=None, help='Host của MySQL server')
    hybrid_parser.add_argument('--mysql_user', type=str, default=None, help='Username MySQL')
    hybrid_parser.add_argument('--mysql_password', type=str, default=None, help='Password MySQL')
    hybrid_parser.add_argument('--mysql_port', type=int, default=None, help='Port của MySQL server')
    hybrid_parser.add_argument('--mysql_database', type=str, default=None, help='Tên database MySQL')
    hybrid_parser.add_argument('--lm_studio_url', type=str, default=None, help='URL của LM Studio API')
    hybrid_parser.add_argument('--model_name', type=str, default=None, help='Tên model LLM')
    hybrid_parser.add_argument('--top_k', type=int, default=3, help='Số lượng kết quả tìm kiếm cho tài liệu')
    
    # Lệnh auto: Truy vấn tự động (sử dụng kiến thức model trước, sau đó hybrid nếu cần)
    auto_parser = subparsers.add_parser('auto', help='Truy vấn tự động (model knowledge first)')
    auto_parser.add_argument('--query', type=str, required=True, help='Câu hỏi của người dùng')
    auto_parser.add_argument('--persist_directory', type=str, default='./chroma_db', help='Thư mục lưu trữ vector database')
    auto_parser.add_argument('--mysql_host', type=str, default=None, help='Host của MySQL server')
    auto_parser.add_argument('--mysql_user', type=str, default=None, help='Username MySQL')
    auto_parser.add_argument('--mysql_password', type=str, default=None, help='Password MySQL')
    auto_parser.add_argument('--mysql_port', type=int, default=None, help='Port của MySQL server')
    auto_parser.add_argument('--mysql_database', type=str, default=None, help='Tên database MySQL')
    auto_parser.add_argument('--lm_studio_url', type=str, default=None, help='URL của LM Studio API')
    auto_parser.add_argument('--model_name', type=str, default=None, help='Tên model LLM')
    auto_parser.add_argument('--top_k', type=int, default=3, help='Số lượng kết quả tìm kiếm cho tài liệu')
    
    # Lệnh interactive: Chế độ tương tác
    interactive_parser = subparsers.add_parser('interactive', help='Chế độ tương tác')
    interactive_parser.add_argument('--mode', type=str, choices=['auto', 'hybrid', 'document', 'database'], default='auto', help='Chế độ truy vấn ban đầu')
    interactive_parser.add_argument('--persist_directory', type=str, default='./chroma_db', help='Thư mục lưu trữ vector database')
    interactive_parser.add_argument('--mysql_host', type=str, default=None, help='Host của MySQL server')
    interactive_parser.add_argument('--mysql_user', type=str, default=None, help='Username MySQL')
    interactive_parser.add_argument('--mysql_password', type=str, default=None, help='Password MySQL')
    interactive_parser.add_argument('--mysql_port', type=int, default=None, help='Port của MySQL server')
    interactive_parser.add_argument('--mysql_database', type=str, default=None, help='Tên database MySQL')
    interactive_parser.add_argument('--lm_studio_url', type=str, default=None, help='URL của LM Studio API')
    interactive_parser.add_argument('--model_name', type=str, default=None, help='Tên model LLM')
    interactive_parser.add_argument('--top_k', type=int, default=3, help='Số lượng kết quả tìm kiếm cho tài liệu')
    
    args = parser.parse_args()
    
    # Đọc các biến môi trường nếu cần
    lm_studio_url = args.lm_studio_url if hasattr(args, 'lm_studio_url') and args.lm_studio_url else os.getenv("LM_STUDIO_URL", "http://127.0.0.1:1234")
    model_name = args.model_name if hasattr(args, 'model_name') and args.model_name else os.getenv("MODEL_NAME", "gemma-3-12b-it")
    
    # Thực hiện lệnh tương ứng
    if args.command == 'create':
        create_vector_database(
            docs_dir=args.docs_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            persist_directory=args.persist_directory
        )
    elif args.command == 'document':
        query_document(
            query=args.query,
            persist_directory=args.persist_directory,
            lm_studio_url=lm_studio_url,
            model_name=model_name,
            top_k=args.top_k
        )
    elif args.command == 'database':
        query_database(
            query=args.query,
            mysql_host=args.mysql_host,
            mysql_user=args.mysql_user,
            mysql_password=args.mysql_password,
            mysql_port=args.mysql_port,
            mysql_database=args.mysql_database,
            lm_studio_url=lm_studio_url,
            model_name=model_name
        )
    elif args.command == 'hybrid':
        query_hybrid(
            query=args.query,
            persist_directory=args.persist_directory,
            mysql_host=args.mysql_host,
            mysql_user=args.mysql_user,
            mysql_password=args.mysql_password,
            mysql_port=args.mysql_port,
            mysql_database=args.mysql_database,
            lm_studio_url=lm_studio_url,
            model_name=model_name,
            top_k=args.top_k
        )
    elif args.command == 'auto':
        # Tạo đối tượng HybridQuery
        hybrid_query = HybridQuery(
            lm_studio_url=lm_studio_url,
            model_name=model_name,
            persist_directory=args.persist_directory,
            mysql_host=args.mysql_host,
            mysql_user=args.mysql_user,
            mysql_password=args.mysql_password,
            mysql_port=args.mysql_port,
            mysql_database=args.mysql_database
        )
        
        # Kiểm tra xem model có thể trả lời trực tiếp không
        model_can_answer, model_answer = hybrid_query.evaluate_model_knowledge(args.query)
        
        if model_can_answer:
            # In kết quả từ kiến thức của model
            print("\n" + "="*50)
            print(f"Câu hỏi: {args.query}")
            print("="*50)
            print(f"Trả lời: {model_answer}")
            print("="*50)
            print("Nguồn thông tin: Kiến thức của model")
            print("="*50 + "\n")
        else:
            # Nếu model không biết, sử dụng hybrid
            query_hybrid(
                query=args.query,
                persist_directory=args.persist_directory,
                mysql_host=args.mysql_host,
                mysql_user=args.mysql_user,
                mysql_password=args.mysql_password,
                mysql_port=args.mysql_port,
                mysql_database=args.mysql_database,
                lm_studio_url=lm_studio_url,
                model_name=model_name,
                top_k=args.top_k
            )
    elif args.command == 'interactive':
        interactive_mode(
            mode=args.mode,
            persist_directory=args.persist_directory,
            mysql_host=args.mysql_host,
            mysql_user=args.mysql_user,
            mysql_password=args.mysql_password,
            mysql_port=args.mysql_port,
            mysql_database=args.mysql_database,
            lm_studio_url=lm_studio_url,
            model_name=model_name,
            top_k=args.top_k
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    main()