import os
import logging
import argparse
from document_processor import DocumentProcessor
from document_query import DocumentQuery

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

def interactive_mode(persist_directory: str = "./chroma_db",
                     lm_studio_url: str = "http://127.0.0.1:1234",
                     model_name: str = "gemma-3-12b-it"):
    """
    Chế độ tương tác với người dùng
    
    Args:
        persist_directory: Thư mục lưu trữ vector database
        lm_studio_url: URL của LM Studio API
        model_name: Tên model LLM (mặc định: gemma-3-12b-it)
    """
    print("\n" + "="*50)
    print("CHƯƠNG TRÌNH RAG VỚI LM STUDIO")
    print("="*50)
    print(f"Model đang sử dụng: {model_name}")
    print("Nhập 'exit' hoặc 'quit' để thoát")
    print("="*50 + "\n")
    
    while True:
        query = input("Câu hỏi của bạn: ")
        if query.lower() in ['exit', 'quit']:
            print("Tạm biệt!")
            break
        
        query_document(query, persist_directory, lm_studio_url, model_name)

def main():
    """Hàm chính của chương trình"""
    parser = argparse.ArgumentParser(description="Chương trình RAG với LM Studio")
    
    subparsers = parser.add_subparsers(dest='command', help='Lệnh')
    
    # Lệnh create: Tạo vector database
    create_parser = subparsers.add_parser('create', help='Tạo vector database')
    create_parser.add_argument('--docs_dir', type=str, default='./docs', help='Thư mục chứa tài liệu')
    create_parser.add_argument('--chunk_size', type=int, default=500, help='Kích thước của mỗi đoạn văn bản')
    create_parser.add_argument('--chunk_overlap', type=int, default=50, help='Độ chồng lấp giữa các đoạn')
    create_parser.add_argument('--persist_directory', type=str, default='./chroma_db', help='Thư mục lưu trữ vector database')
    
    # Lệnh query: Truy vấn tài liệu
    query_parser = subparsers.add_parser('query', help='Truy vấn tài liệu')
    query_parser.add_argument('--query', type=str, required=True, help='Câu hỏi của người dùng')
    query_parser.add_argument('--persist_directory', type=str, default='./chroma_db', help='Thư mục lưu trữ vector database')
    query_parser.add_argument('--lm_studio_url', type=str, default='http://127.0.0.1:1234', help='URL của LM Studio API')
    query_parser.add_argument('--model_name', type=str, default='gemma-3-12b-it', help='Tên model LLM (mặc định: gemma-3-12b-it)')
    query_parser.add_argument('--top_k', type=int, default=3, help='Số lượng kết quả tìm kiếm')
    
    # Lệnh interactive: Chế độ tương tác
    interactive_parser = subparsers.add_parser('interactive', help='Chế độ tương tác')
    interactive_parser.add_argument('--persist_directory', type=str, default='./chroma_db', help='Thư mục lưu trữ vector database')
    interactive_parser.add_argument('--lm_studio_url', type=str, default='http://127.0.0.1:1234', help='URL của LM Studio API')
    interactive_parser.add_argument('--model_name', type=str, default='gemma-3-12b-it', help='Tên model LLM (mặc định: gemma-3-12b-it)')
    
    args = parser.parse_args()
    
    # Thực hiện lệnh tương ứng
    if args.command == 'create':
        create_vector_database(
            docs_dir=args.docs_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            persist_directory=args.persist_directory
        )
    elif args.command == 'query':
        query_document(
            query=args.query,
            persist_directory=args.persist_directory,
            lm_studio_url=args.lm_studio_url,
            model_name=args.model_name,
            top_k=args.top_k
        )
    elif args.command == 'interactive':
        interactive_mode(
            persist_directory=args.persist_directory,
            lm_studio_url=args.lm_studio_url,
            model_name=args.model_name
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    main()