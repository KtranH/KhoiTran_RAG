import os
import gradio as gr
import logging
import time
import requests
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

# Đọc các biến môi trường
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://127.0.0.1:1234")
MODEL_NAME = os.getenv("MODEL_NAME", "gemma-3-12b-it")
PERSIST_DIRECTORY = os.getenv("PERSIST_DIRECTORY", "./chroma_db")
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "kt_ai")
TOP_K = int(os.getenv("TOP_K", "3"))

# Màu sắc và CSS tùy chỉnh
DARK_CUSTOM_CSS = """
.container {
    max-width: 1200px;
    margin: auto;
}
.status-connected {
    color: #22c55e;
    font-weight: bold;
}
.status-error {
    color: #ef4444;
    font-weight: bold;
}
.query-info {
    background-color: #1e293b;
    border-left: 4px solid #4f46e5;
    padding: 10px;
    margin: 10px 0;
    border-radius: 4px;
}
.source-info {
    margin-top: 8px;
    padding: 8px;
    background-color: #0f172a;
    border-radius: 4px;
    font-size: 0.9em;
}
.chatbox .message.user {
    background-color: #334155 !important;
}
.chatbox .message.bot {
    background-color: #1e293b !important;
}
code {
    background-color: #0f172a;
    padding: 2px 4px;
    border-radius: 4px;
}
pre {
    background-color: #0f172a;
    padding: 10px;
    border-radius: 4px;
    overflow-x: auto;
}
"""

# Khởi tạo các đối tượng query
def get_document_query():
    return DocumentQuery(
        persist_directory=PERSIST_DIRECTORY,
        lm_studio_url=LM_STUDIO_URL,
        model_name=MODEL_NAME
    )

def get_database_query():
    return DatabaseQuery(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        port=MYSQL_PORT,
        database=MYSQL_DATABASE,
        lm_studio_url=LM_STUDIO_URL,
        model_name=MODEL_NAME
    )

def get_hybrid_query():
    return HybridQuery(
        lm_studio_url=LM_STUDIO_URL,
        model_name=MODEL_NAME,
        persist_directory=PERSIST_DIRECTORY,
        mysql_host=MYSQL_HOST,
        mysql_user=MYSQL_USER,
        mysql_password=MYSQL_PASSWORD,
        mysql_port=MYSQL_PORT,
        mysql_database=MYSQL_DATABASE
    )

# Kiểm tra kết nối đến LM Studio API
def check_connection():
    try:
        response = requests.get(f"{LM_STUDIO_URL}/v1/models", timeout=5)
        if response.status_code == 200:
            return True, "Đã kết nối thành công đến LM Studio API"
        else:
            return False, f"Không thể kết nối đến LM Studio API. Mã lỗi: {response.status_code}"
    except Exception as e:
        return False, f"Lỗi kết nối đến LM Studio API: {str(e)}"

# Kiểm tra kết nối đến MySQL
def check_mysql_connection():
    try:
        import mysql.connector
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            port=MYSQL_PORT,
            database=MYSQL_DATABASE,
            connection_timeout=5
        )
        conn.close()
        return True, "Đã kết nối thành công đến MySQL"
    except Exception as e:
        return False, f"Lỗi kết nối đến MySQL: {str(e)}"

# Đánh giá kiến thức của model
def evaluate_model_knowledge(question):
    """
    Đánh giá xem model có đủ kiến thức để trả lời câu hỏi không
    
    Args:
        question: Câu hỏi của người dùng
            
    Returns:
        Tuple[bool, Optional[str]]: (có_thể_trả_lời, câu_trả_lời)
    """
    logger.info(f"Đánh giá kiến thức model cho câu hỏi: '{question}'")
    
    url = f"{LM_STUDIO_URL}/v1/chat/completions"
    
    system_message = """Bạn là trợ lý AI thông minh. 
Trước khi tôi truy vấn database hoặc tìm kiếm trong tài liệu, hãy đánh giá xem bạn có kiến thức để trả lời câu hỏi này không.
Nếu bạn biết câu trả lời, hãy trả lời ngắn gọn và chính xác.
Nếu bạn không chắc chắn hoặc không biết, hãy chỉ trả lời: "TÔI CẦN TRA CỨU THÊM"
Đừng đoán mò. Nếu không chắc chắn 100%, hãy nói bạn cần tra cứu thêm."""
    
    try:
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": question}
            ],
            "max_tokens": 1000,
            "temperature": 0.3,
            "stream": False
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        result = response.json()
        answer = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        
        # Kiểm tra xem model có cần tra cứu thêm không
        needs_lookup = "TÔI CẦN TRA CỨU THÊM" in answer.upper() or any(phrase in answer.lower() for phrase in [
            "tôi cần tra cứu", "không có thông tin", "không biết", "không chắc chắn",
            "không thể trả lời", "không đủ thông tin", "cần tìm hiểu thêm"
        ])
        
        if not needs_lookup:
            return True, answer
        else:
            return False, None
    except Exception as e:
        logger.error(f"Lỗi khi đánh giá kiến thức model: {e}")
        return False, None

# Hàm xử lý truy vấn
def process_query(message, history, mode, top_k_value, progress=gr.Progress()):
    if not message:
        return "", history
    
    # Hiển thị thông báo đang xử lý
    processing_message = "⏳ Đang xử lý..."
    yield "", history + [[message, processing_message]]
    
    # Cập nhật giá trị TOP_K
    top_k = int(top_k_value)
    
    try:
        start_time = time.time()
        
        # Nếu chế độ là "auto", hỏi model trước
        if mode == "auto":
            model_can_answer, model_answer = evaluate_model_knowledge(message)
            
            if model_can_answer:
                elapsed_time = time.time() - start_time
                response = f"{model_answer}\n\n<div style='font-size: 0.8em; color: gray; text-align: right; margin-top: 10px;'>🧠 Trả lời từ kiến thức đã huấn luyện | ⏱️ {elapsed_time:.2f} giây</div>"
                yield "", history + [[message, response]]
                return
            else:
                # Nếu model không biết, chuyển sang chế độ hybrid
                mode = "hybrid"
        
        # Xử lý theo các chế độ khác nhau
        if mode == "document":
            yield "", history + [[message, "⏳ Đang tìm kiếm thông tin..."]]
            doc_query = get_document_query()
            result = doc_query.query(message, top_k=top_k)
            
            response = f"{result['answer']}\n\n"
            
            # Thêm thông tin về nguồn
            response += "<div class='source-info'>"
            response += "<b>📚 Nguồn tài liệu:</b><br>"
            for i, source in enumerate(result['sources'], 1):
                response += f"  {i}. {source}<br>"
            response += "</div>"
                
        elif mode == "database":
            yield "", history + [[message, "⏳ Đang truy vấn cơ sở dữ liệu..."]]
            db_query = get_database_query()
            result = db_query.query(message)
            
            if result["success"]:
                response = f"<div class='query-info'>"
                response += f"<b>🔍 Truy vấn SQL:</b> <code>{result['sql_query']}</code>"
                response += "</div>"
                
                if isinstance(result["results"], list):
                    if result.get("formatted_results"):
                        response += f"<pre>{result['formatted_results']}</pre>"
                    else:
                        response += f"<b>Số lượng kết quả:</b> {len(result['results'])}"
                else:
                    response += f"<b>Kết quả:</b> {result['results']}"
            else:
                response = f"<span class='status-error'>⚠️ Lỗi: {result['message']}</span>"
                
        else:  # hybrid
            yield "", history + [[message, f"⏳ Đang phân tích và xử lý..."]]
            hybrid_query = get_hybrid_query()
            
            result = hybrid_query.query(message, top_k=top_k)
            
            response = f"{result['answer']}\n\n"
            
            # Thêm thông tin về nguồn
            response += "<div class='source-info'>"
            response += "<b>📚 Nguồn thông tin:</b><br>"
            for i, source in enumerate(result['sources'], 1):
                response += f"  {i}. {source}<br>"
            response += "</div>"
            
            # Thêm thông tin về loại truy vấn
            response += "<div class='query-info'>"
            response += "<b>🔍 Loại truy vấn:</b><br>"
            
            # Kiểm tra nếu đây là câu trả lời từ kiến thức có sẵn của model
            if result["query_type"].get("model_knowledge", False):
                response += "  - 🧠 Kiến thức của model: <span class='status-connected'>Có</span><br>"
                response += "  - 💾 Database: Không<br>"
                response += "  - 📄 Tài liệu: Không<br>"
            else:
                # Hiển thị thông tin nếu không phải từ kiến thức model
                if result["query_type"]["database"]:
                    response += "  - 💾 Database: <span class='status-connected'>Có</span><br>"
                    if result.get("sql_query"):
                        response += f"    <code>SQL: {result['sql_query']}</code><br>"
                else:
                    response += "  - 💾 Database: Không<br>"
                
                if result["query_type"]["document"]:
                    response += "  - 📄 Tài liệu: <span class='status-connected'>Có</span><br>"
                else:
                    response += "  - 📄 Tài liệu: Không<br>"
            
            response += "</div>"
        
        # Thêm thông tin về thời gian xử lý
        elapsed_time = time.time() - start_time
        response += f"<div style='font-size: 0.8em; color: gray; text-align: right; margin-top: 10px;'>⏱️ {elapsed_time:.2f} giây</div>"
                
    except Exception as e:
        logger.error(f"Lỗi khi xử lý truy vấn: {e}", exc_info=True)
        response = f"<span class='status-error'>⚠️ Đã xảy ra lỗi khi xử lý truy vấn: {str(e)}</span>"
    
    yield "", history + [[message, response]]

# Tạo vector database
def create_db(docs_dir, chunk_size, chunk_overlap, progress=gr.Progress()):
    try:
        progress(0, desc="Đang khởi tạo...")
        processor = DocumentProcessor(
            docs_dir=docs_dir,
            chunk_size=int(chunk_size),
            chunk_overlap=int(chunk_overlap),
            persist_directory=PERSIST_DIRECTORY
        )
        
        progress(0.2, desc="Đang xử lý tài liệu...")
        processor.process_all(progress_callback=lambda x: progress(0.2 + 0.7 * x, desc=f"Đang xử lý: {x*100:.0f}%"))
        
        progress(1.0, desc="Hoàn thành!")
        return f"✅ Đã tạo vector database thành công tại {PERSIST_DIRECTORY}"
    except Exception as e:
        logger.error(f"Lỗi khi tạo vector database: {e}", exc_info=True)
        return f"❌ Lỗi khi tạo vector database: {str(e)}"

# Tạo giao diện Gradio
def create_gradio_interface():
    # Kiểm tra kết nối
    lm_connected, lm_status = check_connection()
    db_connected, db_status = check_mysql_connection()
    
    # Định nghĩa các chuỗi HTML cho trạng thái
    lm_connected_html = '<span class="status-connected">✅ Đã kết nối</span>'
    lm_error_html = '<span class="status-error">❌ Lỗi kết nối</span>'
    db_connected_html = '<span class="status-connected">✅ Đã kết nối</span>'
    db_error_html = '<span class="status-error">❌ Lỗi kết nối</span>'
    
    # Định nghĩa chủ đề tối (Dark theme)
    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="slate",
        neutral_hue="slate",
        text_size="sm"
    ).set(
        body_background_fill="#0f172a",  # Nền tối
        block_background_fill="#1e293b",  # Nền block tối
        block_border_width="1px",
        block_border_color="#334155",
        block_radius="8px",
        button_primary_background_fill="#4f46e5",
        button_primary_background_fill_hover="#4338ca",
        button_primary_text_color="#ffffff",
        button_secondary_background_fill="#334155",
        button_secondary_background_fill_hover="#475569",
        button_secondary_text_color="#e2e8f0"
    )
    
    with gr.Blocks(title="RAG Chatbot", theme=theme, css=DARK_CUSTOM_CSS) as demo:
        gr.Markdown(
            """
            # 🤖 RAG Chatbot
            ### Hệ thống hỏi đáp kết hợp RAG (Retrieval-Augmented Generation) và truy vấn cơ sở dữ liệu
            """
        )
        
        with gr.Tabs() as tabs:
            with gr.TabItem("💬 Chat", id=0):
                with gr.Row():
                    with gr.Column(scale=4):
                        chatbot = gr.Chatbot(
                            show_copy_button=True,
                            height=500,
                            avatar_images=("https://t3.ftcdn.net/jpg/03/94/89/90/360_F_394899054_4TMgw6eiMYUfozaZU3Kgr5e0LdH4ZrsU.jpg", "https://www.shutterstock.com/image-vector/chat-bot-icon-virtual-smart-600nw-2478937553.jpg"),
                            bubble_full_width=False,
                            render_markdown=True,
                            elem_classes="chatbox",
                        )
                        
                        with gr.Row():
                            msg = gr.Textbox(
                                placeholder="Nhập câu hỏi của bạn tại đây...",
                                container=False,
                                scale=10,
                                autofocus=True,
                            )
                            submit_btn = gr.Button("🚀 Gửi", variant="primary", scale=1)
                        
                        with gr.Row():
                            clear_btn = gr.Button("🗑️ Xóa lịch sử", variant="secondary")
                            
                    with gr.Column(scale=1):
                        gr.Markdown("### Chế độ truy vấn")
                        mode_selector = gr.Radio(
                            ["auto", "hybrid", "document", "database"],
                            label="Chọn chế độ truy vấn phù hợp với câu hỏi của bạn",
                            value="auto",
                            container=True,
                        )
                        
                        gr.Markdown("### Số lượng kết quả tìm kiếm (top-k)")
                        top_k_slider = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=TOP_K,
                            step=1,
                            label="Số lượng kết quả trả về khi tìm kiếm tài liệu",
                        )
                        
                        with gr.Accordion("ℹ️ Trạng thái hệ thống", open=True):
                            lm_status_component = gr.Markdown(
                                f"**LM Studio API**: {lm_connected_html if lm_connected else lm_error_html}"
                            )
                            db_status_component = gr.Markdown(
                                f"**MySQL Database**: {db_connected_html if db_connected else db_error_html}"
                            )
                            
                            if not lm_connected or not db_connected:
                                gr.Markdown(
                                    f"**Chi tiết lỗi**:\n- LM Studio: {lm_status}\n- MySQL: {db_status}"
                                )
                        
                        with gr.Accordion("🔧 Thông tin hệ thống", open=False):
                            model_info = gr.Markdown(
                                f"""
                                - **Model LLM**: {MODEL_NAME}
                                - **API URL**: {LM_STUDIO_URL}
                                - **Vector DB**: {PERSIST_DIRECTORY}
                                - **Database**: {MYSQL_DATABASE}@{MYSQL_HOST}
                                
                                ### Chế độ truy vấn
                                - **Auto**: Tự động quyết định sử dụng kiến thức được huấn luyện sẵn hoặc truy vấn tài nguyên
                                - **Hybrid**: Kết hợp cả RAG và truy vấn cơ sở dữ liệu
                                - **Document**: Chỉ sử dụng RAG
                                - **Database**: Chỉ truy vấn cơ sở dữ liệu
                                """
                            )
                
            with gr.TabItem("🛠️ Quản lý", id=1):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Tạo Vector Database")
                        with gr.Row():
                            docs_dir = gr.Textbox(value="./docs", label="Thư mục tài liệu")
                        
                        with gr.Row():
                            chunk_size = gr.Number(value=500, label="Kích thước đoạn văn bản", precision=0)
                            chunk_overlap = gr.Number(value=50, label="Độ chồng lấp", precision=0)
                        
                        create_btn = gr.Button("🔨 Tạo Vector Database", variant="primary")
                        create_output = gr.Markdown()
                        
                        gr.Markdown("### Kiểm tra kết nối")
                        with gr.Row():
                            check_lm_btn = gr.Button("🔄 Kiểm tra LM Studio API", variant="secondary")
                            check_db_btn = gr.Button("🔄 Kiểm tra MySQL", variant="secondary")
                        
                        connection_status = gr.Markdown()
                
                with gr.Accordion("📖 Hướng dẫn", open=False):
                    gr.Markdown(
                        """
                        ### Hướng dẫn sử dụng

                        1. **Tạo Vector Database**:
                           - Đảm bảo tài liệu đã được đặt trong thư mục chỉ định
                           - Nhấn nút "Tạo Vector Database" để tạo cơ sở dữ liệu vector từ tài liệu

                        2. **Sử dụng chatbot**:
                           - Chọn chế độ truy vấn phù hợp (auto, hybrid, document, database)
                           - Nhập câu hỏi và nhấn "Gửi" hoặc Enter
                           - Kết quả sẽ được hiển thị trong cửa sổ chat

                        3. **Các chế độ truy vấn**:
                           - **Auto**: Tự động sử dụng kiến thức có sẵn hoặc truy vấn tài nguyên khi cần
                           - **Hybrid**: Kết hợp cả RAG và truy vấn cơ sở dữ liệu
                           - **Document**: Chỉ sử dụng RAG để trả lời từ tài liệu
                           - **Database**: Chỉ truy vấn cơ sở dữ liệu

                        4. **Mẹo tối ưu hiệu suất**:
                           - Đối với câu hỏi đơn giản, hệ thống sẽ trả lời ngay lập tức
                           - Chọn chế độ truy vấn phù hợp để tối ưu thời gian xử lý
                           - Điều chỉnh top-k nhỏ hơn để tăng tốc độ xử lý
                        """
                    )
                
        # Xử lý các sự kiện
        submit_btn.click(
            process_query,
            inputs=[msg, chatbot, mode_selector, top_k_slider],
            outputs=[msg, chatbot],
            api_name="submit",
        )
        
        msg.submit(
            process_query,
            inputs=[msg, chatbot, mode_selector, top_k_slider],
            outputs=[msg, chatbot],
            api_name="submit_enter",
        )
        
        clear_btn.click(
            lambda: (None, []),
            inputs=None,
            outputs=[msg, chatbot],
        )
        
        # Xử lý sự kiện tạo vector database
        create_btn.click(
            create_db,
            inputs=[docs_dir, chunk_size, chunk_overlap],
            outputs=[create_output],
        )
        
        # Xử lý sự kiện kiểm tra kết nối
        check_lm_btn.click(
            lambda: check_connection()[1],
            inputs=None,
            outputs=[connection_status],
        )
        
        check_db_btn.click(
            lambda: check_mysql_connection()[1],
            inputs=None,
            outputs=[connection_status],
        )
        
    return demo

if __name__ == "__main__":
    # Kiểm tra xem vector database đã tồn tại chưa
    if not os.path.exists(PERSIST_DIRECTORY):
        logger.warning(f"Vector database không tồn tại tại {PERSIST_DIRECTORY}. Bạn có thể tạo mới trong tab Quản lý.")
    
    # Khởi động giao diện Gradio
    demo = create_gradio_interface()
    demo.launch(share=False, server_name="127.0.0.1") 