# Hướng dẫn sử dụng Giao diện Gradio cho RAG Chatbot

## Giới thiệu

Giao diện Gradio cung cấp một cách trực quan để tương tác với hệ thống chatbot RAG (Retrieval-Augmented Generation) và truy vấn cơ sở dữ liệu. Giao diện này cho phép bạn:

- Chat với chatbot sử dụng tài liệu và cơ sở dữ liệu
- Tạo và quản lý vector database từ tài liệu
- Chọn chế độ truy vấn (hybrid, document, database)
- Kiểm tra kết nối đến các dịch vụ

## Cài đặt

1. Đảm bảo bạn đã cài đặt tất cả các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

2. Cấu hình các biến môi trường trong file `.env` (sao chép từ `.env.example` nếu cần):

```
LM_STUDIO_URL=http://127.0.0.1:1234
MODEL_NAME=gemma-3-12b-it
PERSIST_DIRECTORY=./chroma_db
MYSQL_HOST=localhost
MYSQL_USER=root
MYSQL_PASSWORD=
MYSQL_PORT=3306
MYSQL_DATABASE=kt_ai
TOP_K=3
```

## Khởi động giao diện Gradio

Chạy lệnh sau để khởi động giao diện Gradio:

```bash
python gradio_interface.py
```

Giao diện sẽ được khởi động tại địa chỉ: `http://127.0.0.1:7860`

## Các tính năng chính

### Tab Chat

- **Chế độ truy vấn**: Chọn giữa "hybrid", "document", hoặc "database"
  - **Hybrid**: Kết hợp cả RAG và truy vấn cơ sở dữ liệu
  - **Document**: Chỉ sử dụng RAG để trả lời từ tài liệu
  - **Database**: Chỉ truy vấn cơ sở dữ liệu

- **Số lượng kết quả tìm kiếm (top-k)**: Điều chỉnh số lượng kết quả trả về khi tìm kiếm tài liệu

- **Trạng thái hệ thống**: Hiển thị trạng thái kết nối đến LM Studio API và MySQL Database

### Tab Quản lý

- **Tạo Vector Database**: 
  - Nhập đường dẫn thư mục tài liệu
  - Điều chỉnh kích thước đoạn văn bản và độ chồng lấp
  - Nhấn nút "Tạo Vector Database" để bắt đầu quá trình

- **Kiểm tra kết nối**:
  - Kiểm tra kết nối đến LM Studio API
  - Kiểm tra kết nối đến MySQL Database

## Hướng dẫn sử dụng

1. **Tạo Vector Database** (trước khi sử dụng chế độ Document hoặc Hybrid):
   - Đảm bảo tài liệu văn bản đã được đặt trong thư mục chỉ định (mặc định: `./docs`)
   - Chuyển đến tab "Quản lý"
   - Nhấn nút "Tạo Vector Database" để tạo cơ sở dữ liệu vector từ tài liệu

2. **Sử dụng Chatbot**:
   - Chọn chế độ truy vấn phù hợp (hybrid, document, database)
   - Nhập câu hỏi vào ô văn bản và nhấn "Gửi" hoặc Enter
   - Kết quả sẽ được hiển thị trong cửa sổ chat

3. **Xem thông tin chi tiết**:
   - Kết quả trả về sẽ hiển thị nguồn tài liệu hoặc câu truy vấn SQL
   - Thời gian xử lý được hiển thị ở cuối mỗi phản hồi

## Xử lý sự cố

- **Vector database không tồn tại**: Sử dụng tab "Quản lý" để tạo vector database mới
- **Lỗi kết nối LM Studio API**: Đảm bảo LM Studio đang chạy và URL được cấu hình đúng
- **Lỗi kết nối MySQL**: Kiểm tra cấu hình MySQL trong file .env

## Hỗ trợ

Nếu bạn gặp vấn đề hoặc có câu hỏi, vui lòng tạo issue trên GitHub repository. 