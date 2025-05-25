# Hướng Dẫn Sử Dụng Hệ Thống RAG (Retrieval-Augmented Generation)

## Giới Thiệu

Hệ thống RAG (Retrieval-Augmented Generation) là một giải pháp kết hợp tìm kiếm tài liệu và tạo văn bản bằng mô hình ngôn ngữ lớn (LLM) để cung cấp câu trả lời chính xác và có nguồn gốc rõ ràng. Hệ thống này có thể truy vấn thông tin từ:

1. **Tài liệu văn bản** (tìm kiếm vector)
2. **Cơ sở dữ liệu MySQL** (tạo và thực thi truy vấn SQL)
3. **Kết hợp cả hai nguồn trên** (hybrid)

## Cấu Trúc Mã Nguồn

Dự án bao gồm các thành phần chính:

- `main.py`: File chính để chạy ứng dụng từ dòng lệnh
- `gradio_interface.py`: Giao diện web sử dụng Gradio
- `document_processor.py`: Xử lý tài liệu và tạo vector database
- `document_query.py`: Truy vấn thông tin từ tài liệu
- `database_query.py`: Truy vấn thông tin từ cơ sở dữ liệu MySQL
- `hybrid_query.py`: Kết hợp truy vấn từ cả tài liệu và cơ sở dữ liệu

## Yêu Cầu Hệ Thống

1. Python 3.8 trở lên
2. LM Studio (hoặc API tương thích với OpenAI để tạo embeddings và truy vấn LLM)
3. MySQL (tùy chọn, nếu sử dụng chức năng truy vấn cơ sở dữ liệu)

## Cài Đặt

1. Clone repository và cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

2. Tạo file `.env` dựa trên mẫu `.env.example`:

```bash
cp .env.example .env
```

3. Chỉnh sửa file `.env` với thông tin cấu hình phù hợp:
   - Thông tin kết nối MySQL
   - URL của LM Studio API
   - Tên model LLM
   - Thư mục lưu trữ vector database
   - Thư mục chứa tài liệu

## Cách Sử Dụng

### 1. Từ Dòng Lệnh (CLI)

#### Tạo Vector Database

```bash
python main.py --create-db --docs-dir ./docs --chunk-size 500 --chunk-overlap 50
```

#### Truy Vấn Tài Liệu

```bash
python main.py --document --query "Câu hỏi của bạn?"
```

#### Truy Vấn Cơ Sở Dữ Liệu

```bash
python main.py --database --query "Câu hỏi về dữ liệu trong MySQL?"
```

#### Truy Vấn Kết Hợp (Hybrid)

```bash
python main.py --hybrid --query "Câu hỏi kết hợp cả tài liệu và database?"
```

#### Chế Độ Tương Tác

```bash
python main.py --interactive --mode hybrid
```

### 2. Sử Dụng Giao Diện Web (Gradio)

```bash
python gradio_interface.py
```

Giao diện web sẽ được chạy tại địa chỉ http://127.0.0.1:7860 và cung cấp các chức năng:
- Truy vấn tài liệu
- Truy vấn cơ sở dữ liệu
- Truy vấn kết hợp
- Tạo và quản lý vector database

## Quy Trình Hoạt Động

### 1. Xử Lý Tài Liệu (`document_processor.py`)

- Đọc tài liệu từ thư mục chỉ định
- Chia nhỏ tài liệu thành các đoạn (chunks)
- Tạo embeddings cho các đoạn văn bản
- Lưu trữ trong vector database (Chroma)

### 2. Truy Vấn Tài Liệu (`document_query.py`)

- Đánh giá loại câu hỏi (cần tài liệu hay kiến thức chung)
- Tìm kiếm tài liệu liên quan bằng similarity search
- Kết hợp kết quả tìm kiếm với câu hỏi
- Truy vấn LLM để tạo câu trả lời

### 3. Truy Vấn Cơ Sở Dữ Liệu (`database_query.py`)

- Đánh giá xem câu hỏi có thể trả lời bằng SQL
- Tự động tạo truy vấn SQL từ câu hỏi tự nhiên
- Thực thi truy vấn SQL trên cơ sở dữ liệu MySQL
- Định dạng và trả về kết quả

### 4. Truy Vấn Kết Hợp (`hybrid_query.py`)

- Phân tích câu hỏi để xác định nguồn thông tin cần thiết
- Truy vấn cả hai nguồn (tài liệu và cơ sở dữ liệu)
- Tổng hợp thông tin từ nhiều nguồn
- Trả về câu trả lời với nguồn gốc rõ ràng

## Lớp LMStudioEmbeddings

Lớp này sử dụng API của LM Studio để tạo embeddings cho văn bản, giúp xây dựng vector database. Nó hoạt động với mô hình embedding như `text-embedding-nomic-embed-text-v1.5-embedding`.

## Tùy Chỉnh

### Thay Đổi Mô Hình LLM

Sửa biến `MODEL_NAME` trong file `.env` hoặc truyền tham số khi chạy:

```bash
python main.py --hybrid --query "Câu hỏi?" --model-name "gemma-3-7b-it"
```

### Thay Đổi Kích Thước Chunk

```bash
python main.py --create-db --chunk-size 1000 --chunk-overlap 100
```

### Điều Chỉnh Số Lượng Kết Quả Tìm Kiếm

```bash
python main.py --document --query "Câu hỏi?" --top-k 5
```

## Lưu Ý Quan Trọng

1. **LM Studio**: Cần chạy LM Studio với API hoạt động ở cổng mặc định (1234)
2. **Vector Database**: Cần tạo trước khi truy vấn tài liệu
3. **MySQL**: Cần cấu hình đúng thông tin kết nối trong file `.env`
4. **Tài Liệu**: Nên đặt tất cả tài liệu text (.txt) trong thư mục `./docs`

## Các Tham Số Dòng Lệnh

### main.py

```
--create-db           Tạo vector database
--docs-dir            Thư mục chứa tài liệu
--chunk-size          Kích thước mỗi đoạn văn bản
--chunk-overlap       Độ chồng lấp giữa các đoạn
--document            Chế độ truy vấn tài liệu
--database            Chế độ truy vấn cơ sở dữ liệu
--hybrid              Chế độ truy vấn kết hợp
--query               Câu hỏi cần truy vấn
--interactive         Chế độ tương tác
--mode                Chế độ truy vấn (document/database/hybrid)
--top-k               Số lượng kết quả tìm kiếm
--model-name          Tên mô hình LLM
```

## Ví Dụ Sử Dụng

### Truy Vấn Tài Liệu

```bash
python main.py --document --query "Quy trình đăng ký visa du học là gì?"
```

### Truy Vấn Cơ Sở Dữ Liệu

```bash
python main.py --database --query "Liệt kê 5 khách hàng có doanh số cao nhất"
```

### Truy Vấn Kết Hợp

```bash
python main.py --hybrid --query "Các sản phẩm bán chạy nhất trong tháng 6 có được đề cập trong tài liệu marketing không?"
``` 