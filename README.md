# Hybrid RAG + Database (Retrieval Augmented Generation với MySQL)

![RAG Demo](https://img.shields.io/badge/RAG-Demo-blue)
![Python](https://img.shields.io/badge/Python-3.9+-brightgreen)
![LangChain](https://img.shields.io/badge/LangChain-0.1.0-orange)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.4.22-yellow)
![MySQL](https://img.shields.io/badge/MySQL-8.0+-blue)
![Gemma](https://img.shields.io/badge/Gemma--3--12B--IT-Model-purple)

Hệ thống kết hợp RAG (Retrieval Augmented Generation) và Database Query tích hợp với LM Studio, cho phép truy vấn thông tin từ cả tài liệu văn bản và cơ sở dữ liệu bằng ngôn ngữ tự nhiên.

## Tổng quan

Dự án này xây dựng một hệ thống hybrid kết hợp RAG (Retrieval Augmented Generation) và truy vấn database, tích hợp với LM Studio để tạo ra câu trả lời chính xác từ nhiều nguồn dữ liệu. Dự án sử dụng các công nghệ sau:

- **LangChain**: Framework để xây dựng ứng dụng sử dụng LLM
- **ChromaDB**: Vector database để lưu trữ và truy vấn embeddings
- **MySQL**: Cơ sở dữ liệu quan hệ để lưu trữ dữ liệu có cấu trúc
- **Sentence Transformers**: Tạo embeddings cho các đoạn văn bản
- **LM Studio**: Cung cấp API cho Large Language Model (LLM) chạy cục bộ
- **Gemma-3-12b-it**: Mô hình ngôn ngữ lớn của Google để xử lý câu trả lời

## Cách hoạt động

Hệ thống hoạt động theo quy trình kết hợp RAG và truy vấn database với khả năng tự động phân loại câu hỏi:

1. **Đánh giá loại câu hỏi**:
   - Phân tích câu hỏi để xác định xem nó yêu cầu thông tin từ database, tài liệu, hay cả hai
   - Nếu là kiến thức chung (như lịch sử, khoa học, nhân vật nổi tiếng), truy vấn LLM trực tiếp
   - Nếu cần thông tin từ tài liệu, tiến hành quy trình RAG
   - Nếu cần thông tin từ database, tiến hành quy trình truy vấn MySQL
   - Nếu cần cả hai, thực hiện cả hai quy trình và kết hợp kết quả

2. **Xử lý tài liệu (Document Processing)**:
   - Tải tài liệu văn bản từ thư mục `docs`
   - Chia nhỏ tài liệu thành các đoạn (chunks) với kích thước và độ chồng lấp có thể tùy chỉnh
   - Tạo embeddings cho từng đoạn sử dụng API của LM Studio
   - Lưu embeddings vào ChromaDB

3. **Truy vấn tài liệu (Document Querying)**:
   - Chuyển đổi câu hỏi của người dùng thành embedding
   - Tìm kiếm các đoạn văn bản tương tự nhất trong vector database
   - Kết hợp đoạn văn bản tìm được vào prompt và gửi đến LM Studio
   - Sử dụng model Gemma-3-12b-it để tạo câu trả lời chất lượng cao
   - Trả về câu trả lời cùng với nguồn tài liệu

4. **Truy vấn database (Database Querying)**:
   - Phân tích câu hỏi bằng LLM để tạo câu truy vấn SQL tương ứng
   - Thực thi truy vấn SQL trên MySQL database
   - Định dạng kết quả truy vấn để LLM có thể hiểu và xử lý
   - Sử dụng model Gemma-3-12b-it để tạo câu trả lời dựa trên kết quả truy vấn
   - Trả về câu trả lời kèm theo truy vấn SQL và kết quả

5. **Kết hợp kết quả (Hybrid Results)**:
   - Trong trường hợp câu hỏi cần thông tin từ cả database và tài liệu
   - Kết hợp ngữ cảnh từ cả hai nguồn
   - Sử dụng LLM để tổng hợp một câu trả lời toàn diện
   - Trả về câu trả lời cùng với thông tin về các nguồn dữ liệu

## Cài đặt

### Yêu cầu

- Python 3.9 trở lên
- MySQL Server 8.0 trở lên
- LM Studio đã cài đặt và đang chạy trên máy tính của bạn

### Bước 1: Clone dự án

```bash
git clone https://github.com/yourusername/hybrid-rag-database.git
cd hybrid-rag-database
```

### Bước 2: Tạo môi trường ảo và cài đặt thư viện

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

### Bước 3: Cấu hình môi trường

Tạo file `.env` từ file `.env.example`:

```bash
cp .env.example .env
```

Chỉnh sửa file `.env` để thiết lập các thông số:
- Cấu hình LM Studio (URL, tên model)
- Cấu hình MySQL (host, port, user, password, database)
- Cấu hình RAG (thư mục tài liệu, kích thước chunk, v.v.)

### Bước 4: Chuẩn bị tài liệu và database

1. **Tài liệu**: Đặt tất cả tài liệu văn bản (`.txt`) vào thư mục `docs`. Dự án hỗ trợ tài liệu dạng text, và mỗi tài liệu nên được lưu dưới dạng UTF-8.

2. **Database**: Đảm bảo MySQL server đang chạy và database đã được tạo:
   ```bash
   mysql -u root -p
   CREATE DATABASE kt_ai;
   USE kt_ai;
   # Tạo các bảng cần thiết và nhập dữ liệu
   ```

### Bước 5: Cài đặt và chạy LM Studio

1. Tải và cài đặt [LM Studio](https://lmstudio.ai/)
2. Tải các mô hình sau từ LM Studio:
   - Model embedding: `text-embedding-nomic-embed-text-v1.5-embedding`
   - Model sinh câu trả lời: `gemma-3-12b-it`
3. Khởi chạy Local Server API trong LM Studio:
   - Vào tab "Local Server"
   - Chọn mô hình `gemma-3-12b-it`
   - Nhấn "Start Server"
   - Mặc định server sẽ chạy tại `http://127.0.0.1:1234`

## Sử dụng

Dự án có năm chế độ sử dụng chính:

### 1. Tạo Vector Database

```bash
python main.py create --docs_dir ./docs --chunk_size 500 --chunk_overlap 50 --persist_directory ./chroma_db
```

Các tham số:
- `--docs_dir`: Thư mục chứa tài liệu (mặc định: `./docs`)
- `--chunk_size`: Kích thước của mỗi đoạn văn bản (mặc định: 500 ký tự)
- `--chunk_overlap`: Độ chồng lấp giữa các đoạn (mặc định: 50 ký tự)
- `--persist_directory`: Thư mục lưu trữ vector database (mặc định: `./chroma_db`)

### 2. Truy vấn tài liệu (Document)

```bash
python main.py document --query "Điều kiện để được nhận bằng tốt nghiệp là gì?" --top_k 3
```

Các tham số:
- `--query`: Câu hỏi của người dùng (bắt buộc)
- `--persist_directory`: Thư mục lưu trữ vector database (mặc định: `./chroma_db`)
- `--lm_studio_url`: URL của LM Studio API (mặc định: từ .env hoặc `http://127.0.0.1:1234`)
- `--model_name`: Tên model LLM (mặc định: từ .env hoặc `gemma-3-12b-it`)
- `--top_k`: Số lượng kết quả tìm kiếm (mặc định: 3)

### 3. Truy vấn database (Database)

```bash
python main.py database --query "Có bao nhiêu sinh viên đạt điểm A trong môn Toán?"
```

Các tham số:
- `--query`: Câu hỏi của người dùng (bắt buộc)
- `--mysql_host`: Host của MySQL server (mặc định: từ .env hoặc `localhost`)
- `--mysql_user`: Username MySQL (mặc định: từ .env hoặc `root`)
- `--mysql_password`: Password MySQL (mặc định: từ .env)
- `--mysql_port`: Port của MySQL server (mặc định: từ .env hoặc `3306`)
- `--mysql_database`: Tên database MySQL (mặc định: từ .env hoặc `kt_ai`)
- `--lm_studio_url`: URL của LM Studio API (mặc định: từ .env hoặc `http://127.0.0.1:1234`)
- `--model_name`: Tên model LLM (mặc định: từ .env hoặc `gemma-3-12b-it`)

### 4. Truy vấn hybrid (Database + Document)

```bash
python main.py hybrid --query "Quy định về điểm thi và số sinh viên đạt điểm A trong kỳ vừa qua là gì?"
```

Các tham số:
- `--query`: Câu hỏi của người dùng (bắt buộc)
- `--persist_directory`: Thư mục lưu trữ vector database (mặc định: `./chroma_db`)
- Các tham số MySQL và LM Studio tương tự như trên

### 5. Chế độ tương tác

```bash
python main.py interactive --mode hybrid
```

Các tham số:
- `--mode`: Chế độ truy vấn ban đầu (`hybrid`, `document`, `database`) (mặc định: `hybrid`)
- Các tham số khác tương tự như trên

Trong chế độ tương tác, bạn có thể:
- Nhập câu hỏi để truy vấn
- Nhập `mode` để chuyển đổi giữa các chế độ truy vấn (hybrid, document, database)
- Nhập `exit` hoặc `quit` để thoát

## Cấu trúc dự án

```
.
├── main.py                  # Điểm vào chính của chương trình
├── document_processor.py    # Xử lý tài liệu và tạo vector database
├── document_query.py        # Truy vấn tài liệu và tạo câu trả lời
├── database_query.py        # Kết nối và truy vấn MySQL database
├── hybrid_query.py          # Kết hợp truy vấn từ database và tài liệu
├── requirements.txt         # Các thư viện cần thiết
├── .env.example             # Mẫu file cấu hình môi trường
├── chroma_db/               # Thư mục lưu trữ vector database
└── docs/                    # Thư mục chứa tài liệu
    ├── quy_dinh_dao_tao.txt
    ├── quy_dinh_bao_mat.txt
    └── quy_dinh_lam_viec.txt
```

## Tùy chỉnh nâng cao

### Tối ưu hóa truy vấn database

Để tối ưu hóa truy vấn database, bạn có thể điều chỉnh các tham số sau:

1. **Cache schema**: Hệ thống sẽ tự động cache schema của database để tránh truy vấn lặp lại, bạn có thể tùy chỉnh cách này trong file `database_query.py`:

```python
def get_table_schema(self, connection=None) -> Dict[str, List]:
    # Trả về từ cache nếu đã có
    if self._schema_info:
        return self._schema_info
        
    # ... code để lấy schema từ database ...
    
    # Lưu vào cache
    self._schema_info = schema_info
    
    return schema_info
```

2. **Temperature cho SQL generation**: Giảm temperature khi tạo SQL để có kết quả nhất quán hơn:

```python
payload = {
    "model": self.model_name,
    "messages": [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Yêu cầu: {question}"}
    ],
    "max_tokens": 512,
    "temperature": 0.2,  # Giảm temperature để có SQL ổn định hơn
    "stream": False
}
```

### Tự động đánh giá loại câu hỏi (Smart Query Routing)

Hệ thống có khả năng tự động phân loại câu hỏi để quyết định xem cần truy vấn database, tài liệu, hay cả hai:

1. **Đánh giá nhu cầu database**: Tùy chỉnh trong `database_query.py`:

```python
def evaluate_sql_query_type(self, question: str) -> bool:
    # ...
    system_message = """Bạn là một trợ lý thông minh giúp phân loại câu hỏi. 
Nhiệm vụ của bạn là xác định xem một câu hỏi có yêu cầu thông tin từ cơ sở dữ liệu hay không.

Phân loại câu hỏi thành một trong hai loại:
1. Câu hỏi liên quan đến dữ liệu hoặc số liệu cụ thể có thể truy vấn từ database
2. Câu hỏi không liên quan đến dữ liệu trong database

Trả lời chỉ với "DATABASE" cho loại 1 hoặc "NON_DATABASE" cho loại 2. Không giải thích lý do."""
    # ...
```

2. **Đánh giá nhu cầu tài liệu**: Tùy chỉnh trong `document_query.py`:

```python
def evaluate_query_type(self, query: str) -> bool:
    # ...
    system_message = """Bạn là một trợ lý thông minh giúp phân loại câu hỏi. 
Nhiệm vụ của bạn là xác định xem một câu hỏi có yêu cầu thông tin từ tài liệu cụ thể hay không.

Phân loại câu hỏi thành một trong hai loại:
1. Câu hỏi về kiến thức chung hoặc câu hỏi mà bạn đã biết câu trả lời (như về lịch sử, khoa học, văn hóa, nhân vật nổi tiếng, v.v.)
2. Câu hỏi về quy định, hướng dẫn, hoặc thông tin cụ thể có thể có trong tài liệu

Trả lời chỉ với "GENERAL" cho loại 1 hoặc "DOCUMENT" cho loại 2. Không giải thích lý do."""
    # ...
```

### Kết hợp kết quả từ nhiều nguồn

Khi câu hỏi cần thông tin từ cả database và tài liệu, hệ thống sẽ kết hợp thông tin từ cả hai nguồn. Bạn có thể tùy chỉnh cách kết hợp trong `hybrid_query.py`:

```python
def _synthesize_hybrid_answer(self, question: str, combined_context: str) -> str:
    # ...
    system_message = """Bạn là một trợ lý thông minh và hữu ích.
Nhiệm vụ của bạn là tổng hợp thông tin từ nhiều nguồn (database và tài liệu) để trả lời câu hỏi của người dùng.
Hãy phân tích cả dữ liệu số liệu từ database và thông tin từ tài liệu để cung cấp câu trả lời toàn diện nhất.
Kết hợp thông tin từ các nguồn khác nhau một cách hợp lý và logic.
Ưu tiên dữ liệu cụ thể từ database nếu có, và bổ sung thêm thông tin từ tài liệu để giải thích hoặc mở rộng.
Nếu có sự mâu thuẫn giữa các nguồn, hãy nêu rõ điều này và giải thích sự khác biệt."""
    # ...
```

### Thay đổi mô hình và cấu hình

Bạn có thể dễ dàng thay đổi cấu hình qua file `.env` hoặc tham số dòng lệnh:

1. **Thay đổi mô hình embedding và sinh câu trả lời**:
   - Điều chỉnh `MODEL_NAME` trong file `.env`
   - Hoặc sử dụng tham số `--model_name` khi chạy chương trình

2. **Thay đổi cấu hình MySQL**:
   - Điều chỉnh các tham số MySQL trong file `.env`
   - Hoặc sử dụng các tham số `--mysql_*` khi chạy chương trình

3. **Thay đổi chế độ truy vấn mặc định**:
   - Điều chỉnh `DEFAULT_QUERY_MODE` trong file `.env`
   - Hoặc sử dụng tham số `--mode` khi chạy chương trình ở chế độ interactive

## Lưu ý

- Đảm bảo LM Studio đang chạy trước khi sử dụng chương trình
- Đảm bảo MySQL server đang chạy và database đã được cấu hình đúng
- Nếu sử dụng vector database lần đầu, chạy lệnh `python main.py create` trước
- Cấu hình xác thực MySQL đúng trong file `.env` hoặc tham số dòng lệnh

## Liên hệ

Nếu bạn có bất kỳ câu hỏi hoặc đề xuất nào, vui lòng tạo issue hoặc liên hệ với tôi qua [email](mailto:hoangkhoi230@gmail.com). 