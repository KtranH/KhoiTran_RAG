# RAG (Retrieval Augmented Generation) với LM Studio

![RAG Demo](https://img.shields.io/badge/RAG-Demo-blue)
![Python](https://img.shields.io/badge/Python-3.9+-brightgreen)
![LangChain](https://img.shields.io/badge/LangChain-0.1.0-orange)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.4.22-yellow)
![Gemma](https://img.shields.io/badge/Gemma--3--12B--IT-Model-purple)

Hệ thống RAG (Retrieval Augmented Generation) tích hợp với LM Studio, cho phép truy vấn thông tin từ tài liệu văn bản bằng ngôn ngữ tự nhiên.

## Tổng quan

Dự án này xây dựng một hệ thống RAG (Retrieval Augmented Generation) tích hợp với LM Studio để tạo ra câu trả lời chính xác dựa trên tài liệu cục bộ. Dự án sử dụng các công nghệ sau:

- **LangChain**: Framework để xây dựng ứng dụng sử dụng LLM
- **ChromaDB**: Vector database để lưu trữ và truy vấn embeddings
- **Sentence Transformers**: Tạo embeddings cho các đoạn văn bản
- **LM Studio**: Cung cấp API cho Large Language Model (LLM) chạy cục bộ
- **Gemma-3-12b-it**: Mô hình ngôn ngữ lớn của Google để xử lý câu trả lời

## Cách hoạt động

Hệ thống hoạt động theo quy trình RAG tiêu chuẩn với bổ sung tính năng đánh giá loại câu hỏi:

1. **Đánh giá loại câu hỏi**:
   - Phân tích câu hỏi để xác định xem nó yêu cầu thông tin từ tài liệu hay là kiến thức chung
   - Nếu là kiến thức chung (như lịch sử, khoa học, nhân vật nổi tiếng), truy vấn LLM trực tiếp
   - Nếu cần thông tin từ tài liệu, tiến hành quy trình RAG

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

## Cài đặt

### Yêu cầu

- Python 3.9 trở lên
- LM Studio đã cài đặt và đang chạy trên máy tính của bạn

### Bước 1: Clone dự án

```bash
git clone https://github.com/yourusername/rag-lm-studio.git
cd rag-lm-studio
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

### Bước 3: Chuẩn bị tài liệu

Đặt tất cả tài liệu văn bản (`.txt`) vào thư mục `docs`. Dự án hiện hỗ trợ tài liệu dạng text, và mỗi tài liệu nên được lưu dưới dạng UTF-8.

### Bước 4: Cài đặt và chạy LM Studio

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

Dự án có ba chế độ sử dụng chính:

### 1. Tạo Vector Database

```bash
python main.py create --docs_dir ./docs --chunk_size 500 --chunk_overlap 50 --persist_directory ./chroma_db
```

Các tham số:
- `--docs_dir`: Thư mục chứa tài liệu (mặc định: `./docs`)
- `--chunk_size`: Kích thước của mỗi đoạn văn bản (mặc định: 500 ký tự)
- `--chunk_overlap`: Độ chồng lấp giữa các đoạn (mặc định: 50 ký tự)
- `--persist_directory`: Thư mục lưu trữ vector database (mặc định: `./chroma_db`)

### 2. Truy vấn một câu hỏi cụ thể

```bash
python main.py query --query "Điều kiện để được nhận bằng tốt nghiệp là gì?" --model_name "gemma-3-12b-it" --lm_studio_url http://127.0.0.1:1234 --top_k 3
```

Các tham số:
- `--query`: Câu hỏi của người dùng (bắt buộc)
- `--persist_directory`: Thư mục lưu trữ vector database (mặc định: `./chroma_db`)
- `--lm_studio_url`: URL của LM Studio API (mặc định: `http://127.0.0.1:1234`)
- `--model_name`: Tên model LLM (mặc định: `gemma-3-12b-it`)
- `--top_k`: Số lượng kết quả tìm kiếm (mặc định: 3)

### 3. Chế độ tương tác

```bash
python main.py interactive --model_name "gemma-3-12b-it" --lm_studio_url http://127.0.0.1:1234
```

Các tham số:
- `--persist_directory`: Thư mục lưu trữ vector database (mặc định: `./chroma_db`)
- `--lm_studio_url`: URL của LM Studio API (mặc định: `http://127.0.0.1:1234`)
- `--model_name`: Tên model LLM (mặc định: `gemma-3-12b-it`)

## Cấu trúc dự án

```
.
├── main.py                  # Điểm vào chính của chương trình
├── document_processor.py    # Xử lý tài liệu và tạo vector database
├── document_query.py        # Truy vấn tài liệu và tạo câu trả lời
├── requirements.txt         # Các thư viện cần thiết
├── chroma_db/               # Thư mục lưu trữ vector database
└── docs/                    # Thư mục chứa tài liệu
    ├── quy_dinh_dao_tao.txt
    ├── quy_dinh_bao_mat.txt
    └── quy_dinh_lam_viec.txt
```

## Tùy chỉnh nâng cao

### Tự động đánh giá loại câu hỏi (Smart Query Routing)

Hệ thống có khả năng tự động phân loại câu hỏi để quyết định xem có cần truy vấn tài liệu hay không:

- **Câu hỏi kiến thức chung**: Truy vấn LLM trực tiếp không qua RAG (nhanh hơn)
- **Câu hỏi về tài liệu**: Sử dụng quy trình RAG đầy đủ

Bạn có thể tùy chỉnh logic phân loại bằng cách sửa đổi phương thức `evaluate_query_type` trong file `document_query.py`:

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

Bạn có thể điều chỉnh system message này để thay đổi cách hệ thống phân loại câu hỏi.

### Chế độ trả lời kết hợp (Hybrid Answering)

Hệ thống được cấu hình để có thể sử dụng kiến thức riêng của model LLM khi không tìm thấy thông tin trong tài liệu hoặc khi câu hỏi là về kiến thức chung. Bạn có thể điều chỉnh cách hệ thống xử lý bằng cách sửa đổi system message trong file `document_query.py`:

```python
system_message = f"""Bạn là một trợ lý thông minh và hữu ích.

Nếu câu hỏi liên quan đến tài liệu được cung cấp, hãy ưu tiên sử dụng thông tin từ các tài liệu đó để trả lời. 
Nếu không tìm thấy thông tin liên quan trong tài liệu hoặc câu hỏi là về kiến thức chung, bạn có thể sử dụng kiến thức riêng để trả lời.

Đối với câu hỏi về các quy định cụ thể, hãy chỉ dựa vào thông tin trong tài liệu được cung cấp.
Đối với câu hỏi kiến thức chung không liên quan đến tài liệu, hãy trả lời dựa trên hiểu biết của bạn."""
```

Bạn có thể thay đổi hướng dẫn này để:
- Yêu cầu model chỉ sử dụng thông tin từ tài liệu (strict RAG)
- Cho phép model sử dụng kiến thức riêng trong mọi trường hợp
- Tùy chỉnh cách model phân biệt giữa câu hỏi cần thông tin từ tài liệu và câu hỏi kiến thức chung

### Thay đổi mô hình embedding

Mặc định, dự án sử dụng mô hình `text-embedding-nomic-embed-text-v1.5-embedding` từ LM Studio. Bạn có thể thay đổi mô hình trong file `document_processor.py`:

```python
payload = {
    "input": text,
    "model": "text-embedding-nomic-embed-text-v1.5-embedding"  # Thay đổi mô hình ở đây
}
```

### Thay đổi mô hình sinh câu trả lời

Mặc định, dự án sử dụng mô hình `gemma-3-12b-it`, nhưng bạn có thể dễ dàng chuyển sang mô hình khác bằng cách sử dụng tham số `--model_name`:

```bash
python main.py interactive --model_name "mistral-7b-instruct-v0.2" --lm_studio_url http://127.0.0.1:1234
```

### Thay đổi prompt template

Bạn có thể tùy chỉnh template cho prompt trong file `document_query.py`.

## Xử lý lỗi phổ biến

### Không kết nối được đến LM Studio

Đảm bảo:
- LM Studio đã được cài đặt và đang chạy
- Server API trong LM Studio đã được khởi động
- URL API đúng (mặc định: `http://127.0.0.1:1234`)
- Mô hình được chọn trong LM Studio trùng khớp với tham số `--model_name`

### Không tìm thấy tài liệu

Kiểm tra:
- Thư mục `docs` chứa các file text (`.txt`)
- Encoding của file là UTF-8
- Vector database đã được tạo bằng lệnh `python main.py create`

### Lỗi khi trích xuất câu trả lời

Kiểm tra:
- Mô hình được chọn có hỗ trợ chat completions API
- Định dạng prompt phù hợp với mô hình

## Liên hệ

Nếu bạn có bất kỳ câu hỏi hoặc đề xuất nào, vui lòng tạo issue hoặc liên hệ với tôi qua [email](mailto:hoangkhoi230@gmail.com). 