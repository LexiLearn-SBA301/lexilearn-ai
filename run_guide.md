# Hướng dẫn Vận hành Hệ thống RAG & Kiểm tra Dữ liệu MongoDB

Tài liệu này cung cấp toàn bộ các câu lệnh cần thiết để vận hành luồng hoạt động của hệ thống RAG Văn học Việt Nam (Nạp dữ liệu $\rightarrow$ Truy vấn $\rightarrow$ Đánh giá) và các phương pháp kiểm tra dữ liệu lưu trữ trong MongoDB.

---

## I. Chuẩn bị Môi trường

Đảm bảo bạn đã kích hoạt môi trường ảo Python và khởi động các dịch vụ phụ trợ:

1. **Kích hoạt môi trường ảo (PowerShell / CMD):**
   ```powershell
   # PowerShell
   .venv\Scripts\Activate.ps1
   
   # CMD
   .venv\Scripts\activate.bat
   ```
2. **Kiểm tra dịch vụ cục bộ:**
   * **MongoDB:** Đảm bảo MongoDB đang chạy tại địa chỉ mặc định `mongodb://localhost:27017` (bạn có thể kiểm tra qua dịch vụ Windows Services hoặc mở MongoDB Compass).
   * **Ollama:** Đảm bảo Ollama đang khởi chạy và mô hình sinh câu trả lời `qwen2.5:3b` đã được tải xuống (`ollama pull qwen2.5:3b`).

3. **Cài đặt Tesseract OCR và Poppler (Cho tính năng nạp PDF quét ảnh):**
   * Hệ thống yêu cầu cài đặt phần mềm bên ngoài cho hệ điều hành Windows để hỗ trợ nạp các tệp PDF dạng ảnh quét:
     * **Tesseract OCR**: Tải từ [UB-Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki). Lúc cài đặt phải chọn bổ sung gói ngôn ngữ `Vietnamese`. (Cập nhật đường dẫn vào `TESSERACT_CMD` trong `.env` nếu cài khác mặc định `C:\Program Files\Tesseract-OCR\tesseract.exe`).
     * **Poppler**: Tải bản pre-built cho Windows từ [poppler-windows releases](https://github.com/oschwartz10612/poppler-windows/releases). Giải nén và cập nhật đường dẫn tới thư mục `bin` vào biến `POPPLER_PATH` trong `.env`.

## II. Các Câu Lệnh Chạy Luồng Hoạt Động (CLI)

Chúng ta sử dụng tệp `main.py` làm trung tâm định tuyến dòng lệnh.

### 1. Luồng Nạp Dữ Liệu (Ingestion Pipeline)
Lệnh này quét thư mục chứa các tệp sách giáo khoa dạng PDF, chạy qua pipeline xử lý cấu trúc $\rightarrow$ chia đoạn (chunking) $\rightarrow$ sinh vector nhúng (embedding) $\rightarrow$ ghi vào MongoDB.

* **Nạp toàn bộ thư mục sách giáo khoa mặc định (thư mục `docs`):**
  ```powershell
  python main.py --ingest --pdf docs
  ```
* **Nạp một tệp PDF đơn lẻ cụ thể:**
  ```powershell
  python main.py --ingest --pdf docs/sach-giao-khoa-ngu-van-12-tap-1-co-ban.pdf
  ```
  *(Lưu ý: CLI sẽ hiển thị tiến trình nạp thời gian thực tự động cập nhật từ MongoDB).*

### 2. Luồng Hỏi Đáp RAG (Query & Synthesis)
Gửi câu hỏi trực tiếp để hệ thống thực hiện tìm kiếm lai (Hybrid Search), kiểm duyệt bảo mật (`InjectionGuard`) và gọi LLM tạo câu trả lời.

* **Truy vấn tự do không bộ lọc:**
  ```powershell
  python main.py --query "Hình ảnh người lính Tây Tiến trong bài thơ Tây Tiến của Quang Dũng hiện lên như thế nào?"
  ```
* **Truy vấn kết hợp bộ lọc Lớp học (`--lop`) và Tác phẩm (`--work`):**
  ```powershell
  python main.py --query "Phân tích tâm trạng nhân vật Tràng khi nhặt được vợ" --lop 12 --work "Vợ Nhặt"
  ```

### 3. Luồng Đánh Giá Hiệu Năng (Evaluation Suite)
Tự động gửi câu hỏi từ tệp `ground_truth.json` (100 câu hỏi văn học đã biên soạn), so khớp kết quả trích xuất và tính toán điểm số **Hit Rate @ 5** và **MRR**:

```powershell
python main.py --evaluate
```

---

## III. Hướng Dẫn Kiểm Tra Dữ Liệu Trong MongoDB

Bạn có 2 cách để kiểm tra dữ liệu hiện có trong Database:

### Cách 1: Sử dụng MongoDB Compass hoặc MongoUI (Giao diện đồ họa trực quan)
1. Khởi động phần mềm **MongoDB Compass** hoặc **MongoUI** của bạn.
2. Kết nối tới chuỗi Connection String mặc định: `mongodb://localhost:27017`
3. Tìm và chọn cơ sở dữ liệu mang tên **`rag_db`**.
4. Truy cập 2 collection chính để xem dữ liệu:
   * **`document_chunks`**: Chứa dữ liệu của các đoạn văn bản trích xuất:
     * `chunk_id`: Mã định danh duy nhất (ví dụ: `sach-giao-khoa-ngu-van-12..._p003_c02`).
     * `content`: Nội dung văn bản tiếng Việt của chunk.
     * `embedding`: Mảng số thực 1024 chiều biểu diễn ngữ nghĩa.
     * `search_text`: Văn bản đã loại bỏ dấu tiếng Việt để phục vụ Full-text Search.
     * `metadata`: Chứa tên tác phẩm, tác giả, lớp, học kì...
     * `is_active`: Trạng thái active (`true`/`false`).
   * **`ingestion_jobs`**: Chứa lịch sử các lần chạy nạp dữ liệu (job_id, status: pending/running/done/error, processed_files, errors list).

---

### Cách 2: Sử dụng Command Line qua MongoDB Shell (`mongosh`)
Nếu bạn thích dùng dòng lệnh, hãy mở CMD/PowerShell và thực thi các lệnh sau:

1. **Khởi động Mongo Shell:**
   ```powershell
   mongosh
   ```
2. **Chọn cơ sở dữ liệu:**
   ```javascript
   use rag_db
   ```
3. **Hiển thị danh sách các collection:**
   ```javascript
   show collections
   ```
4. **Đếm tổng số chunk đang hoạt động (active):**
   ```javascript
   db.document_chunks.countDocuments({ is_active: true })
   ```
5. **Đếm số lượng chunk của một tác phẩm cụ thể:**
   ```javascript
   db.document_chunks.countDocuments({ "metadata.ten_tac_pham": "Tây Tiến", is_active: true })
   ```
6. **Truy vấn 1 chunk mẫu đầu tiên (ẩn vector `embedding` để tránh dài màn hình):**
   ```javascript
   db.document_chunks.findOne({ is_active: true }, { embedding: 0 })
   ```
7. **Xem danh sách các Ingestion Job bị lỗi:**
   ```javascript
   db.ingestion_jobs.find({ status: "error" })
   ```
8. **Thoát shell:**
   ```javascript
   exit
   ```
