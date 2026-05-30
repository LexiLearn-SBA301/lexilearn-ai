import os
import sys
import argparse

# Reconfigure stdout to handle Vietnamese characters properly in Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Add src folder to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from core.pdf_reader import PDFReader
from core.structure_detector import StructureDetector
from core.semantic_chunker import SemanticChunker
from core.chunk_validator import ChunkValidator

def run_integration_pipeline(pdf_path: str, limit_print: int = 15):
    if not os.path.exists(pdf_path):
        print(f"Error: File không tồn tại tại đường dẫn: {pdf_path}")
        return

    print("=" * 80)
    print(f"BẮT ĐẦU CHẠY PIPELINE CHO TỆP: {os.path.basename(pdf_path)}")
    print("=" * 80)

    # 1. Đọc PDF
    print("\n[Bước 1/4] Đang đọc file PDF và trích xuất bố cục văn bản...")
    reader = PDFReader()
    elements = reader.read(pdf_path)
    print(f"-> Đã trích xuất thành công: {len(elements)} phần tử (paragraphs, tables, headings).")

    # 2. Nhận diện cấu trúc phân cấp
    print("\n[Bước 2/4] Đang xây dựng cây đề mục phân cấp...")
    detector = StructureDetector()
    sections = detector.detect(elements)
    print(f"-> Đã nhận diện được: {len(sections)} phân mục chính/phụ.")

    # 3. Chia tách ngữ nghĩa
    print("\n[Bước 3/4] Đang thực hiện Semantic Chunking...")
    chunker = SemanticChunker()
    chunks = chunker.chunk(sections)
    print(f"-> Đã sinh ra: {len(chunks)} chunks ngữ nghĩa.")

    # 4. Kiểm duyệt và chấm điểm chất lượng
    print("\n[Bước 4/4] Đang kiểm duyệt chất lượng từng chunk...")
    validator = ChunkValidator()
    validated_chunks = validator.validate(chunks)
    
    passed_chunks = [vc for vc in validated_chunks if vc.validation.passed]
    failed_chunks = [vc for vc in validated_chunks if not vc.validation.passed]
    print(f"-> Kiểm duyệt hoàn tất:")
    print(f"   - Số chunk ĐẠT YÊU CẦU: {len(passed_chunks)} ({len(passed_chunks)/len(chunks)*100:.1f}%)")
    print(f"   - Số chunk BỊ TỪ CHỐI: {len(failed_chunks)} ({len(failed_chunks)/len(chunks)*100:.1f}%)")

    # --- Thống kê chất lượng ---
    print("\n" + "=" * 40 + " THỐNG KÊ CHI TIẾT " + "=" * 40)
    scores = [vc.validation.quality_score for vc in validated_chunks]
    avg_score = sum(scores) / len(scores) if scores else 0
    print(f"  * Điểm chất lượng trung bình: {avg_score:.2f} / 100")
    print(f"  * Điểm cao nhất: {max(scores):.2f} | Điểm thấp nhất: {min(scores):.2f}")

    # Đếm phân bổ kiểu nội dung
    ct_counts = {}
    for vc in validated_chunks:
        ct = vc.chunk.content_type
        ct_counts[ct] = ct_counts.get(ct, 0) + 1
    
    print("\n  * Phân bổ kiểu nội dung:")
    for ct, count in sorted(ct_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"    - {ct:<12}: {count} chunks")

    # Đếm phân bổ lỗi & cảnh báo
    errors_dist = {}
    warnings_dist = {}
    for vc in validated_chunks:
        for err in vc.validation.errors:
            errors_dist[err] = errors_dist.get(err, 0) + 1
        for warn in vc.validation.warnings:
            warnings_dist[warn] = warnings_dist.get(warn, 0) + 1

    print("\n  * Thống kê Lỗi (khiến chunk bị từ chối):")
    if errors_dist:
        for err, count in sorted(errors_dist.items(), key=lambda x: x[1], reverse=True):
            print(f"    - [LỖI] {err:<30}: {count} lần")
    else:
        print("    - Không phát hiện lỗi nghiêm trọng nào.")

    print("\n  * Thống kê Cảnh báo (trừ điểm chất lượng):")
    if warnings_dist:
        for warn, count in sorted(warnings_dist.items(), key=lambda x: x[1], reverse=True):
            print(f"    - [CẢNH BÁO] {warn:<35}: {count} lần")
    else:
        print("    - Không có cảnh báo nào.")

    # --- Hiển thị trực quan dữ liệu chunk ---
    print("\n" + "=" * 36 + f" HIỂN THỊ MẪU {limit_print} CHUNKS ĐẠT CHẤT LƯỢNG " + "=" * 36)
    
    for idx, vc in enumerate(passed_chunks[:limit_print]):
        c = vc.chunk
        v = vc.validation
        print(f"\n[CHUNK #{idx + 1:02d}] ID: {c.chunk_id}")
        print(f"  * Đề mục   : {c.title}")
        print(f"  * Thuộc mục: {c.section_title} > {c.subsection_title or 'N/A'}")
        print(f"  * Kiểu     : {c.content_type:<10} | Trang: {c.page_start}-{c.page_end} | Token: {c.token_count} | Điểm: {v.quality_score:.1f}")
        print(f"  * Nhãn tags: {c.tags}")
        if c.has_overlap:
            print(f"  * Overlap  : Có (Kế thừa từ chunk: {c.overlap_from_chunk})")
        if v.warnings:
            print(f"  * Cảnh báo : {v.warnings}")
        
        # In nội dung chunk (được thụt lề và bọc trong khung)
        print("  * Nội dung :")
        content_lines = c.content.strip().split("\n")
        # Chỉ in tối đa 8 dòng đầu để tránh làm đầy màn hình
        max_lines_to_show = 8
        for line_idx, line in enumerate(content_lines[:max_lines_to_show]):
            print(f"      | {line}")
        if len(content_lines) > max_lines_to_show:
            print(f"      | ... (còn tiếp {len(content_lines) - max_lines_to_show} dòng)")
        print("-" * 90)

    if len(passed_chunks) > limit_print:
        print(f"\n... Đã ẩn {len(passed_chunks) - limit_print} chunks hợp lệ khác để đảm bảo màn hình gọn gàng.")
    
    print("\n" + "=" * 80)
    print("HOÀN THÀNH KIỂM TRA PIPELINE.")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy thử nghiệm tích hợp toàn bộ pipeline tiền xử lý RAG Ngữ Văn.")
    parser.add_argument(
        "--pdf", 
        type=str, 
        default="d:/Project/SBA/rag-service/docs/sach-giao-khoa-ngu-van-12-tap-2-co-ban.pdf",
        help="Đường dẫn tới file PDF cần xử lý"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=10,
        help="Số lượng chunks tối đa muốn hiển thị nội dung chi tiết trên màn hình"
    )
    args = parser.parse_args()
    
    # Kiểm tra đường dẫn mặc định, nếu không tìm thấy thử quét thư mục docs để tìm file PDF đầu tiên
    pdf_to_run = args.pdf
    if not os.path.exists(pdf_to_run):
        docs_dir = "d:/Project/SBA/rag-service/docs"
        if os.path.exists(docs_dir):
            pdf_files = [os.path.join(docs_dir, f) for f in os.listdir(docs_dir) if f.endswith(".pdf")]
            if pdf_files:
                pdf_to_run = pdf_files[0]
                print(f"Lưu ý: Không tìm thấy file mặc định. Tự động chọn file PDF đầu tiên tìm thấy: {pdf_to_run}")
    
    run_integration_pipeline(pdf_to_run, args.limit)
