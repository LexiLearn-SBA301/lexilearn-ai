import os
import sys
import numpy as np
from PIL import Image
import pdfplumber
from dotenv import load_dotenv
import pytest

# Load environment
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(dotenv_path=os.path.join(project_root, ".env"))
DEEPDOC_ROOT = os.getenv("DEEPDOC_PATH")

@pytest.mark.skipif(not DEEPDOC_ROOT or not os.path.exists(DEEPDOC_ROOT), reason="DEEPDOC_PATH not configured or directory does not exist")
def test_deepdoc_vietocr_integration():
    if DEEPDOC_ROOT not in sys.path:
        sys.path.insert(0, DEEPDOC_ROOT)

    from module import LayoutRecognizer, OCR

    # Force working directory for deepdoc path resolving
    original_cwd = os.getcwd()
    os.chdir(DEEPDOC_ROOT)
    try:
        layout_recognizer = LayoutRecognizer("layout")
        ocr = OCR()
    finally:
        os.chdir(original_cwd)

    pdf_path = os.path.join(project_root, "docs", "sach-giao-khoa-ngu-van-10-tap-1-co-ban.pdf")
    if not os.path.exists(pdf_path):
        pytest.skip(f"Test PDF not found at {pdf_path}")

    with pdfplumber.open(pdf_path) as pdf:
        img = pdf.pages[5].to_image(resolution=300).original

    # Test layout recognition
    layouts = layout_recognizer.forward([img], thr=0.5)[0]
    assert len(layouts) > 0, "No layout regions detected"

    # Test OCR recognition
    ocr_results = ocr(np.array(img))
    assert len(ocr_results) > 0, "OCR extracted 0 text boxes"
    
    # Check if first text box contains part of the heading "TỔNG QUAN VĂN HỌC VIỆT NAM"
    first_text = ocr_results[0][1][0]
    assert "TỔNG QUAN" in first_text or "VĂN HỌC" in first_text
