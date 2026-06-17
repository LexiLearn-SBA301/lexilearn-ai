import os
import pdfplumber
import pytest

def test_pdfplumber_image_conversion():
    pdf_path = r"docs/sach-giao-khoa-ngu-van-10-tap-1-co-ban.pdf"
    if not os.path.exists(pdf_path):
        pytest.skip(f"Test PDF not found at {pdf_path}")
        
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[5]  # page 6
        img = page.to_image(resolution=150).original
        assert img is not None
        assert img.size[0] > 0 and img.size[1] > 0
