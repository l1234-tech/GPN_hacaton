# baseline/ocr_utils.py
from __future__ import annotations

import os
import re
from functools import lru_cache

import fitz
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from paddleocr import PaddleOCR

from layout_utils import PageItem


os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


@lru_cache(maxsize=1)
def get_ocr_engine() -> PaddleOCR:
    return PaddleOCR(
        use_angle_cls=True, 
        lang="ru,en",
        show_log=False,
        det_db_thresh=0.3,
        det_db_box_thresh=0.5,
        rec_batch_num=4,
    )


def normalize_ocr_text(text: str) -> str:
    """Нормализация OCR текста."""
    if not text:
        return ""
    
    text = text.replace("\xa0", " ")
    text = re.sub(r"[«»]", '"', text)
    text = re.sub(r"[–—]", "-", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?%)\]])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    
    return text.strip()


def page_to_numpy(page: fitz.Page, scale: float = 2.0) -> np.ndarray:
    """Конвертация страницы в numpy array."""
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    image = _preprocess_image_for_ocr(image)
    return np.array(image)


def _preprocess_image_for_ocr(image: Image.Image) -> Image.Image:
    """Предобработка для улучшения OCR."""
    if image.mode != 'L':
        image = image.convert('L')
    
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.8)
    
    return image


def _is_garbage_text(text: str) -> bool:
    """Проверка, является ли текст мусором."""
    if not text or len(text) < 3:
        return True
    
    # Только цифры и спецсимволы
    if re.match(r'^[0-9\s\-\.\,°²³\*\+\=\:\;\|\/\(\)]+$', text):
        return True
    
    # Много спецсимволов
    special_count = sum(1 for c in text if not c.isalnum() and c not in ' -')
    if special_count / len(text) > 0.35:
        return True
    
    # Повторяющиеся символы
    if re.search(r'(.)\1{5,}', text):
        return True
    
    # Почти без гласных
    letters = [c for c in text.lower() if c.isalpha()]
    if letters and len(letters) > 3:
        vowels = sum(1 for c in letters if c in 'aeiouаеёиоуыэюя')
        if vowels / len(letters) < 0.20:
            return True
    
    return False


def _is_watermark_text(text: str) -> bool:
    """Проверка водяных знаков."""
    normalized = text.lower().strip()
    
    markers = {
        'draft', 'черновик', 'confidential', 'sample', 'proof', 'watermark',
        'копия', 'copy', 'internal', 'restricted'
    }
    
    words = set(normalized.split())
    if any(marker in words for marker in markers):
        return True
    
    patterns = [
        r'^стр\.\s*\d+$',
        r'^страница\s*\d+$', 
        r'^page\s*\d+$',
        r'^\d+\s*/\s*\d+$',
    ]
    
    for pattern in patterns:
        if re.match(pattern, normalized, re.IGNORECASE):
            return True
    
    return False


def ocr_page_items(page: fitz.Page, min_confidence: float = 0.70) -> list[PageItem]:
    """OCR с жесткой фильтрацией."""
    try:
        image = page_to_numpy(page, scale=2.5)
        result = get_ocr_engine().ocr(image, cls=True)
    except Exception:
        return []

    lines = []
    scale_x = page.rect.width / image.shape[1]
    scale_y = page.rect.height / image.shape[0]

    for block in result:
        for line in block:
            points, payload = line
            text = normalize_ocr_text(payload[0])
            confidence = payload[1]
            
            # Жесткая фильтрация
            if len(text) < 3 or confidence < min_confidence:
                continue
            if _is_garbage_text(text):
                continue
            if _is_watermark_text(text):
                continue
            
            xs = [point[0] * scale_x for point in points]
            ys = [point[1] * scale_y for point in points]
            bbox = (min(xs), min(ys), max(xs), max(ys))
            lines.append((bbox, text))

    # Объединение в параграфы
    if not lines:
        return []
    
    lines.sort(key=lambda x: (x[0][1], x[0][0]))
    
    merged = []
    current_bbox = None
    current_lines = []
    
    for bbox, text in lines:
        if current_bbox is None:
            current_bbox = bbox
            current_lines = [text]
            continue
        
        _, prev_y1, _, _ = current_bbox
        _, curr_y0, _, _ = bbox
        gap = curr_y0 - prev_y1
        
        if gap < 12:
            current_lines.append(text)
            current_bbox = (
                min(current_bbox[0], bbox[0]),
                min(current_bbox[1], bbox[1]),
                max(current_bbox[2], bbox[2]),
                max(current_bbox[3], bbox[3]),
            )
        else:
            content = "\n".join(current_lines)
            merged.append(PageItem(kind="text", bbox=current_bbox, content=content))
            current_bbox = bbox
            current_lines = [text]
    
    if current_bbox and current_lines:
        content = "\n".join(current_lines)
        merged.append(PageItem(kind="text", bbox=current_bbox, content=content))
    
    return merged


def is_page_mostly_scanned(page: fitz.Page) -> bool:
    """Проверка, отсканирована ли страница."""
    blocks = page.get_text("dict").get("blocks", [])
    text_chars = 0
    
    for block in blocks:
        if block.get("type") == 0:
            text = ""
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text += span.get("text", "")
            text_chars += len(text)
    
    return text_chars < 100
