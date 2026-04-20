# baseline/ocr_utils.py
from __future__ import annotations

import os
import re
from functools import lru_cache

import fitz
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

from .layout_utils import PageItem


os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


@lru_cache(maxsize=1)
def get_ocr_engine():
    """Ленивая инициализация OCR движка."""
    from paddleocr import PaddleOCR
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
    
    # Нормализация пробелов
    text = re.sub(r"\s+", " ", text)
    
    # Исправление пунктуации
    text = re.sub(r"\s+([,.;:!?%)\]])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    
    return text.strip()


def _fix_common_ocr_errors(text: str) -> str:
    """Исправление типичных ошибок OCR."""
    # Цифры и буквы
    replacements = {
        '0': 'O', '1': 'I', '2': 'Z', '3': 'B', '4': 'A',
        '5': 'S', '6': 'G', '7': 'T', '8': 'B', '9': 'g',
        'l': 'I', 'I': 'l',  # Зависит от контекста
    }
    
    # Простые замены (осторожно, чтобы не испортить правильный текст)
    for old, new in replacements.items():
        # Заменяем только если это изолированный символ
        text = re.sub(rf'\b{re.escape(old)}\b', new, text)
    
    return text


def page_to_numpy(page: fitz.Page, scale: float = 2.0, enhance: bool = True) -> np.ndarray:
    """Улучшенная конвертация страницы в numpy array с предобработкой."""
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    if enhance:
        image = _preprocess_image_for_ocr(image)
    
    return np.array(image)


def _preprocess_image_for_ocr(image: Image.Image) -> Image.Image:
    """Предобработка изображения для улучшения OCR."""
    # Конвертация в grayscale
    if image.mode != 'L':
        image = image.convert('L')
    
    # Увеличение контрастности
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)
    
    # Увеличение резкости
    image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
    
    # Бинаризация (пороговая обработка)
    image = image.point(lambda x: 0 if x < 128 else 255, 'L')
    
    return image


def ocr_page_items(page: fitz.Page, min_confidence: float = 0.4) -> list[PageItem]:
    """Улучшенное OCR с фильтрацией и объединением."""
    try:
        image = page_to_numpy(page, scale=2.5)  # Высокое разрешение для текста
        result = get_ocr_engine().ocr(image, cls=True)
    except Exception as e:
        print(f"OCR failed: {e}")
        return []

    lines: list[tuple[tuple[float, float, float, float], str, float]] = []
    scale_x = page.rect.width / image.shape[1]
    scale_y = page.rect.height / image.shape[0]

    for block in result:
        for line in block:
            points, payload = line
            text = normalize_ocr_text(payload[0])
            confidence = payload[1]
            
            if len(text) < 2 or confidence < min_confidence:
                continue
            
            # Фильтрация водяных знаков
            if _is_watermark_text(text):
                continue
            
            xs = [point[0] * scale_x for point in points]
            ys = [point[1] * scale_y for point in points]
            bbox = (min(xs), min(ys), max(xs), max(ys))
            lines.append((bbox, text, confidence))

    # Сортировка по позиции
    lines.sort(key=lambda item: (item[0][1], item[0][0]))
    
    # Умное объединение в параграфы
    merged = _merge_ocr_lines_into_paragraphs(lines, page.rect.width)
    
    return merged


def _is_watermark_text(text: str) -> bool:
    """Определение, является ли текст водяным знаком."""
    normalized = text.lower().strip()
    
    # Водяные знаки
    watermark_markers = {
        'draft', 'черновик', 'confidential', 'sample', 'proof', 'watermark',
        'копия', 'copy', 'internal', 'confidential', 'restricted'
    }
    
    # Паттерны страниц
    page_patterns = [
        r'^стр\.\s*\d+$',
        r'^страница\s*\d+$', 
        r'^page\s*\d+$',
        r'^\d+\s*/\s*\d+$',  # 1 / 10
    ]
    
    # Проверка маркеров
    words = set(normalized.split())
    if any(marker in words for marker in watermark_markers):
        return True
    
    # Проверка паттернов
    for pattern in page_patterns:
        if re.match(pattern, normalized, re.IGNORECASE):
            return True
    
    return False


def _merge_ocr_lines_into_paragraphs(
    lines: list[tuple[tuple[float, float, float, float], str, float]], 
    page_width: float
) -> list[PageItem]:
    """Умное объединение OCR строк в параграфы."""
    if not lines:
        return []
    
    merged: list[PageItem] = []
    current_bbox: tuple[float, float, float, float] | None = None
    current_lines: list[str] = []
    current_confidences: list[float] = []
    
    for bbox, text, confidence in lines:
        if current_bbox is None:
            current_bbox = bbox
            current_lines = [text]
            current_confidences = [confidence]
            continue

        # Определение, принадлежит ли строка текущему параграфу
        _, prev_top, _, prev_bottom = current_bbox
        x0, y0, x1, y1 = bbox
        
        # Расстояние между строками
        gap = y0 - prev_bottom
        
        # Ширина текста
        text_width = x1 - x0
        is_full_width = text_width > page_width * 0.7
        
        # Логика объединения
        same_paragraph = (
            gap < 15 or  # Маленький разрыв
            (gap < 25 and is_full_width) or  # Полноширинный текст
            (gap < 20 and _lines_have_similar_alignment(current_bbox, bbox))  # Выровненные строки
        )
        
        if same_paragraph:
            current_lines.append(text)
            current_confidences.append(confidence)
            current_bbox = (
                min(current_bbox[0], x0),
                min(current_bbox[1], y0),
                max(current_bbox[2], x1),
                max(current_bbox[3], y1),
            )
        else:
            # Создаем PageItem из накопленных строк
            avg_confidence = sum(current_confidences) / len(current_confidences)
            content = "\n".join(current_lines)
            
            merged.append(PageItem(
                kind="text", 
                bbox=current_bbox, 
                content=content
            ))
            
            # Начинаем новый параграф
            current_bbox = bbox
            current_lines = [text]
            current_confidences = [confidence]

    # Добавляем последний параграф
    if current_bbox is not None and current_lines:
        content = "\n".join(current_lines)
        merged.append(PageItem(kind="text", bbox=current_bbox, content=content))

    return merged


def _lines_have_similar_alignment(
    bbox1: tuple[float, float, float, float], 
    bbox2: tuple[float, float, float, float]
) -> bool:
    """Проверка схожести выравнивания строк."""
    x0_1, _, x1_1, _ = bbox1
    x0_2, _, x1_2, _ = bbox2
    
    # Левый край
    left_diff = abs(x0_1 - x0_2)
    # Ширина
    width_diff = abs((x1_1 - x0_1) - (x1_2 - x0_2))
    
    return left_diff < 20 and width_diff < 50


def is_page_mostly_scanned(page: fitz.Page) -> bool:
    """Определение, является ли страница преимущественно отсканированной."""
    blocks = page.get_text("dict").get("blocks", [])
    
    text_chars = 0
    total_chars = 0
    
    for block in blocks:
        if block.get("type") == 0:  # Текстовый блок
            text = ""
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text += span.get("text", "")
            text_chars += len(text)
        elif block.get("type") == 1:  # Изображение
            # Изображения могут содержать текст
            pass
    
    # Если очень мало текста, считаем страницу отсканированной
    return text_chars < 100
    # Увеличение резкости
    image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
    
    # Бинаризация (пороговая обработка)
    image = image.point(lambda x: 0 if x < 128 else 255, 'L')
    
    return image


def ocr_page_items(page: fitz.Page, min_confidence: float = 0.4) -> list[PageItem]:
    """Улучшенное OCR с фильтрацией и объединением."""
    try:
        image = page_to_numpy(page, scale=2.5)  # Высокое разрешение для текста
        result = get_ocr_engine().ocr(image, cls=True)
    except Exception as e:
        print(f"OCR failed: {e}")
        return []

    lines: list[tuple[tuple[float, float, float, float], str, float]] = []
    scale_x = page.rect.width / image.shape[1]
    scale_y = page.rect.height / image.shape[0]

    for block in result:
        for line in block:
            points, payload = line
            text = normalize_ocr_text(payload[0])
            confidence = payload[1]
            
            if len(text) < 2 or confidence < min_confidence:
                continue
            
            # Фильтрация водяных знаков
            if _is_watermark_text(text):
                continue
            
            xs = [point[0] * scale_x for point in points]
            ys = [point[1] * scale_y for point in points]
            bbox = (min(xs), min(ys), max(xs), max(ys))
            lines.append((bbox, text, confidence))

    # Сортировка по позиции
    lines.sort(key=lambda item: (item[0][1], item[0][0]))
    
    # Умное объединение в параграфы
    merged = _merge_ocr_lines_into_paragraphs(lines, page.rect.width)
    
    return merged


def _is_watermark_text(text: str) -> bool:
    """Определение, является ли текст водяным знаком."""
    normalized = text.lower().strip()
    
    # Водяные знаки
    watermark_markers = {
        'draft', 'черновик', 'confidential', 'sample', 'proof', 'watermark',
        'копия', 'copy', 'internal', 'confidential', 'restricted'
    }
    
    # Паттерны страниц
    page_patterns = [
        r'^стр\.\s*\d+$',
        r'^страница\s*\d+$', 
        r'^page\s*\d+$',
        r'^\d+\s*/\s*\d+$',  # 1 / 10
    ]
    
    # Проверка маркеров
    words = set(normalized.split())
    if any(marker in words for marker in watermark_markers):
        return True
    
    # Проверка паттернов
    for pattern in page_patterns:
        if re.match(pattern, normalized, re.IGNORECASE):
            return True
    
    return False


def _merge_ocr_lines_into_paragraphs(
    lines: list[tuple[tuple[float, float, float, float], str, float]], 
    page_width: float
) -> list[PageItem]:
    """Умное объединение OCR строк в параграфы."""
    if not lines:
        return []
    
    merged: list[PageItem] = []
    current_bbox: tuple[float, float, float, float] | None = None
    current_lines: list[str] = []
    current_confidences: list[float] = []
    
    for bbox, text, confidence in lines:
        if current_bbox is None:
            current_bbox = bbox
            current_lines = [text]
            current_confidences = [confidence]
            continue

        # Определение, принадлежит ли строка текущему параграфу
        _, prev_top, _, prev_bottom = current_bbox
        x0, y0, x1, y1 = bbox
        
        # Расстояние между строками
        gap = y0 - prev_bottom
        
        # Ширина текста
        text_width = x1 - x0
        is_full_width = text_width > page_width * 0.7
        
        # Логика объединения
        same_paragraph = (
            gap < 15 or  # Маленький разрыв
            (gap < 25 and is_full_width) or  # Полноширинный текст
            (gap < 20 and _lines_have_similar_alignment(current_bbox, bbox))  # Выровненные строки
        )
        
        if same_paragraph:
            current_lines.append(text)
            current_confidences.append(confidence)
            current_bbox = (
                min(current_bbox[0], x0),
                min(current_bbox[1], y0),
                max(current_bbox[2], x1),
                max(current_bbox[3], y1),
            )
        else:
            # Создаем PageItem из накопленных строк
            avg_confidence = sum(current_confidences) / len(current_confidences)
            content = "\n".join(current_lines)
            
            merged.append(PageItem(
                kind="text", 
                bbox=current_bbox, 
                content=content
            ))
            
            # Начинаем новый параграф
            current_bbox = bbox
            current_lines = [text]
            current_confidences = [confidence]

    # Добавляем последний параграф
    if current_bbox is not None and current_lines:
        content = "\n".join(current_lines)
        merged.append(PageItem(kind="text", bbox=current_bbox, content=content))

    return merged


def _lines_have_similar_alignment(
    bbox1: tuple[float, float, float, float], 
    bbox2: tuple[float, float, float, float]
) -> bool:
    """Проверка схожести выравнивания строк."""
    x0_1, _, x1_1, _ = bbox1
    x0_2, _, x1_2, _ = bbox2
    
    # Левый край
    left_diff = abs(x0_1 - x0_2)
    # Ширина
    width_diff = abs((x1_1 - x0_1) - (x1_2 - x0_2))
    
    return left_diff < 20 and width_diff < 50


def is_page_mostly_scanned(page: fitz.Page) -> bool:
    """Определение, является ли страница преимущественно отсканированной."""
    blocks = page.get_text("dict").get("blocks", [])
    
    text_chars = 0
    total_chars = 0
    
    for block in blocks:
        if block.get("type") == 0:  # Текстовый блок
            text = ""
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text += span.get("text", "")
            text_chars += len(text)
        elif block.get("type") == 1:  # Изображение
            # Изображения могут содержать текст
            pass
    
    # Если очень мало текста, считаем страницу отсканированной
    return text_chars < 100