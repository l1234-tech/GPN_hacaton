# baseline/layout_utils.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import fitz


EDGE_MARGIN = 48


@dataclass(frozen=True)
class PageItem:
    kind: str
    bbox: tuple[float, float, float, float]
    content: str


def bbox_area(bbox: tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = bbox
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def intersection_area(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    lx0, ly0, lx1, ly1 = left
    rx0, ry0, rx1, ry1 = right
    x0 = max(lx0, rx0)
    y0 = max(ly0, ry0)
    x1 = min(lx1, rx1)
    y1 = min(ly1, ry1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


def overlaps(
    bbox: tuple[float, float, float, float],
    others: Iterable[tuple[float, float, float, float]],
    threshold: float,
) -> bool:
    area = bbox_area(bbox)
    if area <= 0:
        return False
    for other in others:
        if intersection_area(bbox, other) / area >= threshold:
            return True
    return False


def block_max_font_size(block: dict) -> float:
    return max(
        (
            span.get("size", 0.0)
            for line in block.get("lines", [])
            for span in line.get("spans", [])
        ),
        default=0.0,
    )


def clean_inline_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+([,.;:!?%)\]])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    return text.strip()


def join_text_lines(lines: list[str]) -> str:
    if not lines:
        return ""

    merged: list[str] = [lines[0].strip()]
    for raw_line in lines[1:]:
        line = raw_line.strip()
        if not line:
            continue

        prev = merged[-1].rstrip()
        if re.search(r"[\wа-яёА-ЯЁ]-$", prev) and re.match(r"^[\wа-яёА-ЯЁ]", line):
            merged[-1] = prev[:-1] + line
            continue

        if re.search(r"[A-Za-zА-Яа-яЁё0-9]$", prev) and re.match(r"^[a-zа-яё0-9]", line):
            merged[-1] = f"{prev} {line}"
            continue

        if re.search(r"[,:;]$", prev):
            merged[-1] = f"{prev} {line}"
            continue

        merged.append(line)

    return "\n".join(clean_inline_text(part) for part in merged if part.strip())


def extract_block_text(block: dict) -> str:
    lines: list[str] = []
    for line in block.get("lines", []):
        spans = [span.get("text", "") for span in line.get("spans", [])]
        text = clean_inline_text("".join(spans))
        if text:
            lines.append(text)
    return join_text_lines(lines)


def is_rotated_text_block(block: dict) -> bool:
    for line in block.get("lines", []):
        direction = line.get("dir", (1.0, 0.0))
        if abs(direction[0] - 1.0) > 0.03 or abs(direction[1]) > 0.03:
            return True
    return False


def normalize_repeat_key(text: str) -> str:
    text = text.lower().replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\bстр\.\s*\d+\b", "", text)
    text = re.sub(r"\bpage\s*\d+\b", "", text)
    text = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", "", text)
    text = re.sub(r"\b\d{1,4}\b", "", text)
    text = re.sub(r"\[\s*\d+\s*\]$", "", text)
    return text.strip(" .-")


def collect_repeated_margin_texts(doc: fitz.Document) -> set[str]:
    occurrences: dict[str, set[int]] = {}

    for page_index, page in enumerate(doc):
        blocks = page.get_text("dict").get("blocks", [])
        page_width = page.rect.width
        page_height = page.rect.height

        for block in blocks:
            if block.get("type") != 0:
                continue

            text = extract_block_text(block)
            if not text:
                continue

            x0, y0, x1, y1 = block["bbox"]
            in_margin = (
                y0 <= EDGE_MARGIN
                or y1 >= page_height - EDGE_MARGIN
                or x0 <= EDGE_MARGIN / 2
                or x1 >= page_width - EDGE_MARGIN / 2
            )
            if not in_margin:
                continue

            key = normalize_repeat_key(text)
            if len(key) < 6:
                continue
            occurrences.setdefault(key, set()).add(page_index)

    min_repeat_pages = 2 if doc.page_count < 4 else max(2, doc.page_count // 3)
    return {key for key, pages in occurrences.items() if len(pages) >= min_repeat_pages}


def is_probable_watermark(text: str) -> bool:
    normalized = text.lower()
    watermark_markers = (
        "draft",
        "черновик",
        "confidential",
        "sample",
        "proof",
        "watermark",
    )
    return any(marker in normalized for marker in watermark_markers)


def is_margin_text_block(
    block: dict,
    page_rect: fitz.Rect,
    repeated_margin_texts: set[str],
) -> bool:
    x0, y0, x1, y1 = block["bbox"]
    font_size = block_max_font_size(block)
    normalized = normalize_repeat_key(extract_block_text(block))

    if normalized in repeated_margin_texts:
        return True
    if y0 <= 36 and font_size < 14:
        return True
    if y1 >= page_rect.height - 24 and font_size < 12:
        return True
    if x1 >= page_rect.width - 18 and font_size < 12:
        return True
    return False


def detect_heading_level(text: str, block: dict, prev_block: dict = None) -> tuple[str, str] | None:
    """Улучшенное определение уровня заголовка с учетом контекста."""
    if not text or "\n" in text:
        return None
    if len(text) > 150:  # Увеличен лимит для длинных заголовков
        return None

    text_lower = text.lower().strip()
    
    # Исключения - не заголовки
    if any(phrase in text_lower for phrase in [
        'стр.', 'страница', 'page', 'рис.', 'табл.', 'table', 'figure'
    ]):
        return None
    
    max_size = block_max_font_size(block)
    font_flags = block.get('lines', [{}])[0].get('spans', [{}])[0].get('flags', 0)
    is_bold = bool(font_flags & 16)  # Проверка на жирный шрифт
    
    # Анализ предыдущего блока для контекста
    prev_size = block_max_font_size(prev_block) if prev_block else 0
    
    # Логика определения заголовков
    if max_size >= 24 or (max_size >= 20 and is_bold):
        return ("#", text)
    elif max_size >= 18 or (max_size >= 16 and is_bold and max_size > prev_size):
        return ("##", text)
    elif (max_size >= 14 and len(text) <= 100 and 
          (is_bold or max_size > prev_size + 2)):
        return ("###", text)
    elif (max_size >= 12 and len(text) <= 80 and 
          text[0].isupper() and not text.endswith('.')):
        return ("####", text)
    
    return None


def is_probable_watermark(text: str, block: dict, page_rect: fitz.Rect) -> bool:
    """Улучшенное определение водяных знаков."""
    normalized = text.lower().strip()
    
    # Явные маркеры водяных знаков
    watermark_markers = (
        "draft", "черновик", "confidential", "sample", "proof", "watermark",
        "копия", "copy", "internal", "restricted", "confidential", "do not distribute"
    )
    
    if any(marker in normalized for marker in watermark_markers):
        return True
    
    # Анализ позиции и размера
    x0, y0, x1, y1 = block["bbox"]
    font_size = block_max_font_size(block)
    
    # Водяные знаки часто в углах или по центру с маленьким шрифтом
    in_corner = (
        (x0 < 50 and y0 < 50) or  # Левый верхний
        (x1 > page_rect.width - 50 and y0 < 50) or  # Правый верхний
        (x0 < 50 and y1 > page_rect.height - 50) or  # Левый нижний
        (x1 > page_rect.width - 50 and y1 > page_rect.height - 50)  # Правый нижний
    )
    
    if in_corner and font_size < 12:
        return True
    
    # Повторяющийся текст в верхней/нижней части
    in_margin = y0 <= 60 or y1 >= page_rect.height - 60
    if in_margin and font_size < 14 and len(normalized) < 50:
        return True
    
    return False


def is_margin_text_block(
    block: dict,
    page_rect: fitz.Rect,
    repeated_margin_texts: set[str],
    margin_threshold: float = 48,
) -> bool:
    """Улучшенное определение текста в полях."""
    x0, y0, x1, y1 = block["bbox"]
    font_size = block_max_font_size(block)
    normalized = normalize_repeat_key(extract_block_text(block))

    # Повторяющийся текст
    if normalized in repeated_margin_texts:
        return True
    
    # Позиционные критерии
    in_vertical_margin = y0 <= margin_threshold or y1 >= page_rect.height - margin_threshold
    in_horizontal_margin = x0 <= margin_threshold / 2 or x1 >= page_rect.width - margin_threshold / 2
    
    # Размер шрифта
    small_font = font_size < 12
    
    # Короткий текст
    short_text = len(normalized) < 30
    
    if in_vertical_margin and (small_font or short_text):
        return True
    
    if in_horizontal_margin and small_font:
        return True
    
    return False


def page_is_mostly_raster(page: fitz.Page) -> bool:
    """Определение, является ли страница преимущественно растровой."""
    blocks = page.get_text("dict").get("blocks", [])
    text_chars = 0
    image_area = 0.0
    page_area = page.rect.width * page.rect.height

    for block in blocks:
        if block.get("type") == 0:  # Текстовый блок
            text_chars += len(extract_block_text(block))
        elif block.get("type") == 1:  # Изображение
            image_area += bbox_area(tuple(block["bbox"]))
        elif block.get("type") == 2:  # Векторная графика (может содержать текст)
            # Некоторые векторные объекты могут быть текстом
            pass

    # Если очень мало текста и много изображений - считаем растровой
    text_ratio = text_chars / max(page_area, 1.0)
    image_ratio = image_area / max(page_area, 1.0)
    
    return text_chars < 50 and image_ratio > 0.5


def order_page_items(items: list[PageItem], page_width: float) -> list[PageItem]:
    """Улучшенное упорядочивание элементов страницы."""
    if len(items) < 2:
        return sorted(items, key=lambda item: (round(item.bbox[1], 1), round(item.bbox[0], 1)))

    # Группировка по типам и позициям
    full_width: list[PageItem] = []
    left_column: list[PageItem] = []
    right_column: list[PageItem] = []
    center = page_width / 2
    
    # Анализ распределения элементов
    left_positions = []
    right_positions = []
    
    for item in items:
        x0, y0, x1, y1 = item.bbox
        width = x1 - x0
        
        if width > page_width * 0.75:  # Полноширинный элемент
            full_width.append(item)
        elif x1 <= center + 20:  # Левый столбец
            left_column.append(item)
            left_positions.append(x0)
        elif x0 >= center - 20:  # Правый столбец
            right_column.append(item)
            right_positions.append(x0)
        else:  # Неопределенная позиция
            full_width.append(item)
    
    # Определение, есть ли реальные столбцы
    has_columns = (
        len(left_column) >= 2 and len(right_column) >= 2 and
        len(left_positions) > 0 and len(right_positions) > 0 and
        max(left_positions) - min(left_positions) < 50 and  # Консистентное выравнивание
        max(right_positions) - min(right_positions) < 50
    )
    
    if not has_columns:
        # Одноколоночный layout
        return sorted(items, key=lambda item: (round(item.bbox[1], 1), round(item.bbox[0], 1)))
    
    # Многоколоночный layout
    full_width.sort(key=lambda item: (item.bbox[1], item.bbox[0]))
    left_column.sort(key=lambda item: (item.bbox[1], item.bbox[0]))
    right_column.sort(key=lambda item: (item.bbox[1], item.bbox[0]))
    
    ordered: list[PageItem] = []
    
    # Полноширинные элементы в начале
    ordered.extend([item for item in full_width if item.bbox[1] < min(
        left_column[0].bbox[1] if left_column else float('inf'),
        right_column[0].bbox[1] if right_column else float('inf')
    )])
    
    # Чередуем левый и правый столбцы
    left_idx = 0
    right_idx = 0
    
    while left_idx < len(left_column) or right_idx < len(right_column):
        # Добавляем из левого столбца
        if left_idx < len(left_column):
            ordered.append(left_column[left_idx])
            left_idx += 1
        
        # Добавляем из правого столбца
        if right_idx < len(right_column):
            ordered.append(right_column[right_idx])
            right_idx += 1
    
    # Оставшиеся полноширинные элементы
    ordered.extend([item for item in full_width if item not in ordered])
    
    return ordered


def merge_overlapping_text_blocks(items: list[PageItem]) -> list[PageItem]:
    """Объединение перекрывающихся текстовых блоков."""
    if not items:
        return items
    
    merged = []
    sorted_items = sorted(items, key=lambda item: (item.bbox[1], item.bbox[0]))
    
    current = sorted_items[0]
    
    for item in sorted_items[1:]:
        if (item.kind == "text" and current.kind == "text" and 
            _bboxes_overlap_significantly(current.bbox, item.bbox)):
            # Объединяем блоки
            current = PageItem(
                kind="text",
                bbox=_merge_bboxes(current.bbox, item.bbox),
                content=current.content + "\n" + item.content
            )
        else:
            merged.append(current)
            current = item
    
    merged.append(current)
    return merged


def _bboxes_overlap_significantly(
    bbox1: tuple[float, float, float, float], 
    bbox2: tuple[float, float, float, float],
    threshold: float = 0.3
) -> bool:
    """Проверка значительного перекрытия bbox."""
    area1 = bbox_area(bbox1)
    area2 = bbox_area(bbox2)
    overlap = intersection_area(bbox1, bbox2)
    
    return (overlap / min(area1, area2)) > threshold


def _merge_bboxes(
    bbox1: tuple[float, float, float, float], 
    bbox2: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    """Объединение двух bbox."""
    x0 = min(bbox1[0], bbox2[0])
    y0 = min(bbox1[1], bbox2[1])
    x1 = max(bbox1[2], bbox2[2])
    y1 = max(bbox1[3], bbox2[3])
    return (x0, y0, x1, y1)