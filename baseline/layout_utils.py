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


def detect_heading_level(text: str, block: dict) -> tuple[str, str] | None:
    """Возвращает (префикс_заголовка, чистый_текст) или None."""
    if not text or "\n" in text:
        return None
    if len(text) > 140:
        return None

    max_size = block_max_font_size(block)
    if max_size >= 22:
        return ("#", text)
    if max_size >= 17:
        return ("##", text)
    if max_size >= 14 and len(text) <= 90:
        return ("###", text)
    return None


def page_is_mostly_raster(page: fitz.Page) -> bool:
    blocks = page.get_text("dict").get("blocks", [])
    text_chars = 0
    image_area = 0.0
    page_area = page.rect.width * page.rect.height

    for block in blocks:
        if block.get("type") == 0:
            text_chars += len(extract_block_text(block))
        elif block.get("type") == 1:
            image_area += bbox_area(tuple(block["bbox"]))

    if text_chars < 40 and image_area / max(page_area, 1.0) > 0.6:
        return True
    return False


def order_page_items(items: list[PageItem], page_width: float) -> list[PageItem]:
    if len(items) < 4:
        return sorted(items, key=lambda item: (round(item.bbox[1], 1), round(item.bbox[0], 1)))

    full_width: list[PageItem] = []
    left: list[PageItem] = []
    right: list[PageItem] = []
    center = page_width / 2

    for item in items:
        x0, _, x1, _ = item.bbox
        width = x1 - x0
        if width > page_width * 0.72:
            full_width.append(item)
        elif x1 <= center + 18:
            left.append(item)
        elif x0 >= center - 18:
            right.append(item)
        else:
            full_width.append(item)

    if len(left) < 2 or len(right) < 2:
        return sorted(items, key=lambda item: (round(item.bbox[1], 1), round(item.bbox[0], 1)))

    full_width.sort(key=lambda item: (item.bbox[1], item.bbox[0]))
    left.sort(key=lambda item: (item.bbox[1], item.bbox[0]))
    right.sort(key=lambda item: (item.bbox[1], item.bbox[0]))

    ordered: list[PageItem] = []
    ordered.extend([item for item in full_width if item.bbox[1] < min(left[0].bbox[1], right[0].bbox[1])])
    ordered.extend(left)
    ordered.extend(right)
    ordered.extend([item for item in full_width if item not in ordered])
    return ordered