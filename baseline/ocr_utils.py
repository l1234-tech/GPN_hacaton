# baseline/ocr_utils.py
from __future__ import annotations

import os
import re
from functools import lru_cache

import fitz
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR

from layout_utils import PageItem


os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


@lru_cache(maxsize=1)
def get_ocr_engine() -> PaddleOCR:
    return PaddleOCR(use_angle_cls=True, lang="ru", show_log=False)


def normalize_ocr_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
    text = re.sub(r"\s+([,.;:!?%)\]])", r"\1", text)
    return text.strip()


def page_to_numpy(page: fitz.Page, scale: float = 2.0) -> np.ndarray:
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return np.array(image)


def ocr_page_items(page: fitz.Page) -> list[PageItem]:
    image = page_to_numpy(page)
    result = get_ocr_engine().ocr(image, cls=True)

    lines: list[tuple[tuple[float, float, float, float], str]] = []
    scale_x = page.rect.width / image.shape[1]
    scale_y = page.rect.height / image.shape[0]

    for block in result:
        for line in block:
            points, payload = line
            text = normalize_ocr_text(payload[0])
            confidence = payload[1]
            if len(text) < 2 or confidence < 0.45:
                continue
            xs = [point[0] * scale_x for point in points]
            ys = [point[1] * scale_y for point in points]
            bbox = (min(xs), min(ys), max(xs), max(ys))
            lines.append((bbox, text))

    lines.sort(key=lambda item: (item[0][1], item[0][0]))
    merged: list[PageItem] = []
    current_bbox: tuple[float, float, float, float] | None = None
    current_lines: list[str] = []

    for bbox, text in lines:
        if current_bbox is None:
            current_bbox = bbox
            current_lines = [text]
            continue

        _, prev_top, _, prev_bottom = current_bbox
        x0, y0, x1, y1 = bbox
        gap = y0 - prev_bottom
        same_paragraph = gap < 14

        if same_paragraph:
            current_lines.append(text)
            current_bbox = (
                min(current_bbox[0], x0),
                min(current_bbox[1], y0),
                max(current_bbox[2], x1),
                max(current_bbox[3], y1),
            )
            continue

        merged.append(PageItem(kind="text", bbox=current_bbox, content="\n".join(current_lines)))
        current_bbox = bbox
        current_lines = [text]

    if current_bbox is not None and current_lines:
        merged.append(PageItem(kind="text", bbox=current_bbox, content="\n".join(current_lines)))

    return merged