# baseline/image_utils.py
from __future__ import annotations

import io
from pathlib import Path

import fitz
from PIL import Image

from layout_utils import PageItem, overlaps


MIN_IMAGE_SIDE = 80
IMAGE_OVERLAP_THRESHOLD = 0.35


def save_image_bytes(raw_bytes: bytes, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(io.BytesIO(raw_bytes)) as image:
        image.convert("RGB").save(output_path, format="PNG")


def save_page_clip(page: fitz.Page, bbox: tuple[float, float, float, float], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=fitz.Rect(bbox), alpha=False).save(output_path)


def extract_page_images(
    page: fitz.Page,
    table_bboxes: list[tuple[float, float, float, float]],
    output_images_dir: Path,
    doc_id: int,
    image_counter_start: int,
    skip_large_background: bool,
) -> tuple[list[PageItem], int, list[tuple[float, float, float, float]]]:
    items: list[PageItem] = []
    image_bboxes: list[tuple[float, float, float, float]] = []
    image_counter = image_counter_start
    page_area = page.rect.width * page.rect.height
    page_dict = page.get_text("dict")

    for block in page_dict.get("blocks", []):
        if block.get("type") != 1:
            continue

        bbox = tuple(block["bbox"])
        x0, y0, x1, y1 = bbox
        area_ratio = ((x1 - x0) * (y1 - y0)) / max(page_area, 1.0)
        if (x1 - x0) < MIN_IMAGE_SIDE or (y1 - y0) < MIN_IMAGE_SIDE:
            continue
        if area_ratio > 0.45:
            continue
        if skip_large_background and area_ratio > 0.25:
            continue
        if overlaps(bbox, table_bboxes, IMAGE_OVERLAP_THRESHOLD):
            continue

        raw_bytes = block.get("image")
        if not raw_bytes:
            continue

        image_counter += 1
        filename = f"doc_{doc_id}_image_{image_counter}.png"
        save_image_bytes(raw_bytes, output_images_dir / filename)
        image_bboxes.append(bbox)
        items.append(PageItem(kind="image", bbox=bbox, content=f"![Image](images/{filename})"))

    for drawing_rect in page.cluster_drawings():
        bbox = tuple(drawing_rect)
        x0, y0, x1, y1 = bbox
        area_ratio = ((x1 - x0) * (y1 - y0)) / max(page_area, 1.0)
        if (x1 - x0) < 120 or (y1 - y0) < 120:
            continue
        if area_ratio > 0.45:
            continue
        if skip_large_background and area_ratio > 0.25:
            continue
        if overlaps(bbox, table_bboxes, IMAGE_OVERLAP_THRESHOLD):
            continue
        if overlaps(bbox, image_bboxes, IMAGE_OVERLAP_THRESHOLD):
            continue

        image_counter += 1
        filename = f"doc_{doc_id}_image_{image_counter}.png"
        save_page_clip(page, bbox, output_images_dir / filename)
        image_bboxes.append(bbox)
        items.append(PageItem(kind="image", bbox=bbox, content=f"![Image](images/{filename})"))

    return items, image_counter, image_bboxes