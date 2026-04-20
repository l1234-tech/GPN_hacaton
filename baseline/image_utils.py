# image_utils.py
import fitz
from pathlib import Path
from typing import List, Dict

from .layout_utils import PageItem

MIN_IMAGE_DIMENSION = 50


def _get_image_bboxes(page: fitz.Page) -> Dict[int, tuple[float, float, float, float]]:
    result: Dict[int, tuple[float, float, float, float]] = {}
    page_dict = page.get_text("dict")
    for block in page_dict.get("blocks", []):
        if block.get("type") != 1:
            continue
        image_xref = block.get("image")
        if image_xref is not None:
            result[image_xref] = tuple(block.get("bbox", (0.0, 0.0, page.rect.width, page.rect.height)))
    return result


def _save_image(pix: fitz.Pixmap, path: Path) -> None:
    if pix.n - pix.alpha >= 4:
        pix = fitz.Pixmap(fitz.csRGB, pix)
    pix.save(str(path))


def extract_all_images(page: fitz.Page, doc_id: str, img_counter: int, out_dir: Path) -> tuple[List[PageItem], int]:
    """Извлекает все изображения и рендерит векторную графику."""
    items: List[PageItem] = []
    visited_xrefs = set()
    xref_bboxes = _get_image_bboxes(page)

    # 1. Растровые изображения
    for img in page.get_images(full=True):
        xref = img[0]
        if xref in visited_xrefs:
            continue
        visited_xrefs.add(xref)

        try:
            pix = fitz.Pixmap(page.parent, xref)
        except Exception:
            continue

        if pix.width < MIN_IMAGE_DIMENSION or pix.height < MIN_IMAGE_DIMENSION:
            continue

        fname = f"doc_{doc_id}_image_{img_counter}.png"
        out_path = out_dir / fname
        _save_image(pix, out_path)

        bbox = xref_bboxes.get(xref, (0.0, 0.0, page.rect.width, page.rect.height))
        items.append(PageItem(kind="image", bbox=bbox, content=f"![Image]({fname})"))
        img_counter += 1

    drawings = page.cluster_drawings()
    for rect in drawings:
        if rect.width < MIN_IMAGE_DIMENSION or rect.height < MIN_IMAGE_DIMENSION:
            continue

        fname = f"doc_{doc_id}_image_{img_counter}.png"
        out_path = out_dir / fname
        pix = page.get_pixmap(clip=rect, matrix=fitz.Matrix(2, 2))
        if pix.width < MIN_IMAGE_DIMENSION or pix.height < MIN_IMAGE_DIMENSION:
            continue

        _save_image(pix, out_path)
        bbox = (rect.x0, rect.y0, rect.x1, rect.y1)
        items.append(PageItem(kind="image", bbox=bbox, content=f"![Image]({fname})"))
        img_counter += 1

    unique_items: List[PageItem] = []
    seen_bboxes: List[tuple[float, float, float, float]] = []
    for item in items:
        if any(
            abs(item.bbox[0] - other[0]) < 2
            and abs(item.bbox[1] - other[1]) < 2
            and abs(item.bbox[2] - other[2]) < 2
            and abs(item.bbox[3] - other[3]) < 2
            for other in seen_bboxes
        ):
            continue
        seen_bboxes.append(item.bbox)
        unique_items.append(item)

    return unique_items, img_counter


def extract_page_images(
    page: fitz.Page,
    table_bboxes: List[tuple[float, float, float, float]],
    output_images_dir: Path,
    doc_id: int,
    image_counter_start: int,
    skip_large_background: bool = False,
) -> tuple[List[PageItem], int, List[tuple[float, float, float, float]]]:
    output_images_dir.mkdir(parents=True, exist_ok=True)
    items, image_counter = extract_all_images(page, doc_id, image_counter_start, output_images_dir)
    image_bboxes = [item.bbox for item in items]

    if skip_large_background:
        page_area = page.rect.width * page.rect.height
        filtered: List[PageItem] = []
        kept_bboxes: List[tuple[float, float, float, float]] = []
        for item in items:
            x0, y0, x1, y1 = item.bbox
            if (x1 - x0) * (y1 - y0) >= page_area * 0.85:
                continue
            filtered.append(item)
            kept_bboxes.append(item.bbox)
        return filtered, image_counter, kept_bboxes

    return items, image_counter, image_bboxes
