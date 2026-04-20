from __future__ import annotations

import argparse
from pathlib import Path

import fitz

try:
    from .image_utils import extract_page_images
    from .layout_utils import (
        PageItem,
        collect_repeated_margin_texts,
        detect_heading_level,
        extract_block_text,
        is_margin_text_block,
        is_probable_watermark,
        is_rotated_text_block,
        order_page_items,
        overlaps,
        page_is_mostly_raster,
    )
    from .ocr_utils import ocr_page_items
    from .postprocess import normalize_markdown, normalize_text_block
    from .table_utils import extract_tables
except ImportError:
    from image_utils import extract_page_images
    from layout_utils import (
        PageItem,
        collect_repeated_margin_texts,
        detect_heading_level,
        extract_block_text,
        is_margin_text_block,
        is_probable_watermark,
        is_rotated_text_block,
        order_page_items,
        overlaps,
        page_is_mostly_raster,
    )
    from ocr_utils import ocr_page_items
    from postprocess import normalize_markdown, normalize_text_block
    from table_utils import extract_tables


TABLE_OVERLAP_THRESHOLD = 0.2
IMAGE_OVERLAP_THRESHOLD = 0.35


def doc_id_from_stem(stem: str) -> int:
    parts = stem.rsplit("_", 1)
    if len(parts) == 2:
        try:
            return int(parts[1])
        except ValueError:
            pass
    return 0


def collect_text_items(
    page: fitz.Page,
    repeated_margin_texts: set[str],
    table_bboxes: list[tuple[float, float, float, float]],
    image_bboxes: list[tuple[float, float, float, float]],
) -> list[PageItem]:
    items: list[PageItem] = []
    page_dict = page.get_text("dict")

    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue

        bbox = tuple(block["bbox"])
        text = extract_block_text(block)
        if not text:
            continue
        if is_rotated_text_block(block):
            continue
        if is_probable_watermark(text):
            continue
        if is_margin_text_block(block, page.rect, repeated_margin_texts):
            continue
        if overlaps(bbox, table_bboxes, TABLE_OVERLAP_THRESHOLD):
            continue
        if overlaps(bbox, image_bboxes, IMAGE_OVERLAP_THRESHOLD):
            continue

        heading = detect_heading_level(text, block)
        if heading:
            text = f"{heading} {text}"
        items.append(PageItem(kind="text", bbox=bbox, content=normalize_text_block(text)))

    return items


def convert_pdf(pdf_path: Path, output_dir: Path) -> None:
    stem = pdf_path.stem
    doc_id = doc_id_from_stem(stem)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    with fitz.open(pdf_path) as doc:
        repeated_margin_texts = collect_repeated_margin_texts(doc)
        image_counter = 0
        document_parts: list[str] = []

        for page in doc:
            table_candidates = extract_tables(pdf_path, page.number)
            table_items = [PageItem(kind="table", bbox=candidate.bbox, content=candidate.markdown) for candidate in table_candidates]
            table_bboxes = [candidate.bbox for candidate in table_candidates]

            raster_page = page_is_mostly_raster(page)
            if raster_page:
                image_items = []
                image_bboxes = []
            else:
                image_items, image_counter, image_bboxes = extract_page_images(
                    page=page,
                    table_bboxes=table_bboxes,
                    output_images_dir=images_dir,
                    doc_id=doc_id,
                    image_counter_start=image_counter,
                    skip_large_background=False,
                )

            if raster_page:
                text_items = ocr_page_items(page)
            else:
                text_items = collect_text_items(page, repeated_margin_texts, table_bboxes, image_bboxes)

            page_items = order_page_items(table_items + image_items + text_items, page.rect.width)
            page_content = "\n\n".join(item.content for item in page_items if item.content.strip())
            if page_content.strip():
                document_parts.append(page_content)

    markdown = normalize_markdown("\n\n".join(document_parts))
    (output_dir / f"{stem}.md").write_text(markdown, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid PDF -> Markdown baseline for the GPN hackathon")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory with PDF files")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for Markdown results")
    parser.add_argument("--max-files", type=int, default=None, help="Optional limit for debugging")
    parser.add_argument("--skip-existing", action="store_true", help="Skip Markdown files that already exist")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "images").mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(args.input_dir.glob("*.pdf"))
    if args.max_files is not None:
        pdf_files = pdf_files[: args.max_files]

    if not pdf_files:
        print("No PDF files found.")
        return

    for pdf_path in pdf_files:
        out_md = args.output_dir / f"{pdf_path.stem}.md"
        if args.skip_existing and out_md.exists():
            print(f"{pdf_path.name}: SKIP")
            continue

        try:
            convert_pdf(pdf_path, args.output_dir)
            print(f"{pdf_path.name}: OK")
        except Exception as exc:
            print(f"{pdf_path.name}: ERROR: {exc}")


if __name__ == "__main__":
    main()
