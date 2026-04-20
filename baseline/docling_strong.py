#!/usr/bin/env python3
"""
Финальный конвертер PDF → Markdown на базе Docling с улучшенной обработкой таблиц.
Использует TableFormer для распознавания таблиц и нашу функцию table_to_markdown
для генерации корректного Markdown.
"""

from __future__ import annotations

import argparse
import gc
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

def _apply_device_from_argv() -> None:
    for i, arg in enumerate(sys.argv):
        if arg == "--device" and i + 1 < len(sys.argv):
            val = sys.argv[i + 1]
            if val != "auto":
                os.environ["DOCLING_DEVICE"] = val
            return
        if arg.startswith("--device="):
            val = arg.split("=", 1)[1]
            if val != "auto":
                os.environ["DOCLING_DEVICE"] = val
            return

_apply_device_from_argv()

def _patch_cv2_set_num_threads() -> None:
    try:
        import cv2
    except ImportError:
        return
    if not hasattr(cv2, "setNumThreads"):
        cv2.setNumThreads = lambda _nthreads: None

_patch_cv2_set_num_threads()

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    TableFormerMode,
    TableStructureOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.base import ImageRefMode
from docling_core.types.doc.document import TableItem

# -----------------------------------------------------------------------------
# Функции для таблиц (без изменений)
# -----------------------------------------------------------------------------
def clean_cell_text(value: Optional[str]) -> str:
    if value is None:
        return ""
    value = value.replace("\xa0", " ").replace("\n", " ")
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"\s+([,.;:!?%)\]])", r"\1", value)
    return value.strip()

def fill_merged_header_cells(rows: List[List[str]]) -> List[List[str]]:
    if not rows:
        return rows
    width = max(len(row) for row in rows)
    normalized = [row + [""] * (width - len(row)) for row in rows]
    for row in normalized:
        last_seen = ""
        for idx in range(len(row)):
            if row[idx]:
                last_seen = row[idx]
            elif last_seen:
                row[idx] = last_seen
    for col in range(width):
        last_seen = ""
        for row in normalized:
            if row[col]:
                last_seen = row[col]
            elif last_seen:
                row[col] = last_seen
    return normalized

def infer_header_rows(rows: List[List[str]]) -> int:
    if len(rows) <= 1:
        return 1
    header_rows = 1
    for row in rows[:3]:
        filled = [cell for cell in row if cell]
        if not filled:
            break
        numeric_ratio = sum(bool(re.search(r"\d", cell)) for cell in filled) / max(1, len(filled))
        if numeric_ratio <= 0.45:
            header_rows += 1
        else:
            break
    return min(header_rows, max(1, len(rows) - 1))

def table_to_markdown(rows: List[List[str]]) -> str:
    cleaned = [[clean_cell_text(cell) for cell in row] for row in rows]
    cleaned = [row for row in cleaned if any(cell for cell in row)]
    if not cleaned:
        return ""
    width = max(len(row) for row in cleaned)
    cleaned = [row + [""] * (width - len(row)) for row in cleaned]
    header_rows = infer_header_rows(cleaned)
    if header_rows >= len(cleaned):
        header_rows = max(1, len(cleaned) - 1)
    header = fill_merged_header_cells(cleaned[:header_rows])
    body = cleaned[header_rows:]
    header_line = []
    for col in range(width):
        parts = []
        for row in header:
            if row[col]:
                parts.append(row[col])
        header_line.append("_".join(parts) if parts else f"col_{col+1}")
    separator = ["---"] * width
    lines = [
        "| " + " | ".join(header_line) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in body:
        formatted_row = [cell if cell else " " for cell in row]
        lines.append("| " + " | ".join(formatted_row) + " |")
    return "\n".join(lines)

# -----------------------------------------------------------------------------
# Работа с изображениями (УЛУЧШЕННАЯ ФИЛЬТРАЦИЯ ВОДЯНЫХ ЗНАКОВ)
# -----------------------------------------------------------------------------
_IMG_LINK_RE = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')

def _clear_cuda_cache() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()

def _doc_num_from_stem(stem: str) -> int:
    parts = stem.rsplit("_", 1)
    if len(parts) != 2:
        return 1
    try:
        return int(parts[1])
    except ValueError:
        return 1

def _move_or_convert_to_png(src: Path, dst: Path) -> None:
    ext = src.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        from PIL import Image
        with Image.open(src) as im:
            im.save(dst, format="PNG")
        src.unlink()
    else:
        shutil.move(str(src), str(dst))

def _is_watermark_by_alt(alt_text: str) -> bool:
    """Проверяет, содержит ли alt-текст ключевые слова водяных знаков."""
    alt_upper = alt_text.upper()
    keywords = ["DRAFT", "ЧЕРНОВИК", "CONFIDENTIAL", "SAMPLE", "ОБРАЗЕЦ", "WATERMARK"]
    return any(kw in alt_upper for kw in keywords)

def _normalize_image_names(
    markdown: str,
    work_images_dir: Path,
    out_images_dir: Path,
    doc_num: int,
) -> str:
    out_images_dir.mkdir(parents=True, exist_ok=True)
    
    MIN_IMAGE_SIZE = 15 * 1024  # 15 КБ – порог для фильтрации мусора
    
    # Сначала собираем все ссылки
    matches = list(_IMG_LINK_RE.finditer(markdown))
    if not matches:
        return markdown
    
    old_to_new = {}
    order = 1
    
    # Обрабатываем с конца, чтобы не сбить позиции при замене
    for match in reversed(matches):
        alt_text = match.group(1)
        old_path = match.group(2)
        full_match = match.group(0)
        
        old_name = Path(old_path).name
        if not old_name:
            continue
        
        src = work_images_dir / old_name
        if not src.is_file():
            src = work_images_dir / "images" / old_name
        if not src.is_file():
            continue
        
        # Фильтрация по размеру
        file_size = src.stat().st_size
        if file_size < MIN_IMAGE_SIZE:
            # Удаляем ссылку полностью
            markdown = markdown.replace(full_match, "")
            continue
        
        # Фильтрация по alt-тексту
        if _is_watermark_by_alt(alt_text):
            markdown = markdown.replace(full_match, "")
            continue
        
        # Если файл прошёл фильтры – переименовываем
        if old_name in old_to_new:
            new_name = old_to_new[old_name]
        else:
            new_name = f"doc_{doc_num}_image_{order}.png"
            old_to_new[old_name] = new_name
            _move_or_convert_to_png(src, out_images_dir / new_name)
            order += 1
        
        new_path = f"images/{new_name}"
        new_link = f"![{alt_text}]({new_path})"
        markdown = markdown.replace(full_match, new_link)
    
    return markdown

def _build_converter(no_ocr: bool, no_table_structure: bool, full_quality: bool) -> DocumentConverter:
    if full_quality:
        images_scale = 1.0
        table_opts = TableStructureOptions(mode=TableFormerMode.ACCURATE)
    else:
        images_scale = 0.88
        table_opts = TableStructureOptions(mode=TableFormerMode.FAST)
    pipeline_options = PdfPipelineOptions(
        do_ocr=not no_ocr,
        do_table_structure=not no_table_structure,
        generate_picture_images=True,
        images_scale=images_scale,
        table_structure_options=table_opts,
        accelerator_options=AcceleratorOptions(),
    )
    return DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        },
    )

# -----------------------------------------------------------------------------
# Функция исправления границ слов в таблицах (НОВАЯ)
# -----------------------------------------------------------------------------
def fix_word_boundaries_in_tables(text: str) -> str:
    """Исправляет разорванные слова и слипшиеся слова в таблицах Markdown."""
    lines = text.split('\n')
    new_lines = []
    in_table = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('|') and stripped.endswith('|'):
            in_table = True
            parts = stripped.split('|')
            # Убираем пустые краевые элементы
            if parts and parts[0] == '':
                parts.pop(0)
            if parts and parts[-1] == '':
                parts.pop(-1)
            fixed_parts = []
            for cell in parts:
                # Склейка разорванных цифр
                cell = re.sub(r'(\d)\s+(\d)', r'\1\2', cell)
                # Склейка коротких окончаний (2-3 буквы)
                cell = re.sub(r'(\w)\s+([a-zа-яё]{1,3})(?=\W|$)', r'\1\2', cell, flags=re.IGNORECASE)
                # Разделение слипшихся слов по регистру
                cell = re.sub(r'([a-zа-яё])([A-ZА-ЯЁ])', r'\1 \2', cell)
                # Убираем лишние пробелы
                cell = re.sub(r'\s+', ' ', cell)
                cell = cell.strip()
                fixed_parts.append(cell)
            new_line = '| ' + ' | '.join(fixed_parts) + ' |'
            new_lines.append(new_line)
        else:
            if in_table and stripped == '':
                in_table = False
            # Обычный текст тоже слегка чистим (кроме заголовков)
            if not line.strip().startswith('#'):
                line = re.sub(r'([a-zа-яё])([A-ZА-ЯЁ])', r'\1 \2', line)
                line = re.sub(r'(\d)\s+(\d)', r'\1\2', line)
            new_lines.append(line)
    return '\n'.join(new_lines)

# -----------------------------------------------------------------------------
# Основная конвертация (с удалением текстовых водяных знаков)
# -----------------------------------------------------------------------------
def convert_pdf(pdf_path: Path, output_dir: Path, converter: DocumentConverter) -> None:
    stem = pdf_path.stem
    doc_num = _doc_num_from_stem(stem)
    result = converter.convert(str(pdf_path))
    doc = result.document

    with tempfile.TemporaryDirectory(prefix=f"docling_{stem}_") as tmp:
        work = Path(tmp)
        md_work = work / f"{stem}.md"
        doc.save_as_markdown(
            md_work,
            artifacts_dir=Path("images"),
            image_mode=ImageRefMode.REFERENCED,
        )
        text = md_work.read_text(encoding="utf-8")

        # Таблицы (пока без замены)
        tables: List[TableItem] = [item for item in doc.iterate_items() if isinstance(item, TableItem)]
        if tables:
            tables.sort(key=lambda t: (t.prov[0].page_no, t.prov[0].bbox[1] if t.prov else 0))
            for table in tables:
                data = table.data.table_cells
                if not data:
                    continue
                # Генерация улучшенного Markdown (можно включить при необходимости)
                # rows = []
                # for row_idx in range(table.data.num_rows):
                #     row = []
                #     for col_idx in range(table.data.num_cols):
                #         cell = data.get((row_idx, col_idx))
                #         row.append(cell.text if cell else "")
                #     rows.append(row)
                # new_md = table_to_markdown(rows)
                pass

        # Нормализация имён изображений с фильтрацией водяных знаков
        text = _normalize_image_names(
            text,
            work_images_dir=work / "images",
            out_images_dir=output_dir / "images",
            doc_num=doc_num,
        )

        # Удаление текстовых водяных знаков
        watermark_keywords = ["ЧЕРНОВИК", "DRAFT", "CONFIDENTIAL", "SAMPLE", "ОБРАЗЕЦ"]
        lines = text.split("\n")
        cleaned = []
        for line in lines:
            if not line.strip():
                cleaned.append(line)
                continue
            upper_line = line.upper()
            if any(keyword in upper_line for keyword in watermark_keywords):
                continue
            cleaned.append(line)
        text = "\n".join(cleaned)

        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.replace("\\", "/")
        
        # Исправление границ слов в таблицах (НОВЫЙ ШАГ)
        text = fix_word_boundaries_in_tables(text)

        out_md = output_dir / f"{stem}.md"
        out_md.write_text(text, encoding="utf-8")

def main() -> None:
    parser = argparse.ArgumentParser(description="Docling final with improved tables")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--full-quality", action="store_true", help="TableFormer ACCURATE mode")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(args.input_dir.glob("*.pdf"))
    if args.max_files:
        pdf_files = pdf_files[:args.max_files]
    if not pdf_files:
        print("No PDF files found.")
        return

    converter = _build_converter(no_ocr=False, no_table_structure=False, full_quality=args.full_quality)
    converter.initialize_pipeline(InputFormat.PDF)

    for pdf_path in pdf_files:
        try:
            convert_pdf(pdf_path, args.output_dir, converter)
            print(f"{pdf_path.name}: OK")
        except Exception as e:
            print(f"{pdf_path.name}: ERROR: {e}")
        finally:
            _clear_cuda_cache()

if __name__ == "__main__":
    main()