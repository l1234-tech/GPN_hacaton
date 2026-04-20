#!/usr/bin/env python3
"""
ULTRA PDF → Markdown Converter (Debug Mode)
Запуск: python baseline/docling_strong.py --input-dir dataset/public/pdfs --output-dir results
"""

import argparse
import os
import re
import sys
import traceback
from pathlib import Path
from collections import defaultdict

import fitz
import pdfplumber

# Абсолютные импорты (работают при запуске из корня проекта)
from baseline.table_utils import extract_tables_with_bbox, detect_borderless_tables, rows_to_markdown, merge_tables_across_pages
from baseline.image_utils import extract_all_images
from baseline.table_model import ParsedTable
from baseline.layout_utils import collect_repeated_margin_texts, detect_heading_level, is_probable_watermark, merge_overlapping_text_blocks, page_is_mostly_raster
from baseline.postprocess import normalize_markdown, normalize_text_block
from baseline.ocr_utils import ocr_page_items, is_page_mostly_scanned

def remove_text_overlapping_tables(words, tables):
    """Удаляет слова, попадающие внутрь bounding box таблиц."""
    valid_words = []
    for w in words:
        wbbox = (w["x0"], w["top"], w["x1"], w["bottom"])
        inside = False
        for t in tables:
            # Простая проверка пересечения
            if (wbbox[0] > t.bbox[0] and wbbox[2] < t.bbox[2] and
                wbbox[1] > t.bbox[1] and wbbox[3] < t.bbox[3]):
                inside = True
                break
        if not inside:
            valid_words.append(w)
    return valid_words

def convert_pdf(pdf_path: Path, output_dir: Path) -> bool:
    """Конвертирует один PDF с улучшенной обработкой. Возвращает True при успехе."""
    print(f" Обработка: {pdf_path.name}", flush=True)
    try:
        doc_id = pdf_path.stem
        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        doc = fitz.open(pdf_path)
        pdf = pdfplumber.open(pdf_path)
        
        # Сбор повторяющихся текстов в полях для фильтрации
        repeated_margin_texts = collect_repeated_margin_texts(doc)
        
        pages_md = []
        img_counter = 1
        all_tables = []  # Для межстраничного объединения таблиц

        for i, page in enumerate(doc):
            pl_page = pdf.pages[i]
            page_md_parts = []

            # 1. Таблицы с улучшенной обработкой
            page_tables = extract_tables_with_bbox(pl_page)
            page_tables.extend(detect_borderless_tables(pl_page))
            all_tables.extend(page_tables)
            
            # 2. Определение типа страницы
            raster_page = is_page_mostly_scanned(page) or page_is_mostly_raster(page)
            
            if raster_page:
                # OCR для растровых страниц
                ocr_items = ocr_page_items(page)
                text_content = "\n\n".join(item.content for item in ocr_items if item.content.strip())
                if text_content:
                    page_md_parts.append(text_content)
            else:
                # Извлечение текста с улучшенной фильтрацией
                words = pl_page.extract_words(keep_blank_chars=True)
                clean_words = remove_text_overlapping_tables(words, page_tables)
                
                if clean_words:
                    # Улучшенная сборка текста с учетом заголовков
                    text_blocks = _build_text_blocks_with_headings(clean_words, page, repeated_margin_texts)
                    if text_blocks:
                        page_md_parts.append("\n\n".join(text_blocks))

            # 3. Добавляем таблицы страницы
            for t in page_tables:
                if t.markdown:
                    page_md_parts.append(t.markdown)

            # 4. Изображения с улучшенной обработкой
            if not raster_page:
                imgs, img_counter = extract_all_images(page, doc_id, img_counter, images_dir)
                page_md_parts.extend(item.content for item in imgs)

            if page_md_parts:
                pages_md.append("\n\n".join(page_md_parts))

        # Попытка объединить таблицы между страницами
        if len(all_tables) > 1:
            merged_tables = merge_tables_across_pages(all_tables)
            if len(merged_tables) < len(all_tables):
                print(f"  Объединено таблиц: {len(all_tables)} → {len(merged_tables)}")
                # TODO: Обновить содержимое страниц с объединенными таблицами

        doc.close()
        pdf.close()

        # Финальная постобработка
        full_content = "\n\n---\n\n".join(pages_md)
        normalized_content = normalize_markdown(full_content)

        # Сохраняем результат
        out_path = output_dir / f"{doc_id}.md"
        out_path.write_text(normalized_content, encoding="utf-8")
        print(f"✅ Готово: {out_path.name}")
        return True

    except Exception as e:
        print(f"❌ Ошибка в {pdf_path.name}: {e}")
        traceback.print_exc()
        return False


def _build_text_blocks_with_headings(words, page, repeated_margin_texts):
    """Строит текстовые блоки с распознаванием заголовков."""
    if not words:
        return []
    
    # Группировка слов в блоки
    blocks = []
    current_block = []
    last_y = -1.0
    
    for w in words:
        if abs(w["top"] - last_y) > 8.0:  # Новый блок
            if current_block:
                blocks.append(current_block)
            current_block = [w]
            last_y = w["top"]
        else:
            current_block.append(w)
    
    if current_block:
        blocks.append(current_block)
    
    # Преобразование в текст с анализом заголовков
    text_blocks = []
    prev_block_text = ""
    
    for block_words in blocks:
        block_text = " ".join(w["text"] for w in block_words)
        block_text = normalize_text_block(block_text)
        
        if not block_text:
            continue
            
        # Простой анализ для определения заголовков
        if (len(block_text) < 100 and 
            block_text[0].isupper() and 
            not block_text.endswith('.') and
            len(block_words) <= 10):
            # Возможный заголовок
            if len(block_text) < 60:
                block_text = f"## {block_text}"
            elif len(block_text) < 90:
                block_text = f"### {block_text}"
        
        # Фильтрация водяных знаков
        if is_probable_watermark(block_text, {"bbox": _words_to_bbox(block_words)}, page.rect):
            continue
            
        text_blocks.append(block_text)
        prev_block_text = block_text
    
    return text_blocks


def _words_to_bbox(words):
    """Вычисляет bbox для группы слов."""
    if not words:
        return (0, 0, 0, 0)
    
    x0 = min(w["x0"] for w in words)
    y0 = min(w["top"] for w in words)
    x1 = max(w["x1"] for w in words)
    y1 = max(w["bottom"] for w in words)
    
    return (x0, y0, x1, y1)

def main():
    parser = argparse.ArgumentParser(description="PDF → Markdown с улучшенным score")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()

    # Проверка путей
    if not args.input_dir.exists():
        print(f" Директория не найдена: {args.input_dir.absolute()}")
        return
    if not args.input_dir.is_dir():
        print(f"🚫 Путь не является директорией: {args.input_dir}")
        return

    pdf_files = sorted(args.input_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"📭 В {args.input_dir} не найдено ни одного .pdf файла")
        return

    if args.max_files:
        pdf_files = pdf_files[:args.max_files]

    print(f"🚀 Найдено PDF: {len(pdf_files)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    for p in pdf_files:
        if convert_pdf(p, args.output_dir):
            success_count += 1

    print(f"\n🏁 Завершено. Обработано: {success_count}/{len(pdf_files)}")
    print(f"📂 Результаты: {args.output_dir.absolute()}")

if __name__ == "__main__":
    main()