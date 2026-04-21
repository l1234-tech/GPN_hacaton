import os
import sys
import argparse
import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Set

# Проверка зависимостей
try:
    import pdfplumber
    import fitz  # PyMuPDF
    from PIL import Image
    import io
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    print("Установите зависимости: pip install pdfplumber pymupdf pillow")
    sys.exit(1)

# ==========================================
# КОНФИГУРАЦИЯ
# ==========================================
MIN_TABLE_ROWS = 2
IMAGE_MIN_SIZE = 50
WATERMARK_KEYWORDS = [
    "confidential", "draft", "секретно", "образец", 
    "page", "страница", "of", "из", "www.", "http://"
]

# ==========================================
# УТИЛИТЫ ДЛЯ ТЕКСТА (PRO FEATURES)
# ==========================================

def is_watermark_or_garbage(text: str) -> bool:
    if not text or not text.strip():
        return True
    t = text.strip().lower()
    
    # Ключевые слова
    for kw in WATERMARK_KEYWORDS:
        if kw in t and len(t) < 60:
            return True
            
    # Мусорные строки (только символы)
    if re.match(r'^[\s\-_=.*]+$', t):
        return True
        
    # Вертикальный мусор (набор одиночных букв через пробел)
    # Пример: "Я И Н Е Н А" -> удаляем
    words = t.split()
    if len(words) > 3 and all(len(w) == 1 and w.isalpha() for w in words):
        return True
        
    # Повторяющийся текст
    if len(words) > 3:
        unique_words = set(words)
        if len(unique_words) == 1 and len(words) > 4:
            return True
            
    return False

def fix_broken_words(text: str) -> str:
    """
    Исправляет разорванные слова внутри ячеек.
    Пример: "естественны й" -> "естественный", "3987, 88" -> "3987,88"
    """
    if not text:
        return text
    
    # 1. Склейка букв, разделенных пробелом, если это похоже на разорванное слово
    # Паттерн: буква + пробел + буква (кириллица или латиница)
    # Осторожно, чтобы не склеить разные слова. 
    # Эвристика: если между буквами только один пробел и длина получившегося слова разумна
    
    # Простой вариант: убираем пробелы внутри чисел с запятой (3987, 88 -> 3987,88)
    text = re.sub(r'(\d)\s*,\s*(\d)', r'\1,\2', text)
    text = re.sub(r'(\d)\s*\.\s*(\d)', r'\1.\2', text)
    
    # Убираем пробелы перед "руб.", "°", "±"
    text = re.sub(r'\s+(руб\.|°|±|%|USD|EUR)', r'\1', text)
    
    return text

def clean_cell_content(cell_text: str) -> str:
    if not cell_text:
        return ""
    
    # Базовая очистка
    text = re.sub(r'\n', ' ', cell_text)
    text = re.sub(r'\s+', ' ', text)
    
    # Применение умных фиксов
    text = fix_broken_words(text)
    
    return text.strip()

def get_table_hash(rows: List[List[str]]) -> str:
    if not rows:
        return ""
    sample = rows[:3]
    normalized = []
    for row in sample:
        norm_row = "|".join([str(cell).strip().lower() for cell in row])
        normalized.append(norm_row)
    content = ":::".join(normalized)
    return hashlib.md5(content.encode('utf-8')).hexdigest()

# ==========================================
# ОБРАБОТКА ТАБЛИЦ
# ==========================================

def extract_tables_with_pdfplumber(page: pdfplumber.page.Page) -> List[Dict[str, Any]]:
    tables = []
    
    # Используем стандартный extract_tables(), который стабилен во всех версиях
    # Он автоматически определяет сетку
    raw_tables = page.extract_tables()
    
    if not raw_tables:
        return tables

    for t_idx, table_rows in enumerate(raw_tables):
        if not table_rows or len(table_rows) < MIN_TABLE_ROWS:
            continue
        
        # Получаем bounding box таблицы (примерно)
        # В старой версии api нет прямого bbox у extract_tables, но можно вычислить по словам
        # Для простоты берем всю ширину страницы и высоту, занятую таблицей (эвристика)
        # Или пробуем найти координаты через объекты, если нужно точно.
        # Здесь используем заглушку bbox, так как для логики склейки важнее данные.
        bbox = (0, page.height * 0.1, page.width, page.height * 0.9) 

        cleaned_rows = []
        for row in table_rows:
            cleaned_row = []
            row_has_content = False
            for cell in row:
                c_text = clean_cell_content(cell) if cell else ""
                
                # Фильтрация мусора
                if is_watermark_or_garbage(c_text):
                    cleaned_row.append("") 
                else:
                    cleaned_row.append(c_text)
                    row_has_content = True
            
            if row_has_content:
                cleaned_rows.append(cleaned_row)
        
        if len(cleaned_rows) >= MIN_TABLE_ROWS:
            tables.append({
                'data': cleaned_rows,
                'bbox': bbox,
                'page_num': page.page_number,
                'height': page.height,
                'top_text': None # Будет заполнено позже
            })
            
    return tables

def find_context_title(page: pdfplumber.page.Page, table_bbox: Tuple[float, float, float, float]) -> Optional[str]:
    """
    Пытается найти заголовок таблицы, анализируя текст над ней.
    """
    if not table_bbox:
        return None
        
    top = table_bbox[1]
    # Берем область над таблицей (50-100 пикселей)
    search_box = (table_bbox[0], max(0, top - 100), table_bbox[2], top)
    
    # Извлекаем текст из этой области
    # В pdfplumber можно crop страницу
    try:
        crop_page = page.crop(search_box)
        text = crop_page.extract_text()
        if text:
            lines = [l.strip() for l in text.split('\n') if l.strip()]
            if lines:
                # Берем последнюю непустую строку перед таблицей
                candidate = lines[-1]
                if not is_watermark_or_garbage(candidate) and len(candidate) < 150:
                    return candidate
    except Exception:
        pass
    return None

def merge_split_tables(all_tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not all_tables:
        return []
    
    merged_tables = []
    # Сортировка
    all_tables.sort(key=lambda x: (x['page_num'], x['bbox'][1]))
    
    current_table = None
    
    for i, table in enumerate(all_tables):
        if current_table is None:
            current_table = table
            continue
        
        prev = current_table
        curr = table
        
        can_merge = False
        curr_body = []
        
        # Проверка количества столбцов
        if len(prev['data'][0]) == len(curr['data'][0]):
            prev_header = prev['data'][0]
            curr_first_row = curr['data'][0]
            
            # Дубликат заголовка?
            if prev_header == curr_first_row:
                can_merge = True
                curr_body = curr['data'][1:]
            else:
                # Просто продолжение? Проверяем схожесть стилей или просто склеиваем, если близко по смыслу
                # Для надежности склеиваем, если структура совпадает
                can_merge = True
                curr_body = curr['data']
        
        if can_merge:
            # Проверка на полный дубликат
            if get_table_hash(prev['data']) == get_table_hash(curr['data']) and len(prev['data']) == len(curr['data']):
                continue
            
            new_data = prev['data'][:]
            if curr_body:
                new_data.extend(curr_body)
            
            current_table = {
                'data': new_data,
                'bbox': prev['bbox'],
                'page_num': prev['page_num'],
                'height': prev['height'],
                'merged_pages': list(set(prev.get('merged_pages', [prev['page_num']]) + [curr['page_num']])),
                'title': prev.get('title') # Сохраняем титул
            }
        else:
            merged_tables.append(current_table)
            current_table = table
            
    if current_table:
        merged_tables.append(current_table)
        
    return merged_tables

def format_table_to_markdown(table_data: List[List[str]], title: Optional[str] = None) -> str:
    if not table_data:
        return ""
    
    lines = []
    
    if title:
        lines.append(f"### {title}")
        lines.append("")
    
    num_cols = len(table_data[0])
    
    header = [cell.replace("|", "\\|") for cell in table_data[0]]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * num_cols) + " |")
    
    for row in table_data[1:]:
        while len(row) < num_cols:
            row.append("")
        row = row[:num_cols]
        clean_row = [cell.replace("|", "\\|") for cell in row]
        lines.append("| " + " | ".join(clean_row) + " |")
    
    return "\n".join(lines)

# ==========================================
# ОБРАБОТКА ИЗОБРАЖЕНИЙ
# ==========================================

def extract_images_from_doc(doc_path: str, doc_index: int, output_images_dir: str) -> List[Dict[str, Any]]:
    saved_images = []
    doc_fit = fitz.open(doc_path)
    img_counter = 1
    
    for page_num, page in enumerate(doc_fit):
        image_list = page.get_images(full=True)
        
        for img_info in image_list:
            xref = img_info[0]
            try:
                base_image = page.parent.extract_image(xref)
                if not base_image:
                    continue
                
                image_bytes = base_image["image"]
                img_ext = base_image["ext"]
                
                img = Image.open(io.BytesIO(image_bytes))
                
                if img.width < IMAGE_MIN_SIZE or img.height < IMAGE_MIN_SIZE:
                    continue
                
                if img.mode in ('RGBA', 'LA', 'P'):
                    img = img.convert('RGB')
                
                filename = f"doc_{doc_index}_image_{img_counter}.png"
                filepath = os.path.join(output_images_dir, filename)
                
                img.save(filepath, "PNG")
                
                saved_images.append({
                    'filename': filename,
                    'path': filepath,
                    'page': page_num + 1,
                    'caption': f"Рис. {img_counter}"
                })
                
                img_counter += 1
                
            except Exception as e:
                continue
                
    doc_fit.close()
    return saved_images

# ==========================================
# ОСНОВНОЙ ПРОЦЕСС
# ==========================================

def process_document(pdf_path: str, doc_index: int, output_md_path: str, images_dir: str) -> bool:
    stats = {"tables": 0, "images": 0, "pages": 0}
    
    # 1. Извлечение таблиц и текста
    all_tables_raw = []
    text_blocks = []
    
    with pdfplumber.open(pdf_path) as pdf:
        stats["pages"] = len(pdf.pages)
        for page in pdf.pages:
            # Текст
            p_text = page.extract_text()
            if p_text:
                text_blocks.append(p_text)
            
            # Таблицы
            page_tables = extract_tables_with_pdfplumber(page)
            
            # Поиск заголовков для каждой таблицы
            for tbl in page_tables:
                title = find_context_title(page, tbl['bbox'])
                tbl['title'] = title
            
            all_tables_raw.extend(page_tables)
    
    # Обработка таблиц
    final_tables = merge_split_tables(all_tables_raw)
    stats["tables"] = len(final_tables)
    
    # Изображения
    images = extract_images_from_doc(pdf_path, doc_index, images_dir)
    stats["images"] = len(images)
    
    # Генерация MD
    md_lines = []
    
    # Заголовок документа
    full_text = "\n".join(text_blocks)
    first_lines = [l for l in full_text.split('\n') if l.strip() and not is_watermark_or_garbage(l)]
    
    if first_lines:
        title = first_lines[0].strip()
        if len(title) > 100:
            title = " ".join(title.split()[:10]) + "..."
        # Фильтр вертикального мусора в заголовке
        words = title.split()
        if len(words) > 3 and all(len(w) == 1 for w in words):
            title = "Документ" # Если заголовок оказался мусором
            
        md_lines.append(f"# {title}")
        md_lines.append("")
        
        intro = "\n".join(first_lines[1:5])
        if intro:
            md_lines.append(intro)
            md_lines.append("")
    
    # Таблицы
    if final_tables:
        md_lines.append("## Сводные данные")
        md_lines.append("")
        
        for table in final_tables:
            md_table = format_table_to_markdown(table['data'], title=table.get('title'))
            md_lines.append(md_table)
            md_lines.append("")
            
    # Изображения
    if images:
        for img in images:
            md_lines.append(f"![{img['caption']}](images/{img['filename']})")
            md_lines.append(f"{img['caption']}")
            md_lines.append("")
            md_lines.append("----")
            md_lines.append("")
    
    # Запись
    try:
        with open(output_md_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(md_lines))
        return True, stats
    except Exception as e:
        print(f"Error saving: {e}")
        return False, stats

def main():
    parser = argparse.ArgumentParser(description="PDF to Markdown Converter PRO")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()
    
    input_dir = args.input_dir
    output_dir = args.output_dir
    images_dir = os.path.join(output_dir, "images")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    
    pdf_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')])
    
    if args.max_files:
        pdf_files = pdf_files[:args.max_files]
        
    print(f"Запуск PRO версии. Файлов: {len(pdf_files)}")
    
    total_stats = {"tables": 0, "images": 0, "pages": 0}
    success_count = 0
    
    for idx, filename in enumerate(pdf_files):
        doc_index = idx + 1
        print(f"[{idx+1}/{len(pdf_files)}] {filename}...", end=" ")
        
        input_path = os.path.join(input_dir, filename)
        output_filename = f"document_{doc_index:03d}.md"
        output_path = os.path.join(output_dir, output_filename)
        
        success, stats = process_document(input_path, doc_index, output_path, images_dir)
        
        if success:
            success_count += 1
            total_stats['tables'] += stats['tables']
            total_stats['images'] += stats['images']
            total_stats['pages'] += stats['pages']
            print(f"OK (Таблиц: {stats['tables']}, Картинки: {stats['images']})")
        else:
            print("FAIL")
            
    print("-" * 30)
    print(f"Готово! Успешно: {success_count}/{len(pdf_files)}")
    print(f"Всего обработано страниц: {total_stats['pages']}")
    print(f"Всего извлечено таблиц: {total_stats['tables']}")
    print(f"Всего сохранено изображений: {total_stats['images']}")
    print(f"Результат в: {output_dir}")

if __name__ == "__main__":
    main()