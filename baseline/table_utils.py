# baseline/table_utils.py
import re
from typing import List, Tuple, Dict, Optional, Set
import pdfplumber
from collections import defaultdict
from .table_model import ParsedTable, TableCell

def extract_tables_with_bbox(page: pdfplumber.page.Page) -> List[ParsedTable]:
    """Извлекает таблицы с рамками и их BBox с улучшенной обработкой."""
    tables = []
    try:
        found_tables = page.find_tables()
        for table in found_tables:
            extracted = table.extract()
            if not extracted or len(extracted) < 2:
                continue
            # Очистка данных таблицы
            cleaned_data = _clean_table_data(extracted)
            md = rows_to_markdown(cleaned_data)
            if md:
                tables.append(ParsedTable(
                    page_num=page.page_number,
                    bbox=table.bbox,
                    cells=_build_cells_from_table(cleaned_data, table.bbox),
                    markdown=md,
                    is_borderless=False
                ))
    except Exception as e:
        print(f"Warning: find_tables failed: {e}")
    
    # Фоллбек на старый метод
    if not tables:
        extracted = page.extract_tables()
        for table_data in extracted:
            if not table_data or len(table_data) < 2:
                continue
            cleaned_data = _clean_table_data(table_data)
            md = rows_to_markdown(cleaned_data)
            if md:
                # Улучшенный bbox расчет
                bbox = _calculate_table_bbox_from_data(cleaned_data, page)
                tables.append(ParsedTable(
                    page_num=page.page_number,
                    bbox=bbox,
                    cells=_build_cells_from_table(cleaned_data, bbox),
                    markdown=md,
                    is_borderless=False
                ))
    
    # Дополнительно ищем таблицы без границ
    borderless = detect_borderless_tables(page)
    for bt in borderless:
        # Проверяем, не пересекается ли с уже найденными
        if not any(t.overlaps_with_bbox(bt.bbox, 0.5) for t in tables):
            tables.append(bt)
    
    return tables

def detect_borderless_tables(page: pdfplumber.page.Page, vertical_threshold=30.0, horizontal_threshold=10.0) -> List[ParsedTable]:
    """Улучшенный поиск таблиц без границ с кластеризацией и анализом структуры."""
    words = page.extract_words(keep_blank_chars=True, use_text_flow=True)
    if not words:
        return []

    # Группировка по строкам с учетом вертикального выравнивания
    rows_dict = defaultdict(list)
    for w in words:
        y_key = round(w['top'], 1)
        rows_dict[y_key].append(w)
    
    sorted_y_keys = sorted(rows_dict.keys())
    
    found_tables = []
    current_block_words = []
    last_col_count = 0
    table_start_y = None
    
    for y_key in sorted_y_keys:
        row_words = sorted(rows_dict[y_key], key=lambda w: w['x0'])
        
        # Разбиваем строку на ячейки с адаптивным порогом
        cell_texts = _split_row_into_cells_adaptive(row_words, vertical_threshold)
        non_empty_cells = [c for c in cell_texts if c.strip()]
        col_count = len(non_empty_cells)
        
        if col_count >= 2:
            if last_col_count == 0:
                current_block_words = [row_words]
                table_start_y = y_key
            elif _is_compatible_row(col_count, last_col_count, non_empty_cells):
                current_block_words.append(row_words)
            else:
                if len(current_block_words) >= 3 and _is_valid_table_structure(current_block_words):
                    found_tables.append(_build_table_from_words_advanced(page, current_block_words, table_start_y, y_key))
                current_block_words = [row_words]
                table_start_y = y_key
            last_col_count = col_count
        else:
            if len(current_block_words) >= 3 and _is_valid_table_structure(current_block_words):
                found_tables.append(_build_table_from_words_advanced(page, current_block_words, table_start_y, y_key))
            current_block_words = []
            last_col_count = 0
            table_start_y = None

    # Последний блок
    if len(current_block_words) >= 3 and _is_valid_table_structure(current_block_words):
        found_tables.append(_build_table_from_words_advanced(page, current_block_words, table_start_y, sorted_y_keys[-1]))
        
    return found_tables

def _clean_table_data(table_data: List[List[str]]) -> List[List[str]]:
    """Очистка данных таблицы от артефактов с валидацией."""
    cleaned = []
    for row in table_data:
        cleaned_row = []
        has_valid_cell = False
        
        for cell in row:
            if cell is None:
                cell = ""
            
            # Очистка пробелов
            cell = re.sub(r'\s+', ' ', str(cell)).strip()
            
            # Валидация ячейки
            if _is_garbage_cell(cell):
                cell = ""
            else:
                cell = _remove_watermarks(cell)
                has_valid_cell = True
            
            cleaned_row.append(cell)
        
        # Добавляем только если есть хотя бы одна осмысленная ячейка
        if has_valid_cell:
            cleaned.append(cleaned_row)
    
    return cleaned


def _is_garbage_cell(text: str) -> bool:
    """Проверка, является ли содержимое ячейки мусором."""
    if not text or len(text) < 2:
        return False  # Пустые ячейки нормальны
    
    # Очень длинный текст в ячейке
    if len(text) > 300:
        return True
    
    # Только цифры и базовые спецсимволы - обычно нормально
    if re.match(r'^[0-9\s\-\.\,°²³\*\+\=\:\;\|\/\(\)]+$', text):
        return False
    
    # Много странных спецсимволов
    special_count = sum(1 for c in text if not c.isalnum() and c not in ' -.,;:()%')
    if special_count / len(text) > 0.4:
        return True
    
    # Почти всё спецсимволы
    alnum_count = sum(1 for c in text if c.isalnum())
    if alnum_count / len(text) < 0.15:
        return True
    
    return False

def _remove_watermarks(text: str) -> str:
    """Удаляет водяные знаки и повторяющийся текст."""
    # Простые паттерны водяных знаков
    patterns = [
        r'\b(?:draft|черновик|confidential|sample|proof|watermark)\b',
        r'\bстр\.\s*\d+\b',
        r'\bpage\s*\d+\b',
        r'\b\d{4}-\d{2}-\d{2}\b',
        r'\[\s*\d+\s*\]',
    ]
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', text).strip()

def _calculate_table_bbox_from_data(table_data: List[List[str]], page: pdfplumber.page.Page) -> Tuple[float, float, float, float]:
    """Расчет bbox таблицы из данных."""
    words = page.extract_words()
    if not words:
        return (0, 0, page.width, page.height)
    
    # Находим слова, соответствующие ячейкам таблицы
    table_words = []
    for row in table_data:
        for cell in row:
            if cell.strip():
                # Ищем слова, содержащиеся в ячейке
                for word in words:
                    if cell.strip() in word['text'] or word['text'] in cell.strip():
                        table_words.append(word)
    
    if not table_words:
        return (0, 0, page.width, page.height)
    
    x0 = min(w['x0'] for w in table_words)
    top = min(w['top'] for w in table_words)
    x1 = max(w['x1'] for w in table_words)
    bottom = max(w['bottom'] for w in table_words)
    
    return (x0, top, x1, bottom)

def _build_cells_from_table(table_data: List[List[str]], bbox: Tuple[float, float, float, float]) -> List[TableCell]:
    """Создает объекты TableCell из данных таблицы."""
    cells = []
    x0, top, x1, bottom = bbox
    row_height = (bottom - top) / len(table_data) if table_data else 1
    col_width = (x1 - x0) / max(len(row) for row in table_data) if table_data else 1
    
    for i, row in enumerate(table_data):
        for j, cell_text in enumerate(row):
            if cell_text.strip():
                cell_bbox = (
                    x0 + j * col_width,
                    top + i * row_height,
                    x0 + (j + 1) * col_width,
                    top + (i + 1) * row_height
                )
                cells.append(TableCell(
                    text=cell_text,
                    row=i,
                    col=j,
                    bbox=cell_bbox
                ))
    return cells

def _split_row_into_cells_adaptive(words: list, base_threshold: float) -> list:
    """Адаптивное разбиение строки на ячейки с учетом плотности слов."""
    if not words:
        return []
    
    # Вычисляем среднее расстояние между словами
    gaps = []
    for i in range(1, len(words)):
        gap = words[i]['x0'] - words[i-1]['x1']
        if gap > 0:
            gaps.append(gap)
    
    if gaps:
        avg_gap = sum(gaps) / len(gaps)
        threshold = max(base_threshold, avg_gap * 1.5)  # Адаптивный порог
    else:
        threshold = base_threshold
    
    cells = []
    current_cell_words = [words[0]]
    current_x1 = words[0]['x1']
    
    for w in words[1:]:
        if w['x0'] - current_x1 > threshold:
            cells.append(" ".join(cw['text'] for cw in current_cell_words))
            current_cell_words = [w]
            current_x1 = w['x1']
        else:
            current_cell_words.append(w)
            current_x1 = max(current_x1, w['x1'])
    
    cells.append(" ".join(cw['text'] for cw in current_cell_words))
    return cells

def _is_compatible_row(new_col_count: int, last_col_count: int, cells: List[str]) -> bool:
    """Проверяет совместимость строки с предыдущими."""
    if abs(new_col_count - last_col_count) <= 1:
        return True
    # Дополнительная проверка на пустые ячейки в конце
    if new_col_count > last_col_count and all(not cell.strip() for cell in cells[last_col_count:]):
        return True
    return False

def _is_valid_table_structure(block_rows_words: List[List[Dict]]) -> bool:
    """Проверяет, является ли блок слов валидной таблицей."""
    if len(block_rows_words) < 3:
        return False
    
    # Проверяем консистентность колонок
    col_counts = [len(_split_row_into_cells_adaptive(row, 30.0)) for row in block_rows_words]
    avg_cols = sum(col_counts) / len(col_counts)
    return all(abs(c - avg_cols) <= 1 for c in col_counts)

def _build_table_from_words_advanced(page: pdfplumber.page.Page, block_rows_words: List[List[Dict]], start_y: float, end_y: float) -> ParsedTable:
    """Улучшенное построение таблицы из слов с учетом структуры."""
    all_words_flat = [w for row in block_rows_words for w in row]
    
    if not all_words_flat:
        return None
        
    x0 = min(w['x0'] for w in all_words_flat)
    top = min(w['top'] for w in all_words_flat)
    x1 = max(w['x1'] for w in all_words_flat)
    bottom = max(w['bottom'] for w in all_words_flat)
    
    # Строим данные таблицы
    table_data = []
    for row_words in block_rows_words:
        row_words_sorted = sorted(row_words, key=lambda w: w['x0'])
        cells = _split_row_into_cells_adaptive(row_words_sorted, 30.0)
        cleaned_cells = [re.sub(r'\s+', ' ', c).strip() for c in cells if c.strip()]
        if cleaned_cells:
            table_data.append(cleaned_cells)
    
    md = rows_to_markdown(table_data)
    
    return ParsedTable(
        page_num=page.page_number,
        bbox=(x0, top, x1, bottom),
        cells=_build_cells_from_table(table_data, (x0, top, x1, bottom)),
        markdown=md,
        is_borderless=True
    )

def merge_tables_across_pages(tables: List[ParsedTable]) -> List[ParsedTable]:
    """Объединяет таблицы, разорванные между страницами."""
    if not tables:
        return tables
    
    merged = []
    current_table = None
    
    for table in sorted(tables, key=lambda t: (t.page_num, t.bbox[1])):
        if current_table is None:
            current_table = table
            continue
        
        # Проверяем, можно ли объединить
        if (_tables_can_merge(current_table, table) and 
            _content_suggests_continuation(current_table, table)):
            current_table = _merge_two_tables(current_table, table)
        else:
            merged.append(current_table)
            current_table = table
    
    if current_table:
        merged.append(current_table)
    
    return merged

def _tables_can_merge(t1: ParsedTable, t2: ParsedTable) -> bool:
    """Проверяет, можно ли объединить две таблицы."""
    # Проверяем страницы и позицию
    if t1.page_num >= t2.page_num:
        return False
    
    # Проверяем структуру колонок
    if not t1.cells or not t2.cells:
        return False
    
    t1_cols = max(cell.col for cell in t1.cells) + 1
    t2_cols = max(cell.col for cell in t2.cells) + 1
    
    return abs(t1_cols - t2_cols) <= 1

def _content_suggests_continuation(t1: ParsedTable, t2: ParsedTable) -> bool:
    """Проверяет, указывает ли контент на продолжение таблицы."""
    # Простая эвристика: если последняя строка первой таблицы похожа на заголовок второй
    if not t1.cells or not t2.cells:
        return False
    
    last_row_t1 = [cell for cell in t1.cells if cell.row == max(c.row for c in t1.cells)]
    first_row_t2 = [cell for cell in t2.cells if cell.row == min(c.row for c in t2.cells)]
    
    # Сравниваем типы данных (числа, текст и т.д.)
    return _rows_have_similar_structure(last_row_t1, first_row_t2)

def _rows_have_similar_structure(row1: List[TableCell], row2: List[TableCell]) -> bool:
    """Проверяет схожесть структуры строк."""
    if len(row1) != len(row2):
        return False
    
    # Простая проверка типов данных
    def get_cell_type(cell: TableCell) -> str:
        text = cell.text.strip()
        if re.match(r'^\d+(\.\d+)?$', text):
            return 'number'
        elif re.match(r'^\d{1,2}[./-]\d{1,2}[./-]\d{2,4}$', text):
            return 'date'
        else:
            return 'text'
    
    types1 = [get_cell_type(cell) for cell in row1]
    types2 = [get_cell_type(cell) for cell in row2]
    
    return types1 == types2

def _merge_two_tables(t1: ParsedTable, t2: ParsedTable) -> ParsedTable:
    """Объединяет две таблицы."""
    # Объединяем bbox
    x0 = min(t1.bbox[0], t2.bbox[0])
    top = min(t1.bbox[1], t2.bbox[1])
    x1 = max(t1.bbox[2], t2.bbox[2])
    bottom = max(t1.bbox[3], t2.bbox[3])
    
    # Объединяем ячейки
    all_cells = t1.cells + t2.cells
    
    # Корректируем номера строк для второй таблицы
    max_row_t1 = max(cell.row for cell in t1.cells) + 1
    for cell in all_cells:
        if cell in t2.cells:
            cell.row += max_row_t1
    
    # Объединяем markdown
    md1_lines = t1.markdown.split('\n')
    md2_lines = t2.markdown.split('\n')
    
    # Удаляем заголовок из второй таблицы
    if len(md2_lines) > 2:
        md2_lines = md2_lines[2:]
    
    combined_md = '\n'.join(md1_lines + md2_lines)
    
    return ParsedTable(
        page_num=t1.page_num,  # Используем номер первой страницы
        bbox=(x0, top, x1, bottom),
        cells=all_cells,
        markdown=combined_md,
        is_borderless=t1.is_borderless or t2.is_borderless
    )

def rows_to_markdown(rows: List[List[str]]) -> str:
    """Конвертация в Markdown с валидацией структуры."""
    if not rows or len(rows) < 2:
        return ""
    
    # Очистка
    cleaned_rows = []
    max_cols = 0
    for row in rows:
        cleaned_row = [re.sub(r'\s+', ' ', cell).strip() if cell else "" for cell in row]
        if any(cleaned_row):  # Не полностью пустые
            cleaned_rows.append(cleaned_row)
            max_cols = max(max_cols, len(cleaned_row))
    
    if len(cleaned_rows) < 2 or max_cols < 2:
        return ""
    
    # Проверка: таблица не должна быть полностью числами/спецсимволами
    non_empty_cells = sum(1 for row in cleaned_rows for cell in row if cell.strip())
    if non_empty_cells == 0:
        return ""
    
    # Выравнивание
    aligned_rows = [row + [""] * (max_cols - len(row)) for row in cleaned_rows]
    
    # Проверка заголовка - если первая строка слишком короткая/странная, пропускаем
    header = aligned_rows[0]
    if all(len(cell) > 50 or (not cell.strip()) for cell in header):
        return ""
    
    # Markdown
    lines = []
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * max_cols) + " |")
    for row in aligned_rows[1:]:
        lines.append("| " + " | ".join(row) + " |")
    
    return "\n".join(lines)