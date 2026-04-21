import re
from typing import List, Tuple, Optional, Dict, Any
import pdfplumber
from pdfplumber.table import TableFinder, TableSettings

def is_valid_cell_text(text: str) -> bool:
    """
    Проверка, является ли текст ячейки валидным (не мусор и не артефакт).
    Разрешены: кириллица, латиница, цифры, основные знаки препинания и спецсимволы (±, °, ≈, ₽, руб).
    """
    if not text or not text.strip():
        return False
    
    text = text.strip()
    
    # Если текст очень короткий (1-2 символа), проверяем строже
    if len(text) < 3:
        # Разрешаем короткие слова, числа, даты, обозначения
        if re.match(r'^[a-zA-Zа-яА-Я0-9°±≈]+$', text):
            return True
        # Если это просто знак препинания или мусорный символ - отклоняем
        if re.match(r'^[^a-zA-Zа-яА-Я0-9]+$', text):
            return False

    # Подсчет "плохих" символов
    bad_count = 0
    total_count = len(text)
    
    for char in text:
        code = ord(char)
        # Разрешенные диапазоны:
        # Базовая латиница (ASCII)
        if 0x0020 <= code <= 0x007F:
            continue
        # Кириллица (основная и дополнительная)
        if 0x0400 <= code <= 0x052F:
            continue
        # Латиница-1 Дополнение (диакритика)
        if 0x00C0 <= code <= 0x00FF:
            continue
        # Специальные разрешенные символы
        if char in '°²³₽$€£¥©®™±≈×÷¬¶§':
            continue
        # Основные знаки препинания
        if char in '.,;:!?()[]{}\'"«»„"–—/\\|@#%&*+-=_<>':
            continue
            
        # Все остальное считаем подозрительным
        bad_count += 1

    # Порог мусора: если более 20% символов непонятные - считаем ячейку битой
    # Исключение: очень короткие строки, где 1 символ уже много
    if total_count > 0:
        if bad_count / total_count > 0.20:
            return False
            
    return True

def clean_cell_text(text: str) -> str:
    """
    Очистка текста ячейки от артефактов переноса строк и лишних пробелов.
    Сохраняет важные спецсимволы.
    """
    if not text:
        return ""
    
    # Замена переносов строк внутри ячейки на пробел (если это не список)
    text = re.sub(r'\s*\n\s*', ' ', text)
    
    # Удаление множественных пробелов
    text = re.sub(r'\s+', ' ', text)
    
    # Исправление частых артефактов OCR (если они есть), но аккуратно
    # Например, замена '0' на 'О' в русском тексте часто ошибочна, оставляем как есть
    
    return text.strip()

def extract_tables_from_page(page: pdfplumber.page.Page) -> List[Dict[str, Any]]:
    """
    Извлечение таблиц со страницы с использованием нескольких стратегий.
    Возвращает список словарей с данными таблицы и её координатами.
    """
    tables_data = []
    
    # Стратегия 1: Стандартные таблицы с явными границами
    # Используем настройки для лучшего обнаружения
    settings = TableSettings(
        vertical_strategy='lines',
        horizontal_strategy='lines',
        explicit_vertical_lines=None,
        explicit_horizontal_lines=None,
        snap_tolerance=3,
        join_tolerance=3,
        edge_min_length=3,
        min_words_vertical=1, # Таблица может быть из 1 слова в столбце
        min_words_horizontal=1,
        intersection_tolerance=3
    )
    
    try:
        finder = TableFinder(page, settings)
        for table in finder.tables:
            rows = table.extract()
            if not rows:
                continue
                
            # Фильтрация и очистка данных
            cleaned_rows = []
            has_valid_data = False
            
            for row in rows:
                cleaned_row = []
                row_has_content = False
                for cell in row:
                    if is_valid_cell_text(cell):
                        cleaned_cell = clean_cell_text(cell)
                        cleaned_row.append(cleaned_cell)
                        row_has_content = True
                    else:
                        cleaned_row.append("") # Пустая ячейка, но сохраняем структуру
                
                if row_has_content:
                    cleaned_rows.append(cleaned_row)
                    has_valid_data = True
            
            if has_valid_data and len(cleaned_rows) > 1: # Минимум заголовок + 1 строка
                tables_data.append({
                    'data': cleaned_rows,
                    'bbox': table.bbox,
                    'page': page.page_number
                })
    except Exception as e:
        # Если стандартный метод упал, пробуем упрощенный
        pass

    # Стратегия 2: Поиск таблиц без явных границ (по выравниванию текста)
    # Актуально, если стратегии 1 не хватило
    if not tables_data:
        try:
            # Пробуем найти по тексту с группировкой
            words = page.chars
            if not words:
                return tables_data
                
            # Эвристика: ищем группы слов, выровненных по сетке
            # Это упрощенная реализация, можно расширить
            pass 
        except Exception:
            pass
            
    return tables_data

def format_table_to_markdown(table_data: List[List[str]]) -> str:
    """
    Преобразование списка списков (таблицы) в Markdown формат.
    """
    if not table_data:
        return ""
    
    md_lines = []
    
    # Заголовок
    header = table_data[0]
    md_lines.append("| " + " | ".join(header) + " |")
    
    # Разделитель
    md_lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    
    # Тело таблицы
    for row in table_data[1:]:
        # Выравниваем длину строки с заголовком
        while len(row) < len(header):
            row.append("")
        row = row[:len(header)]
        
        # Экранирование вертикальных черт внутри ячеек
        cleaned_row = [cell.replace("|", "\\|") for cell in row]
        md_lines.append("| " + " | ".join(cleaned_row) + " |")
    
    return "\n".join(md_lines)