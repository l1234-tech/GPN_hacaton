# baseline/postprocess.py
import re
from typing import List, Dict, Any

def clean_cell_text(value: str | None) -> str:
    """Очистка текста ячейки таблицы."""
    if value is None:
        return ""
    value = str(value).replace("\xa0", " ").replace("\n", " ").replace("\t", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip()

def table_to_markdown(rows: List[List[str]]) -> str:
    """Конвертация таблицы в Markdown."""
    if not rows or not any(any(cell.strip() for cell in row) for row in rows):
        return ""
    
    # Очистка строк
    cleaned_rows = []
    for row in rows:
        cleaned_row = [clean_cell_text(cell) for cell in row]
        if any(cleaned_row):
            cleaned_rows.append(cleaned_row)
    
    if len(cleaned_rows) < 2:
        return ""
    
    # Выравнивание колонок
    max_cols = max(len(row) for row in cleaned_rows)
    aligned_rows = [row + [""] * (max_cols - len(row)) for row in cleaned_rows]
    
    # Создание Markdown
    header = aligned_rows[0]
    body = aligned_rows[1:]
    
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |"
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    
    return "\n".join(lines)

def normalize_markdown(text: str) -> str:
    """Комплексная нормализация Markdown текста."""
    if not text:
        return ""
    
    # Разделение на строки
    lines = text.split('\n')
    normalized_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Обработка заголовков
        if line.startswith('#'):
            line = _normalize_header(line)
        
        # Обработка списков
        elif re.match(r'^[-*+]\s', line):
            line = _normalize_list_item(line)
        
        # Обработка таблиц (уже обработаны отдельно)
        
        # Общая очистка текста
        else:
            line = _normalize_text_line(line)
        
        normalized_lines.append(line)
    
    # Объединение параграфов
    result = _merge_paragraphs(normalized_lines)
    
    # Финальная очистка
    result = _final_cleanup(result)
    
    return result.strip() + "\n"

def _normalize_header(line: str) -> str:
    """Нормализация заголовков."""
    # Убираем лишние # в начале
    match = re.match(r'^(#{1,6})\s*(.+)$', line)
    if match:
        hashes, content = match.groups()
        content = _normalize_text_line(content)
        return f"{hashes} {content}"
    return line

def _normalize_list_item(line: str) -> str:
    """Нормализация элементов списка."""
    match = re.match(r'^([-*+])\s*(.+)$', line)
    if match:
        marker, content = match.groups()
        content = _normalize_text_line(content)
        return f"{marker} {content}"
    return line

def _normalize_text_line(line: str) -> str:
    """Нормализация обычной текстовой строки."""
    # Удаление водяных знаков
    line = _remove_watermarks(line)
    
    # Исправление дефисного переноса
    line = _fix_hyphenation(line)
    
    # Нормализация пробелов
    line = _normalize_spaces(line)
    
    # Исправление пунктуации
    line = _fix_punctuation(line)
    
    return line

def _remove_watermarks(text: str) -> str:
    """Удаление водяных знаков и повторяющегося текста."""
    # Распространенные водяные знаки
    watermark_patterns = [
        r'\b(?:draft|черновик|confidential|sample|proof|watermark|копия)\b',
        r'\bстр\.\s*\d+\b',
        r'\bстраница\s*\d+\b',
        r'\bpage\s*\d+\b',
        r'\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b',  # Даты
        r'\[\s*\d+\s*\]',  # Ссылки типа [1]
        r'\b\d{4}-\d{2}-\d{2}\b',  # Даты ISO
    ]
    
    for pattern in watermark_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    return text

def _fix_hyphenation(text: str) -> str:
    """Исправление дефисного переноса слов."""
    # Паттерн: слово- + перенос строки + продолжение
    text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
    
    # Паттерн: слово в конце строки с дефисом
    text = re.sub(r'(\w)-\s*$', r'\1', text, flags=re.MULTILINE)
    
    return text

def _normalize_spaces(text: str) -> str:
    """Нормализация пробелов и отступов."""
    # Замена неразрывных пробелов
    text = text.replace('\xa0', ' ')
    
    # Удаление лишних пробелов
    text = re.sub(r'\s+', ' ', text)
    
    # Удаление пробелов в начале и конце
    text = text.strip()
    
    return text

def _fix_punctuation(text: str) -> str:
    """Исправление пунктуации."""
    # Пробел перед знаками препинания
    text = re.sub(r'\s+([,.;:!?])', r'\1', text)
    
    # Пробел после открывающих скобок
    text = re.sub(r'([(\[])\s+', r'\1', text)
    
    # Пробел перед закрывающими скобками
    text = re.sub(r'\s+([)\]])', r'\1', text)
    
    return text

def _merge_paragraphs(lines: List[str]) -> str:
    """Объединение коротких строк в параграфы."""
    if not lines:
        return ""
    
    merged = [lines[0]]
    for line in lines[1:]:
        # Если предыдущая строка не заголовок/список и текущая короткая - объединяем
        if (not re.match(r'^(#|[-*+])', merged[-1]) and 
            not re.match(r'^(#|[-*+])', line) and
            len(line) < 80):
            merged[-1] += " " + line
        else:
            merged.append(line)
    
    return "\n\n".join(merged)

def _final_cleanup(text: str) -> str:
    """Финальная очистка текста."""
    # Удаление пустых строк в начале и конце
    text = text.strip()
    
    # Замена множественных переносов строк
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Удаление пробелов в конце строк
    lines = [line.rstrip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    return text

def normalize_text_block(text: str) -> str:
    """Нормализация блока текста с учетом контекста."""
    if not text:
        return ""
    
    # Разбиение на предложения
    sentences = _split_into_sentences(text)
    
    # Нормализация каждого предложения
    normalized_sentences = []
    for sentence in sentences:
        normalized = _normalize_text_line(sentence)
        if normalized:
            normalized_sentences.append(normalized)
    
    # Объединение предложений
    result = " ".join(normalized_sentences)
    
    # Финальная обработка
    result = _final_cleanup(result)
    
    return result

def _split_into_sentences(text: str) -> List[str]:
    """Разбиение текста на предложения."""
    # Простое разбиение по точкам, восклицательным и вопросительным знакам
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def postprocess_table_markdown(markdown: str) -> str:
    """Постобработка Markdown таблиц."""
    if not markdown:
        return ""
    
    lines = markdown.split('\n')
    processed_lines = []
    
    for line in lines:
        # Очистка ячеек от лишних пробелов
        if '|' in line:
            cells = [cell.strip() for cell in line.split('|')]
            # Удаление пустых ячеек в конце
            while cells and not cells[-1].strip():
                cells.pop()
            if cells:
                processed_lines.append('|'.join(cells))
        else:
            processed_lines.append(line)
    
    return '\n'.join(processed_lines)