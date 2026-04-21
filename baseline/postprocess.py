# baseline/postprocess.py
from __future__ import annotations

import re


def _maybe_join_broken(m: re.Match) -> str:
    a, b = m.group(1), m.group(2)
    # Keep space if second char is uppercase (likely separate word)
    if b.isupper():
        return m.group(0)
    return a + b


def clean_cell_text(value: str | None) -> str:
    if value is None:
        return ""
    value = value.replace("\xa0", " ")
    value = re.sub(r"\s*\n\s*", " ", value)
    value = re.sub(r"[ \t]+", " ", value)
    # Fix broken words split across lines/columns: "есте ственный" → "естественный"
    value = re.sub(r"([а-яёА-ЯЁa-zA-Z])\s([а-яёА-ЯЁa-zA-Z])", _maybe_join_broken, value)
    value = re.sub(r"\s+([,.;:!?%°²³)\]])", r"\1", value)
    return value.strip()


def fill_merged_header_cells(rows: list[list[str]]) -> list[list[str]]:
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


def infer_header_rows(rows: list[list[str]]) -> int:
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


def table_to_markdown(rows: list[list[str]]) -> str:
    # Удаляем полностью пустые строки и столбцы
    rows = [row for row in rows if any(cell.strip() for cell in row)]
    if not rows:
        return ""
    # Транспонируем, удаляем пустые столбцы, возвращаем обратно
    transposed = list(zip(*rows))
    transposed = [list(col) for col in transposed if any(cell.strip() for cell in col)]
    if not transposed:
        return ""
    rows = [list(row) for row in zip(*transposed)]

    cleaned = [[clean_cell_text(cell) for cell in row] for row in rows]
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


def normalize_text_block(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
    text = re.sub(r"\s+([,.;:!?%)\]])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_markdown(text: str) -> str:
    text = text.replace("\r\n", "\n")
    # Удаление строк с водяными знаками (на всякий случай)
    watermark_keywords = ["ЧЕРНОВИК", "DRAFT", "CONFIDENTIAL", "SAMPLE", "ОБРАЗЕЦ",
                          "КОНФИДЕНЦИАЛЬНО", "НЕ ДЛЯ РАСПРОСТРАНЕНИЯ"]
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        upper = line.upper()
        if any(kw in upper for kw in watermark_keywords):
            continue
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)

    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.replace("\\", "/")
    text = re.sub(r"(?<!\n)(#{1,6}\s)", r"\n\n\1", text)
    text = re.sub(r"(?<!\n)(\|)", r"\n\n\1", text)
    text = re.sub(r"(?<!\n)(!\[Image\]\(images/)", r"\n\n\1", text)
    return text.strip() + "\n"


def merge_split_tables_in_markdown(text: str) -> str:
    """Склеивает соседние таблицы с одинаковым числом столбцов."""
    lines = text.split('\n')
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith('|') and stripped.endswith('|') and '---' not in stripped:
            start = i
            while i < len(lines) and lines[i].strip() != '':
                i += 1
            end = i
            current = lines[start:end]
            # Смотрим следующую непустую строку
            k = end + 1
            while k < len(lines) and lines[k].strip() == '':
                k += 1
            if k < len(lines) and lines[k].strip().startswith('|'):
                m = k
                while m < len(lines) and lines[m].strip() != '':
                    m += 1
                next_tbl = lines[k:m]
                # Проверяем число столбцов
                sep_curr = next((ln for ln in current if '---' in ln), None)
                sep_next = next((ln for ln in next_tbl if '---' in ln), None)
                if sep_curr and sep_next:
                    if sep_curr.count('|') == sep_next.count('|'):
                        # Склеиваем: убираем заголовок второй таблицы
                        sep_idx = next((idx for idx, ln in enumerate(next_tbl) if '---' in ln), None)
                        if sep_idx is not None:
                            body = next_tbl[sep_idx+1:]
                            new_lines.extend(current + body)
                            i = m
                            if m < len(lines) and lines[m].strip() == '':
                                new_lines.append(lines[m])
                                i = m + 1
                            continue
            new_lines.extend(current)
            if end < len(lines):
                new_lines.append(lines[end])
            i = end + 1
        else:
            new_lines.append(line)
            i += 1
    return '\n'.join(new_lines)