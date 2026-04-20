# baseline/postprocess.py
from __future__ import annotations

import re


def clean_cell_text(value: str | None) -> str:
    if value is None:
        return ""
    value = value.replace("\xa0", " ").replace("\n", " ")
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"\s+([,.;:!?%)\]])", r"\1", value)
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
    """Преобразует таблицу (список списков) в корректный Markdown."""
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

    header_line: list[str] = []
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
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.upper() in ("ЧЕРНОВИК", "DRAFT", "CONFIDENTIAL", "SAMPLE"):
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