from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pdfplumber


@dataclass(frozen=True)
class TableCandidate:
    bbox: tuple[float, float, float, float]
    markdown: str


def _clean_cell(text: str | None) -> str:
    if text is None:
        return ""
    text = text.replace("\xa0", " ")
    # Newlines inside cell → space
    text = re.sub(r"\s*\n\s*", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"[ \t]+", " ", text)
    # Fix broken words: "естественны й" → "естественный"
    # Pattern: letter immediately followed by space then letter where both are same script
    text = re.sub(r"([а-яёА-ЯЁa-zA-Z])\s([а-яёА-ЯЁa-zA-Z])", _maybe_join_broken, text)
    # Fix broken numbers: "3 987,88" — only if no other word context
    text = re.sub(r"(\d)\s*,\s*(\d)", r"\1,\2", text)
    text = re.sub(r"(\d)\s+(\d{3})\b", r"\1 \2", text)  # keep thousands separator spaces
    # Fix space before punctuation
    text = re.sub(r"\s+([,.;:!?%°²³])", r"\1", text)
    return text.strip()


def _maybe_join_broken(m: re.Match) -> str:
    """
    Join two letters separated by a single space only when second letter is lowercase.
    This catches PDF column-wrap artifacts like "естественны й" or "Multi-tie red".
    Preserves "New York", "Q1 Result" etc. where second word starts with uppercase.
    """
    a, b = m.group(1), m.group(2)
    # Keep the space if second char is uppercase (likely start of a new word)
    if b.isupper():
        return m.group(0)
    return a + b


def _propagate_merged_cells(rows: list[list[str]]) -> list[list[str]]:
    """
    pdfplumber returns None for cells that are part of a horizontal or vertical merge.
    Propagate the last seen non-empty value rightward and downward to fill them.
    This matches the hackathon requirement: split merged cells by copying content.
    """
    if not rows:
        return rows

    width = max(len(row) for row in rows)
    # Pad all rows to same width
    padded = [row + [None] * (width - len(row)) for row in rows]  # type: ignore[list-item]

    # Horizontal propagation: fill None with previous non-None in same row
    for row in padded:
        last = ""
        for j in range(width):
            val = row[j]
            if val is not None and val.strip():
                last = val
            elif val is None or not val.strip():
                row[j] = last

    # Vertical propagation: fill empty cells with value from row above
    for j in range(width):
        last = ""
        for row in padded:
            val = row[j]
            if val and val.strip():
                last = val
            elif not val:
                row[j] = last

    return padded  # type: ignore[return-value]


def _rows_to_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""

    # Remove fully empty rows
    rows = [r for r in rows if any(c.strip() for c in r)]
    if not rows:
        return ""

    width = max(len(r) for r in rows)
    # Normalize row widths
    rows = [r + [""] * (width - len(r)) for r in rows]

    # Escape pipe chars inside cells
    def fmt(cell: str) -> str:
        return cell.replace("|", "\\|")

    header = rows[0]
    body = rows[1:]

    lines = [
        "| " + " | ".join(fmt(c) for c in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(fmt(c) for c in row) + " |")
    return "\n".join(lines)


def extract_tables(pdf_path: Path, page_number: int) -> List[TableCandidate]:
    """
    Extract tables from a PDF page using pdfplumber.
    Returns list of TableCandidate with bbox and markdown string.
    """
    candidates: List[TableCandidate] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_number >= len(pdf.pages):
                return candidates
            page = pdf.pages[page_number]
            tables = page.find_tables()
            for tbl in tables:
                try:
                    raw_rows = tbl.extract()
                    if not raw_rows or len(raw_rows) < 2:
                        continue

                    # Propagate merged cells (None → copied content)
                    propagated = _propagate_merged_cells(raw_rows)

                    # Clean each cell
                    cleaned = [[_clean_cell(cell) for cell in row] for row in propagated]

                    # Skip tables where almost all cells are empty
                    total = sum(len(r) for r in cleaned)
                    filled = sum(1 for r in cleaned for c in r if c.strip())
                    if total > 0 and filled / total < 0.1:
                        continue

                    md = _rows_to_markdown(cleaned)
                    if not md:
                        continue

                    bbox = tbl.bbox
                    candidates.append(TableCandidate(bbox=bbox, markdown=md))
                except Exception:
                    continue
    except Exception:
        pass
    return candidates
