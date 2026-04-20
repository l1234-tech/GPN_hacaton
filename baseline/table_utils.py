# baseline/table_utils.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import pdfplumber

from layout_utils import bbox_area, intersection_area
from postprocess import clean_cell_text, table_to_markdown


@dataclass(frozen=True)
class TableCandidate:
    bbox: tuple[float, float, float, float]
    markdown: str
    score: float
    rows: List[List[str]]


def _looks_like_real_table(
    data: list[list[str]],
    bbox: tuple[float, float, float, float],
    page_area: float,
) -> bool:
    if len(data) < 2:
        return False
    width = max(len(row) for row in data)
    if width < 2:
        return False

    filled = sum(bool(clean_cell_text(cell)) for row in data for cell in row)
    total = max(1, sum(len(row) for row in data))
    fill_ratio = filled / total
    area_ratio = bbox_area(bbox) / max(page_area, 1.0)

    if fill_ratio < 0.12:
        return False
    if area_ratio > 0.7:
        return False
    return True


def _cluster_positions(values: list[float], tolerance: float) -> list[float]:
    if not values:
        return []
    values = sorted(values)
    clusters: list[list[float]] = [[values[0]]]
    for value in values[1:]:
        if abs(value - clusters[-1][-1]) <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return [sum(cluster) / len(cluster) for cluster in clusters]


def _group_words_to_rows(words: list[dict], tolerance: float = 5.0) -> list[list[dict]]:
    rows: list[list[dict]] = []
    for word in sorted(words, key=lambda item: (item["top"], item["x0"])):
        if not rows or abs(word["top"] - rows[-1][0]["top"]) > tolerance:
            rows.append([word])
        else:
            rows[-1].append(word)
    return rows


def _rows_to_text_table(page: pdfplumber.page.Page, rows: list[list[dict]]) -> TableCandidate | None:
    rows = list(rows)
    while rows and clean_cell_text(" ".join(word["text"] for word in rows[0])).lower().startswith(("раздел", "глава", "рис.")):
        rows.pop(0)
    while rows and clean_cell_text(" ".join(word["text"] for word in rows[-1])).lower().startswith(("раздел", "глава", "рис.")):
        rows.pop()
    if len(rows) < 3:
        return None

    x_values = [word["x0"] for row in rows for word in row]
    anchors = _cluster_positions(x_values, tolerance=35)
    if len(anchors) < 2:
        return None

    structured_rows: list[list[str]] = []
    dense_rows = 0

    for row in rows:
        cells = [""] * len(anchors)
        for word in sorted(row, key=lambda item: item["x0"]):
            index = min(range(len(anchors)), key=lambda idx: abs(word["x0"] - anchors[idx]))
            if abs(word["x0"] - anchors[index]) > 60:
                continue
            piece = clean_cell_text(word["text"])
            if not piece:
                continue
            cells[index] = f"{cells[index]} {piece}".strip()

        non_empty = sum(bool(cell) for cell in cells)
        if non_empty >= 2:
            dense_rows += 1
        structured_rows.append(cells)

    if dense_rows < max(2, len(rows) - 1):
        return None

    x0 = min(word["x0"] for row in rows for word in row)
    top = min(word["top"] for row in rows for word in row)
    x1 = max(word["x1"] for row in rows for word in row)
    bottom = max(word["bottom"] for row in rows for word in row)
    bbox = (x0, top, x1, bottom)
    markdown = table_to_markdown(structured_rows)
    score = dense_rows + len(anchors)
    return TableCandidate(bbox=bbox, markdown=markdown, score=score, rows=structured_rows)


def _extract_borderless_tables(page: pdfplumber.page.Page) -> list[TableCandidate]:
    words = [
        word
        for word in page.extract_words(use_text_flow=False, keep_blank_chars=False)
        if word["x0"] > 20
        and word["x1"] < page.width - 20
        and word["top"] > 40
        and word["bottom"] < page.height - 40
    ]
    rows = _group_words_to_rows(words)
    candidates: list[TableCandidate] = []

    indexed_rows = [(index, row) for index, row in enumerate(rows) if len(row) >= 3]
    for start in range(len(indexed_rows)):
        for end in range(start + 3, min(len(indexed_rows), start + 30) + 1):
            subset = indexed_rows[start:end]
            if subset[-1][0] - subset[0][0] > len(subset) + 8:
                continue
            candidate = _rows_to_text_table(page, [row for _, row in subset])
            if candidate is not None:
                candidates.append(candidate)

    return candidates


def _extract_ruled_tables(page: pdfplumber.page.Page) -> list[TableCandidate]:
    candidates: list[TableCandidate] = []
    page_area = page.width * page.height
    strategies = [
        {"vertical_strategy": "lines", "horizontal_strategy": "lines"},
        {"vertical_strategy": "lines_strict", "horizontal_strategy": "lines_strict"},
        {"vertical_strategy": "text", "horizontal_strategy": "text"},
    ]

    for settings in strategies:
        for table in page.find_tables(table_settings=settings):
            data = table.extract() or []
            bbox = tuple(table.bbox)
            if not _looks_like_real_table(data, bbox, page_area):
                continue
            markdown = table_to_markdown(data)
            if not markdown:
                continue
            score = len(data) * max(len(row) for row in data)
            candidates.append(TableCandidate(bbox=bbox, markdown=markdown, score=score, rows=data))
    return candidates


def _dedupe_tables(candidates: list[TableCandidate]) -> list[TableCandidate]:
    unique: list[TableCandidate] = []
    for candidate in sorted(candidates, key=lambda item: (-item.score, item.bbox[1], item.bbox[0])):
        duplicate = False
        for existing in unique:
            overlap = intersection_area(candidate.bbox, existing.bbox) / max(
                1.0,
                min(bbox_area(candidate.bbox), bbox_area(existing.bbox)),
            )
            if overlap > 0.3:
                duplicate = True
                break
        if not duplicate:
            unique.append(candidate)
    return sorted(unique, key=lambda item: (item.bbox[1], item.bbox[0]))


def extract_tables(pdf_path: Path, page_number: int, page_width: float, page_height: float) -> list[TableCandidate]:
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_number]
        candidates = _extract_ruled_tables(page)
        candidates.extend(_extract_borderless_tables(page))
        return _dedupe_tables(candidates)