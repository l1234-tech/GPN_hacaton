# baseline/table_model.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import pypdfium2 as pdfium
from gmft import AutoTableDetector, CroppedTable

from postprocess import table_to_markdown


@dataclass(frozen=True)
class TableCandidate:
    bbox: tuple[float, float, float, float]
    markdown: str
    score: float
    rows: List[List[str]]


def _clean_dataframe_cells(rows: List[List[str]]) -> List[List[str]]:
    """Удаляет пустые строки и столбцы, нормализует пробелы."""
    # Удаляем полностью пустые строки
    rows = [row for row in rows if any(cell and str(cell).strip() for cell in row)]
    if not rows:
        return []
    # Удаляем столбцы, где все ячейки пусты
    col_count = max(len(row) for row in rows)
    valid_cols = []
    for col_idx in range(col_count):
        if any(
            col_idx < len(row) and row[col_idx] and str(row[col_idx]).strip()
            for row in rows
        ):
            valid_cols.append(col_idx)
    if not valid_cols:
        return []
    cleaned = []
    for row in rows:
        new_row = []
        for col_idx in valid_cols:
            val = str(row[col_idx]) if col_idx < len(row) else ""
            new_row.append(val.strip())
        cleaned.append(new_row)
    return cleaned


class GMFTTableExtractor:
    def __init__(self, confidence_threshold: float = 0.5, page_scale: float = 2.5):
        self.detector = AutoTableDetector()
        self.confidence_threshold = confidence_threshold
        self.page_scale = page_scale

    def extract_tables_from_page(
        self,
        pdf_path: Path,
        page_number: int,
        page_width: float,
        page_height: float,
    ) -> List[TableCandidate]:
        candidates = []
        try:
            pdf = pdfium.PdfDocument(str(pdf_path))
            if page_number >= len(pdf):
                return []
            page = pdf[page_number]
            pil_image = page.render(scale=self.page_scale).to_pil()

            detected_tables = self.detector.extract(pil_image)
            print(f"    [gmft] page {page_number+1}: found {len(detected_tables)} tables")
            for det_table in detected_tables:
                if det_table.confidence < self.confidence_threshold:
                    continue
                try:
                    cropped = CroppedTable.from_detected(det_table, pil_image)
                    df = cropped.df()
                    rows = [df.columns.tolist()] + df.values.tolist()
                    rows = _clean_dataframe_cells(rows)
                    if len(rows) < 2:
                        continue
                    scale_x = page_width / pil_image.width
                    scale_y = page_height / pil_image.height
                    bbox = (
                        det_table.bbox[0] * scale_x,
                        det_table.bbox[1] * scale_y,
                        det_table.bbox[2] * scale_x,
                        det_table.bbox[3] * scale_y,
                    )
                    markdown = table_to_markdown(rows)
                    score = det_table.confidence * len(rows) * max(len(row) for row in rows)
                    candidates.append(TableCandidate(bbox=bbox, markdown=markdown, score=score, rows=rows))
                except Exception as e:
                    print(f"    [gmft] error processing table: {e}")
                    continue
        except Exception as e:
            print(f"    [gmft] page {page_number+1} failed: {e}")
        return candidates


def extract_tables_with_gmft(pdf_path: Path, page_number: int, page_width: float, page_height: float) -> List[TableCandidate]:
    extractor = GMFTTableExtractor()
    return extractor.extract_tables_from_page(pdf_path, page_number, page_width, page_height)