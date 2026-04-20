# baseline/table_utils.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import fitz
import pandas as pd
import torch
from PIL import Image
from transformers import AutoImageProcessor, TableTransformerForObjectDetection

from layout_utils import bbox_area, intersection_area
from postprocess import table_to_markdown


@dataclass(frozen=True)
class TableCandidate:
    bbox: tuple[float, float, float, float]
    markdown: str
    score: float
    rows: List[List[str]]


class TableExtractor:
    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model = TableTransformerForObjectDetection.from_pretrained(
            "microsoft/table-transformer-structure-recognition"
        ).to(self.device)
        self.processor = AutoImageProcessor.from_pretrained(
            "microsoft/table-transformer-structure-recognition"
        )

    def extract_tables_from_page(
        self,
        page: fitz.Page,
        page_width: float,
        page_height: float,
        confidence_threshold: float = 0.7,
    ) -> List[TableCandidate]:
        """Извлекает таблицы со страницы с помощью Table Transformer."""
        # Рендерим страницу в PIL Image
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([img.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs, threshold=confidence_threshold, target_sizes=target_sizes
        )[0]

        # Группируем ячейки по строкам и столбцам
        rows: dict[int, List[dict]] = {}
        cols: dict[int, List[dict]] = {}
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            bbox = box.tolist()
            label_id = label.item()
            # label_id: 0 - table row, 1 - table column, 2 - table spanning cell (пока пропускаем)
            if label_id == 0:  # row
                row_num = len(rows)
                rows.setdefault(row_num, []).append({"bbox": bbox, "score": score.item()})
            elif label_id == 1:  # column
                col_num = len(cols)
                cols.setdefault(col_num, []).append({"bbox": bbox, "score": score.item()})

        if not rows or not cols:
            return []

        # Сортируем строки и столбцы по координатам
        sorted_rows = sorted(rows.items(), key=lambda kv: kv[1][0]["bbox"][1])  # по Y
        sorted_cols = sorted(cols.items(), key=lambda kv: kv[1][0]["bbox"][0])  # по X

        # Создаём сетку ячеек
        grid = [[None for _ in range(len(sorted_cols))] for _ in range(len(sorted_rows))]
        # Извлекаем текст из каждой ячейки (используем OCR? Пока просто текст из страницы по bbox)
        for r_idx, (_, row_data) in enumerate(sorted_rows):
            for c_idx, (_, col_data) in enumerate(sorted_cols):
                # Пересечение bbox строки и столбца даёт ячейку
                row_bbox = row_data[0]["bbox"]
                col_bbox = col_data[0]["bbox"]
                cell_bbox = (
                    max(row_bbox[0], col_bbox[0]),
                    max(row_bbox[1], col_bbox[1]),
                    min(row_bbox[2], col_bbox[2]),
                    min(row_bbox[3], col_bbox[3]),
                )
                # Нормализуем координаты к размеру страницы
                scale_x = page_width / pix.width
                scale_y = page_height / pix.height
                cell_rect = fitz.Rect(
                    cell_bbox[0] * scale_x,
                    cell_bbox[1] * scale_y,
                    cell_bbox[2] * scale_x,
                    cell_bbox[3] * scale_y,
                )
                # Извлекаем текст внутри rect
                text = page.get_textbox(cell_rect).strip()
                grid[r_idx][c_idx] = text if text else ""

        # Преобразуем сетку в Markdown
        markdown = table_to_markdown(grid)
        if not markdown:
            return []

        # Вычисляем общий bbox таблицы
        table_x0 = min(col[1][0]["bbox"][0] for col in sorted_cols) * scale_x
        table_y0 = min(row[1][0]["bbox"][1] for row in sorted_rows) * scale_y
        table_x1 = max(col[1][0]["bbox"][2] for col in sorted_cols) * scale_x
        table_y1 = max(row[1][0]["bbox"][3] for row in sorted_rows) * scale_y
        bbox = (table_x0, table_y0, table_x1, table_y1)

        score = float(results["scores"].mean())
        return [TableCandidate(bbox=bbox, markdown=markdown, score=score, rows=grid)]


def extract_tables(
    pdf_path: Path, page_number: int, page_width: float, page_height: float
) -> List[TableCandidate]:
    """Основная функция для извлечения таблиц."""
    with fitz.open(pdf_path) as doc:
        page = doc[page_number]
        extractor = TableExtractor()
        return extractor.extract_tables_from_page(page, page_width, page_height)