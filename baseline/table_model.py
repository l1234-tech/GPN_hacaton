# table_model.py
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

@dataclass
class TableCell:
    text: str
    row: int
    col: int
    rowspan: int = 1
    colspan: int = 1
    bbox: Optional[Tuple[float, float, float, float]] = None

@dataclass
class ParsedTable:
    """Расширенная модель таблицы с поддержкой объединенных ячеек и метаданных."""
    page_num: int
    bbox: Tuple[float, float, float, float]  # (x0, top, x1, bottom)
    cells: List[TableCell] = field(default_factory=list)
    markdown: str = ""
    is_borderless: bool = False
    confidence: float = 1.0  # Уверенность в распознавании таблицы
    merged_cells: List[Tuple[int, int, int, int]] = field(default_factory=list)  # (row, col, rowspan, colspan)

    def overlaps_with_bbox(self, other_bbox: Tuple[float, float, float, float], threshold: float = 0.5) -> bool:
        """Улучшенная проверка пересечения с учетом площади."""
        x0, top, x1, bottom = self.bbox
        ox0, otop, ox1, obottom = other_bbox
        
        # Вычисляем пересечение
        inter_x0 = max(x0, ox0)
        inter_top = max(top, otop)
        inter_x1 = min(x1, ox1)
        inter_bottom = min(bottom, obottom)
        
        if inter_x1 < inter_x0 or inter_bottom < inter_top:
            return False
            
        inter_area = (inter_x1 - inter_x0) * (inter_bottom - inter_top)
        self_area = (x1 - x0) * (bottom - top)
        other_area = (ox1 - ox0) * (obottom - otop)
        
        if self_area == 0 or other_area == 0:
            return False
        
        # Пересечение считается значимым, если оно превышает порог для любой из областей
        return (inter_area / self_area > threshold) or (inter_area / other_area > threshold)

    def get_table_dimensions(self) -> Tuple[int, int]:
        """Возвращает (rows, cols) таблицы."""
        if not self.cells:
            return (0, 0)
        max_row = max(cell.row + cell.rowspan for cell in self.cells)
        max_col = max(cell.col + cell.colspan for cell in self.cells)
        return (max_row, max_col)

    def is_cell_merged(self, row: int, col: int) -> bool:
        """Проверяет, является ли ячейка частью объединенной."""
        for m_row, m_col, rowspan, colspan in self.merged_cells:
            if (m_row <= row < m_row + rowspan and 
                m_col <= col < m_col + colspan and 
                not (row == m_row and col == m_col)):
                return True
        return False

    def get_cell_at(self, row: int, col: int) -> Optional[TableCell]:
        """Возвращает ячейку в указанной позиции."""
        for cell in self.cells:
            if (cell.row <= row < cell.row + cell.rowspan and 
                cell.col <= col < cell.col + cell.colspan):
                return cell
        return None

    def validate_structure(self) -> bool:
        """Проверяет целостность структуры таблицы."""
        if not self.cells:
            return False
        
        rows, cols = self.get_table_dimensions()
        
        # Проверяем, что все позиции заполнены
        for r in range(rows):
            for c in range(cols):
                if not self.is_cell_merged(r, c) and self.get_cell_at(r, c) is None:
                    return False
        
        return True

    def to_dict(self) -> dict:
        """Сериализация таблицы в словарь."""
        return {
            'page_num': self.page_num,
            'bbox': self.bbox,
            'cells': [
                {
                    'text': cell.text,
                    'row': cell.row,
                    'col': cell.col,
                    'rowspan': cell.rowspan,
                    'colspan': cell.colspan,
                    'bbox': cell.bbox
                } for cell in self.cells
            ],
            'markdown': self.markdown,
            'is_borderless': self.is_borderless,
            'confidence': self.confidence,
            'merged_cells': self.merged_cells
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ParsedTable':
        """Десериализация из словаря."""
        cells = [
            TableCell(
                text=cell_data['text'],
                row=cell_data['row'],
                col=cell_data['col'],
                rowspan=cell_data.get('rowspan', 1),
                colspan=cell_data.get('colspan', 1),
                bbox=tuple(cell_data['bbox']) if cell_data.get('bbox') else None
            ) for cell_data in data['cells']
        ]
        
        return cls(
            page_num=data['page_num'],
            bbox=tuple(data['bbox']),
            cells=cells,
            markdown=data.get('markdown', ''),
            is_borderless=data.get('is_borderless', False),
            confidence=data.get('confidence', 1.0),
            merged_cells=data.get('merged_cells', [])
        )