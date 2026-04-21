import os
import fitz  # PyMuPDF
from PIL import Image
import io
from typing import List, Tuple, Dict, Any

def extract_images_from_page(page: fitz.Page, doc_index: int, output_images_dir: str) -> List[Dict[str, Any]]:
    """
    Извлечение изображений из страницы PDF.
    Возвращает список метаданных о сохраненных изображениях.
    """
    saved_images = []
    image_list = page.get_images(full=True)
    
    img_counter = 1
    
    for img_index, img_info in enumerate(image_list):
        xref = img_info[0]
        
        try:
            base_image = page.parent.extract_image(xref)
            if not base_image:
                continue
                
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            # Конвертация в PNG для унификации
            img = Image.open(io.BytesIO(image_bytes))
            
            # Фильтр по размеру (игнорируем иконки и шум)
            if img.width < 50 or img.height < 50:
                continue
                
            # Формирование имени файла: doc_<N>_image_<K>.png
            # N - номер документа (1-based), K - порядковый номер картинки в документе
            filename = f"doc_{doc_index}_image_{img_counter}.png"
            filepath = os.path.join(output_images_dir, filename)
            
            # Сохранение
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')
            img.save(filepath, "PNG")
            
            saved_images.append({
                'filename': filename,
                'path': filepath,
                'bbox': None, # Можно добавить поиск координат, если нужно для позиционирования
                'caption': f"Рис. {img_counter}" # Базовая подпись
            })
            
            img_counter += 1
            
        except Exception as e:
            print(f"Ошибка при извлечении изображения {xref}: {e}")
            continue
            
    return saved_images

def render_vector_graphics(page: fitz.Page, doc_index: int, output_images_dir: str, existing_images_count: int) -> List[Dict[str, Any]]:
    """
    Рендеринг векторной графики (диаграмм, графиков), которая не извлекается как растр.
    Делает снимок области страницы, если там есть сложные объекты.
    В данной базовой версии пропускаем, чтобы не дублировать, 
    так как get_images обычно захватывает и встроенные растры графиков.
    """
    return []