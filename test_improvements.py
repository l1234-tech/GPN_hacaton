#!/usr/bin/env python3
"""
Простой тест улучшенных функций PDF парсинга
"""

import sys
from pathlib import Path

# Добавляем путь к baseline
sys.path.insert(0, str(Path(__file__).parent))

try:
    from baseline.ocr_utils import normalize_ocr_text, _is_watermark_text
    from baseline.layout_utils import detect_heading_level, is_probable_watermark
    from baseline.postprocess import normalize_text_block
    from baseline.table_utils import merge_tables_across_pages
    print("✅ Все импорты успешны")

    # Тест нормализации OCR текста
    test_text = "Hello  world!"
    normalized = normalize_ocr_text(test_text)
    assert normalized == "Hello world!", f"Expected 'Hello world!', got '{normalized}'"
    print("✅ OCR нормализация работает")

    # Тест обнаружения водяных знаков
    assert _is_watermark_text("стр. 1") == True
    assert _is_watermark_text("Договор") == False
    print("✅ Фильтрация водяных знаков работает")

    # Тест определения заголовков
    heading = detect_heading_level("ГЛАВА 1", {"lines": [{"spans": [{"size": 24, "flags": 0}]}]})
    assert heading is not None and heading[0] == "#", f"Expected heading, got {heading}"
    print("✅ Определение заголовков работает")

    # Тест постобработки текста
    processed = normalize_text_block("Hello   world!")
    assert processed == "Hello world!", f"Expected 'Hello world!', got '{processed}'"
    print("✅ Постобработка текста работает")

    print("\n🎉 Все тесты пройдены! Улучшения успешно интегрированы.")

except Exception as e:
    print(f"❌ Ошибка: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)