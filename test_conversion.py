#!/usr/bin/env python3
"""
Тест обработки одного PDF файла с улучшениями
"""

import sys
from pathlib import Path

# Проверяем наличие PDF файлов
pdf_dir = Path("dataset/public/pdfs")
if not pdf_dir.exists():
    print("❌ Директория с PDF не найдена")
    sys.exit(1)

pdf_files = list(pdf_dir.glob("*.pdf"))
if not pdf_files:
    print("❌ PDF файлы не найдены")
    sys.exit(1)

# Берем первый PDF для теста
test_pdf = pdf_files[0]
print(f"📄 Тестируем на файле: {test_pdf.name}")

# Создаем временную директорию для результатов
output_dir = Path("test_output")
output_dir.mkdir(exist_ok=True)
images_dir = output_dir / "images"
images_dir.mkdir(exist_ok=True)

try:
    # Импортируем и тестируем
    from baseline.docling_baseline import convert_pdf
    print("✅ Импорт docling_baseline успешен")

    # Конвертируем
    convert_pdf(test_pdf, output_dir)
    print("✅ Конвертация завершена успешно")

    # Проверяем результат
    result_file = output_dir / f"{test_pdf.stem}.md"
    if result_file.exists():
        content = result_file.read_text(encoding='utf-8')
        print(f"✅ Результат сохранен: {len(content)} символов")
        print(f"📝 Первые 200 символов:\n{content[:200]}...")
    else:
        print("❌ Файл результата не найден")

except Exception as e:
    print(f"❌ Ошибка: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n🎉 Тест завершен успешно!")