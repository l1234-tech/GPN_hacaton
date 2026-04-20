#!/usr/bin/env python3
"""
Проверка формальной корректности сгенерированных Markdown-файлов.
"""

import argparse
import re
from pathlib import Path

def check_md_file(md_path: Path, images_dir: Path) -> tuple[bool, list[str]]:
    errors = []
    try:
        text = md_path.read_text(encoding="utf-8")
    except Exception as e:
        return False, [f"Не удалось прочитать: {e}"]

    # Проверяем, что файл не пустой
    if not text.strip():
        errors.append("Файл пуст")

    # Проверяем, что все ссылки на изображения ведут на существующие файлы
    img_links = re.findall(r'!\[.*?\]\((images/[^)]+)\)', text)
    for link in img_links:
        # Убираем возможный ведущий слеш
        rel_path = link.lstrip('/')
        img_path = images_dir.parent / rel_path  # images_dir уже внутри output_dir
        if not img_path.exists():
            errors.append(f"Изображение не найдено: {link}")

    # Проверяем, что имена изображений соответствуют шаблону doc_<N>_image_<M>.png
    pattern = re.compile(r'doc_\d+_image_\d+\.png')
    for link in img_links:
        filename = Path(link).name
        if not pattern.match(filename):
            errors.append(f"Некорректное имя изображения: {filename}")

    # Проверяем, что в файле нет явных артефактов (например, незакрытых HTML-тегов)
    if re.search(r'<[^>]+>', text):
        errors.append("Обнаружены HTML-теги (возможно, не до конца сконвертировано)")

    # Проверяем, что файл начинается не с пустой строки (не критично, но для информации)
    if text.startswith('\n'):
        errors.append("Файл начинается с пустой строки")

    return len(errors) == 0, errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    results_dir = args.results_dir
    images_dir = results_dir / "images"
    md_files = sorted(results_dir.glob("*.md"))

    if not md_files:
        print("Нет .md файлов для проверки.")
        return

    all_ok = True
    for md_path in md_files:
        ok, errors = check_md_file(md_path, images_dir)
        if ok:
            print(f"{md_path.name}: OK")
        else:
            all_ok = False
            print(f"{md_path.name}: ОШИБКИ")
            for err in errors:
                print(f"  - {err}")

    if all_ok:
        print("\nВсе файлы прошли формальную проверку.")
    else:
        print("\nОбнаружены проблемы в некоторых файлах.")


if __name__ == "__main__":
    main()