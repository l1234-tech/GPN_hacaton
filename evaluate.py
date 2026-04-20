#!/usr/bin/env python3
"""
Оценка качества парсинга PDF в Markdown.
Сравнивает предсказанные файлы из results/ с эталонными из ground_truth/.
"""

import argparse
import re
from pathlib import Path
import Levenshtein


def text_score(pred: str, gt: str) -> float:
    if not pred and not gt:
        return 1.0
    return 1 - Levenshtein.distance(pred, gt) / max(len(pred), len(gt), 1)


def table_score(pred: str, gt: str) -> float:
    pred_tables = re.findall(r'\|.*?\|', pred)
    gt_tables = re.findall(r'\|.*?\|', gt)

    if not pred_tables or not gt_tables:
        return 0.0

    score = 0.0
    for p, g in zip(pred_tables, gt_tables):
        score += text_score(p, g)

    return score / max(len(gt_tables), 1)


def image_score(pred: str, gt: str) -> float:
    pred_imgs = re.findall(r'!\[.*?\]\((.*?)\)', pred)
    gt_imgs = re.findall(r'!\[.*?\]\((.*?)\)', gt)

    if not pred_imgs or not gt_imgs:
        return 0.0

    return min(len(pred_imgs), len(gt_imgs)) / max(len(pred_imgs), len(gt_imgs))


def structure_score(pred: str, gt: str) -> float:
    pred_h = re.findall(r'^#+ .*', pred, re.MULTILINE)
    gt_h = re.findall(r'^#+ .*', gt, re.MULTILINE)

    if not pred_h and not gt_h:
        return 1.0
    if not pred_h or not gt_h:
        return 0.0

    return text_score("\n".join(pred_h), "\n".join(gt_h))


def total_score(pred: str, gt: str) -> float:
    return (
        0.4 * table_score(pred, gt) +
        0.3 * text_score(pred, gt) +
        0.2 * structure_score(pred, gt) +
        0.1 * image_score(pred, gt)
    )


def evaluate_all(pred_dir: Path, gt_dir: Path) -> None:
    pred_files = sorted(pred_dir.glob("*.md"))
    if not pred_files:
        print("Нет .md файлов в директории предсказаний.")
        return

    total_scores = []
    print(f"{'Файл':<20} {'Total':<8} {'Tables':<8} {'Text':<8} {'Struct':<8} {'Images':<8}")
    print("-" * 65)

    for pred_path in pred_files:
        gt_path = gt_dir / pred_path.name
        if not gt_path.exists():
            print(f"{pred_path.name:<20} эталон не найден")
            continue

        pred_text = pred_path.read_text(encoding="utf-8")
        gt_text = gt_path.read_text(encoding="utf-8")

        t_score = table_score(pred_text, gt_text)
        txt_score = text_score(pred_text, gt_text)
        str_score = structure_score(pred_text, gt_text)
        img_score = image_score(pred_text, gt_text)
        total = total_score(pred_text, gt_text)

        total_scores.append(total)

        print(f"{pred_path.name:<20} {total:.4f}   {t_score:.4f}   {txt_score:.4f}   {str_score:.4f}   {img_score:.4f}")

    if total_scores:
        avg = sum(total_scores) / len(total_scores)
        print("-" * 65)
        print(f"{'СРЕДНЕЕ':<20} {avg:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Оценка метрик хакатона")
    parser.add_argument("--pred-dir", type=Path, default=Path("results"), help="Папка с предсказаниями")
    parser.add_argument("--gt-dir", type=Path, default=Path("dataset/public/ground_truth"), help="Папка с эталонами")
    args = parser.parse_args()

    evaluate_all(args.pred_dir, args.gt_dir)


if __name__ == "__main__":
    main()