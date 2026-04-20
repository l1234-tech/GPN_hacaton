import Levenshtein
import re


def text_score(pred, gt):
    return 1 - Levenshtein.distance(pred, gt) / max(len(pred), len(gt), 1)


def table_score(pred, gt):
    pred_tables = re.findall(r'\|.*?\|', pred)
    gt_tables = re.findall(r'\|.*?\|', gt)

    if not pred_tables or not gt_tables:
        return 0

    score = 0
    for p, g in zip(pred_tables, gt_tables):
        score += text_score(p, g)

    return score / max(len(gt_tables), 1)


def image_score(pred, gt):
    pred_imgs = re.findall(r'!\[.*?\]\((.*?)\)', pred)
    gt_imgs = re.findall(r'!\[.*?\]\((.*?)\)', gt)

    if not pred_imgs or not gt_imgs:
        return 0

    return min(len(pred_imgs), len(gt_imgs)) / max(len(pred_imgs), len(gt_imgs))


def structure_score(pred, gt):
    pred_h = re.findall(r'^#+ .*', pred, re.MULTILINE)
    gt_h = re.findall(r'^#+ .*', gt, re.MULTILINE)

    return text_score("\n".join(pred_h), "\n".join(gt_h))


def total_score(pred, gt):
    return (
        0.4 * table_score(pred, gt) +
        0.3 * text_score(pred, gt) +
        0.2 * structure_score(pred, gt) +
        0.1 * image_score(pred, gt)
    )