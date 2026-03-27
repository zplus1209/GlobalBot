"""
Reading order sorting using hantian/layoutreader (LayoutLMv3-based).
Input : list of OCR result dicts with "bbox": [xmin, ymin, xmax, ymax]
Output: same list re-ordered by reading order + "position" field added
"""

from __future__ import annotations

from loguru import logger
from model.utils import download_model

_MODEL_REPO = "hantian/layoutreader"
_MAX_BOXES  = 512          # LayoutLMv3 token limit


def _load_layoutreader():
    from transformers import LayoutLMv3ForTokenClassification
    model_dir = download_model(model_name="layoutreader", repo_id=_MODEL_REPO)
    model = LayoutLMv3ForTokenClassification.from_pretrained(model_dir)
    model.eval()
    return model


def _normalize_boxes(bboxes: list[list[int]], img_w: int, img_h: int) -> list[list[int]]:
    """Scale pixel bbox to 0-1000 range as required by LayoutReader."""
    result = []
    for xmin, ymin, xmax, ymax in bboxes:
        result.append([
            int(xmin / img_w * 1000),
            int(ymin / img_h * 1000),
            int(xmax / img_w * 1000),
            int(ymax / img_h * 1000),
        ])
    return result


def _boxes2inputs(boxes: list[list[int]]) -> dict:
    """Convert bbox list to LayoutLMv3 token inputs (bbox only, no text)."""
    import torch

    # Pad/clip to max length
    boxes = boxes[:_MAX_BOXES]
    n     = len(boxes)

    # LayoutLMv3 expects: [CLS] + tokens + [SEP]
    # Use dummy word_ids (0) since we only use layout
    input_ids      = [0] + [0] * n + [2]          # CLS + tokens + SEP
    bbox_input     = [[0, 0, 0, 0]] + boxes + [[1000, 1000, 1000, 1000]]
    attention_mask = [1] * len(input_ids)

    return {
        "input_ids":      torch.tensor([input_ids],  dtype=torch.long),
        "bbox":           torch.tensor([bbox_input], dtype=torch.long),
        "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
    }


def _parse_logits(logits, n: int) -> list[int]:
    """
    Greedy decode: pick the highest-scoring un-assigned position at each step.
    Returns list of length n: orders[i] = reading position of box i.
    """
    import torch

    # logits shape: (seq_len, num_labels) where num_labels == max_position
    # slice off CLS/SEP, take first n tokens
    token_logits = logits[1: n + 1]           # (n, num_labels)
    probs        = torch.softmax(token_logits, dim=-1)

    assigned   = set()
    orders     = [0] * n

    for i in range(n):
        # mask already-assigned positions
        row = probs[i].clone()
        for a in assigned:
            row[a] = -1.0
        pos = int(row.argmax().item())
        # clamp to valid range
        pos = min(pos, n - 1)
        while pos in assigned:
            pos = (pos + 1) % n
        orders[i] = pos
        assigned.add(pos)

    return orders


class ReadingOrderSorter:
    def __init__(self):
        self._model = None

    def _load(self):
        if self._model is None:
            logger.info("Loading LayoutReader model...")
            self._model = _load_layoutreader()
            logger.info("LayoutReader loaded.")

    def sort(
        self,
        ocr_results: list[dict],
        img_w: int,
        img_h: int,
    ) -> list[dict]:
        """
        Sort OCR results by reading order and add "position" field.

        Args:
            ocr_results : list of dicts, each has "bbox": [xmin,ymin,xmax,ymax]
            img_w, img_h: original image dimensions (pixels) for normalization

        Returns:
            New list sorted in reading order, each item gets:
              "position": int  (1-based reading index)
        """
        if not ocr_results:
            return []

        self._load()

        bboxes     = [r["bbox"] for r in ocr_results]
        norm_boxes = _normalize_boxes(bboxes, img_w, img_h)

        # If more than _MAX_BOXES, fall back to heuristic for the excess
        n_model    = min(len(norm_boxes), _MAX_BOXES)
        model_part = norm_boxes[:n_model]
        extra_part = ocr_results[n_model:]

        import torch
        inputs = _boxes2inputs(model_part)
        inputs = {k: v.to(next(self._model.parameters()).device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self._model(**inputs).logits.cpu().squeeze(0)

        orders = _parse_logits(logits, n_model)

        # Build sorted list for model part
        indexed   = list(zip(orders, ocr_results[:n_model]))
        sorted_items = [item for _, item in sorted(indexed, key=lambda x: x[0])]

        # Heuristic fallback for overflow: top→bottom, left→right
        if extra_part:
            logger.warning(
                f"{len(extra_part)} boxes exceeded LayoutReader limit ({_MAX_BOXES}), "
                "using heuristic sort for overflow."
            )
            extra_sorted = sorted(extra_part, key=lambda r: (r["bbox"][1], r["bbox"][0]))
            sorted_items.extend(extra_sorted)

        # Attach 1-based position
        result = []
        for pos, item in enumerate(sorted_items, start=1):
            result.append({**item, "position": pos})

        return result

_sorter: ReadingOrderSorter | None = None

def sort_reading_order(
    ocr_results: list[dict],
    img_w: int,
    img_h: int,
) -> list[dict]:
    """
    Module-level convenience function.
    Lazily initialises the sorter singleton.
    """
    global _sorter
    if _sorter is None:
        _sorter = ReadingOrderSorter()
    return _sorter.sort(ocr_results, img_w, img_h)