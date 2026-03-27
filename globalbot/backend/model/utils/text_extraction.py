import os
import re
from pathlib import Path

import cv2
import yaml
import numpy as np
from loguru import logger
from paddleocr import PaddleOCR

from model.ocr import LightOnOCRParser
from model.utils import download_model, ensure_dir, ModelPath


# Crop nho hon nguong nay (px^2) → dung Paddle text, khong qua LightOnOCR
_SMALL_CROP_AREA = 200 * 60

# Hallucination: qua nhieu lenh LaTeX phuc tap → fallback Paddle
_HALLUC_PATTERN  = re.compile(
    r"\\(frac|int|nabla|partial|mathcal|sqrt|sum|prod|oint)\s*\{",
    re.IGNORECASE,
)
_HALLUC_MAX_HITS = 2


_PADDLE_MODELS = [
    ModelPath.paddle_det,
    ModelPath.paddle_textline_ori,
    ModelPath.paddle_rec,
    ModelPath.paddle_doc_ori,
    ModelPath.paddle_uvdoc,
]


def _download_paddle_models() -> dict[str, str]:
    paths = {}
    for name in _PADDLE_MODELS:
        paths[name] = download_model(name, repo_mode="paddleocr")
    return paths


def _build_yaml(save_dir: str) -> str:
    ensure_dir(save_dir)
    yaml_path = os.path.join(save_dir, "PaddleOCR.yaml")

    ocr = PaddleOCR()
    ocr.export_paddlex_config_to_yaml(yaml_path)

    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    def _patch(node):
        if isinstance(node, dict):
            if "model_name" in node:
                node["model_dir"] = download_model(node["model_name"], repo_mode="paddleocr")
            for v in node.values():
                _patch(v)
        elif isinstance(node, list):
            for item in node:
                _patch(item)

    _patch(config)

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True)

    return yaml_path


def _is_hallucinated(text: str) -> bool:
    return len(_HALLUC_PATTERN.findall(text)) > _HALLUC_MAX_HITS


class TextOCR:
    def __init__(self, ocr_backend: LightOnOCRParser | None = None):
        model_paths = _download_paddle_models()
        det_path    = model_paths[ModelPath.paddle_det]
        yaml_path   = os.path.join(det_path, "PaddleOCR.yaml")

        if not Path(yaml_path).exists():
            logger.debug("Tao moi PaddleOCR.yaml")
            yaml_path = _build_yaml(det_path)

        self._yaml_path = yaml_path
        self._pipeline  = PaddleOCR(
            use_doc_orientation_classify=True,
            use_doc_unwarping=False,
            use_textline_orientation=True,
            paddlex_config=yaml_path,
        )
        self.ocr_backend = ocr_backend or LightOnOCRParser()

    # ------------------------------------------------------------------
    # geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _poly_to_bbox(poly) -> tuple[int, int, int, int]:
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        return min(xs), min(ys), max(xs), max(ys)

    @staticmethod
    def _expand_poly(poly, ratio: float) -> np.ndarray:
        pts    = np.array(poly, dtype=np.float32)
        center = pts.mean(axis=0)
        return (pts - center) * (1 + ratio) + center

    def _get_pad_ratio(self, w: int, h: int) -> float:
        size = min(w, h)
        if size < 32:
            return 0.2
        if size > 200:
            return 0.05
        return 0.1

    def _crop_poly(self, image: np.ndarray, poly) -> np.ndarray | None:
        xmin, ymin, xmax, ymax = self._poly_to_bbox(poly)
        pad = self._get_pad_ratio(xmax - xmin, ymax - ymin)
        pts = self._expand_poly(poly, pad).astype(np.float32)

        s    = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        tl   = pts[np.argmin(s)]
        br   = pts[np.argmax(s)]
        tr   = pts[np.argmin(diff)]
        bl   = pts[np.argmax(diff)]

        rect = np.array([tl, tr, br, bl], dtype=np.float32)
        w    = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
        h    = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))

        if w <= 0 or h <= 0:
            return None

        dst    = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
        M      = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (w, h))
        return warped if warped.size > 0 else None

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def run_det(self, path: str) -> list[dict]:
        """Detect + recognize bang PaddleOCR, tra ve list bbox/text/score."""
        results = []
        for res in self._pipeline.predict(input=path):
            for text, score, box in zip(res["rec_texts"], res["rec_scores"], res["rec_polys"]):
                coords = box.astype(int).tolist()
                xmin, ymin, xmax, ymax = self._poly_to_bbox(coords)
                results.append({
                    "text":  text,
                    "bbox":  [xmin, ymin, xmax, ymax],
                    "poly":  coords,
                    "score": float(score),
                })
        return results

    def run_rec(self, path: str) -> list[dict]:
        """
        Detect bang PaddleOCR, phan loai tung crop:
          - Crop nho  (area < _SMALL_CROP_AREA) → Paddle text
          - Crop lon                             → LightOnOCR (fallback Paddle neu hallucinate)
        """
        image = cv2.imread(path)

        # Thu thap ket qua Paddle
        paddle_results: list[dict] = []
        for res in self._pipeline.predict(input=path):
            for text, score, box in zip(res["rec_texts"], res["rec_scores"], res["rec_polys"]):
                coords = box.astype(int).tolist()
                xmin, ymin, xmax, ymax = self._poly_to_bbox(coords)
                paddle_results.append({
                    "text":  text,
                    "score": float(score),
                    "poly":  coords,
                    "bbox":  [xmin, ymin, xmax, ymax],
                })

        if not paddle_results:
            logger.warning(f"Khong tim thay vung chu nao trong: {path}")
            return []

        # Crop va phan loai
        small_idxs:  list[int]         = []   # dung Paddle
        large_idxs:  list[int]         = []   # dung LightOnOCR
        crops:       list[np.ndarray]  = []   # chi cho large

        for i, pr in enumerate(paddle_results):
            crop = self._crop_poly(image, pr["poly"])
            if crop is None:
                small_idxs.append(i)          # fallback Paddle neu crop loi
                continue

            h, w = crop.shape[:2]
            if w * h < _SMALL_CROP_AREA:
                small_idxs.append(i)
            else:
                large_idxs.append(i)
                crops.append(crop)

        logger.info(
            f"Tong {len(paddle_results)} crop: "
            f"{len(small_idxs)} nho (Paddle), {len(large_idxs)} lon (LightOnOCR)"
        )

        # Chay LightOnOCR batch
        lighton_texts: list[str] = []
        if crops:
            lighton_texts = self.ocr_backend.parse_batch(crops)

        # Ghep ket qua
        lighton_map = dict(zip(large_idxs, lighton_texts))

        results = []
        for i, pr in enumerate(paddle_results):
            if i in lighton_map:
                lo_text = lighton_map[i]
                if _is_hallucinated(lo_text):
                    logger.debug(f"Crop {i}: LightOnOCR hallucinate → fallback Paddle")
                    text, backend = pr["text"], "paddle-fallback"
                else:
                    text, backend = lo_text, "lighton"
            else:
                text, backend = pr["text"], "paddle"

            results.append({
                "text":    text,
                "bbox":    pr["bbox"],
                "poly":    pr["poly"],
                "score":   pr["score"],
                "backend": backend,
            })

        return results

    def run_rec_ordered(self, path: str) -> list[dict]:
        """
        run_rec() + sort by reading order via LayoutReader.
        Moi item co them:
          "position": int  (1-based, thu tu doc tu trai sang phai, tren xuong duoi)
        """
        from model.reading_order import sort_reading_order

        results = self.run_rec(path)
        if not results:
            return []

        img              = cv2.imread(path)
        img_h, img_w     = img.shape[:2]

        return sort_reading_order(results, img_w, img_h)


if __name__ == "__main__":
    text    = TextOCR()
    results = text.run_rec_ordered("/mnt/data1/home/staging/workspace/zplus/146.jpg")
    for r in results:
        print(f"[{r['position']:>3}] [{r['backend']}] {r['text']}")