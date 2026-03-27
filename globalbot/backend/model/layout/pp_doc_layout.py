"""
PP-DocLayoutV3 layout detector.

Model : PaddlePaddle/PP-DocLayoutV3_safetensors
Doc   : https://huggingface.co/docs/transformers/model_doc/pp_doclayout_v3

Output da duoc model sap xep theo reading order (tren xuong duoi, trai sang phai).
Khong can post-sort them; chi lay theo thu tu tra ve.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from loguru import logger
from PIL import Image as PILImage

from model.utils import download_model
from model.layout.base import LayoutRegion, image_to_base64

_MODEL_REPO = "PaddlePaddle/PP-DocLayoutV3_safetensors"
_DEFAULT_THRESHOLD = 0.5


class PPDocLayoutDetector:
    """
    Wrapper cho PP-DocLayoutV3.
    Load model lazy (chi load khi goi detect lan dau).
    """

    def __init__(self, threshold: float = _DEFAULT_THRESHOLD):
        self.threshold   = threshold
        self._model      = None
        self._processor  = None
        self._device     = None

    def _load(self):
        if self._model is not None:
            return

        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        logger.info("Loading PP-DocLayoutV3...")
        model_dir = download_model(
            model_name="PP-DocLayoutV3",
            repo_id=_MODEL_REPO,
        )

        self._device    = "cuda" if torch.cuda.is_available() else "cpu"
        self._processor = AutoImageProcessor.from_pretrained(model_dir)
        self._model     = AutoModelForObjectDetection.from_pretrained(model_dir).to(self._device)
        self._model.eval()
        logger.info(f"PP-DocLayoutV3 loaded on {self._device}.")

    # ------------------------------------------------------------------

    def detect(self, image: np.ndarray) -> list[LayoutRegion]:
        """
        Detect layout regions tu anh BGR (cv2).

        Returns:
            list[LayoutRegion] da sap xep theo reading order cua model
            (thu tu trong list = thu tu doc thuc te).
        """
        self._load()

        pil_img = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        h, w    = image.shape[:2]

        inputs  = self._processor(images=[pil_img], return_tensors="pt")
        inputs  = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        # post_process_object_detection tra ve bbox pixel coords (xmin,ymin,xmax,ymax)
        # thu tu trong results chinh la reading order cua model
        results = self._processor.post_process_object_detection(
            outputs,
            threshold=self.threshold,
            target_sizes=torch.tensor([[h, w]]),
        )[0]

        regions: list[LayoutRegion] = []
        for idx, (score, label_id, box) in enumerate(
            zip(results["scores"], results["labels"], results["boxes"])
        ):
            label_name = self._model.config.id2label[label_id.item()]
            bbox       = [round(v) for v in box.tolist()]   # [xmin,ymin,xmax,ymax]

            region = LayoutRegion.build(
                position    = idx + 1,          # 1-based, thu tu model tra ve = thu tu doc
                region_type = label_name,
                score       = round(score.item(), 4),
                bbox        = bbox,
                full_image  = image,
            )
            regions.append(region)

        logger.debug(f"Detected {len(regions)} layout regions.")
        return regions

    def detect_from_path(self, path: str) -> list[LayoutRegion]:
        """Convenience wrapper nhan duong dan file."""
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Khong the doc anh: {path}")
        return self.detect(image)


_detector: PPDocLayoutDetector | None = None


def detect_layout(
    image: np.ndarray,
    threshold: float = _DEFAULT_THRESHOLD,
) -> list[LayoutRegion]:
    """
    Module-level convenience function.
    Tra ve list[LayoutRegion] theo reading order.
    """
    global _detector
    if _detector is None or _detector.threshold != threshold:
        _detector = PPDocLayoutDetector(threshold=threshold)
    return _detector.detect(image)


def detect_layout_from_path(
    path: str,
    threshold: float = _DEFAULT_THRESHOLD,
) -> list[LayoutRegion]:
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Khong the doc anh: {path}")
    return detect_layout(image, threshold)

if __name__ == "__main__":
    regions = detect_layout_from_path("/mnt/data1/home/staging/workspace/zplus/146.jpg")
    for r in regions:
        print(f"[{r.position}] {r.region_type} ({r.score:.2f}) bbox={r.bbox}")