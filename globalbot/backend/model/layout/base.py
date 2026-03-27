"""Base class và dataclass cho layout detection."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field

import cv2
import numpy as np


def image_to_base64(image: np.ndarray, ext: str = ".jpg") -> str:
    """Convert numpy BGR image sang base64 string."""
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        raise ValueError("Khong the encode anh sang base64")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


@dataclass
class LayoutRegion:
    """Mot vung layout duoc detect tu tai lieu."""

    position:    int                    # thu tu doc (1-based, chinh xac theo model)
    region_type: str                    # text | paragraph_title | figure | table | ...
    score:       float                  # confidence score
    bbox:        list[int]              # [xmin, ymin, xmax, ymax] pixel coords
    image:       np.ndarray = field(repr=False)   # crop BGR
    base64:      str        = field(repr=False)   # base64 JPEG

    @classmethod
    def build(
        cls,
        position: int,
        region_type: str,
        score: float,
        bbox: list[int],
        full_image: np.ndarray,
    ) -> "LayoutRegion":
        xmin, ymin, xmax, ymax = bbox
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(full_image.shape[1], xmax)
        ymax = min(full_image.shape[0], ymax)

        cropped = full_image[ymin:ymax, xmin:xmax]
        return cls(
            position=position,
            region_type=region_type,
            score=score,
            bbox=bbox,
            image=cropped,
            base64=image_to_base64(cropped),
        )

    def to_dict(self, include_image: bool = False) -> dict:
        d = {
            "position":    self.position,
            "region_type": self.region_type,
            "score":       self.score,
            "bbox":        self.bbox,
            "base64":      self.base64,
        }
        if include_image:
            d["image"] = self.image
        return d