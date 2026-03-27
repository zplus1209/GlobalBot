from __future__ import annotations

import base64
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import fitz
from loguru import logger

from model.layout import PPDocLayoutDetector, LayoutRegion
from model.ocr.base import TextOCR
from model.reading_order import sort_reading_order

_VISUAL_TYPES = {"figure", "chart", "image", "table", "formula", "equation", "seal", "stamp"}
_TEXT_TYPES = {
    "text", "paragraph_title", "title", "header", "footer",
    "footnote", "caption", "number", "abstract", "content",
    "reference", "paragraph",
}
_TABLE_TYPES = {"table"}
_FORMULA_TYPES = {"formula", "equation"}
_CHART_TYPES = {"chart"}
_LIST_ORIGINS = {"number", "footnote"}


@dataclass
class RegionContext:
    position: int
    region_type: str
    bbox: list[int]
    base64_img: str
    ocr_text: str = ""
    needs_vlm: bool = False
    crop_path: str = ""


@dataclass
class PageContext:
    image_path: str
    page: Optional[int]
    img_w: int
    img_h: int
    regions: list[RegionContext] = field(default_factory=list)
    ocr_blocks: list[dict] = field(default_factory=list)

    def visual_regions(self) -> list[RegionContext]:
        return [r for r in self.regions if r.needs_vlm]

    def text_summary(self) -> str:
        return "\n".join(
            f"[{r.position}] {r.region_type} | {r.ocr_text[:100]}"
            for r in self.regions
        )


def _encode_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _convert_doc_to_images(doc_path: str, dpi: int = 150) -> list[str]:
    suffix = Path(doc_path).suffix.lower()
    image_paths = []

    if suffix == ".pdf":
        pdf = fitz.open(doc_path)
        tmp_dir = tempfile.mkdtemp(prefix="ade_pages_")
        for i, page in enumerate(pdf):
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            out = f"{tmp_dir}/page_{i:04d}.png"
            pix.save(out)
            image_paths.append(out)
        pdf.close()

    elif suffix in {".doc", ".docx"}:
        import subprocess
        tmp_dir = tempfile.mkdtemp(prefix="ade_pages_")
        subprocess.run(
            ["libreoffice", "--headless", "--convert-to", "pdf",
             "--outdir", tmp_dir, doc_path],
            check=True, capture_output=True,
        )
        pdf_path = next(Path(tmp_dir).glob("*.pdf"), None)
        if pdf_path is None:
            raise RuntimeError(f"LibreOffice failed to convert {doc_path}")
        return _convert_doc_to_images(str(pdf_path), dpi=dpi)

    elif suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}:
        image_paths = [doc_path]

    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    return image_paths


def process_page(
    image_path: str,
    page: Optional[int] = None,
    threshold: float = 0.5,
    crops_dir: str = "/tmp/ade_crops",
) -> PageContext:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    img_h, img_w = image.shape[:2]
    Path(crops_dir).mkdir(parents=True, exist_ok=True)

    detector = PPDocLayoutDetector(threshold=threshold)
    raw_regions: list[LayoutRegion] = detector.detect(image)
    for i, r in enumerate(raw_regions, start=1):
        r.position = i

    ocr_engine = TextOCR()
    all_boxes = []

    for r in raw_regions:
        if r.region_type not in _TEXT_TYPES:
            continue
        x0, y0, x1, y1 = r.bbox
        crop = image[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        tmp_crop = f"/tmp/_ocr_{r.position}.jpg"
        cv2.imwrite(tmp_crop, crop)
        for b in ocr_engine.run_rec(tmp_crop):
            bx0, by0, bx1, by1 = b["bbox"]
            b["bbox"] = [bx0 + x0, by0 + y0, bx1 + x0, by1 + y0]
            b["region_id"] = r.position
            all_boxes.append(b)

    sorted_boxes = sort_reading_order(all_boxes, img_w, img_h) if all_boxes else []
    ocr_map: dict[int, str] = {}
    for b in sorted_boxes:
        rid = b.get("region_id")
        if rid and b.get("text"):
            ocr_map[rid] = ocr_map.get(rid, "") + " " + b["text"].strip()

    regions = []
    for r in raw_regions:
        needs_vlm = r.region_type in _VISUAL_TYPES
        crop_path = ""

        if needs_vlm:
            x0, y0, x1, y1 = r.bbox
            crop_path = f"{crops_dir}/page{page or 0}_region{r.position}_{r.region_type}.jpg"
            cv2.imwrite(crop_path, image[y0:y1, x0:x1])

        regions.append(RegionContext(
            position=r.position,
            region_type=r.region_type,
            bbox=r.bbox,
            base64_img=r.base64 if needs_vlm else "",
            ocr_text=ocr_map.get(r.position, "").strip(),
            needs_vlm=needs_vlm,
            crop_path=crop_path,
        ))

    return PageContext(
        image_path=image_path,
        page=page,
        img_w=img_w,
        img_h=img_h,
        regions=regions,
        ocr_blocks=sorted_boxes,
    )


def load_document(
    doc_path: str,
    threshold: float = 0.5,
    crops_dir: str = "/tmp/ade_crops",
    dpi: int = 150,
) -> list[PageContext]:
    image_paths = _convert_doc_to_images(doc_path, dpi=dpi)
    pages = []
    for i, img_path in enumerate(image_paths):
        logger.info(f"Processing page {i + 1}/{len(image_paths)}")
        ctx = process_page(img_path, page=i + 1, threshold=threshold, crops_dir=crops_dir)
        pages.append(ctx)
    return pages
