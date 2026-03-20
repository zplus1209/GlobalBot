from __future__ import annotations

import base64
import glob
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import model_validator

from globalbot.backend.base import BaseComponent, Document
from globalbot.backend.utils.hashing import make_doc_id


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class ParsedElement:
    element_type: str          # "text" | "table" | "image"
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    page_no: int = 0
    element_id: str = ""

    def __post_init__(self):
        if not self.element_id:
            self.element_id = make_doc_id(
                self.content[:200],
                source=self.metadata.get("source", ""),
                extra=f"p{self.page_no}_{self.element_type}",
            )


@dataclass
class ParsedPDF:
    source: str
    texts: List[ParsedElement] = field(default_factory=list)
    tables: List[ParsedElement] = field(default_factory=list)
    images: List[ParsedElement] = field(default_factory=list)
    markdown: str = ""

    @property
    def all_elements(self) -> List[ParsedElement]:
        return self.texts + self.tables + self.images


# ── PyMuPDF4LLM parser ───────────────────────────────────────────────────────

class PyMuPDF4LLMParser(BaseComponent):
    """
    PDF parser using pymupdf4llm.to_markdown(page_chunks=True).

    page_chunks=True returns list[dict] per page:
      - "text"      : full page markdown (includes table markdown + image refs)
      - "tables"    : list of {bbox, row_count, col_count} — position only, NO content
      - "images"    : list from page.get_image_info() — NO caption
      - "metadata"  : {page_number (1-based), file_path, page_count, ...}
      - "toc_items" : TOC entries pointing to this page
      - "graphics"  : vector graphics bboxes

    Strategy:
      - tables content  → extract markdown tables from chunk["text"]
      - images content  → extract image refs ![...](path) from chunk["text"]
      - text content    → chunk["text"] stripped of table rows and image refs

    Install: pip install pymupdf4llm
    """

    extract_images: bool = False
    extract_tables: bool = True
    write_images: bool = False          # True → save images to disk, False → keep in text
    image_dir: Optional[str] = None
    image_format: str = "png"
    dpi: int = 150
    table_strategy: str = "lines_strict"
    ignore_images: bool = False
    ignore_graphics: bool = False
    margins: float = 0

    @model_validator(mode="after")
    def _set_name(self) -> "PyMuPDF4LLMParser":
        if not self.name:
            self.name = "pymupdf4llm"
        return self

    def parse(self, pdf_path: str) -> ParsedPDF:
        import pymupdf.layout   # activates PyMuPDF layout engine (full box classes)
        import pymupdf4llm

        pdf_path = str(Path(pdf_path).resolve())
        source = os.path.basename(pdf_path)
        self.log.info("pymupdf4llm.parse.start", source=source)

        img_dir = ""
        if self.write_images:
            img_dir = self.image_dir or tempfile.mkdtemp(prefix="pymupdf_img_")
            os.makedirs(img_dir, exist_ok=True)

        # page_chunks=True → list[dict], one dict per page
        # tables/images inside chunk contain position metadata only, not text content
        # actual markdown content (table rows, image refs) is inside chunk["text"]
        chunks: List[Dict] = pymupdf4llm.to_markdown(
            pdf_path,
            page_chunks=True,
            write_images=self.write_images,
            image_path=img_dir,
            image_format=self.image_format,
            dpi=self.dpi,
            table_strategy=self.table_strategy,
            ignore_images=self.ignore_images if not self.extract_images else False,
            ignore_graphics=self.ignore_graphics,
            margins=self.margins,
        )

        result = ParsedPDF(source=source)
        full_md: List[str] = []

        for chunk in chunks:
            meta = chunk.get("metadata", {})
            # metadata["page_number"] is 1-based per docs
            page_no: int = meta.get("page_number", 1) - 1   # convert to 0-based
            md_text: str = chunk.get("text", "")
            base_meta = {
                "source": source,
                "page_no": page_no,
                "page_number_1based": page_no + 1,
            }
            full_md.append(md_text)

            # ── tables: chunk["tables"] has only bbox/row/col counts
            #    actual markdown table content lives inside chunk["text"]
            if self.extract_tables:
                table_blocks = chunk.get("tables", [])
                md_tables = self._extract_md_tables(md_text)
                for i, tbl_md in enumerate(md_tables):
                    # attach bbox from chunk["tables"][i] if available
                    bbox = table_blocks[i]["bbox"] if i < len(table_blocks) else None
                    result.tables.append(ParsedElement(
                        element_type="table",
                        content=tbl_md,
                        metadata=self._sanitize_meta({
                            **base_meta,
                            "format": "markdown",
                            "row_count": table_blocks[i].get("row_count") if i < len(table_blocks) else None,
                            "col_count": table_blocks[i].get("col_count") if i < len(table_blocks) else None,
                            "bbox": bbox,
                        }),
                        page_no=page_no,
                    ))

            # ── images: chunk["images"] is page.get_image_info() output
            #    each entry: {number, bbox, size, digest, ...} — no caption
            #    caption / alt text lives in chunk["text"] as ![alt](path)
            if self.extract_images:
                img_infos = chunk.get("images", [])
                img_refs = self._extract_image_refs(md_text)
                for i, (alt, path) in enumerate(img_refs):
                    extra = img_infos[i] if i < len(img_infos) else {}
                    content = alt.strip() if alt.strip() else f"[Image on page {page_no + 1}]"
                    result.images.append(ParsedElement(
                        element_type="image",
                        content=content,
                        metadata=self._sanitize_meta({
                            **base_meta,
                            "img_caption": alt,
                            "image_path": path,
                            "image_base64": self._encode_image(path),
                            "width": extra.get("width", 0),
                            "height": extra.get("height", 0),
                            "bbox": extra.get("bbox"),
                        }),
                        page_no=page_no,
                    ))

            # ── text: strip markdown table rows and image refs from chunk["text"]
            clean = self._extract_prose(
                md_text,
                strip_tables=self.extract_tables,
                strip_images=self.extract_images,
            )
            if clean.strip():
                result.texts.append(ParsedElement(
                    element_type="text",
                    content=clean,
                    metadata=self._sanitize_meta({
                        **base_meta,
                        "toc_items": chunk.get("toc_items", []),
                    }),
                    page_no=page_no,
                ))

        result.markdown = "\n\n".join(full_md)
        self.log.info(
            "pymupdf4llm.parse.done",
            source=source,
            texts=len(result.texts),
            tables=len(result.tables),
            images=len(result.images),
        )
        return result

    # ── markdown extraction helpers ───────────────────────────────────────────

    def _sanitize_meta(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        ChromaDB metadata restrictions:
          - No None values
          - No empty lists / non-empty lists (lists not supported at all)
          - No nested dicts
          - No tuples
          Only str, int, float, bool are allowed.
        """
        out: Dict[str, Any] = {}
        for k, v in meta.items():
            if v is None:
                out[k] = ""
            elif isinstance(v, (list, tuple)):
                # serialize non-empty sequences to string, empty → ""
                out[k] = str(v) if v else ""
            elif isinstance(v, dict):
                out[k] = str(v)
            elif isinstance(v, (str, int, float, bool)):
                out[k] = v
            else:
                out[k] = str(v)
        return out

    def _extract_md_tables(self, md: str) -> List[str]:
        """Extract complete markdown table blocks (consecutive lines starting with |)."""
        tables: List[str] = []
        current: List[str] = []
        for line in md.splitlines():
            if line.strip().startswith("|"):
                current.append(line)
            else:
                if current:
                    tables.append("\n".join(current))
                    current = []
        if current:
            tables.append("\n".join(current))
        return tables

    def _extract_image_refs(self, md: str) -> List[tuple]:
        """Extract (alt_text, path) from ![alt](path) patterns."""
        return re.findall(r"!\[([^\]]*)\]\(([^)]+)\)", md)

    def _extract_prose(self, md: str, strip_tables: bool, strip_images: bool) -> str:
        """Return text with table rows and/or image refs removed."""
        lines: List[str] = []
        in_table = False
        for line in md.splitlines():
            # skip table rows
            if strip_tables and line.strip().startswith("|"):
                in_table = True
                continue
            # blank line after table block
            if in_table and not line.strip():
                in_table = False
                continue
            in_table = False
            # remove inline image refs
            if strip_images:
                line = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", line)
            lines.append(line)
        return "\n".join(lines)

    def _encode_image(self, img_path: str) -> str:
        if not img_path or not os.path.exists(img_path):
            return ""
        try:
            return base64.b64encode(Path(img_path).read_bytes()).decode()
        except Exception:
            return ""

    def run(self, pdf_path: str, **kwargs: Any) -> ParsedPDF:
        return self.parse(pdf_path)

    def to_documents(self, parsed: ParsedPDF) -> Dict[str, List[Document]]:
        return {
            kind: [
                Document(page_content=el.content, metadata=el.metadata, doc_id=el.element_id)
                for el in getattr(parsed, kind)
            ]
            for kind in ("texts", "tables", "images")
        }


# ── MinerU 2.7.x parser — optional, higher accuracy ─────────────────────────

class MinerUParser(BaseComponent):
    """
    MinerU 2.7.x PDF parser.

    Setup (one-time):
        pip install mineru[all]
        mineru-models-download          # for pipeline backend

    backend:
        "pipeline"          — CPU layout model, fast after download
        "vlm-transformers"  — vision LLM, no extra download needed

    MinerU 2.7.x notes:
        - Flag: -b <backend>  (not -m)
        - Add --device cpu to avoid GPU crash when no GPU
        - Output: output_dir/{pdf_name}/{method_subdir}/
          e.g.    /tmp/out/QT.KT.03/hybrid_auto/
    """

    backend: str = "pipeline"
    parse_method: str = "auto"
    output_dir: Optional[str] = None
    lang: Optional[str] = None
    device: str = "cpu"
    extract_images: bool = True
    extract_tables: bool = True
    timeout: int = 600

    @model_validator(mode="after")
    def _set_name(self) -> "MinerUParser":
        if not self.name:
            self.name = f"mineru/{self.backend}"
        return self

    def parse(self, pdf_path: str) -> ParsedPDF:
        pdf_path = str(Path(pdf_path).resolve())
        source = os.path.basename(pdf_path)
        self.log.info("mineru.parse.start", source=source, backend=self.backend, device=self.device)

        output_dir = self.output_dir or tempfile.mkdtemp(prefix="mineru_")
        pdf_name = Path(pdf_path).stem

        content_list = self._run_cli(pdf_path, output_dir, pdf_name)
        if not content_list:
            self.log.warning("mineru.empty_content", source=source, output_dir=output_dir)

        result = self._parse_content_list(content_list, source, output_dir, pdf_name)
        self.log.info(
            "mineru.parse.done", source=source,
            texts=len(result.texts), tables=len(result.tables), images=len(result.images),
        )
        return result

    def _run_cli(self, pdf_path: str, output_dir: str, pdf_name: str) -> List[Dict]:
        cmd = [
            "mineru", "-p", pdf_path, "-o", output_dir,
            "-b", self.backend, "--device", self.device,
        ]
        if self.parse_method != "auto":
            cmd += ["-m", self.parse_method]
        if self.lang:
            cmd += ["--lang", self.lang]

        self.log.info("mineru.cli.run", cmd=" ".join(cmd))
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"MinerU timed out after {self.timeout}s")

        if r.returncode != 0:
            if "doclayout_yolo" in r.stderr:
                raise RuntimeError(
                    "MinerU pipeline thiếu model. Chạy: mineru-models-download\n"
                    "Hoặc đổi backend='vlm-transformers'"
                )
            raise RuntimeError(f"MinerU failed (code {r.returncode}): {r.stderr[-300:]}")

        return self._find_content_list(output_dir, pdf_name)

    def _find_content_list(self, output_dir: str, pdf_name: str) -> List[Dict]:
        # MinerU 2.7.x puts output in output_dir/{pdf_name}/{method_subdir}/
        # glob to find content_list regardless of subdir depth
        for jf in sorted(glob.glob(
            os.path.join(output_dir, "**", "*content_list*.json"), recursive=True
        )):
            data = self._load_json(jf)
            if isinstance(data, list):
                self.log.info("mineru.json_found", path=jf)
                return data

        # fallback: any json list
        for jf in sorted(Path(output_dir).glob("**/*.json")):
            data = self._load_json(str(jf))
            if isinstance(data, list) and data and isinstance(data[0], dict):
                self.log.info("mineru.json_fallback", path=str(jf))
                return data

        self.log.warning("mineru.json_not_found", output_dir=output_dir)
        return []

    def _load_json(self, path: str) -> Any:
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def _find_image_dir(self, output_dir: str, pdf_name: str) -> str:
        for d in sorted(Path(output_dir).glob(f"{pdf_name}/**/images")):
            if d.is_dir():
                return str(d)
        return os.path.join(output_dir, pdf_name, "images")

    def _parse_content_list(self, content_list, source, output_dir, pdf_name) -> ParsedPDF:
        result = ParsedPDF(source=source)
        image_base_dir = self._find_image_dir(output_dir, pdf_name)

        for item in content_list:
            itype = item.get("type", "")
            page_no = item.get("page_no", item.get("page_idx", 0))
            base_meta = {"source": source, "page_no": page_no}

            if itype == "text":
                text = item.get("text", "").strip()
                if not text:
                    continue
                result.texts.append(ParsedElement(
                    element_type="text", content=text,
                    metadata={**base_meta, "category": item.get("text_level", "paragraph")},
                    page_no=page_no,
                ))

            elif itype == "table" and self.extract_tables:
                caption  = item.get("table_caption", item.get("caption", ""))
                body     = item.get("table_body", item.get("body", item.get("text", "")))
                footnote = item.get("table_footnote", item.get("footnote", ""))
                content  = "\n".join(filter(None, [caption, body, footnote]))
                if not content.strip():
                    continue
                img_path = self._resolve_img(item, image_base_dir, output_dir, pdf_name)
                result.tables.append(ParsedElement(
                    element_type="table", content=content,
                    metadata={**base_meta, "table_caption": caption, "table_body": body,
                               "image_path": img_path,
                               "format": "html" if "<table" in body else "markdown"},
                    page_no=page_no,
                ))

            elif itype in ("image", "figure") and self.extract_images:
                caption  = item.get("img_caption", item.get("figure_caption", item.get("caption", "")))
                footnote = item.get("img_footnote", item.get("footnote", ""))
                img_path = self._resolve_img(item, image_base_dir, output_dir, pdf_name)
                content  = "\n".join(filter(None, [caption, footnote])) or f"[Image on page {page_no}]"
                result.images.append(ParsedElement(
                    element_type="image", content=content,
                    metadata={**base_meta, "img_caption": caption, "image_path": img_path,
                               "image_base64": self._encode_image(img_path) if img_path else ""},
                    page_no=page_no,
                ))

        for md_path in sorted(Path(output_dir).glob(f"{pdf_name}/**/*.md")):
            result.markdown = md_path.read_text(encoding="utf-8")
            break

        return result

    def _resolve_img(self, item, image_base_dir, output_dir, pdf_name) -> str:
        raw = item.get("img_path", item.get("image_path", item.get("figure_path", "")))
        if not raw:
            return ""
        for c in [raw,
                  os.path.join(image_base_dir, raw),
                  os.path.join(image_base_dir, os.path.basename(raw)),
                  os.path.join(output_dir, pdf_name, raw)]:
            if c and os.path.exists(c):
                return c
        return ""

    def _encode_image(self, img_path: str) -> str:
        if not img_path or not os.path.exists(img_path):
            return ""
        try:
            return base64.b64encode(Path(img_path).read_bytes()).decode()
        except Exception:
            return ""

    def run(self, pdf_path: str, **kwargs: Any) -> ParsedPDF:
        return self.parse(pdf_path)

    def to_documents(self, parsed: ParsedPDF) -> Dict[str, List[Document]]:
        return {
            kind: [
                Document(page_content=el.content, metadata=el.metadata, doc_id=el.element_id)
                for el in getattr(parsed, kind)
            ]
            for kind in ("texts", "tables", "images")
        }