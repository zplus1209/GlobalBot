from __future__ import annotations

import json
from typing import Any, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from loguru import logger

from model.ade.pipeline import (
    PageContext, RegionContext,
    _TEXT_TYPES, _TABLE_TYPES, _FORMULA_TYPES, _CHART_TYPES,
    _LIST_ORIGINS,
)
from model.ade.tools import (
    TextAnalysisTool,
    TableAnalysisTool,
    ImageAnalysisTool,
    ChartAnalysisTool,
)


_FORMULA_PROMPT = (
    "Extract this mathematical formula as LaTeX. Respond ONLY with valid JSON:\n"
    '{"latex": "", "description": ""}'
)


def _build_system_context(page_ctx: PageContext) -> str:
    lines = [
        "Document regions (reading order, top→bottom):",
        f"Page: {page_ctx.page or 'N/A'}  |  Size: {page_ctx.img_w}×{page_ctx.img_h}",
        "",
    ]
    for r in page_ctx.regions:
        lines.append(f"  [{r.position}] type={r.region_type}  bbox={r.bbox}  ocr={r.ocr_text[:80]!r}")
    return "\n".join(lines)


class ADEAgent:
    def __init__(
        self,
        llm: Any,
        vlm: Optional[Any] = None,
        verbose: bool = False,
    ):
        self.llm = llm
        self.vlm = vlm or llm
        self.verbose = verbose

        self.text_tool = TextAnalysisTool(llm=self.llm)
        self.table_tool = TableAnalysisTool(vlm=self.vlm)
        self.image_tool = ImageAnalysisTool(vlm=self.vlm)
        self.chart_tool = ChartAnalysisTool(vlm=self.vlm)

    def _safe_json(self, text: str) -> dict:
        try:
            s, e = text.find("{"), text.rfind("}") + 1
            return json.loads(text[s:e])
        except Exception:
            return {}

    def _call_vlm_raw(self, b64: str, prompt: str) -> str:
        msg = HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ])
        resp = self.vlm.invoke([msg])
        return resp.content if hasattr(resp, "content") else str(resp)

    def _process_region(self, r: RegionContext, page: Optional[int]) -> Optional[dict]:
        if self.verbose:
            logger.info(f"  Region {r.position} [{r.region_type}]")

        block = {"bbox": r.bbox, "page": page, "origin_label": r.region_type}

        if not r.needs_vlm:
            if not r.ocr_text:
                return None
            analysis = self.text_tool._run(
                region_id=r.position,
                ocr_text=r.ocr_text,
                region_type=r.region_type,
            )
            block.update({
                "label": "list_item" if r.region_type in _LIST_ORIGINS else "text",
                "content": r.ocr_text,
                "summary": analysis.get("summary", ""),
                "description": analysis.get("description", ""),
            })

        elif r.region_type in _TABLE_TYPES:
            result = self.table_tool._run(
                region_id=r.position,
                image_base64=r.base64_img,
                ocr_text=r.ocr_text,
            )
            block.update({
                "label": "table",
                "content": r.ocr_text,
                "table_title": result.get("table_title", ""),
                "column_headers": result.get("column_headers", []),
                "rows": result.get("rows", []),
                "notes": result.get("notes", ""),
                "table_footnote": result.get("raw_footnote", ""),
                "image_path": r.crop_path,
            })

        elif r.region_type in _FORMULA_TYPES:
            raw = self._call_vlm_raw(r.base64_img, _FORMULA_PROMPT)
            parsed = self._safe_json(raw)
            block.update({
                "label": "formula",
                "content": parsed.get("latex", r.ocr_text),
                "description": parsed.get("description", ""),
            })

        elif r.region_type in _CHART_TYPES:
            result = self.chart_tool._run(
                region_id=r.position,
                image_base64=r.base64_img,
                ocr_text=r.ocr_text,
            )
            block.update({
                "label": "chart",
                "content": r.ocr_text,
                "image_caption": result.get("title", ""),
                "image_footnote": json.dumps(result, ensure_ascii=False)[:1000],
                "chart_data": {
                    "trend": result.get("trend", ""),
                    "x_axis": result.get("x_axis", {}),
                    "y_axis": result.get("y_axis", {}),
                    "key_data_points": result.get("key_data_points", []),
                },
                "image_path": r.crop_path,
            })

        else:
            result = self.image_tool._run(
                region_id=r.position,
                image_base64=r.base64_img,
                region_type=r.region_type,
            )
            block.update({
                "label": "image",
                "content": r.ocr_text,
                "image_caption": result.get("description", ""),
                "image_footnote": json.dumps(result, ensure_ascii=False)[:1000],
                "figure_type": result.get("figure_type", ""),
                "key_elements": result.get("key_elements", []),
                "purpose": result.get("purpose", ""),
                "image_path": r.crop_path,
            })

        return block

    def process_page(self, page_ctx: PageContext) -> list[dict]:
        if self.verbose:
            logger.info(f"[ADEAgent] Page {page_ctx.page} — {len(page_ctx.regions)} regions")
            logger.debug(_build_system_context(page_ctx))

        results = []
        for r in page_ctx.regions:
            block = self._process_region(r, page_ctx.page)
            if block is not None:
                results.append(block)

        return results

    def process_pages(self, pages: list[PageContext]) -> list[dict]:
        all_blocks = []
        for page_ctx in pages:
            blocks = self.process_page(page_ctx)
            all_blocks.extend(blocks)
        return all_blocks
