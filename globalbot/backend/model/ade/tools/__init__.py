from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field


class TextAnalysisInput(BaseModel):
    region_id: int = Field(description="Region position index")
    ocr_text: str = Field(description="OCR extracted text content")
    region_type: str = Field(description="Region label type")


class TableAnalysisInput(BaseModel):
    region_id: int
    image_base64: str = Field(description="Base64 encoded crop image")
    ocr_text: str = Field(description="Raw OCR text from the table region")


class ImageAnalysisInput(BaseModel):
    region_id: int
    image_base64: str
    region_type: str


class ChartAnalysisInput(BaseModel):
    region_id: int
    image_base64: str
    ocr_text: str = Field(description="OCR text overlaid on chart if any")


def _call_vlm(vlm, image_base64: str, prompt: str) -> str:
    msg = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
    ])
    return vlm.invoke([msg]).content


def _safe_json(text: str) -> dict:
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        return json.loads(text[start:end])
    except Exception:
        return {}


class TextAnalysisTool(BaseTool):
    name: str = "analysis_text"
    description: str = (
        "Analyzes a text region from a document. "
        "Returns summary and description of the text content."
    )
    args_schema: type[BaseModel] = TextAnalysisInput
    llm: Any = Field(exclude=True)

    def _run(self, region_id: int, ocr_text: str, region_type: str) -> dict:
        if not ocr_text.strip():
            return {"region_id": region_id, "summary": "", "description": ""}

        prompt = (
            f"You are a document analysis assistant. Given this {region_type} text:\n\n"
            f"{ocr_text}\n\n"
            "Respond ONLY with valid JSON:\n"
            '{"summary": "<one sentence summary>", "description": "<detailed description>"}'
        )

        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            parsed = _safe_json(content)
            return {
                "region_id": region_id,
                "summary": parsed.get("summary", ""),
                "description": parsed.get("description", ""),
            }
        except Exception:
            return {"region_id": region_id, "summary": "", "description": ""}

    async def _arun(self, **kwargs):
        return self._run(**kwargs)


class TableAnalysisTool(BaseTool):
    name: str = "analysis_table"
    description: str = (
        "Analyzes a table region from a document image crop. "
        "Returns headers, rows, values, notes extracted by VLM."
    )
    args_schema: type[BaseModel] = TableAnalysisInput
    vlm: Any = Field(exclude=True)

    _PROMPT = (
        "Extract this table as structured JSON. Respond ONLY with valid JSON:\n"
        '{"table_title": "", "column_headers": [], "rows": [], "notes": ""}'
    )

    def _run(self, region_id: int, image_base64: str, ocr_text: str) -> dict:
        try:
            raw = _call_vlm(self.vlm, image_base64, self._PROMPT)
            parsed = _safe_json(raw)
            return {
                "region_id": region_id,
                "table_title": parsed.get("table_title", ""),
                "column_headers": parsed.get("column_headers", []),
                "rows": parsed.get("rows", []),
                "notes": parsed.get("notes", ""),
                "raw_footnote": json.dumps(parsed, ensure_ascii=False)[:1200],
            }
        except Exception:
            return {"region_id": region_id, "table_title": "", "column_headers": [], "rows": [], "notes": ""}

    async def _arun(self, **kwargs):
        return self._run(**kwargs)


class ImageAnalysisTool(BaseTool):
    name: str = "analysis_image"
    description: str = (
        "Analyzes an image or figure region from a document. "
        "Returns figure_type, descriptions, key_elements, annotations, purpose."
    )
    args_schema: type[BaseModel] = ImageAnalysisInput
    vlm: Any = Field(exclude=True)

    _PROMPT = (
        "Describe this figure in detail. Respond ONLY with valid JSON:\n"
        '{"figure_type": "", "description": "", "key_elements": [], "annotations": [], "purpose": ""}'
    )

    def _run(self, region_id: int, image_base64: str, region_type: str) -> dict:
        try:
            raw = _call_vlm(self.vlm, image_base64, self._PROMPT)
            parsed = _safe_json(raw)
            return {
                "region_id": region_id,
                "figure_type": parsed.get("figure_type", region_type),
                "description": parsed.get("description", ""),
                "key_elements": parsed.get("key_elements", []),
                "annotations": parsed.get("annotations", []),
                "purpose": parsed.get("purpose", ""),
            }
        except Exception:
            return {"region_id": region_id, "figure_type": region_type, "description": "", "key_elements": []}

    async def _arun(self, **kwargs):
        return self._run(**kwargs)


class ChartAnalysisTool(BaseTool):
    name: str = "analysis_chart"
    description: str = (
        "Analyzes a chart region from a document. "
        "Returns title, trend, x_axis, y_axis, data_points."
    )
    args_schema: type[BaseModel] = ChartAnalysisInput
    vlm: Any = Field(exclude=True)

    _PROMPT = (
        "Analyze this chart. Respond ONLY with valid JSON:\n"
        '{"title": "", "trend": "", "x_axis": {"label": "", "values": []}, '
        '"y_axis": {"label": "", "unit": ""}, "key_data_points": []}'
    )

    def _run(self, region_id: int, image_base64: str, ocr_text: str) -> dict:
        try:
            raw = _call_vlm(self.vlm, image_base64, self._PROMPT)
            parsed = _safe_json(raw)
            return {
                "region_id": region_id,
                "title": parsed.get("title", ""),
                "trend": parsed.get("trend", ""),
                "x_axis": parsed.get("x_axis", {}),
                "y_axis": parsed.get("y_axis", {}),
                "key_data_points": parsed.get("key_data_points", []),
            }
        except Exception:
            return {"region_id": region_id, "title": "", "trend": ""}

    async def _arun(self, **kwargs):
        return self._run(**kwargs)
