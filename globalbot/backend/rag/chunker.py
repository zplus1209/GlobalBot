from __future__ import annotations

import json
import uuid
from typing import Optional

from langchain_core.documents import Document


_TEXT_LABELS = {"text", "list_item", "formula"}
_VISUAL_LABELS = {"image", "chart", "table"}


def _chunk_id() -> str:
    return str(uuid.uuid4())


def _meta(block: dict, doc_id: Optional[str]) -> dict:
    return {
        "doc_id": doc_id or "",
        "label": block.get("label", "text"),
        "origin_label": block.get("origin_label", ""),
        "page": block.get("page") or 0,
        "bbox": json.dumps(block.get("bbox", [])),
        "image_path": block.get("image_path", ""),
        "chunk_id": _chunk_id(),
    }


def chunk_blocks(blocks: list[dict], doc_id: Optional[str] = None) -> list[Document]:
    docs = []

    for block in blocks:
        label = block.get("label", "text")
        page = block.get("page")
        bbox = block.get("bbox", [])

        if label in _TEXT_LABELS:
            content = block.get("content", "").strip()
            summary = block.get("summary", "")
            description = block.get("description", "")

            if not content:
                continue

            text_parts = [content]
            if summary:
                text_parts.append(f"[SUMMARY] {summary}")
            if description:
                text_parts.append(f"[DESC] {description}")

            docs.append(Document(
                page_content="\n".join(text_parts),
                metadata={
                    **_meta(block, doc_id),
                    "content_type": "text",
                },
            ))

        elif label == "table":
            headers = block.get("column_headers", [])
            rows = block.get("rows", [])
            title = block.get("table_title", "")
            notes = block.get("notes", "")

            lines = []
            if title:
                lines.append(f"Table: {title}")
            if headers:
                lines.append("Columns: " + " | ".join(str(h) for h in headers))
            for row in rows:
                if isinstance(row, list):
                    lines.append(" | ".join(str(c) for c in row))
                else:
                    lines.append(str(row))
            if notes:
                lines.append(f"Notes: {notes}")

            if not lines:
                continue

            docs.append(Document(
                page_content="\n".join(lines),
                metadata={
                    **_meta(block, doc_id),
                    "content_type": "table",
                    "table_title": title,
                },
            ))

        elif label == "image":
            caption = block.get("image_caption", "")
            purpose = block.get("purpose", "")
            key_elements = block.get("key_elements", [])
            figure_type = block.get("figure_type", "")

            lines = []
            if figure_type:
                lines.append(f"Figure type: {figure_type}")
            if caption:
                lines.append(f"Description: {caption}")
            if purpose:
                lines.append(f"Purpose: {purpose}")
            if key_elements:
                lines.append("Key elements: " + ", ".join(str(e) for e in key_elements))

            if not lines:
                continue

            docs.append(Document(
                page_content="\n".join(lines),
                metadata={
                    **_meta(block, doc_id),
                    "content_type": "image",
                    "figure_type": figure_type,
                },
            ))

        elif label == "chart":
            caption = block.get("image_caption", "")
            chart_data = block.get("chart_data", {})
            trend = chart_data.get("trend", "")
            x_axis = chart_data.get("x_axis", {})
            y_axis = chart_data.get("y_axis", {})

            lines = []
            if caption:
                lines.append(f"Chart title: {caption}")
            if trend:
                lines.append(f"Trend: {trend}")
            if x_axis:
                lines.append(f"X-axis: {json.dumps(x_axis)}")
            if y_axis:
                lines.append(f"Y-axis: {json.dumps(y_axis)}")

            if not lines:
                continue

            docs.append(Document(
                page_content="\n".join(lines),
                metadata={
                    **_meta(block, doc_id),
                    "content_type": "chart",
                },
            ))

    return docs
