from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Optional


def to_json(blocks: list[dict], indent: int = 2) -> str:
    return json.dumps(blocks, ensure_ascii=False, indent=indent)


def to_markdown(blocks: list[dict]) -> str:
    lines = []
    for b in blocks:
        label = b.get("label", "text")
        page = b.get("page")
        page_tag = f" *(p.{page})*" if page else ""

        if label == "text":
            content = b.get("content", "")
            if content:
                lines.append(content + page_tag)
                lines.append("")

        elif label == "list_item":
            content = b.get("content", "")
            if content:
                lines.append(f"- {content}{page_tag}")

        elif label == "table":
            title = b.get("table_title", "")
            headers = b.get("column_headers", [])
            rows = b.get("rows", [])
            notes = b.get("notes", "")

            if title:
                lines.append(f"**{title}**{page_tag}")
            if headers:
                lines.append("| " + " | ".join(str(h) for h in headers) + " |")
                lines.append("| " + " | ".join("---" for _ in headers) + " |")
                for row in rows:
                    if isinstance(row, list):
                        lines.append("| " + " | ".join(str(c) for c in row) + " |")
            if notes:
                lines.append(f"> {notes}")
            lines.append("")

        elif label == "formula":
            content = b.get("content", "")
            desc = b.get("description", "")
            lines.append(f"$$\n{content}\n$${page_tag}")
            if desc:
                lines.append(f"*{desc}*")
            lines.append("")

        elif label in {"image", "chart"}:
            caption = b.get("image_caption", "")
            path = b.get("image_path", "")
            if path and Path(path).exists():
                lines.append(f"![{caption}]({path}){page_tag}")
            elif caption:
                lines.append(f"**[{label.upper()}]** {caption}{page_tag}")
            lines.append("")

    return "\n".join(lines)


def to_visual_html(
    blocks: list[dict],
    image_paths: Optional[dict[int, str]] = None,
) -> str:
    def _b64_img(path: str) -> str:
        try:
            with open(path, "rb") as f:
                data = base64.b64encode(f.read()).decode()
            ext = Path(path).suffix.lstrip(".")
            return f"data:image/{ext};base64,{data}"
        except Exception:
            return ""

    label_colors = {
        "text": "#2563eb",
        "list_item": "#7c3aed",
        "table": "#d97706",
        "image": "#059669",
        "chart": "#dc2626",
        "formula": "#0891b2",
    }

    items_html = []
    for i, b in enumerate(blocks):
        label = b.get("label", "text")
        page = b.get("page", "?")
        bbox = b.get("bbox", [])
        color = label_colors.get(label, "#6b7280")

        header = (
            f'<div class="block-header" style="background:{color}">'
            f'<span class="label">{label.upper()}</span>'
            f'<span class="meta">Page {page} | bbox {bbox}</span>'
            f'</div>'
        )

        body_parts = []

        if label in {"text", "list_item", "formula"}:
            content = b.get("content", "")
            summary = b.get("summary", "")
            if content:
                body_parts.append(f'<p class="content">{content}</p>')
            if summary:
                body_parts.append(f'<p class="summary"><em>{summary}</em></p>')

        elif label == "table":
            title = b.get("table_title", "")
            headers = b.get("column_headers", [])
            rows = b.get("rows", [])
            if title:
                body_parts.append(f'<p class="table-title"><strong>{title}</strong></p>')
            if headers:
                th_row = "".join(f"<th>{h}</th>" for h in headers)
                tr_rows = "".join(
                    "<tr>" + "".join(f"<td>{c}</td>" for c in (r if isinstance(r, list) else [r])) + "</tr>"
                    for r in rows
                )
                body_parts.append(
                    f'<table><thead><tr>{th_row}</tr></thead><tbody>{tr_rows}</tbody></table>'
                )

        elif label in {"image", "chart"}:
            caption = b.get("image_caption", "")
            path = b.get("image_path", "")
            if path:
                src = _b64_img(path)
                if src:
                    body_parts.append(f'<img src="{src}" alt="{caption}" class="region-img">')
            if caption:
                body_parts.append(f'<p class="caption">{caption}</p>')

            if label == "chart":
                chart_data = b.get("chart_data", {})
                if chart_data.get("trend"):
                    body_parts.append(f'<p class="trend">Trend: {chart_data["trend"]}</p>')

        body = '<div class="block-body">' + "".join(body_parts) + "</div>"
        items_html.append(f'<div class="block" data-label="{label}" data-page="{page}">{header}{body}</div>')

    page_imgs_html = ""
    if image_paths:
        for pg, img_path in sorted(image_paths.items()):
            src = _b64_img(img_path)
            if src:
                page_imgs_html += (
                    f'<div class="page-container">'
                    f'<h3>Page {pg}</h3>'
                    f'<img src="{src}" class="page-img" id="page-{pg}">'
                    f'</div>'
                )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ADE — Document Extraction Result</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #f8fafc; color: #1e293b; }}
.layout {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0; min-height: 100vh; }}
.pane {{ padding: 1.5rem; overflow-y: auto; max-height: 100vh; }}
.pane-left {{ background: #fff; border-right: 1px solid #e2e8f0; }}
.pane-right {{ background: #f8fafc; }}
h2 {{ font-size: 1.1rem; font-weight: 700; margin-bottom: 1rem; color: #334155; letter-spacing: .05em; text-transform: uppercase; }}
.block {{ border: 1px solid #e2e8f0; border-radius: 8px; margin-bottom: 1rem; overflow: hidden; background: #fff; }}
.block-header {{ display: flex; justify-content: space-between; align-items: center; padding: .4rem .8rem; }}
.label {{ color: #fff; font-size: .7rem; font-weight: 700; letter-spacing: .08em; }}
.meta {{ color: rgba(255,255,255,.8); font-size: .65rem; }}
.block-body {{ padding: .8rem 1rem; }}
.content {{ font-size: .9rem; line-height: 1.6; margin-bottom: .5rem; }}
.summary {{ font-size: .8rem; color: #64748b; }}
.caption {{ font-size: .8rem; color: #475569; margin-top: .4rem; font-style: italic; }}
.trend {{ font-size: .8rem; color: #64748b; }}
.table-title {{ font-size: .9rem; margin-bottom: .5rem; }}
table {{ width: 100%; border-collapse: collapse; font-size: .8rem; }}
th {{ background: #f1f5f9; padding: .3rem .6rem; text-align: left; border: 1px solid #e2e8f0; }}
td {{ padding: .3rem .6rem; border: 1px solid #e2e8f0; }}
.region-img {{ max-width: 100%; border-radius: 4px; margin-top: .5rem; }}
.page-img {{ max-width: 100%; border-radius: 6px; margin-bottom: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,.1); }}
.page-container h3 {{ font-size: .95rem; color: #475569; margin-bottom: .5rem; }}
</style>
</head>
<body>
<div class="layout">
  <div class="pane pane-left">
    <h2>Extracted Regions ({len(blocks)})</h2>
    {"".join(items_html)}
  </div>
  <div class="pane pane-right">
    <h2>Source Pages</h2>
    {page_imgs_html if page_imgs_html else '<p style="color:#94a3b8">No page images available.</p>'}
  </div>
</div>
</body>
</html>"""
