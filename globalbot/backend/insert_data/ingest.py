from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from model.ade import load_document, ADEAgent, to_json, to_markdown, to_visual_html
from rag.chunker import chunk_blocks
from rag.core import RAG


def ingest_document(
    doc_path: str,
    rag: RAG,
    ade_agent: ADEAgent,
    doc_id: Optional[str] = None,
    output_dir: Optional[str] = None,
    return_outputs: bool = False,
) -> dict:
    doc_id = doc_id or str(uuid.uuid4())
    logger.info(f"[Ingest] Starting document: {doc_path}")

    pages = load_document(doc_path)

    blocks = ade_agent.process_pages(pages)
    for b in blocks:
        b["doc_id"] = doc_id

    chunks = chunk_blocks(blocks, doc_id=doc_id)
    logger.info(f"[Ingest] {len(blocks)} blocks → {len(chunks)} chunks")

    rag.add_documents(chunks)
    logger.info(f"[Ingest] Stored {len(chunks)} chunks in vector DB")

    result = {
        "doc_id": doc_id,
        "doc_path": doc_path,
        "pages_processed": len(pages),
        "blocks_extracted": len(blocks),
        "chunks_stored": len(chunks),
    }

    if output_dir or return_outputs:
        out_dir = Path(output_dir or "/tmp/ade_output") / doc_id
        out_dir.mkdir(parents=True, exist_ok=True)

        (out_dir / "extraction.json").write_text(
            to_json(blocks), encoding="utf-8"
        )
        (out_dir / "extraction.md").write_text(
            to_markdown(blocks), encoding="utf-8"
        )

        page_images = {
            ctx.page: ctx.image_path
            for ctx in pages
            if ctx.page and ctx.image_path
        }
        (out_dir / "visual.html").write_text(
            to_visual_html(blocks, image_paths=page_images),
            encoding="utf-8",
        )

        result["output_dir"] = str(out_dir)
        logger.info(f"[Ingest] Outputs saved to {out_dir}")

        if return_outputs:
            result["json"] = to_json(blocks)
            result["markdown"] = to_markdown(blocks)

    return result
