from __future__ import annotations

from hashlib import md5
from typing import Any, Optional

def compute_args_hash(*args: Any) -> str:
    args_str = "".join([str(arg) for arg in args])
    try:
        return md5(args_str.encode("utf-8")).hexdigest()
    except UnicodeEncodeError:
        safe_bytes = args_str.encode("utf-8", errors="replace")
        return md5(safe_bytes).hexdigest()

def make_doc_id(text: str, source: Optional[str] = None, extra: Optional[str] = None) -> str:
    parts = [text]
    if source:
        parts.append(source)
    if extra:
        parts.append(extra)
    return f"doc-{compute_args_hash(*parts)}"

def make_chunk_id(text: str, parent_doc_id: Optional[str] = None, chunk_index: Optional[int] = None) -> str:
    parts = [text]
    if parent_doc_id:
        parts.append(parent_doc_id)
    if chunk_index is not None:
        parts.append(str(chunk_index))
    return f"chunk-{compute_args_hash(*parts)}"

def make_query_id(query: str) -> str:
    return f"query-{compute_args_hash(query)}"

def id_type(id_str: str) -> str:
    for prefix in ("doc", "chunk", "query"):
        if id_str.startswith(f"{prefix}-"):
            return prefix
    return "unknown"
