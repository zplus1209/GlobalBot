from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


UPLOAD_DIR = Path("./uploads")
META_DIR   = Path("./uploads/.meta")
CHAT_DIR   = META_DIR / "chats"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)
CHAT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class DocumentRecord:
    doc_id:          str
    filename:        str
    file_path:       str
    mime_type:       str
    pages:           int          = 0
    blocks_count:    int          = 0
    chunks_count:    int          = 0
    status:          str          = "pending"   # pending | processing | ready | error
    error:           Optional[str] = None
    created_at:      float        = field(default_factory=time.time)
    updated_at:      float        = field(default_factory=time.time)

    def save(self):
        self.updated_at = time.time()
        meta_path = META_DIR / f"{self.doc_id}.json"
        meta_path.write_text(json.dumps(asdict(self)), encoding="utf-8")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def load(cls, doc_id: str) -> Optional["DocumentRecord"]:
        meta_path = META_DIR / f"{doc_id}.json"
        if not meta_path.exists():
            return None
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        return cls(**data)


class DocumentStore:
    def list_all(self) -> list[DocumentRecord]:
        records = []
        for f in META_DIR.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                records.append(DocumentRecord(**data))
            except Exception:
                continue
        return sorted(records, key=lambda r: r.created_at, reverse=True)

    def get(self, doc_id: str) -> Optional[DocumentRecord]:
        return DocumentRecord.load(doc_id)

    def delete(self, doc_id: str) -> bool:
        rec = DocumentRecord.load(doc_id)
        if rec is None:
            return False
        Path(rec.file_path).unlink(missing_ok=True)
        (META_DIR / f"{doc_id}.json").unlink(missing_ok=True)
        crops = Path(f"./uploads/{doc_id}_crops")
        if crops.exists():
            shutil.rmtree(crops)
        return True


@dataclass
class ChatSessionRecord:
    chat_id: str
    mode: str = "knowledge"
    title: str = "New chat"
    messages: list[dict] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def save(self):
        self.updated_at = time.time()
        (CHAT_DIR / f"{self.chat_id}.json").write_text(
            json.dumps(asdict(self), ensure_ascii=False),
            encoding="utf-8",
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def load(cls, chat_id: str) -> Optional["ChatSessionRecord"]:
        path = CHAT_DIR / f"{chat_id}.json"
        if not path.exists():
            return None
        return cls(**json.loads(path.read_text(encoding="utf-8")))


class ChatStore:
    def list_all(self, mode: Optional[str] = None) -> list[ChatSessionRecord]:
        items: list[ChatSessionRecord] = []
        for f in CHAT_DIR.glob("*.json"):
            try:
                obj = ChatSessionRecord(**json.loads(f.read_text(encoding="utf-8")))
                if mode and obj.mode != mode:
                    continue
                items.append(obj)
            except Exception:
                continue
        return sorted(items, key=lambda x: x.updated_at, reverse=True)

    def get(self, chat_id: str) -> Optional[ChatSessionRecord]:
        return ChatSessionRecord.load(chat_id)

    def delete(self, chat_id: str) -> bool:
        rec = ChatSessionRecord.load(chat_id)
        if not rec:
            return False
        (CHAT_DIR / f"{chat_id}.json").unlink(missing_ok=True)
        return True
    

store = DocumentStore()
chat_store = ChatStore()