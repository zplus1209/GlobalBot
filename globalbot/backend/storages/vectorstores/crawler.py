from __future__ import annotations

import time
from typing import Any, Dict, Generator, List, Optional, Set
from urllib.parse import urljoin, urlparse

from pydantic import Field, model_validator

from globalbot.backend.base import BaseComponent, Document


class WebCrawler(BaseComponent):
    start_urls: List[str] = Field(default_factory=list)
    max_depth: int = 2
    max_pages: int = 100
    delay: float = 0.5
    allowed_domains: Optional[List[str]] = None
    exclude_patterns: List[str] = Field(default_factory=list)
    chunk_size: int = 1000
    chunk_overlap: int = 200
    user_agent: str = "GlobalBot/1.0"

    _visited: Set[str] = set()

    def model_post_init(self, __context: Any) -> None:
        self._visited = set()

    @model_validator(mode="after")
    def _set_name(self) -> "WebCrawler":
        if not self.name:
            self.name = "web_crawler"
        return self

    def _is_allowed(self, url: str) -> bool:
        parsed = urlparse(url)
        if not parsed.scheme.startswith("http"):
            return False
        if self.allowed_domains:
            return any(domain in parsed.netloc for domain in self.allowed_domains)
        return True

    def _is_excluded(self, url: str) -> bool:
        return any(pat in url for pat in self.exclude_patterns)

    def _fetch(self, url: str) -> Optional[str]:
        import requests

        try:
            headers = {"User-Agent": self.user_agent}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.text
        except Exception as e:
            self.log.warning("crawler.fetch.error", url=url, error=str(e)[:80])
            return None

    def _parse(self, html: str, base_url: str) -> tuple[str, List[str]]:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        text = "\n".join(line for line in text.splitlines() if line.strip())

        links = []
        for a in soup.find_all("a", href=True):
            href = urljoin(base_url, a["href"])
            href = href.split("#")[0]
            if href not in self._visited and self._is_allowed(href) and not self._is_excluded(href):
                links.append(href)

        return text, links

    def _chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Document]:
        from globalbot.backend.utils.hashing import make_doc_id

        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            if chunk_text.strip():
                doc_id = make_doc_id(chunk_text, source=metadata.get("source", ""))
                chunks.append(Document(
                    page_content=chunk_text,
                    metadata={**metadata, "chunk_start": start},
                    doc_id=doc_id,
                ))
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def crawl(self) -> Generator[Document, None, None]:
        queue = [(url, 0) for url in self.start_urls]
        page_count = 0

        while queue and page_count < self.max_pages:
            url, depth = queue.pop(0)

            if url in self._visited:
                continue
            self._visited.add(url)

            self.log.info("crawler.fetch", url=url, depth=depth, pages=page_count)
            html = self._fetch(url)
            if not html:
                continue

            text, links = self._parse(html, url)
            page_count += 1

            parsed = urlparse(url)
            metadata = {
                "source": url,
                "domain": parsed.netloc,
                "depth": depth,
                "crawler": "WebCrawler",
            }

            chunks = self._chunk_text(text, metadata)
            for chunk in chunks:
                yield chunk

            if depth < self.max_depth:
                for link in links:
                    if link not in self._visited:
                        queue.append((link, depth + 1))

            time.sleep(self.delay)

        self.log.info("crawler.done", total_pages=page_count, total_urls=len(self._visited))

    def run(self, **kwargs: Any) -> List[Document]:
        return list(self.crawl())


class MongoDBWebIngestor(BaseComponent):
    crawler: WebCrawler
    connection_string: str = "mongodb://localhost:27017"
    database_name: str = "globalbot_db"
    collection_name: str = "web_docs"
    batch_size: int = 50

    _mongo_collection: Any = None

    @model_validator(mode="after")
    def _init_mongo(self) -> "MongoDBWebIngestor":
        from pymongo import MongoClient

        client = MongoClient(self.connection_string)
        db = client[self.database_name]
        self._mongo_collection = db[self.collection_name]
        self._mongo_collection.create_index("doc_id", unique=True, background=True)
        self._mongo_collection.create_index("metadata.source", background=True)
        if not self.name:
            self.name = f"mongodb_ingestor/{self.collection_name}"
        return self

    def _save_batch(self, docs: List[Document]) -> int:
        from pymongo import UpdateOne

        ops = [
            UpdateOne(
                {"doc_id": doc.doc_id},
                {"$set": {
                    "doc_id": doc.doc_id,
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                }},
                upsert=True,
            )
            for doc in docs
        ]
        result = self._mongo_collection.bulk_write(ops, ordered=False)
        return result.upserted_count + result.modified_count

    def run(self, **kwargs: Any) -> Dict[str, Any]:
        total_saved = 0
        batch: List[Document] = []

        for doc in self.crawler.crawl():
            batch.append(doc)
            if len(batch) >= self.batch_size:
                saved = self._save_batch(batch)
                total_saved += saved
                self.log.info("ingestor.batch_saved", saved=saved, total=total_saved)
                batch = []

        if batch:
            saved = self._save_batch(batch)
            total_saved += saved

        self.log.info("ingestor.done", total_saved=total_saved)
        return {"total_saved": total_saved, "total_crawled": len(self.crawler._visited)}

    def get_docs(self, filters: Optional[Dict[str, Any]] = None, limit: int = 100) -> List[Document]:
        query = filters or {}
        cursor = self._mongo_collection.find(query).limit(limit)
        docs = []
        for record in cursor:
            docs.append(Document(
                page_content=record.get("text", ""),
                metadata=record.get("metadata", {}),
                doc_id=record.get("doc_id", ""),
            ))
        return docs