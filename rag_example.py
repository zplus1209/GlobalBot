"""
RAG Example — Ollama local (không cần API key)

Cài đặt:
    pip install pymupdf4llm chromadb langchain-text-splitters rank-bm25
    ollama pull nomic-embed-text:v1.5
    ollama pull qwen3.5:35b

Cách dùng:
    python rag_example_ollama.py ingest path/to/file.pdf
    python rag_example_ollama.py ask "nội dung chính là gì?"
    python rag_example_ollama.py chat

Khi MinerU sẵn sàng (chạy `mineru-models-download` xong):
    python rag_example_ollama.py --mineru ingest path/to/file.pdf
"""

import argparse
import sys
from pathlib import Path

OLLAMA_URL   = "http://localhost:11434"
EMBED_MODEL  = "nomic-embed-text:v1.5"
CHAT_MODEL   = "qwen3.5:35b"
CHROMA_DIR   = "./chroma_local_db"
COLLECTION   = "globalbot_local"


def _build_parser(use_mineru: bool, mineru_backend: str):
    if use_mineru:
        from globalbot.backend.loaders.mineru_loader import MinerUParser
        return MinerUParser(backend=mineru_backend)
    from globalbot.backend.loaders.mineru_loader import PyMuPDF4LLMParser
    return PyMuPDF4LLMParser(extract_tables=True, extract_images=False)


def build_ingestion_pipeline(use_mineru: bool = False, mineru_backend: str = "pipeline"):
    from globalbot.backend.embeddings.ollama_impl import OllamaEmbeddings
    from globalbot.backend.storages.vectorstores.chroma_impl import ChromaVectorStore
    from globalbot.backend.chunkings.fixed import RecursiveChunker
    from globalbot.backend.loaders.ingestion_pipeline import PDFIngestionPipeline

    parser_name = f"MinerU({mineru_backend})" if use_mineru else "PyMuPDF4LLM"
    print(f"[OK] Parser: {parser_name}")

    pipeline = PDFIngestionPipeline(
        vectorstore=ChromaVectorStore(collection_name=COLLECTION, persist_directory=CHROMA_DIR),
        embeddings=OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL),
        chunker=RecursiveChunker(chunk_size=500, chunk_overlap=100),
        parser=_build_parser(use_mineru, mineru_backend),
        index_tables=True,
        index_images=False,
        table_collection=f"{COLLECTION}_tables",
        batch_size=8,
    )
    return pipeline


def build_chat_pipeline():
    from globalbot.backend.embeddings.ollama_impl import OllamaEmbeddings
    from globalbot.backend.storages.vectorstores.chroma_impl import ChromaVectorStore
    from globalbot.backend.llms.chats.ollama_impl import ChatOllama
    from globalbot.backend.storages.retrieval import RAGRetriever
    from globalbot.backend.loaders.chat_pipeline import RAGChatPipeline

    pipeline = RAGChatPipeline(
        retriever=RAGRetriever(
            vectorstore=ChromaVectorStore(collection_name=COLLECTION, persist_directory=CHROMA_DIR),
            embeddings=OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL),
            top_k=5,
        ),
        llm=ChatOllama(model=CHAT_MODEL, base_url=OLLAMA_URL),
        return_sources=True,
        max_history=6,
    )
    print(f"[OK] Chat pipeline — {CHAT_MODEL}")
    return pipeline


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_ingest(args):
    pipeline = build_ingestion_pipeline(args.mineru, args.mineru_backend)

    paths = []
    for p in args.pdf:
        path = Path(p)
        if path.is_dir():
            paths.extend(path.glob("**/*.pdf"))
        elif path.exists():
            paths.append(path)
        else:
            print(f"[WARN] Không tìm thấy: {p}")

    if not paths:
        print("[ERROR] Không có PDF.")
        sys.exit(1)

    for pdf in paths:
        print(f"\n>> {pdf.name}")
        try:
            stats = pipeline.ingest(str(pdf))
            p = stats["parsed"]
            i = stats["indexed"]
            print(f"   Parsed : texts={p['texts']}, tables={p['tables']}, images={p['images']}")
            print(f"   Indexed: {i}")
        except RuntimeError as e:
            print(f"   [ERROR] {e}")
            sys.exit(1)

    print(f"\n[OK] Xong. DB: {CHROMA_DIR}")


def cmd_ask(args):
    pipeline = build_chat_pipeline()
    print(f"\nQuestion: {args.question}\n{'─'*50}")
    msg = pipeline.chat(args.question)
    print(f"\nAnswer:\n{msg.content}")
    if msg.sources:
        print(f"\n{'─'*50}\nSources ({len(msg.sources)}):")
        for i, s in enumerate(msg.sources, 1):
            src  = s.metadata.get("source", "?")
            page = s.metadata.get("page_no", "")
            pstr = f" | p.{page}" if page != "" else ""
            print(f"  [{i}] {src}{pstr} | score={s.score:.3f}")
            print(f"       {s.page_content[:120].strip()}")


def cmd_chat(args):
    pipeline = build_chat_pipeline()
    print(f"\nRAG Chat ({CHAT_MODEL}) — 'reset' xoá history, 'quit' thoát\n")
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not q:
            continue
        if q.lower() in ("quit", "exit"):
            break
        if q.lower() == "reset":
            pipeline.reset_history()
            print("[System] History đã reset.\n")
            continue
        print("Assistant: ", end="", flush=True)
        for chunk in pipeline.stream(q):
            print(chunk, end="", flush=True)
        print("\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="GlobalBot RAG — Ollama local")
    p.add_argument("--mineru", action="store_true",
                   help="Dùng MinerU thay vì PyMuPDF4LLM (cần mineru-models-download)")
    p.add_argument("--mineru-backend", default="pipeline",
                   choices=["pipeline", "vlm-transformers"],
                   help="MinerU backend (default: pipeline)")

    sub = p.add_subparsers(dest="command")

    si = sub.add_parser("ingest", help="Parse PDF và lưu vào DB")
    si.add_argument("pdf", nargs="+")

    sa = sub.add_parser("ask", help="Hỏi một câu")
    sa.add_argument("question")

    sub.add_parser("chat", help="Interactive chat")

    args = p.parse_args()
    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "ask":
        cmd_ask(args)
    elif args.command == "chat":
        cmd_chat(args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()