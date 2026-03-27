from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

load_dotenv()
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")


def create_app(args: argparse.Namespace) -> FastAPI:
    from llms.factory import init_singletons
    from globalbot.api.routes.documents import router as doc_router
    from globalbot.api.routes.chat import router as chat_router
    from globalbot.api.routes.pipeline import router as pipeline_router
    from globalbot.api.routes.knowledge import router as knowledge_router

    init_singletons(
        mode=args.mode,
        model_name=args.model_name,
        model_version=args.model_version,
        api_key=_api_key(args),
        base_url=_base_url(args),
        engine=args.model_engine,
        db_type=args.db,
        embedding_name=args.embedding_model,
    )

    app = FastAPI(title="Document RAG API", version="2.0.0", docs_url="/docs")
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"],
        allow_methods=["*"], allow_headers=["*"],
    )
    app.include_router(doc_router)
    app.include_router(chat_router)
    app.include_router(pipeline_router)
    app.include_router(knowledge_router)
    
    frontend = Path("./frontend")
    if frontend.exists():
        app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

    return app


def _api_key(args):
    m = {"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY", "together": "TOGETHER_API_KEY"}
    k = m.get(args.model_name)
    return os.getenv(k) if k else None


def _base_url(args):
    m = {"together": "TOGETHER_BASE_URL", "ollama": "OLLAMA_BASE_URL", "vllm": "VLLM_BASE_URL"}
    k = m.get(args.model_name) or m.get(getattr(args, "model_engine", ""), "")
    return os.getenv(k) if k else None


def main():
    parser = argparse.ArgumentParser()
    g = parser.add_argument_group("Model")
    g.add_argument("--mode", choices=["online", "offline"], default="offline")
    g.add_argument("--model_name", default="gemini")
    g.add_argument("--model_engine", default="ollama")
    g.add_argument("--model_version", required=True)
    g = parser.add_argument_group("Infra")
    g.add_argument("--db", choices=["chromadb", "qdrant", "mongodb"], default="chromadb")
    g.add_argument("--embedding_model", default="Alibaba-NLP/gte-multilingual-base")
    g = parser.add_argument_group("Server")
    g.add_argument("--host", default="0.0.0.0")
    g.add_argument("--port", type=int, default=5002)
    args = parser.parse_args()

    import uvicorn
    app = create_app(args)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
