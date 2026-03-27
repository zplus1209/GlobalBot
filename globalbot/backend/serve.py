from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent

for _p in [str(_THIS_DIR), str(_PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

load_dotenv()
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token


def create_app(args: argparse.Namespace | None = None) -> FastAPI:
    from globalbot.backend.llms.factory import init_singletons
    from globalbot.api.routes.documents import router as doc_router
    from globalbot.api.routes.chat import router as chat_router
    from globalbot.api.routes.pipeline import router as pipeline_router
    from globalbot.api.routes.knowledge import router as knowledge_router

    if args is None:
        args = SimpleNamespace(
            mode=os.getenv("MODE", "offline"),
            model_name=os.getenv("MODEL_NAME", "gemini"),
            model_engine=os.getenv("MODEL_ENGINE", "ollama"),
            model_version=os.getenv("MODEL_VERSION", "qwen3.5:9b"),
            db=os.getenv("DB", "chromadb"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "Alibaba-NLP/gte-multilingual-base"),
        )

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

    allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
    origins = [o.strip() for o in allowed_origins.split(",")]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(doc_router)
    app.include_router(chat_router)
    app.include_router(pipeline_router)
    app.include_router(knowledge_router)

    frontend = _THIS_DIR.parent / "frontend"
    if frontend.exists():
        app.mount("/", StaticFiles(directory=str(frontend), html=True), name="frontend")

    return app


def _api_key(args: argparse.Namespace) -> str | None:
    mapping = {
        "gemini": "GEMINI_API_KEY",
        "openai": "OPENAI_API_KEY",
        "together": "TOGETHER_API_KEY",
    }
    key = mapping.get(args.model_name)
    return os.getenv(key) if key else None


def _base_url(args: argparse.Namespace) -> str | None:
    mapping = {
        "together": "TOGETHER_BASE_URL",
        "ollama": "OLLAMA_BASE_URL",
        "vllm": "VLLM_BASE_URL",
    }
    key = mapping.get(args.model_name) or mapping.get(getattr(args, "model_engine", ""))
    return os.getenv(key) if key else None


def main() -> None:
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
    g.add_argument("--port", type=int, default=3000)

    args = parser.parse_args()

    import uvicorn
    uvicorn.run(
        "globalbot.backend.serve:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=False,
    )


if __name__ == "__main__":
    main()