from __future__ import annotations

from typing import Any, Literal, Optional

_llm: Any = None
_rag: Any = None


def init_singletons(
    mode: Literal["online", "offline"],
    model_name: str,
    model_version: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    engine: Optional[str] = None,
    db_type: str = "chromadb",
    embedding_name: str = "Alibaba-NLP/gte-multilingual-base",
    **rag_kwargs,
) -> None:
    global _llm, _rag

    _llm = build_llm(mode, model_name, model_version, api_key, base_url, engine)

    # BUG FIX: dùng import tương đối để tránh ModuleNotFoundError khi
    # backend dir chưa được thêm vào sys.path trước khi gọi hàm này.
    # serve.py đã đảm bảo backend dir trong sys.path, nên flat-import hoạt động.
    from rag.core import RAG  # flat import — backend dir phải có trong sys.path

    _rag = RAG(llm=_llm, db_type=db_type, embedding_name=embedding_name, **rag_kwargs)


def _llm_instance() -> Any:
    if _llm is None:
        raise RuntimeError(
            "LLM chưa được khởi tạo. Hãy gọi init_singletons() trước."
        )
    return _llm


def _rag_instance() -> Any:
    if _rag is None:
        raise RuntimeError(
            "RAG chưa được khởi tạo. Hãy gọi init_singletons() trước."
        )
    return _rag


def build_llm(
    mode: Literal["online", "offline"],
    model_name: str,
    model_version: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    engine: Optional[str] = None,
    temperature: float = 0.0,
) -> Any:
    if mode == "online":
        return _build_online(model_name, model_version, api_key, base_url, temperature)
    return _build_offline(engine, model_version, base_url, temperature)


def _build_online(
    model_name: str,
    model_version: str,
    api_key: Optional[str],
    base_url: Optional[str],
    temperature: float,
) -> Any:
    if model_name == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model_version,
            google_api_key=api_key,
            temperature=temperature,
        )

    if model_name == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_version,
            api_key=api_key,
            temperature=temperature,
        )

    if model_name == "together":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_version,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
        )

    raise ValueError(f"Unsupported online model: {model_name!r}")


def _build_offline(
    engine: Optional[str],
    model_version: str,
    base_url: Optional[str],
    temperature: float,
) -> Any:
    if engine is None:
        raise ValueError("engine phải được cung cấp cho offline mode")

    if engine == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=model_version,
            base_url=base_url or "http://localhost:11434",
            temperature=temperature,
        )

    if engine == "vllm":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_version,
            base_url=base_url,
            api_key="EMPTY",
            temperature=temperature,
        )

    if engine == "huggingface":
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

        tokenizer = AutoTokenizer.from_pretrained(model_version)
        model = AutoModelForCausalLM.from_pretrained(
            model_version,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=temperature or 0.01,
        )
        hf_llm = HuggingFacePipeline(pipeline=pipe)
        return ChatHuggingFace(llm=hf_llm)

    if engine == "onnx":
        from llms.onnx import ONNXChatWrapper
        return ONNXChatWrapper(model_path=model_version)

    raise ValueError(f"Unsupported offline engine: {engine!r}")