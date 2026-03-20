from __future__ import annotations

import sys
import time
import logging
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

_CONFIGURED = False
_ROOT = "backend"

def setup_logging(level: int = logging.DEBUG, fmt: Optional[str] = None, stream=None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    root = logging.getLogger(_ROOT)
    root.setLevel(level)
    if not root.handlers:
        handler = logging.StreamHandler(stream or sys.stderr)
        handler.setLevel(level)
        handler.setFormatter(
            logging.Formatter(
                fmt or "%(asctime)s  %(levelname)-7s  %(name)-40s  %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        )
        root.addHandler(handler)
        root.propagate = False
    _CONFIGURED = True

def get_logger(name: str) -> logging.Logger:
    setup_logging()
    if not name.startswith(_ROOT):
        name = f"{_ROOT}.{name}"
    return logging.getLogger(name)


class RagLogger:
    def __init__(self, name: str, component: Optional[str] = None):
        self._logger = get_logger(name)
        self._component = component or ""

    def debug(self, event: str, **fields: Any) -> None:
        self._emit(logging.DEBUG, event, **fields)

    def info(self, event: str, **fields: Any) -> None:
        self._emit(logging.INFO, event, **fields)

    def warning(self, event: str, **fields: Any) -> None:
        self._emit(logging.WARNING, event, **fields)

    def error(self, event: str, **fields: Any) -> None:
        self._emit(logging.ERROR, event, **fields)

    def exception(self, event: str, **fields: Any) -> None:
        self._emit(logging.ERROR, event, exc_info=True, **fields)

    def _emit(self, level: int, event: str, exc_info: bool = False, **fields: Any) -> None:
        if not self._logger.isEnabledFor(level):
            return
        prefix = f"[{self._component}] " if self._component else ""
        kv = "  ".join(f"{k}={_fmt(v)}" for k, v in fields.items() if v is not None)
        msg = f"{prefix}{event}"
        if kv:
            msg = f"{msg}  {kv}"
        self._logger.log(level, msg, exc_info=exc_info)

    @contextmanager
    def timed(self, start: str, end: str, **fields: Any) -> Iterator[Dict[str, Any]]:
        self.info(start, **fields)
        ctx: Dict[str, Any] = {}
        t0 = time.perf_counter()
        try:
            yield ctx
        except Exception as exc:
            ms = int((time.perf_counter() - t0) * 1000)
            self.error(
                f"{end}.error",
                duration_ms=ms,
                error=f"{type(exc).__name__}: {str(exc)[:120]}",
                **fields,
            )
            raise
        else:
            ms = int((time.perf_counter() - t0) * 1000)
            self.info(end, duration_ms=ms, **{**fields, **ctx})


class LoggingMixin:
    @property
    def log(self) -> RagLogger:
        attr = "_rag_logger"
        inst = self.__dict__.get(attr)
        if inst is None:
            inst = RagLogger(type(self).__module__, component=type(self).__name__)
            object.__setattr__(self, attr, inst)
        return inst

    def timed(self, start: str, end: str, **kw: Any):
        return self.log.timed(start, end, **kw)


@contextmanager
def timer() -> Iterator[Dict[str, float]]:
    ctx: Dict[str, float] = {}
    t0 = time.perf_counter()
    try:
        yield ctx
    finally:
        ctx["ms"] = (time.perf_counter() - t0) * 1000
        ctx["s"] = ctx["ms"] / 1000

def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    if isinstance(v, str) and len(v) > 80:
        return f"{v[:77]}..."
    if isinstance(v, (list, tuple)):
        return f"[{len(v)} items]"
    return str(v)
