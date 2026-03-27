import os
from pathlib import Path

from loguru import logger
from modelscope import snapshot_download as ms_snapshot_download
from huggingface_hub import snapshot_download as hf_snapshot_download

WEIGHT_DIR = Path.home() / ".weight"


def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def get_model_source() -> str:
    return os.getenv("MODEL_SOURCE", "huggingface")


class ModelPath:
    vlm_root_hf = ""
    vlm_root_modelscope = ""

    paddle_det = "PP-OCRv5_server_det"
    paddle_textline_ori = "PP-LCNet_x1_0_textline_ori"
    paddle_rec = "PP-OCRv5_server_rec"
    paddle_doc_ori = "PP-LCNet_x1_0_doc_ori"
    paddle_uvdoc = "UVDoc"

    ocr = "lightonai/LightOnOCR-2-1B"


REPO_MAPPING: dict[str, dict[str, str]] = {
    "paddleocr": {
        "huggingface": "PaddlePaddle",
        "modelscope": "PaddlePaddle",
    },
    "vlm": {
        "huggingface": ModelPath.vlm_root_hf,
        "modelscope": ModelPath.vlm_root_modelscope,
        "default": ModelPath.vlm_root_hf,
    },
}


def _get_downloader():
    source = get_model_source()
    if source == "huggingface":
        return source, hf_snapshot_download
    if source == "modelscope":
        return source, ms_snapshot_download
    if source == "local":
        return source, None
    raise ValueError(f"MODEL_SOURCE khong hop le: {source}")


def _resolve_repo_id(repo_mode: str, model_name: str, source: str) -> str:
    """Resolve namespace/model_name tu REPO_MAPPING."""
    mapping = REPO_MAPPING.get(repo_mode)
    if mapping is None:
        raise ValueError(
            f"repo_mode khong hop le: {repo_mode!r}. "
            f"Cac gia tri hop le: {list(REPO_MAPPING)}"
        )
    namespace = mapping.get(source) or mapping.get("default")
    if not namespace:
        raise ValueError(
            f"Khong tim thay namespace cho repo_mode={repo_mode!r}, source={source!r}"
        )
    return f"{namespace}/{model_name}"


def _do_download(downloader, repo_id: str, relative_path: str = "") -> str:
    """Thuc hien download, ho tro selective path voi fallback full repo."""
    try:
        if relative_path:
            relative_path = relative_path.strip("/")
            local_dir = downloader(
                repo_id,
                cache_dir=str(WEIGHT_DIR),
                allow_patterns=[relative_path, relative_path + "/*"],
            )
            return str(Path(local_dir) / relative_path)

        return str(downloader(repo_id, cache_dir=str(WEIGHT_DIR)))

    except Exception as e:
        logger.warning(f"Tai selective that bai, fallback full repo: {e}")
        return str(downloader(repo_id, cache_dir=str(WEIGHT_DIR)))


def download_model(
    model_name: str,
    repo_mode: str | None = None,
    repo_id: str | None = None,
    relative_path: str = "",
) -> str:

    ensure_dir(WEIGHT_DIR)
    source, downloader = _get_downloader()

    # --- local ---
    if source == "local":
        base = repo_id or f"{repo_mode}/{model_name}"
        local_path = WEIGHT_DIR / base
        logger.info(f"Dung model local: {local_path}")
        return str(local_path)

    # --- resolve repo_id ---
    if repo_id is not None:
        resolved = repo_id
    elif repo_mode is not None:
        resolved = _resolve_repo_id(repo_mode, model_name, source)
    else:
        raise ValueError("Phai truyen it nhat mot trong hai: repo_mode hoac repo_id")

    logger.info(f"Dang tai model tu {source}: {resolved}")
    return _do_download(downloader, resolved, relative_path)