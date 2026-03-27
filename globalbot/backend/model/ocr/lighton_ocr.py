"""LightOnOCR-2-1B parser."""

import re
from pathlib import Path

import cv2
import numpy as np

from .base import PDFParser
from model.utils import download_model


_MAX_NEW_TOKENS_CROP = 128
_MAX_NEW_TOKENS_PAGE = 4096
_REPEAT_NGRAM = 8
_REPEAT_MAX   = 3


def _sanitize(text: str) -> str:
    if not text:
        return text

    pattern = re.compile(
        r"(.{6,120})(\s*\1){" + str(_REPEAT_MAX) + r",}",
        re.DOTALL,
    )
    m = pattern.search(text)
    if m:
        text = text[: m.start() + len(m.group(1))]

    return text.strip()


class LightOnOCRParser(PDFParser):
    """PDF parser using LightOnOCR-2-1B with transformers."""

    def __init__(self):
        super().__init__()
        self.model     = None
        self.processor = None
        self._device   = None
        self._dtype    = None

    def _load_model(self):
        if self.model is not None and self.processor is not None:
            return

        import torch
        from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._dtype  = torch.bfloat16 if self._device == "cuda" else torch.float32

        model_dir = download_model(model_name=self.display_name, repo_id="lightonai/LightOnOCR-2-1B")
        self.model = LightOnOcrForConditionalGeneration.from_pretrained(
            model_dir,
            torch_dtype=self._dtype,
        ).to(self._device)
        self.model.eval()

        self.processor = LightOnOcrProcessor.from_pretrained(model_dir)

    @classmethod
    def display_name(cls) -> str:
        return "LightOnOCR-2-1B"

    def _build_inputs(self, image: "Path | np.ndarray") -> dict:
        from PIL import Image as PILImage

        if isinstance(image, np.ndarray):
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(rgb)
        else:
            pil_image = PILImage.open(image).convert("RGB")

        conversation = [
            {"role": "user", "content": [{"type": "image", "image": pil_image}]}
        ]
        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        return {
            k: v.to(device=self._device, dtype=self._dtype) if v.is_floating_point() else v.to(self._device)
            for k, v in inputs.items()
        }

    def _generate(self, inputs: dict, max_new_tokens: int) -> str:
        """
        Chay model.generate voi cac params chong lap:
          - repetition_penalty: phat nguoi token da xuat hien
          - no_repeat_ngram_size: cam sinh lai n-gram cu
          - early_stopping: dung som khi gap EOS
        """
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.3,
            no_repeat_ngram_size=_REPEAT_NGRAM,
            early_stopping=True,
        )
        input_len  = inputs["input_ids"].shape[1]
        raw        = self.processor.decode(output_ids[0, input_len:], skip_special_tokens=True)
        return _sanitize(raw)

    def parse(self, image: Path, output_path: Path) -> str:
        """Parse anh tu Path, ghi markdown ra output_path."""
        self._load_model()
        inputs   = self._build_inputs(image)
        markdown = self._generate(inputs, _MAX_NEW_TOKENS_PAGE)
        self._write_output(markdown, output_path)
        return markdown

    def parse_image(self, image: np.ndarray) -> str:
        """Nhan numpy array BGR (output cv2), tra ve markdown string."""
        self._load_model()
        inputs = self._build_inputs(image)
        return self._generate(inputs, _MAX_NEW_TOKENS_CROP)

    def parse_batch(self, images: list[np.ndarray]) -> list[str]:
        """Xu ly nhieu crop, tra ve list markdown theo thu tu."""
        return [self.parse_image(img) for img in images]