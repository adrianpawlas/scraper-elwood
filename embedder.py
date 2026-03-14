"""
embedder.py — SigLIP image and text embeddings.

Model: google/siglip-base-patch16-384
Output: 768-dimensional L2-normalised float vectors.
"""

import logging
import time
from io import BytesIO

import requests
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from config import (
    MODEL_NAME,
    REQUEST_HEADERS,
    REQUEST_TIMEOUT,
    SIGLIP_MAX_TEXT_LENGTH,
)

logger = logging.getLogger(__name__)


class SigLIPEmbedder:
    """Wraps the SigLIP vision-language model for image and text encoding."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading '{model_name}' on {self.device} …")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        logger.info("Model ready.")

    # ── Image embedding ───────────────────────────────────────────────────────

    def embed_image(self, image_url: str, retries: int = 3) -> list[float] | None:
        """
        Download the image at `image_url` and return a normalised 768-dim vector.
        Returns None on failure after `retries` attempts.
        """
        for attempt in range(1, retries + 1):
            try:
                resp = requests.get(
                    image_url,
                    headers=REQUEST_HEADERS,
                    timeout=REQUEST_TIMEOUT,
                )
                resp.raise_for_status()

                image = Image.open(BytesIO(resp.content)).convert("RGB")

                inputs = self.processor(images=image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model.get_image_features(**inputs)
                    if hasattr(outputs, 'pooler_output'):
                        features = outputs.pooler_output
                    else:
                        features = outputs

                features = features / features.norm(dim=-1, keepdim=True)
                return features[0].cpu().float().tolist()

            except Exception as exc:
                logger.warning(
                    f"[Image embed attempt {attempt}/{retries}] "
                    f"Failed for {image_url}: {exc}"
                )
                if attempt < retries:
                    time.sleep(2 ** (attempt - 1))  # exponential back-off

        logger.error(f"Giving up on image embedding for: {image_url}")
        return None

    # ── Text embedding ────────────────────────────────────────────────────────

    def embed_text(self, text: str, retries: int = 2) -> list[float] | None:
        """
        Encode `text` and return a normalised 768-dim vector.
        SigLIP's text encoder is capped at 64 tokens; longer text is truncated.
        """
        for attempt in range(1, retries + 1):
            try:
                inputs = self.processor(
                    text=[text],
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=SIGLIP_MAX_TEXT_LENGTH,
                    return_attention_mask=True,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model.get_text_features(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                    )
                    if hasattr(outputs, 'pooler_output'):
                        features = outputs.pooler_output
                    else:
                        features = outputs

                features = features / features.norm(dim=-1, keepdim=True)
                return features[0].cpu().float().tolist()

            except Exception as exc:
                logger.warning(
                    f"[Text embed attempt {attempt}/{retries}] Failed: {exc}"
                )
                if attempt < retries:
                    time.sleep(1)

        logger.error("Giving up on text embedding.")
        return None
