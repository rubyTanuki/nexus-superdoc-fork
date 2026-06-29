"""Local ONNX MiniLM embedding processor.

Drop-in replacement for ``OpenAIProcessor`` (``src/services/openai_client.py``):
exposes the same ``embed_documents`` / ``embed_query`` surface, but runs
``sentence-transformers/all-MiniLM-L6-v2`` locally via onnxruntime on CPU. No API
calls, no per-token cost. Emits 384-dim, L2-normalized, mean-pooled vectors.

The model weights (~86 MB) are downloaded from the Hugging Face Hub on first use
and cached on disk. Set ``SUPERDOC_MODEL_DIR`` to point the cache at a mounted/baked
path (e.g. a persistent Docker volume or a Lambda layer) so ephemeral containers do
not re-download the model on every run.
"""
from __future__ import annotations

import os
import shutil
import urllib.request
from pathlib import Path

import numpy as np

_HF_BASE = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main"
_ASSETS = {
    "model.onnx": f"{_HF_BASE}/onnx/model.onnx",
    "tokenizer.json": f"{_HF_BASE}/tokenizer.json",
}
# Minimum acceptable sizes catch truncated downloads before onnxruntime tries to parse them.
_ASSET_MIN_SIZES = {
    "model.onnx": 50 * 1024 * 1024,  # fp32 model is ~86 MB
    "tokenizer.json": 100 * 1024,
}
_DOWNLOAD_HEADERS = {"User-Agent": "superdoc/1.0"}

# all-MiniLM-L6-v2 was trained at a max sequence length of 256 tokens; longer inputs
# are truncated so document paragraphs never overflow the model's position embeddings.
_MAX_SEQ_LEN = 256


def _default_cache_dir() -> Path:
    override = os.getenv("SUPERDOC_MODEL_DIR")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "superdoc" / "models" / "all-MiniLM-L6-v2"


class OnnxProcessor:
    """Local MiniLM embedder with the same interface as ``OpenAIProcessor``."""

    DIMENSIONS = 384

    def __init__(self):
        self.model_dir = _default_cache_dir()
        self.onnx_path = str(self.model_dir / "model.onnx")
        self.tokenizer_path = str(self.model_dir / "tokenizer.json")

        self._ensure_assets_present()

        # Imported lazily so the module can be imported (and the download triggered)
        # without paying onnxruntime's import cost until an embedder is constructed.
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(self.tokenizer_path)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        self.tokenizer.enable_truncation(max_length=_MAX_SEQ_LEN)
        self.session = ort.InferenceSession(
            self.onnx_path, providers=["CPUExecutionProvider"]
        )

    @property
    def dimensions(self) -> int:
        return self.DIMENSIONS

    # --- Asset management ------------------------------------------------

    def _asset_is_valid(self, filename: str) -> bool:
        dest = self.model_dir / filename
        min_size = _ASSET_MIN_SIZES.get(filename, 0)
        return dest.exists() and dest.stat().st_size >= min_size

    def _ensure_assets_present(self) -> None:
        if all(self._asset_is_valid(f) for f in _ASSETS):
            return

        self.model_dir.mkdir(parents=True, exist_ok=True)
        print("First-time setup: downloading embedding model from Hugging Face Hub (~86 MB)...")

        for filename, url in _ASSETS.items():
            if self._asset_is_valid(filename):
                continue

            dest = self.model_dir / filename
            dest.unlink(missing_ok=True)  # remove any partial file from a prior interrupted download
            tmp = dest.with_suffix(".tmp")
            print(f"Downloading {filename} ...")
            try:
                req = urllib.request.Request(url, headers=_DOWNLOAD_HEADERS)
                with urllib.request.urlopen(req, timeout=60) as resp:
                    with open(tmp, "wb") as f:
                        shutil.copyfileobj(resp, f)
                tmp.rename(dest)
            except Exception as e:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Failed to download {filename} from Hugging Face Hub.\n"
                    f"URL: {url}\nError: {e}"
                ) from e

            actual = dest.stat().st_size
            min_size = _ASSET_MIN_SIZES.get(filename, 0)
            if actual < min_size:
                dest.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Download of {filename} appears incomplete ({actual / 1024 / 1024:.1f} MB). "
                    f"Expected at least {min_size // 1024 // 1024} MB. Please try again."
                )

        print(f"Embedding model cached to {self.model_dir}")

    # --- Inference -------------------------------------------------------

    def _execute_onnx(self, texts: list[str]) -> list[list[float]]:
        """Tokenize, run the forward pass, mean-pool, and L2-normalize."""
        if not texts:
            return []

        # Fast Rust tokenization.
        encoded = self.tokenizer.encode_batch(texts)

        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        # MiniLM expects a token-type index layer (all zeros for single-sentence input).
        token_type_ids = np.zeros_like(input_ids, dtype=np.int64)

        ort_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
        ort_outputs = self.session.run(None, ort_inputs)
        token_embeddings = ort_outputs[0]  # index 0 = last hidden state (token embeddings)

        # Mean pooling over real (non-padding) tokens.
        input_mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(float)
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        pooled = sum_embeddings / sum_mask

        # Normalize to unit length so dot product == cosine similarity downstream.
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        normalized = pooled / np.clip(norms, a_min=1e-9, a_max=None)

        return normalized.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        cleaned = [t.replace("\n", " ") for t in texts]
        return self._execute_onnx(cleaned)

    def embed_query(self, query: str) -> list[float]:
        return self._execute_onnx([query.replace("\n", " ")])[0]
