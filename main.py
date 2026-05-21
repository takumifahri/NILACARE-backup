from __future__ import annotations

import os
import logging
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Annotated, Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

try:
    from app.labels import LABELS, LABEL_FULL_NAMES
except ModuleNotFoundError:
    try:
        from backend.labels import LABELS, LABEL_FULL_NAMES
    except ModuleNotFoundError:
        from labels import LABELS, LABEL_FULL_NAMES

ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
INPUT_SIZE = (224, 224)
logger = logging.getLogger("uvicorn.error")
VERBOSE_ERRORS = os.getenv("VERBOSE_ERRORS", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

# ── Model Path Resolution ──────────────────────────────────────────────────
# Priority:
#   1. Same directory as main.py   (VPS: /app/MobileNetV2_best.h5)
#   2. <project_root>/models/      (local dev: nilaai/models/MobileNetV2_best.h5)
MODEL_FILENAME = "MobileNetV2_best.h5"
_here = Path(__file__).resolve().parent          # directory of main.py
_candidates = [
    _here / MODEL_FILENAME,                      # e.g. /app/MobileNetV2_best.h5
    _here / "models" / MODEL_FILENAME,           # e.g. /app/models/MobileNetV2_best.h5
    _here.parent / "models" / MODEL_FILENAME,    # e.g. nilaai/models/MobileNetV2_best.h5
]
MODEL_PATH = next((p for p in _candidates if p.exists()), None)

if MODEL_PATH is None:
    _checked = "\n  ".join(str(p) for p in _candidates)
    raise RuntimeError(
        f"Model '{MODEL_FILENAME}' tidak ditemukan. Lokasi yang dicek:\n  {_checked}"
    )

try:
    MODEL = load_model(str(MODEL_PATH), compile=False)
except Exception as exc:
    # ── Keras version compatibility shim ────────────────────────────────────
    # Models saved with Keras 3.x include 'quantization_config' in Dense/Conv
    # configs. Older TF 2.x doesn't know that kwarg and raises:
    #   "Unrecognized keyword arguments passed to Dense: {'quantization_config'}"
    # Fix: patch the built-in layers to silently pop unknown kwargs, then retry.
    _err_msg = str(exc)
    if "quantization_config" in _err_msg or "Unrecognized keyword arguments" in _err_msg:
        import tensorflow as tf

        class _CompatDense(tf.keras.layers.Dense):
            def __init__(self, *args, **kwargs):
                kwargs.pop("quantization_config", None)
                super().__init__(*args, **kwargs)

        class _CompatConv2D(tf.keras.layers.Conv2D):
            def __init__(self, *args, **kwargs):
                kwargs.pop("quantization_config", None)
                super().__init__(*args, **kwargs)

        class _CompatDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
            def __init__(self, *args, **kwargs):
                kwargs.pop("quantization_config", None)
                super().__init__(*args, **kwargs)

        _custom_objects = {
            "Dense": _CompatDense,
            "Conv2D": _CompatConv2D,
            "DepthwiseConv2D": _CompatDepthwiseConv2D,
        }
        try:
            MODEL = load_model(
                str(MODEL_PATH),
                compile=False,
                custom_objects=_custom_objects,
            )
        except Exception as exc2:
            raise RuntimeError(
                f"Gagal load model dari {MODEL_PATH} (dengan compatibility shim): {exc2}"
            ) from exc2
    else:
        raise RuntimeError(f"Gagal load model dari {MODEL_PATH}: {exc}") from exc

# ── Startup validation: model output neurons must equal label count ──
_model_output_size = MODEL.output_shape[-1]
if _model_output_size != len(LABELS):
    raise RuntimeError(
        f"Mismatch: model output size={_model_output_size} "
        f"tapi len(LABELS)={len(LABELS)}. "
        "Pastikan LABELS sesuai dengan class_indices saat training."
    )


DEFAULT_ALLOWED_ORIGINS = [
    "https://nilacareai.takumifahri.my.id",
    "https://nila-care-ai.vercel.app",
    "https://nila-care-neifj1eur-akhfabgss-projects.vercel.app",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]


def _get_allowed_origins() -> list[str]:
    raw_origins = os.getenv("CORS_ORIGINS", "").strip()
    if not raw_origins:
        return DEFAULT_ALLOWED_ORIGINS

    return [origin.strip() for origin in raw_origins.split(",") if origin.strip()]


app = FastAPI(title="NilaCare Inference API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    start = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.exception(
            "request_id=%s method=%s path=%s failed elapsed_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            elapsed_ms,
        )
        raise

    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request_id=%s method=%s path=%s status=%s elapsed_ms=%.2f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


@app.exception_handler(HTTPException)
async def structured_http_exception_handler(
    request: Request,
    exc: HTTPException,
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex[:12])
    error_code = "HTTP_ERROR"
    detail = exc.detail

    if isinstance(exc.detail, dict):
        detail = exc.detail.get("message", exc.detail.get("detail", "Request gagal."))
        error_code = str(exc.detail.get("error_code", error_code))
        request_id = str(exc.detail.get("request_id", request_id))

    return JSONResponse(
        status_code=exc.status_code,
        headers={"X-Request-ID": request_id, **(exc.headers or {})},
        content={
            "detail": detail,
            "error_code": error_code,
            "request_id": request_id,
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex[:12])
    logger.exception(
        "request_id=%s unhandled backend error on %s %s",
        request_id,
        request.method,
        request.url.path,
    )
    detail = "Terjadi error internal di backend inference."
    if VERBOSE_ERRORS:
        detail = f"{detail} ({type(exc).__name__}: {exc})"

    return JSONResponse(
        status_code=500,
        headers={"X-Request-ID": request_id},
        content={
            "detail": detail,
            "error_code": "INTERNAL_SERVER_ERROR",
            "request_id": request_id,
        },
    )


def _looks_like_probabilities(scores: np.ndarray) -> bool:
    if scores.ndim != 1:
        return False

    if np.any(scores < 0.0) or np.any(scores > 1.0):
        return False

    return abs(float(np.sum(scores)) - 1.0) < 1e-3


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp_scores = np.exp(shifted)
    denom = np.sum(exp_scores)
    if denom == 0:
        return exp_scores
    return exp_scores / denom


def _preprocess_image(image_bytes: bytes) -> np.ndarray:
    with Image.open(BytesIO(image_bytes)) as image:
        rgb_image = image.convert("RGB")
        resized = rgb_image.resize(INPUT_SIZE)
        image_array = np.asarray(resized, dtype=np.float32)

    normalized = preprocess_input(image_array)
    batched = np.expand_dims(normalized, axis=0)
    return batched


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/predict",
    responses={
        400: {"description": "Bad Request"},
        500: {"description": "Internal Server Error"},
    },
)
async def predict(
    request: Request,
    file: Annotated[UploadFile, File(...)],
) -> dict[str, Any]:
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex[:12])
    filename = file.filename or "(tanpa nama)"
    logger.info(
        "request_id=%s predict_start filename=%r content_type=%r",
        request_id,
        filename,
        file.content_type,
    )

    if file.content_type not in ALLOWED_MIME_TYPES:
        logger.warning(
            "request_id=%s unsupported_mime filename=%r content_type=%r allowed=%s",
            request_id,
            filename,
            file.content_type,
            sorted(ALLOWED_MIME_TYPES),
        )
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Format file tidak didukung. Gunakan JPEG, PNG, atau WEBP.",
                "error_code": "UNSUPPORTED_IMAGE_TYPE",
                "request_id": request_id,
            },
        )

    try:
        image_bytes = await file.read()
        logger.info(
            "request_id=%s upload_read filename=%r size_bytes=%s",
            request_id,
            filename,
            len(image_bytes),
        )
        input_tensor = _preprocess_image(image_bytes)
        logger.info(
            "request_id=%s preprocess_ok tensor_shape=%s tensor_dtype=%s",
            request_id,
            tuple(input_tensor.shape),
            input_tensor.dtype,
        )
    except Exception:
        logger.exception(
            "request_id=%s preprocess_failed filename=%r content_type=%r",
            request_id,
            filename,
            file.content_type,
        )
        raise HTTPException(
            status_code=400,
            detail={
                "message": "File gambar tidak valid.",
                "error_code": "INVALID_IMAGE_FILE",
                "request_id": request_id,
            },
        )

    try:
        inference_start = time.perf_counter()
        raw_output = MODEL.predict(input_tensor, verbose=0)
        inference_ms = (time.perf_counter() - inference_start) * 1000
        scores = np.asarray(raw_output, dtype=np.float32)
        logger.info(
            "request_id=%s inference_ok raw_shape=%s scores_shape=%s inference_ms=%.2f",
            request_id,
            np.shape(raw_output),
            tuple(scores.shape),
            inference_ms,
        )

        if scores.ndim == 2 and scores.shape[0] == 1:
            scores = scores[0]

        if scores.ndim != 1:
            raise ValueError(f"Bentuk output model tidak didukung: shape={scores.shape}")

        # Runtime guard: model output neurons must equal label count
        if scores.shape[0] != len(LABELS):
            logger.error(
                "request_id=%s output_label_mismatch output_size=%s label_count=%s",
                request_id,
                scores.shape[0],
                len(LABELS),
            )
            raise HTTPException(
                status_code=500,
                detail={
                    "message": (
                        f"Mismatch: model output size={scores.shape[0]} "
                        f"!= len(LABELS)={len(LABELS)}."
                    ),
                    "error_code": "MODEL_LABEL_MISMATCH",
                    "request_id": request_id,
                },
            )

        probabilities = scores if _looks_like_probabilities(scores) else _softmax(scores)

        top_index = int(np.argmax(probabilities))
        confidence = float(probabilities[top_index])
        top_candidates = [
            {
                "label": LABELS[int(index)],
                "confidence": round(float(probabilities[int(index)]), 6),
            }
            for index in np.argsort(probabilities)[::-1][: min(3, len(LABELS))]
        ]

        CONFIDENCE_THRESHOLD = 0.45
        if confidence < CONFIDENCE_THRESHOLD:
            label = "tidak dikenali"
        else:
            label = LABELS[top_index]  # always one of the 5 valid classes

        logger.info(
            "request_id=%s predict_success label=%r confidence=%.6f top_index=%s top_candidates=%s",
            request_id,
            label,
            confidence,
            top_index,
            top_candidates,
        )

        return {
            "label": label,
            "confidence": confidence,
            "request_id": request_id,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(
            "request_id=%s inference_failed filename=%r",
            request_id,
            filename,
        )
        detail = "Gagal melakukan inferensi model."
        if VERBOSE_ERRORS:
            detail = f"{detail} ({type(exc).__name__}: {exc})"

        raise HTTPException(
            status_code=500,
            detail={
                "message": detail,
                "error_code": "MODEL_INFERENCE_FAILED",
                "request_id": request_id,
            },
        )
