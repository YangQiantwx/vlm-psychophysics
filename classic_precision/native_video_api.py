"""Request and response formats for the hosted native-video adapters."""

from __future__ import annotations

import os
import re
from urllib.parse import urlsplit

from .jinwei_release import require

GEMINI = "https://generativelanguage.googleapis.com/v1beta"
QWEN = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
QWEN_ENDPOINT = re.compile(
    r"https://(?:dashscope(?:-intl|-us)?\.aliyuncs\.com|"
    r"[a-z0-9-]+\.[a-z0-9-]+\.maas\.aliyuncs\.com)/compatible-mode/v1"
)
HOSTED = {
    "gemini": ("GEMINI_API_KEY", 24, {"video/mp4"}),
    "qwen": ("DASHSCOPE_API_KEY", 10, {"video/mp4", "video/x-matroska"}),
}
FULL_FRAME_FIELDS = {
    "media_io_kwargs": {"video": {"num_frames": -1, "fps": -1}},
    "mm_processor_kwargs": {"do_sample_frames": False},
}


def vllm_endpoint(base):
    url = urlsplit(base)
    local = url.hostname in ("localhost", "127.0.0.1", "::1")
    require(
        (url.scheme == "https" or (local and url.scheme == "http"))
        and url.hostname
        and url.path == "/v1"
        and not any((url.username, url.password, url.query, url.fragment)),
        "Set VLLM_BASE_URL to the HTTPS endpoint ending in /v1, or use a localhost tunnel",
    )
    return base


def endpoint(provider):
    if provider == "gemini":
        return GEMINI
    base = os.environ.get("DASHSCOPE_BASE_URL", QWEN)
    require(
        bool(QWEN_ENDPOINT.fullmatch(base)),
        "Use an official DashScope compatible-mode HTTPS endpoint",
    )
    return base


def validate_config(config):
    provider = config["provider"]
    model = config["requested_model"]
    if provider == "gemini":
        require(
            bool(re.fullmatch(r"gemini-[A-Za-z0-9_.-]+", model)),
            "Use an exact Gemini model ID",
        )
    base = config["endpoint"]
    require(
        base == GEMINI if provider == "gemini" else bool(QWEN_ENDPOINT.fullmatch(base)),
        "Hosted API endpoint differs",
    )
    require(
        config["video_fps"] == HOSTED[provider][1],
        "Hosted video sampling setting differs",
    )


def request(provider, model, prompt, media, max_tokens, fps):
    ((mime, data),) = media
    require(mime in HOSTED[provider][2], "Unsupported hosted video MIME type")
    if provider == "gemini":
        video = {
            "inlineData": {"mimeType": mime, "data": data},
            "videoMetadata": {"fps": fps},
        }
        return {
            "contents": [{"role": "user", "parts": [video, {"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": max_tokens},
        }
    url = f"data:{mime};base64,{data}"
    require(len(url) < 10_000_000, "DashScope inline video exceeds 10 MB encoded")
    video = {"type": "video_url", "video_url": {"url": url}, "fps": fps}
    return {
        "model": model,
        "stream": False,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "user", "content": [video, {"type": "text", "text": prompt}]}
        ],
    }


def gemini_completion(document):
    """Project the native envelope onto the fields used by the common scorer."""
    (candidate,) = document["candidates"]
    parts = candidate["content"]["parts"]
    text = "".join(part.get("text", "") for part in parts if not part.get("thought"))
    reason = candidate.get("finishReason")
    finish = {"STOP": "stop", "MAX_TOKENS": "length"}.get(reason, reason)
    return {
        "model": document["modelVersion"],
        "id": document.get("responseId"),
        "usage": document.get("usageMetadata"),
        "choices": [
            {
                "finish_reason": finish,
                "message": {
                    "content": text,
                    "tool_calls": any("functionCall" in part for part in parts),
                },
            }
        ],
    }
