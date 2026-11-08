"""Request and response formats for hosted video and frame-list adapters."""

from __future__ import annotations

import base64
from fractions import Fraction
import hashlib
from io import BytesIO
import os
import re
from urllib.parse import urlsplit

from ..catalog import require

GEMINI = "https://generativelanguage.googleapis.com/v1beta"
QWEN = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
ASTRA = "https://api.openai.com/v1"
QWEN_ENDPOINT = re.compile(
    r"https://(?:(?:cn-hongkong\.)?dashscope(?:-intl|-us)?\.aliyuncs\.com|"
    r"[a-z0-9-]+\.[a-z0-9-]+\.maas\.aliyuncs\.com)/compatible-mode/v1"
)
HOSTED = {
    "gemini": ("GEMINI_API_KEY", 24, {"video/mp4"}),
    "qwen": ("DASHSCOPE_API_KEY", 10, {"video/mp4", "video/x-matroska"}),
    "qwen-frames": ("DASHSCOPE_API_KEY", 10, {"video/mp4"}),
    "astra": ("OPENAI_API_KEY", None, {"video/mp4"}),
}
FRAME_LISTS = {"qwen-frames", "astra"}
FULL_FRAME_FIELDS = {
    "media_io_kwargs": {"video": {"num_frames": -1, "fps": -1}},
    "mm_processor_kwargs": {"do_sample_frames": False},
}
SOURCE_FRAMES, SOURCE_FPS = 180, 60
SOURCE_WIDTH, SOURCE_HEIGHT = 768, 512
PNG_COMPRESSION = {"qwen-frames": 6, "astra": 1}


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
    if provider == "astra":
        return ASTRA
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
    if provider == "astra":
        require(model.startswith("gpt-6-astra"), "Use a GPT-6 Astra model ID")
    if provider == "qwen-frames":
        require(
            bool(re.fullmatch(r"qwen3\.[5-8]-(?:max|plus|flash|[0-9][A-Za-z0-9.-]*b)(?:-[0-9-]+)?", model)),
            "Use an exact supported Qwen3.5–3.8 vision model ID, e.g. qwen3.8-max-0902",
        )
    base = config["endpoint"]
    fixed_endpoint = {"gemini": GEMINI, "astra": ASTRA}.get(provider)
    require(
        base == fixed_endpoint if fixed_endpoint else bool(QWEN_ENDPOINT.fullmatch(base)),
        "Hosted API endpoint differs",
    )
    require(
        config["video_fps"] == HOSTED[provider][1],
        "Hosted video sampling setting differs",
    )


def frame_settings(provider):
    if provider in FRAME_LISTS:
        return {
            "delivery": "ordered_lossless_png",
            "frames": SOURCE_FRAMES,
            "source_fps": SOURCE_FPS,
            "requested_fps": HOSTED[provider][1],
            "png_compression": PNG_COMPRESSION[provider],
        }
    return FULL_FRAME_FIELDS


def ordered_pngs(payload, expected_rgb_sha=None, *, compression=6):
    """Decode each original frame once and preserve RGB values in the PNG list."""
    import av
    from PIL import Image

    images = []
    digest = hashlib.sha256()
    with av.open(BytesIO(payload)) as container:
        require(len(container.streams.video) == 1, "Expected exactly one MP4 video stream")
        (stream,) = container.streams.video
        stream.codec_context.thread_count = 1
        require(stream.average_rate == SOURCE_FPS, "Expected a 60 fps video")
        for i, frame in enumerate(container.decode(stream)):
            require(
                (frame.width, frame.height) == (SOURCE_WIDTH, SOURCE_HEIGHT)
                and i < SOURCE_FRAMES,
                "Unexpected frame size or count",
            )
            require(
                frame.pts is not None and frame.time_base is not None
                and frame.pts * frame.time_base == Fraction(i, SOURCE_FPS),
                "Expected timestamps i/60 for all 180 frames in order",
            )
            rgb = frame.to_ndarray(format="rgb24")
            digest.update(rgb)
            png = BytesIO()
            Image.fromarray(rgb).save(png, format="PNG", compress_level=compression)
            images.append("data:image/png;base64," + base64.b64encode(png.getvalue()).decode("ascii"))
    require(len(images) == SOURCE_FRAMES, "Expected all 180 frames")
    rgb_sha = digest.hexdigest()
    require(
        expected_rgb_sha is None or rgb_sha == expected_rgb_sha,
        "Frame-list RGB differs from the verified Protocol X reader output",
    )
    return images, rgb_sha


def frame_list_request(provider, model, prompt, parts, max_tokens, generation):
    if provider == "astra":
        content = []
        for part in parts:
            urls = part["video"] if part["type"] == "video" else [part["image_url"]["url"]]
            content.extend(
                {"type": "input_image", "image_url": url, "detail": "original"}
                for url in urls
            )
        return {
            "model": model,
            "input": [{"role": "user", "content": content + [{"type": "input_text", "text": prompt}]}],
            "max_output_tokens": max_tokens,
            "reasoning": generation["reasoning"],
            "tools": [],
            "tool_choice": "none",
            "store": False,
            "stream": False,
            "truncation": "disabled",
        }
    return {
        "model": model,
        "stream": False,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": parts + [{"type": "text", "text": prompt}]}],
        **generation,
    }


def astra_completion(document):
    output = document["output"]
    parts = [p for item in output if item["type"] == "message" for p in item["content"]]
    content = "".join(p["text"] for p in parts if p["type"] == "output_text")
    status = document["status"]
    incomplete = (document.get("incomplete_details") or {}).get("reason")
    stop = "stop" if status == "completed" else incomplete or status
    if status == "incomplete" and incomplete == "max_output_tokens":
        stop = "length"
    if status == "completed" and any(p["type"] != "output_text" for p in parts):
        stop = "refusal_or_nontext"
    return {
        "model": document["model"],
        "id": document.get("id"),
        "usage": document.get("usage"),
        "choices": [{
            "finish_reason": stop,
            "message": {
                "content": content,
                "tool_calls": any(item["type"] not in {"message", "reasoning"} for item in output),
            },
        }],
    }


def validate_payload_size(provider, base, size):
    if provider == "qwen-frames":
        hostname = urlsplit(base or "").hostname or ""
        workspace = hostname.endswith(".maas.aliyuncs.com")
        limit = 64_000_000 if workspace else 16 * 2**20
        hint = "" if workspace else "; set DASHSCOPE_BASE_URL to the official workspace endpoint"
    elif provider == "astra":
        limit, hint = 512_000_000, ""
    elif provider == "gemini":
        limit, hint = 20_000_000, ""
    else:
        return
    require(size <= limit, f"{provider} request is {size} bytes, above the {limit}-byte limit{hint}")


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
