# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. See LICENSE-FF.txt.
"""Text-to-box grounding through an OpenAI-compatible vision-language endpoint.

The interim segmentation service has no text head of its own (SAM 2.1 takes
boxes and points only), so a text prompt is first turned into boxes by a VLM:
Qwen3.6 served by vLLM at ``http://127.0.0.1:8300/v1`` as model ``qwen36``.
The model is asked for absolute-pixel ``[x_min, y_min, x_max, y_max]`` boxes
in JSON and told the image size; its answer is parsed, rescaled if it reports a
different frame, validated and clipped here. Everything but the HTTP call is
pure Python so the parsing is testable without a model.

Measured on the bench (H200, Qwen3.6-35B-A3B-FP8): the model answers in the
original pixel frame when told the size, echoing ``image_size`` back exactly,
in 0.7-2 s with thinking enabled.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from typing import Any

import requests

DEFAULT_GROUNDING_URL = "http://127.0.0.1:8300/v1"
DEFAULT_GROUNDING_MODEL = "qwen36"
#: Boxes kept per query before SAM; the service keeps the five best masks.
MAX_BOXES = 8
#: Smallest box side (px) worth sending to SAM after clipping.
MIN_BOX_SIDE_PX = 1.0


class GroundingError(RuntimeError):
    """The grounding endpoint failed or answered with something unusable."""


@dataclass(frozen=True)
class GroundingBox:
    """One detected instance in absolute pixels of the queried image."""

    label: str
    box_xyxy: tuple[float, float, float, float]
    confidence: float


NORMALIZED_FRAME = 1000.0
"""Qwen3-VL family models ground in a 0-1000 frame on both axes (measured on Qwen3.6
2026-10-22: a 640x400 head frame got boxes with y up to 718 while echoing image_size
[640, 400]), so the prompt asks for that frame explicitly and the parser treats a
reply that overflows the image while claiming the pixel frame as 0-1000 as well."""


def build_prompt(query: str, width: int, height: int) -> str:
    """The instruction sent with the image; asks for the model's native 0-1000 frame."""
    return (
        f'Find every instance of: "{query}" in the image. '
        "Reply with JSON only, no prose: "
        '{"image_size": [1000, 1000], "objects": [{"label": str, '
        '"box_xyxy": [x_min, y_min, x_max, y_max], "confidence": number}]}. '
        "box_xyxy uses a normalized frame in which the image is 1000 units wide and "
        "1000 units tall (0 <= x <= 1000, 0 <= y <= 1000), regardless of the image's "
        f"pixel size ({int(width)}x{int(height)}); confidence is in [0, 1]. "
        'If there is no such object, "objects" is [].'
    )


def image_data_url(jpeg: bytes) -> str:
    """``data:image/jpeg;base64,...`` for an OpenAI-style ``image_url`` content part."""
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")


def extract_documents(text: str) -> list[Mapping[str, Any]]:
    """Every JSON object in a model reply, in order, tolerating fences, prose and repeats.

    Observed from Qwen3.6 with thinking off: the object alone; the object inside
    a one-element array; several objects back to back or comma-separated (which
    ``json.loads`` rejects as "extra data"); ``bbox_2d`` for ``box_xyxy``. The
    reply is decoded value by value from the first ``{`` or ``[``, arrays are
    flattened, and anything that is not an object is skipped.
    """
    if not isinstance(text, str):
        raise GroundingError("grounding reply is not text")
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else ""
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3]
    starts = [index for index in (stripped.find("{"), stripped.find("[")) if index >= 0]
    if not starts:
        raise GroundingError(f"grounding reply contains no JSON object: {text[:200]!r}")
    decoder = json.JSONDecoder()
    position = min(starts)
    values: list[Any] = []
    error: json.JSONDecodeError | None = None
    while position < len(stripped):
        while position < len(stripped) and stripped[position] in " \t\r\n,]":
            position += 1
        if position >= len(stripped) or stripped[position] not in "{[":
            break
        try:
            value, position = decoder.raw_decode(stripped, position)
        except json.JSONDecodeError as exc:
            if stripped[position] == "[":
                # An array that is not valid as a whole (trailing comma, cut off):
                # step inside it and decode its elements one by one.
                position += 1
                continue
            error = exc
            break
        values.append(value)
    documents: list[Mapping[str, Any]] = []
    for value in values:
        items = value if isinstance(value, list) else [value]
        documents.extend(item for item in items if isinstance(item, Mapping))
    if not documents:
        if error is not None:
            raise GroundingError(f"grounding reply is not valid JSON: {error}")
        raise GroundingError(f"grounding reply holds no JSON object: {text[:200]!r}")
    return documents


def extract_json(text: str) -> Mapping[str, Any]:
    """The first JSON object in a model reply (see :func:`extract_documents`)."""
    return extract_documents(text)[0]


def _finite(values: Any, count: int) -> list[float] | None:
    if isinstance(values, str | bytes) or not isinstance(values, Sequence) or len(values) != count:
        return None
    out: list[float] = []
    for value in values:
        if isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        out.append(number)
    return out


def _reported_scale(
    document: Mapping[str, Any],
    width: int,
    height: int,
    raw_boxes: Sequence[Sequence[float]] = (),
) -> tuple[float, float]:
    """Pixels per reported unit.

    The model is asked for the 0-1000 normalized frame; ``image_size`` names the frame
    it actually used. No ``image_size`` means 0-1000. If it echoes the pixel frame but
    any box overflows the image by more than a pixel, the boxes are 0-1000 anyway
    (observed Qwen3.6 behaviour), so the scale falls back to the normalized frame.
    """
    reported = _finite(document.get("image_size", ()), 2)
    if reported is None or reported[0] <= 0 or reported[1] <= 0:
        reported = [NORMALIZED_FRAME, NORMALIZED_FRAME]
    if abs(reported[0] - width) < 0.5 and abs(reported[1] - height) < 0.5:
        limit_x, limit_y = float(width) + 1.0, float(height) + 1.0
        for raw in raw_boxes:
            if raw[0] > limit_x or raw[2] > limit_x or raw[1] > limit_y or raw[3] > limit_y:
                reported = [NORMALIZED_FRAME, NORMALIZED_FRAME]
                break
    return float(width) / reported[0], float(height) / reported[1]


def parse_boxes(
    text: str,
    width: int,
    height: int,
    *,
    query: str,
    max_boxes: int = MAX_BOXES,
) -> list[GroundingBox]:
    """Boxes from a model reply: parsed, rescaled, clipped to the image and ranked.

    Entries that are not four finite numbers, that are empty after clipping, or
    whose confidence is not a number are dropped rather than failing the whole
    query; a reply with no usable box is an empty list, which the service turns
    into the SAM3 "no segmentation" result.
    """
    documents = extract_documents(text)
    objects: list[Any] = []
    for document in documents:
        listed = document.get("objects", [])
        if listed is None:
            continue
        if isinstance(listed, str | bytes | Mapping) or not isinstance(listed, Sequence):
            raise GroundingError("grounding reply 'objects' must be a list")
        objects.extend(listed)
    sized = next((d for d in documents if "image_size" in d), documents[0])
    raw_boxes = [
        raw
        for item in objects
        if isinstance(item, Mapping)
        for raw in (_finite(item.get("box_xyxy", item.get("bbox_2d", ())), 4),)
        if raw is not None
    ]
    sx, sy = _reported_scale(sized, width, height, raw_boxes)

    boxes: list[GroundingBox] = []
    for item in objects:
        if not isinstance(item, Mapping):
            continue
        raw = _finite(item.get("box_xyxy", item.get("bbox_2d", ())), 4)
        if raw is None:
            continue
        x0, y0, x1, y1 = raw[0] * sx, raw[1] * sy, raw[2] * sx, raw[3] * sy
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        x0, x1 = max(0.0, x0), min(float(width), x1)
        y0, y1 = max(0.0, y0), min(float(height), y1)
        if x1 - x0 < MIN_BOX_SIDE_PX or y1 - y0 < MIN_BOX_SIDE_PX:
            continue
        confidence = item.get("confidence", 1.0)
        if isinstance(confidence, bool) or not isinstance(confidence, int | float):
            continue
        if not math.isfinite(float(confidence)):
            continue
        label = item.get("label", query)
        if not isinstance(label, str) or not label.strip():
            label = query
        boxes.append(
            GroundingBox(
                label=label.strip(),
                box_xyxy=(x0, y0, x1, y1),
                confidence=min(1.0, max(0.0, float(confidence))),
            )
        )
    boxes.sort(key=lambda box: box.confidence, reverse=True)
    return boxes[:max_boxes]


class QwenGrounder:
    """Text grounding through vLLM's OpenAI-compatible chat completions API."""

    def __init__(
        self,
        base_url: str = DEFAULT_GROUNDING_URL,
        *,
        model: str = DEFAULT_GROUNDING_MODEL,
        timeout_s: float = 60.0,
        enable_thinking: bool = True,
        max_tokens: int = 1024,
        session: requests.Session | None = None,
    ) -> None:
        if not base_url.strip():
            raise ValueError("base_url must be non-empty")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = float(timeout_s)
        self.enable_thinking = bool(enable_thinking)
        self.max_tokens = int(max_tokens)
        self.session = session if session is not None else requests.Session()

    def health(self) -> bool:
        try:
            response = self.session.request(
                "GET", f"{self.base_url}/models", timeout=min(self.timeout_s, 10.0)
            )
        except requests.RequestException:
            return False
        return 200 <= response.status_code < 300

    def payload(self, jpeg: bytes, query: str, width: int, height: int) -> dict[str, Any]:
        """The chat-completions request body (exposed so tests can pin it)."""
        return {
            "model": self.model,
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_data_url(jpeg)}},
                        {"type": "text", "text": build_prompt(query, width, height)},
                    ],
                }
            ],
            # vLLM forwards this to the Qwen chat template. Measured on the three
            # bench frames (2026-10-22): thinking on answered 1.0-2.0 s with
            # well-formed JSON every time; off answered 0.3-1.0 s but sometimes
            # looped into truncated JSON (5 s, unparseable) and missed objects.
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }

    def ground(self, jpeg: bytes, query: str, width: int, height: int) -> list[GroundingBox]:
        """Boxes for ``query`` in the JPEG image of the given size (empty when none)."""
        try:
            response = self.session.request(
                "POST",
                f"{self.base_url}/chat/completions",
                json=self.payload(jpeg, query, width, height),
                timeout=self.timeout_s,
            )
        except requests.RequestException as exc:
            raise GroundingError(f"grounding endpoint unreachable: {exc}") from exc
        if not 200 <= response.status_code < 300:
            raise GroundingError(
                f"grounding endpoint returned HTTP {response.status_code}: {response.text[:300]}"
            )
        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise GroundingError(f"grounding reply has no message content: {exc}") from exc
        return parse_boxes(content, width, height, query=query)


__all__ = [
    "DEFAULT_GROUNDING_MODEL",
    "DEFAULT_GROUNDING_URL",
    "MAX_BOXES",
    "GroundingBox",
    "GroundingError",
    "QwenGrounder",
    "build_prompt",
    "extract_documents",
    "extract_json",
    "image_data_url",
    "parse_boxes",
]
