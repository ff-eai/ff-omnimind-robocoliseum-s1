# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. See LICENSE-FF.txt.
"""Turn a GenieSim instruction-board sentence into what a grasp program needs.

The instruction-board tasks share one grammar, ``"Left|Right arm picks up
<description> on the table"``, but the description ranges from a plain noun
("the apple") to something that needs perception or reasoning: a colour or
shape ("red block", "rectangular block"), a size superlative ("the largest
block"), a digit ("the block shaped like number 6"), a category ("drinks",
"stationery"), a disjunction ("the fruit or beverage"), or common sense with no
arm at all ("I want to eat some fruit to supplement Vitamin C"). A geometry
program cannot answer those from the sentence alone, so this module asks the
local Qwen3.6 vision-language model (vLLM, OpenAI-compatible, the same server
``providers/sam2_interim/grounding.py`` uses) to look at the head camera and
name ONE concrete, visible object: a short noun phrase for SAM text
segmentation, fallbacks, its own box, which arm the sentence names, and
whether the object must go back where it was.

Everything but the HTTP call is pure Python: :func:`build_prompt` is the
request text, :func:`parse_reply` turns the model's answer into a
:class:`ResolvedInstruction`, and both are testable without a model. The
parser never raises -- a garbage reply becomes ``ok=False`` with an ``error``
-- because the result is handed straight to a generated program.

Measured on the bench (H200, Qwen3.6-35B-A3B-FP8, 2026-10-22): the model
reports boxes in its native 0-1000 normalized frame on both axes even when
asked for pixels, so the prompt asks for that frame and the pixel box is
derived here from the image size.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
import json
import math
import os
import re
import time
from typing import Any

import requests

from cap_harness.providers.sam2_interim.grounding import (
    GroundingError,
    extract_documents,
    image_data_url,
)

DEFAULT_VLM_URL = "http://127.0.0.1:8300/v1"
DEFAULT_VLM_MODEL = "qwen36"
VLM_URL_ENV = "CAP_HARNESS_G2_VLM_URL"
VLM_MODEL_ENV = "CAP_HARNESS_G2_VLM_MODEL"
NORMALIZED_EXTENT = 1000.0
MAX_ALTERNATIVES = 3
MAX_PHRASE_CHARS = 80
MAX_RATIONALE_CHARS = 300
MAX_RAW_CHARS = 2000

SELECTION_RULES = (
    "named",
    "attribute",
    "largest",
    "smallest",
    "category",
    "common_sense",
    "logic_or",
    "number",
    "position",
)
"""How the object was chosen; ``position`` covers the absolute/relative-position boards."""

_ARM_PATTERN = re.compile(r"\b(left|right)\s+(?:arm|hand)\b", re.IGNORECASE)
_PLACE_BACK_PATTERN = re.compile(
    r"straighten|original position|return(?:s|ed)? it|put (?:it|them) back|place (?:it|them) back",
    re.IGNORECASE,
)
_BOX_KEYS = ("box_xyxy_norm", "box_xyxy", "bbox_2d", "bbox", "box")
_NUMBER = r"(-?\d+(?:\.\d+)?)"
# A box counts only with its fourth number closed off (a bracket, a comma or whitespace after it):
# a reply cut inside that number ("..., 81") would otherwise salvage a wrong coordinate.
_SALVAGE_BOX = re.compile(
    r'"(?:box_xyxy_norm|box_xyxy|bbox_2d|bbox|box)"\s*:\s*\[\s*'
    + r"\s*,\s*".join([_NUMBER] * 4)
    + r"(?=\s*[\],\s])"
)


def vlm_url() -> str:
    """The chat endpoint base URL (``CAP_HARNESS_G2_VLM_URL`` or the bench default)."""
    return os.environ.get(VLM_URL_ENV, "").strip() or DEFAULT_VLM_URL


def vlm_model() -> str:
    """The served model name (``CAP_HARNESS_G2_VLM_MODEL`` or ``qwen36``)."""
    return os.environ.get(VLM_MODEL_ENV, "").strip() or DEFAULT_VLM_MODEL


def arm_in_instruction(instruction: str) -> str | None:
    """``"left"``/``"right"`` when the sentence names an arm or hand, else ``None``.

    Read from the text, not the model: the board always spells the arm out when
    it cares, and a guessed arm would send the wrong gripper.
    """
    match = _ARM_PATTERN.search(instruction or "")
    return match.group(1).lower() if match else None


def place_back_in_instruction(instruction: str) -> bool:
    """True when the sentence asks for the object to be straightened or returned."""
    return _PLACE_BACK_PATTERN.search(instruction or "") is not None


@dataclass(frozen=True, slots=True)
class ResolvedInstruction:
    """What a grasp program needs from an instruction, as plain Python values.

    ``query`` is a short noun phrase naming ONE concrete object visible in the
    image, suitable for SAM text segmentation; ``alternatives`` are fallback
    phrases for the same object. ``box_xyxy_norm`` is the model's own box in a
    0-1000 frame on both axes and ``box_xyxy`` the same box in pixels of the
    image the model saw (``image_size`` = ``(width, height)``); both are
    ``None`` when the model could not point at the object. ``visible`` is the
    model's own claim that the object is in view; a ``False`` comes with a low
    ``confidence``. ``ok`` is False only when no usable answer was obtained,
    with the reason in ``error``. ``salvaged`` marks an answer read from a reply
    that was cut off before its JSON closed (see :func:`salvage_document`).
    """

    ok: bool
    instruction: str
    arm: str | None = None
    query: str = ""
    alternatives: tuple[str, ...] = ()
    box_xyxy_norm: tuple[float, float, float, float] | None = None
    box_xyxy: tuple[int, int, int, int] | None = None
    image_size: tuple[int, int] = (0, 0)
    selection_rule: str = "named"
    place_back: bool = False
    confidence: float = 0.0
    visible: bool = False
    rationale: str = ""
    error: str | None = None
    latency_s: float | None = None
    model: str = ""
    attempts: int = 1
    raw: str = ""
    salvaged: bool = False

    def to_dict(self) -> dict[str, Any]:
        """A JSON-like ``dict`` (lists, not tuples) for the generated-program namespace."""
        data = asdict(self)
        data["alternatives"] = list(self.alternatives)
        data["box_xyxy_norm"] = None if self.box_xyxy_norm is None else list(self.box_xyxy_norm)
        data["box_xyxy"] = None if self.box_xyxy is None else list(self.box_xyxy)
        data["image_size"] = list(self.image_size)
        return data


def build_prompt(instruction: str) -> str:
    """The text sent with the head image: the sentence plus a strict JSON schema.

    Deliberately short. The box is asked for in the model's native 0-1000 frame
    and the pixel size is not mentioned; a longer prompt that explained every
    field and gave the pixel size sent Qwen3.6's thinking into coordinate
    arithmetic that looped until ``max_tokens`` (measured 2026-10-22: 4096
    reasoning tokens and no answer on a plain "the apple"; this wording answers
    the same frame in 1300-2900 reasoning tokens, 5-11 s).
    """
    rules = ", ".join(f'"{rule}"' for rule in SELECTION_RULES)
    return (
        f'Tabletop robot, head camera. Instruction: "{instruction}"\n'
        "Which ONE object on the table does it refer to? Answer with JSON only:\n"
        '{"arm": "left"|"right"|null, "query": str, "alternatives": [str], '
        '"box_xyxy_norm": [x_min, y_min, x_max, y_max]|null, '
        f'"selection_rule": {rules}, '
        '"place_back": bool, "visible": bool, "confidence": 0-1, "rationale": str}\n'
        "arm: the arm the instruction names, null if none. "
        "query: 2-6 words naming the concrete object as it looks here (not the category or "
        "the rule; e.g. 'the big blue block', 'apple'). "
        f"alternatives: up to {MAX_ALTERNATIVES} other phrases for the same object. "
        "box_xyxy_norm: x and y in 0-1000 across the image. "
        "selection_rule: how the object was chosen (named noun, attribute such as colour or "
        "shape, largest/smallest, number, category, logic_or for 'A or B', common_sense for a "
        "need or wish, position). "
        "place_back: true only if it must be returned to its original position or straightened. "
        "If no object matches: visible false, box null, confidence < 0.3; never invent one. "
        "Do not reason about coordinates; estimate the box in one go and keep thinking under "
        "100 words."
    )


def _as_text(value: Any, limit: int) -> str:
    if value is None:
        return ""
    text = value if isinstance(value, str) else str(value)
    return " ".join(text.split())[:limit]


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0", ""}:
            return False
    return default


def _as_unit(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return min(1.0, max(0.0, number))


def _as_arm(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    lowered = value.strip().lower()
    if lowered.startswith("left"):
        return "left"
    if lowered.startswith("right"):
        return "right"
    return None


def _as_rule(value: Any) -> str:
    if not isinstance(value, str):
        return "named"
    lowered = value.strip().lower().replace("-", "_").replace(" ", "_")
    if lowered in SELECTION_RULES:
        return lowered
    aliases = {
        "color": "attribute",
        "colour": "attribute",
        "shape": "attribute",
        "size": "largest",
        "biggest": "largest",
        "or": "logic_or",
        "logic": "logic_or",
        "commonsense": "common_sense",
        "reasoning": "common_sense",
        "digit": "number",
        "numeral": "number",
        "type": "category",
        "class": "category",
        "name": "named",
        "specific": "named",
        "relative_position": "position",
        "absolute_position": "position",
    }
    return aliases.get(lowered, "attribute")


def _as_phrases(value: Any, *, exclude: str) -> tuple[str, ...]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, Sequence):
        return ()
    phrases: list[str] = []
    for item in value:
        text = _as_text(item, MAX_PHRASE_CHARS)
        if text and text.lower() != exclude.lower() and text not in phrases:
            phrases.append(text)
        if len(phrases) == MAX_ALTERNATIVES:
            break
    return tuple(phrases)


def _box_values(document: Mapping[str, Any]) -> Any:
    for key in _BOX_KEYS:
        if key in document:
            return document[key]
    return None


def _as_box(
    value: Any, width: int, height: int
) -> tuple[tuple[float, float, float, float], tuple[int, int, int, int]] | None:
    """A ``(normalized, pixel)`` box pair, or ``None`` when the value is not a usable box.

    The model answers in its 0-1000 frame; a box whose coordinates exceed that
    frame but fit the pixel image is taken as pixels and normalized. A box with
    no area (after clipping) is rejected rather than handed to a segmenter.
    """
    if isinstance(value, Mapping):
        value = [value.get(key) for key in ("x_min", "y_min", "x_max", "y_max")]
    if isinstance(value, str | bytes) or not isinstance(value, Sequence) or len(value) != 4:
        return None
    try:
        numbers = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(number) for number in numbers):
        return None
    x_min, y_min, x_max, y_max = numbers
    if x_min > x_max:
        x_min, x_max = x_max, x_min
    if y_min > y_max:
        y_min, y_max = y_max, y_min
    in_pixels = (
        max(x_max, y_max) > NORMALIZED_EXTENT
        and x_max <= float(width) * 1.05
        and y_max <= float(height) * 1.05
    )
    if in_pixels:
        scale_x, scale_y = NORMALIZED_EXTENT / float(width), NORMALIZED_EXTENT / float(height)
        x_min, x_max = x_min * scale_x, x_max * scale_x
        y_min, y_max = y_min * scale_y, y_max * scale_y
    normalized = tuple(
        min(NORMALIZED_EXTENT, max(0.0, number)) for number in (x_min, y_min, x_max, y_max)
    )
    if normalized[2] - normalized[0] <= 0.0 or normalized[3] - normalized[1] <= 0.0:
        return None
    pixel = (
        int(math.floor(normalized[0] / NORMALIZED_EXTENT * width)),
        int(math.floor(normalized[1] / NORMALIZED_EXTENT * height)),
        int(math.ceil(normalized[2] / NORMALIZED_EXTENT * width)),
        int(math.ceil(normalized[3] / NORMALIZED_EXTENT * height)),
    )
    pixel = (
        min(max(pixel[0], 0), width - 1),
        min(max(pixel[1], 0), height - 1),
        min(max(pixel[2], 1), width),
        min(max(pixel[3], 1), height),
    )
    if pixel[2] <= pixel[0] or pixel[3] <= pixel[1]:
        return None
    return tuple(round(number, 1) for number in normalized), pixel  # type: ignore[return-value]


def _pick_document(documents: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """The reply object that looks most like the schema (the model sometimes nests it)."""
    best = documents[0]
    best_score = -1
    for document in documents:
        candidates = [document]
        for key in ("result", "answer", "object", "target"):
            nested = document.get(key)
            if isinstance(nested, Mapping):
                candidates.append(nested)
        for candidate in candidates:
            score = sum(
                key in candidate for key in ("query", "arm", "selection_rule", "confidence")
            )
            if score > best_score:
                best, best_score = candidate, score
    return best


def _salvaged_string(text: str, key: str) -> str | None:
    match = re.search(r'"' + key + r'"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    if match is None:
        return None
    try:
        return json.loads('"' + match.group(1) + '"')
    except ValueError:
        return match.group(1)


def salvage_document(text: Any) -> dict[str, Any] | None:
    """The complete fields of a reply that stops before its JSON closes, or ``None``.

    With thinking on, Qwen3.6 sometimes spends nearly all of ``max_tokens`` on
    reasoning and the answer is cut mid-object (finish_reason "length"); the
    content is then not empty, so the empty-content retry never ran. Measured
    over every logged instruction-board call (2026-10-28): 71 of ~3700 replies
    were cut this way, 63 of them after a complete ``query`` and 47 after a
    complete box. Only fields whose value is closed are read: a string up to its
    closing quote, a box whose four numbers are all followed by a delimiter, a
    list up to its closing bracket. Without a ``query`` there is nothing to use.
    """
    if not isinstance(text, str):
        return None
    query = _salvaged_string(text, "query")
    if not query:
        return None
    document: dict[str, Any] = {"query": query}
    for key in ("arm", "selection_rule", "rationale"):
        value = _salvaged_string(text, key)
        if value is not None:
            document[key] = value
    box = _SALVAGE_BOX.search(text)
    if box is not None:
        document["box_xyxy_norm"] = [float(value) for value in box.groups()]
    alternatives = re.search(r'"alternatives"\s*:\s*(\[[^\[\]]*\])', text)
    if alternatives is not None:
        try:
            document["alternatives"] = json.loads(alternatives.group(1))
        except ValueError:
            pass
    for key in ("visible", "place_back"):
        flag = re.search(r'"' + key + r'"\s*:\s*(true|false)\b', text)
        if flag is not None:
            document[key] = flag.group(1) == "true"
    confidence = re.search(r'"confidence"\s*:\s*' + _NUMBER + r"(?=\s*[,}\s])", text)
    if confidence is not None:
        document["confidence"] = float(confidence.group(1))
    return document


def failure(
    instruction: str,
    error: str,
    *,
    width: int = 0,
    height: int = 0,
    raw: str = "",
    latency_s: float | None = None,
    model: str = "",
    attempts: int = 1,
) -> ResolvedInstruction:
    """An ``ok=False`` result that still carries what the sentence alone says."""
    return ResolvedInstruction(
        ok=False,
        instruction=instruction,
        arm=arm_in_instruction(instruction),
        place_back=place_back_in_instruction(instruction),
        image_size=(int(width), int(height)),
        error=error,
        raw=raw[:MAX_RAW_CHARS],
        latency_s=latency_s,
        model=model,
        attempts=attempts,
    )


def parse_reply(
    text: str,
    instruction: str,
    width: int,
    height: int,
    *,
    latency_s: float | None = None,
    model: str = "",
    attempts: int = 1,
) -> ResolvedInstruction:
    """Parse the model's reply into a :class:`ResolvedInstruction`; never raises.

    The arm and the place-back flag come from the sentence when it states them
    and from the model only when it does not; the box is validated and clipped;
    a missing ``query`` is a failure because nothing downstream can use it.
    """
    width, height = int(width), int(height)
    raw = text if isinstance(text, str) else ""
    salvaged = False
    try:
        documents = extract_documents(text)
    except GroundingError as exc:
        partial = salvage_document(text)
        if partial is None:
            return failure(
                instruction,
                f"unparseable reply: {exc}",
                width=width,
                height=height,
                raw=raw,
                latency_s=latency_s,
                model=model,
                attempts=attempts,
            )
        documents, salvaged = [partial], True
    document = _pick_document(documents)
    query = _as_text(document.get("query") or document.get("label"), MAX_PHRASE_CHARS)
    visible = _as_bool(document.get("visible"), default=True)
    boxes = _as_box(_box_values(document), width, height) if width > 0 and height > 0 else None
    confidence = _as_unit(document.get("confidence"))
    if not query:
        return failure(
            instruction,
            "reply names no query",
            width=width,
            height=height,
            raw=raw,
            latency_s=latency_s,
            model=model,
            attempts=attempts,
        )
    if boxes is None:
        # No usable box means the model did not actually point at anything,
        # whatever it claimed; the result stays usable (the query can still be
        # segmented) but must not read as a confident sighting.
        visible = False
        confidence = min(confidence, 0.3)
    sentence_arm = arm_in_instruction(instruction)
    return ResolvedInstruction(
        ok=True,
        instruction=instruction,
        arm=sentence_arm if sentence_arm is not None else _as_arm(document.get("arm")),
        query=query,
        alternatives=_as_phrases(document.get("alternatives"), exclude=query),
        box_xyxy_norm=None if boxes is None else boxes[0],
        box_xyxy=None if boxes is None else boxes[1],
        image_size=(width, height),
        selection_rule=_as_rule(document.get("selection_rule")),
        place_back=place_back_in_instruction(instruction)
        or _as_bool(document.get("place_back"), default=False),
        confidence=confidence,
        visible=visible,
        rationale=_as_text(document.get("rationale"), MAX_RATIONALE_CHARS),
        latency_s=latency_s,
        model=model,
        attempts=attempts,
        raw=raw[:MAX_RAW_CHARS],
        salvaged=salvaged,
    )


class InstructionResolver:
    """The HTTP side: one chat-completions call per instruction, parsed defensively."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        model: str | None = None,
        timeout_s: float = 90.0,
        enable_thinking: bool = True,
        max_tokens: int = 4096,
        session: requests.Session | None = None,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self.base_url = (base_url or vlm_url()).rstrip("/")
        self.model = model or vlm_model()
        self.timeout_s = float(timeout_s)
        self.enable_thinking = bool(enable_thinking)
        self.max_tokens = int(max_tokens)
        self.session = session if session is not None else requests.Session()

    def health(self) -> bool:
        """True when ``/models`` answers; never raises."""
        try:
            response = self.session.request(
                "GET", f"{self.base_url}/models", timeout=min(self.timeout_s, 10.0)
            )
        except requests.RequestException:
            return False
        return 200 <= response.status_code < 300

    def payload(
        self, jpeg: bytes, instruction: str, *, enable_thinking: bool | None = None
    ) -> dict[str, Any]:
        """The chat-completions request body (exposed so tests can pin it)."""
        thinking = self.enable_thinking if enable_thinking is None else bool(enable_thinking)
        return {
            "model": self.model,
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_data_url(jpeg)}},
                        {"type": "text", "text": build_prompt(instruction)},
                    ],
                }
            ],
            # vLLM forwards this to the Qwen chat template; thinking is what lets
            # the model work out "something with vitamin C" -> the apple.
            "chat_template_kwargs": {"enable_thinking": thinking},
        }

    def _complete(
        self, jpeg: bytes, instruction: str, *, enable_thinking: bool
    ) -> tuple[str | None, str | None, float]:
        """One chat call: ``(content, error, latency_s)``; ``content`` is None when truncated."""
        started = time.perf_counter()
        try:
            response = self.session.request(
                "POST",
                f"{self.base_url}/chat/completions",
                json=self.payload(jpeg, instruction, enable_thinking=enable_thinking),
                timeout=self.timeout_s,
            )
        except requests.RequestException as exc:
            return None, f"VLM endpoint unreachable: {exc}", time.perf_counter() - started
        latency_s = time.perf_counter() - started
        if not 200 <= response.status_code < 300:
            return (
                None,
                f"VLM endpoint returned HTTP {response.status_code}: {response.text[:300]}",
                latency_s,
            )
        try:
            choice = response.json()["choices"][0]
            content = choice["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            return None, f"VLM reply has no message content: {exc}", latency_s
        if content is None or (isinstance(content, str) and not content.strip()):
            # vLLM returns null content when the reasoning alone used up
            # ``max_tokens`` (finish_reason "length"): the model never answered.
            return None, None, latency_s
        return content if isinstance(content, str) else str(content), None, latency_s

    def resolve(
        self, jpeg: bytes, instruction: str, width: int, height: int
    ) -> ResolvedInstruction:
        """Ask the model about ``instruction`` in the JPEG image; never raises.

        With thinking on, a reply whose reasoning exhausted ``max_tokens`` has
        no content; that call is retried once with thinking off (fast, and
        enough for the plain boards) before giving up. A reply cut off after the
        answer began is salvaged when it already names the object
        (:func:`salvage_document`); one cut before that is retried the same way.
        """
        instruction = " ".join(str(instruction or "").split())
        if not instruction:
            return failure("", "instruction is empty", width=width, height=height, model=self.model)
        if not jpeg:
            return failure(
                instruction, "image is empty", width=width, height=height, model=self.model
            )
        content, error, latency_s = self._complete(
            jpeg, instruction, enable_thinking=self.enable_thinking
        )
        attempts = 1
        if content is None and error is None and self.enable_thinking:
            content, error, retry_s = self._complete(jpeg, instruction, enable_thinking=False)
            latency_s += retry_s
            attempts = 2
        if error is not None:
            return failure(
                instruction,
                error,
                width=width,
                height=height,
                latency_s=latency_s,
                model=self.model,
                attempts=attempts,
            )
        if content is None:
            return failure(
                instruction,
                "VLM reply has no content (reasoning exhausted max_tokens)",
                width=width,
                height=height,
                latency_s=latency_s,
                model=self.model,
                attempts=attempts,
            )
        result = parse_reply(
            content,
            instruction,
            width,
            height,
            latency_s=latency_s,
            model=self.model,
            attempts=attempts,
        )
        if (
            result.ok
            or attempts > 1
            or not self.enable_thinking
            or not str(result.error).startswith("unparseable")
        ):
            return result
        content, error, retry_s = self._complete(jpeg, instruction, enable_thinking=False)
        latency_s += retry_s
        if error is None and content is not None:
            retried = parse_reply(
                content,
                instruction,
                width,
                height,
                latency_s=latency_s,
                model=self.model,
                attempts=2,
            )
            if retried.ok:
                return retried
        return replace(result, latency_s=latency_s, attempts=2)

    def resolve_rgb(self, rgb: Any, instruction: str) -> ResolvedInstruction:
        """:meth:`resolve` for an ``(H, W, 3)`` RGB array (uint8 or float in [0, 1])."""
        import numpy as np

        from cap_harness.g2.codec import encode_jpeg

        try:
            image = np.asarray(rgb)
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError("rgb must have shape (H, W, 3)")
            if np.issubdtype(image.dtype, np.floating):
                image = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
            height, width = int(image.shape[0]), int(image.shape[1])
            jpeg = encode_jpeg(image, quality=90)
        except (ValueError, TypeError) as exc:
            return failure(str(instruction or ""), f"bad image: {exc}", model=self.model)
        return self.resolve(jpeg, instruction, width, height)


__all__ = [
    "DEFAULT_VLM_MODEL",
    "DEFAULT_VLM_URL",
    "MAX_ALTERNATIVES",
    "NORMALIZED_EXTENT",
    "SELECTION_RULES",
    "VLM_MODEL_ENV",
    "VLM_URL_ENV",
    "InstructionResolver",
    "ResolvedInstruction",
    "arm_in_instruction",
    "build_prompt",
    "failure",
    "parse_reply",
    "place_back_in_instruction",
    "salvage_document",
    "vlm_model",
    "vlm_url",
]
