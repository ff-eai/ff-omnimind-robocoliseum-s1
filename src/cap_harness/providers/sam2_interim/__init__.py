# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. See LICENSE-FF.txt.
"""Interim segmentation provider: SAM 2.1 masks behind Qwen text grounding.

A stand-in for the SAM3 service (``cap_harness.providers.sam3.service``) while
the SAM 3.1 weights are gated. It speaks exactly the SAM3 wire format
(``sam3/wire.py`` routes, ``sam3/service.py`` response models), so the
unchanged ``cap_harness.providers.sam3.client.Sam3Provider`` works against it
on the same port (8114) and nothing downstream knows the difference.

What differs is how a text prompt becomes a mask. SAM3 grounds text itself;
SAM 2.1 only takes box and point prompts, so text goes through a vision-language
model first (Qwen3.6 on a vLLM OpenAI-compatible endpoint, ``grounding.py``)
which returns absolute-pixel boxes, and SAM 2.1 (``transformers``'s ``Sam2Model``)
turns each box into a mask. Point prompts go to SAM 2.1 directly, as in SAM3.

Retire this package once ``facebook/sam3.1`` is available: start
``cap_harness.providers.sam3.service`` on the same port instead. Nothing
imports from here except the service's own entry point and its tests.
"""

from .grounding import (
    DEFAULT_GROUNDING_MODEL,
    DEFAULT_GROUNDING_URL,
    GroundingBox,
    GroundingError,
    QwenGrounder,
    build_prompt,
    parse_boxes,
)

__all__ = [
    "DEFAULT_GROUNDING_MODEL",
    "DEFAULT_GROUNDING_URL",
    "GroundingBox",
    "GroundingError",
    "QwenGrounder",
    "build_prompt",
    "parse_boxes",
]
