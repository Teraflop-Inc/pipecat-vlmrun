#
# Copyright (c) 2026, VLM Run
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Draw a realtime result on the frame it came from."""

import base64
import io
from typing import Any

from PIL import Image, ImageDraw

_BOX = (0, 230, 118)
_POINT = (255, 82, 82)
_TEXT_BOX = (64, 156, 255)
_MIN_KEYPOINT_SCORE = 0.3
_MASK_ALPHA = 110
_PALETTE = [
    (0, 230, 118),
    (64, 156, 255),
    (255, 171, 0),
    (213, 0, 249),
    (255, 82, 82),
    (0, 229, 255),
    (255, 235, 59),
    (124, 77, 255),
]


def annotate(image: Image.Image, result: dict[str, Any]) -> Image.Image:
    """Draw one frame's result on a copy of the image.

    Handles the payloads the realtime models return: boxes (``img.detect.bboxes``), instance
    masks (``img.segment.masks``), body and hand keypoints (``img.pose.kpts``,
    ``img.pose.hands3d``), text lines (``doc.ocr.lines``) and plain text from the realtime
    chat models. Coordinates in the result are fractions of the frame's width and height.

    Args:
        image: The frame the result was computed on.
        result: ``VLMRunResultFrame.result``.

    Returns:
        A new RGB image with the result drawn on it.
    """
    canvas = image.convert("RGB")
    mask = result.get("mask")
    if isinstance(mask, dict) and mask.get("data"):
        canvas = _draw_mask(canvas, mask)

    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    for item in result.get("items") or []:
        if not isinstance(item, dict):
            continue
        polygon = item.get("poly_xy")
        box = item.get("bbox_xywh")
        if polygon:
            points = [(x * width, y * height) for x, y in polygon]
            draw.polygon(points, outline=_TEXT_BOX)
            _label(draw, points[0], str(item.get("text", "")), _TEXT_BOX)
        elif box:
            x, y, w, h = box[0] * width, box[1] * height, box[2] * width, box[3] * height
            draw.rectangle([x, y, x + w, y + h], outline=_BOX, width=2)
            name = item.get("label") or item.get("side") or ""
            score = item.get("score")
            text = f"{name} {score:.2f}".strip() if isinstance(score, int | float) else str(name)
            _label(draw, (x, y), text, _BOX)
        scores = item.get("kpts_score") or []
        for index, point in enumerate(item.get("kpts_xy") or []):
            if index < len(scores) and scores[index] < _MIN_KEYPOINT_SCORE:
                continue
            px, py = point[0] * width, point[1] * height
            draw.ellipse([px - 3, py - 3, px + 3, py + 3], fill=_POINT)

    text = result.get("text")
    if isinstance(text, str) and text:
        _caption(draw, text, width, height)
    return canvas


def _label(draw: ImageDraw.ImageDraw, corner: tuple[float, float], text: str, color) -> None:
    if text:
        draw.text((corner[0] + 3, max(0, corner[1] - 11)), text, fill=color)


def _caption(draw: ImageDraw.ImageDraw, text: str, width: int, height: int) -> None:
    line = " ".join(text.split())
    limit = max(20, width // 7)
    if len(line) > limit:
        line = line[: limit - 1] + "…"
    draw.rectangle([0, height - 18, width, height], fill=(0, 0, 0))
    draw.text((4, height - 15), line, fill=(255, 255, 255))


def _draw_mask(canvas: Image.Image, mask: dict[str, Any]) -> Image.Image:
    """Tint each instance of a label-map mask. Pixel value 0 is background."""
    data = mask["data"]
    encoded = data.split(",", 1)[1] if data.startswith("data:") else data
    try:
        labels = Image.open(io.BytesIO(base64.b64decode(encoded)))
    except Exception:
        return canvas
    if labels.mode not in ("L", "P", "I", "I;16"):
        labels = labels.convert("L")
    labels = labels.resize(canvas.size, Image.NEAREST)

    colors = labels.getcolors(maxcolors=1 << 16) or []
    used = sorted(value for _, value in colors if value)
    overlay = Image.new("RGB", canvas.size)
    alpha = Image.new("L", canvas.size, 0)
    for value in used:
        region = labels.point(lambda pixel, value=value: 255 if pixel == value else 0).convert("L")
        color = _PALETTE[(value - 1) % len(_PALETTE)]
        overlay.paste(Image.new("RGB", canvas.size, color), mask=region)
        alpha.paste(_MASK_ALPHA, mask=region)
    return Image.composite(overlay, canvas, alpha)
