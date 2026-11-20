import base64
import io

from PIL import Image

from pipecat_vlmrun import annotate

SIZE = (160, 120)


def blank():
    return Image.new("RGB", SIZE, (20, 20, 20))


def changed(result) -> bool:
    drawn = annotate(blank(), result)
    assert drawn.size == SIZE and drawn.mode == "RGB"
    return drawn.tobytes() != blank().tobytes()


def test_boxes_are_drawn():
    result = {
        "object": "img.detect.bboxes",
        "items": [{"bbox_xywh": [0.1, 0.1, 0.5, 0.5], "label": "cat", "score": 0.8}],
    }
    drawn = annotate(blank(), result)
    assert drawn.getpixel((16, 40)) != (20, 20, 20), "left edge of the box at x = 0.1 * width"


def test_keypoints_below_the_score_floor_are_left_out():
    item = {
        "bbox_xywh": [0, 0, 0, 0],
        "kpts_xy": [[0.5, 0.5], [0.9, 0.9]],
        "kpts_score": [0.9, 0.1],
    }
    drawn = annotate(blank(), {"object": "img.pose.kpts", "items": [item]})
    assert drawn.getpixel((80, 60)) != (20, 20, 20)
    assert drawn.getpixel((144, 108)) == (20, 20, 20)


def test_hand_keypoints_without_scores_are_drawn():
    item = {
        "bbox_xywh": [0.4, 0.4, 0.2, 0.2],
        "kpts_xy": [[0.25, 0.25]],
        "side": "left",
        "score": 0.9,
    }
    drawn = annotate(blank(), {"object": "img.pose.hands3d", "items": [item]})
    assert drawn.getpixel((40, 30)) != (20, 20, 20)


def test_text_lines_are_outlined():
    item = {
        "bbox_xywh": [0.2, 0.2, 0.4, 0.2],
        "text": "BAKERY",
        "score": 0.99,
        "poly_xy": [[0.2, 0.2], [0.6, 0.2], [0.6, 0.4], [0.2, 0.4]],
    }
    assert changed({"object": "doc.ocr.lines", "items": [item]})


def test_mask_instances_are_tinted():
    labels = Image.new("L", (16, 12), 0)
    labels.paste(1, (0, 0, 8, 12))
    out = io.BytesIO()
    labels.save(out, format="PNG")
    data = "data:image/png;base64," + base64.b64encode(out.getvalue()).decode()
    result = {
        "object": "img.segment.masks",
        "items": [],
        "mask": {"format": "png", "height": 12, "width": 16, "data": data},
    }
    drawn = annotate(blank(), result)
    assert drawn.getpixel((10, 60)) != (20, 20, 20), "instance 1 covers the left half"
    assert drawn.getpixel((150, 60)) == (20, 20, 20), "background is untouched"


def test_a_broken_mask_is_ignored():
    result = {"items": [], "mask": {"data": "data:image/png;base64,not-a-png"}}
    assert not changed(result)


def test_text_results_get_a_caption():
    assert changed({"text": "A person at a desk."})


def test_an_empty_result_changes_nothing():
    assert not changed({"object": "img.detect.bboxes", "items": []})


def test_the_input_image_is_not_modified():
    image = blank()
    annotate(image, {"items": [{"bbox_xywh": [0.1, 0.1, 0.5, 0.5], "label": "cat", "score": 0.8}]})
    assert image.tobytes() == blank().tobytes()
