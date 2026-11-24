"""Evaluate rotated-box detections with exact polygon IoU and COCO AP/AR.

Ground truth (--gt) is a COCO-like JSON object with "images", "categories",
and "annotations" lists. Images need an integer "id", categories need an
integer "id" and a unique "name", and annotations need "image_id",
"category_id", and "quad". Predictions (--pred) are a JSON list whose items
need "image_id", "category_id", "quad", and a finite "score". Both files use
the same pixel coordinate system. A "quad" is four (x, y) vertices, supplied
either as four pairs or as eight values. Each polygon is its convex hull,
as in the study evaluator.

Supply predictions after the detector's own score filtering and NMS. The
evaluator makes no threshold or NMS choices. It retains at most 300
detections per image across all categories before COCO category-wise
evaluation. Equal scores preserve input order within an image; COCO's
stable accumulation orders tied detections across images by image ID.
This matches evaluation of study exports already capped at 300 per image.

All metrics are fractions in [0, 1]. Categories with no non-crowd ground
truth have null scores and do not contribute to summary means.

Example:
    python polygon_eval.py --gt val.groundtruth.json --pred val.predictions.json
    python polygon_eval.py --self-test
"""

import argparse
from collections import defaultdict
from contextlib import redirect_stdout
from io import StringIO
import json
import math
from pathlib import Path

import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


MAX_DETECTIONS = 300
IOU_THRESHOLDS = np.linspace(0.50, 0.95, 10)
RECALL_THRESHOLDS = np.linspace(0.0, 1.0, 101)


def _shape(record, label):
    quad = np.asarray(record["quad"], dtype=np.float32)
    if quad.shape not in ((4, 2), (8,)) or not np.isfinite(quad).all():
        raise ValueError(f"{label}: quad must contain four finite (x, y) vertices")
    hull = cv2.convexHull(quad.reshape(4, 2)).reshape(-1, 2)
    area = cv2.contourArea(hull)
    if len(hull) < 3 or not math.isfinite(area) or area <= 0:
        raise ValueError(f"{label}: quad must have a positive finite area")
    return hull, area, hull.min(axis=0), hull.max(axis=0)


def _index(records, label):
    index = {}
    for position, record in enumerate(records):
        value = record["id"]
        if type(value) is not int or value in index:
            raise ValueError(f"{label}[{position}]: id must be a unique integer")
        index[value] = record
    return index


def _geometry_rows(records, images, categories, *, predictions):
    rows = []
    groups = defaultdict(list)
    kind = "prediction" if predictions else "annotation"
    for index, record in enumerate(records):
        label = f"{kind}[{index}]"
        image_id = record["image_id"]
        category_id = record["category_id"]
        if type(image_id) is not int or image_id not in images:
            raise ValueError(f"{label}: unknown image_id {image_id!r}")
        if type(category_id) is not int or category_id not in categories:
            raise ValueError(f"{label}: unknown category_id {category_id!r}")
        hull, area, lower, upper = _shape(record, label)
        row = {
            "id": index + 1,
            "image_id": image_id,
            "category_id": category_id,
            "area": area,
            "iscrowd": 0,
            "_hull": hull,
            "_lower": lower,
            "_upper": upper,
        }
        if predictions:
            score = float(record["score"])
            if not math.isfinite(score):
                raise ValueError(f"{label}: score must be finite")
            row["score"] = score
            groups[image_id].append(row)
        else:
            row["iscrowd"] = int(record.get("iscrowd", 0))
            rows.append(row)

    for detections in groups.values():
        detections.sort(key=lambda row: -row["score"])
        rows.extend(detections[:MAX_DETECTIONS])
    return rows


class _PolygonCOCOeval(COCOeval):
    """Replace COCO's bounding-box IoU with convex polygon IoU."""

    def computeIoU(self, image_id, category_id):
        ground_truth = self._gts[image_id, category_id]
        detections = self._dts[image_id, category_id]
        detections = sorted(detections, key=lambda row: -row["score"])
        detections = detections[:self.params.maxDets[-1]]
        ious = np.zeros((len(detections), len(ground_truth)), dtype=np.float64)
        if not detections or not ground_truth:
            return ious

        lowers = np.stack([row["_lower"] for row in detections])
        uppers = np.stack([row["_upper"] for row in detections])
        for j, target in enumerate(ground_truth):
            candidates = np.flatnonzero(
                np.all(uppers >= target["_lower"], axis=1)
                & np.all(lowers <= target["_upper"], axis=1)
            )
            for i in candidates:
                detection = detections[i]
                area, _ = cv2.intersectConvexConvex(
                    detection["_hull"], target["_hull"]
                )
                # COCO uses intersection over detection area for crowd regions.
                union = detection["area"]
                if not target["iscrowd"]:
                    union += target["area"] - area
                if union > 0:
                    ious[i, j] = min(1.0, max(0.0, area / union))
        return ious


def _mean_valid(values):
    values = values[values > -1]
    return float(values.mean()) if values.size else None


def _metrics(precision, recall):
    return {
        "AP50_95": _mean_valid(precision),
        "AP50": _mean_valid(precision[0]),
        "AP75": _mean_valid(precision[5]),
        "AR300": _mean_valid(recall),
    }


def evaluate(ground_truth, predictions):
    """Evaluate parsed JSON objects and return headline and per-class metrics."""
    images = _index(ground_truth["images"], "images")
    categories = _index(ground_truth["categories"], "categories")
    if not images or not categories:
        raise ValueError("ground truth must define at least one image and category")
    names = [row["name"] for row in categories.values()]
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("category names must be nonempty strings")
    if len(set(names)) != len(names):
        raise ValueError("category names must be unique")
    if not isinstance(predictions, list):
        raise ValueError("predictions must be a JSON list")

    annotations = _geometry_rows(
        ground_truth["annotations"], images, categories, predictions=False
    )
    detections = _geometry_rows(predictions, images, categories, predictions=True)
    common = {"images": list(images.values()), "categories": list(categories.values())}
    reference = COCO()
    outputs = COCO()
    reference.dataset = {**common, "annotations": annotations}
    outputs.dataset = {**common, "annotations": detections}
    with redirect_stdout(StringIO()):
        reference.createIndex()
        outputs.createIndex()
        evaluator = _PolygonCOCOeval(reference, outputs, "bbox")
        evaluator.params.maxDets = [1, 10, MAX_DETECTIONS]
        evaluator.params.iouThrs = IOU_THRESHOLDS
        evaluator.params.recThrs = RECALL_THRESHOLDS
        evaluator.params.areaRng = [[0, float("inf")]]
        evaluator.params.areaRngLbl = ["all"]
        evaluator.evaluate()
        evaluator.accumulate()

    precision = evaluator.eval["precision"][:, :, :, 0, -1]
    recall = evaluator.eval["recall"][:, :, 0, -1]
    result = _metrics(precision, recall)
    result["per_class"] = {
        categories[category_id]["name"]: _metrics(precision[:, :, i], recall[:, i])
        for i, category_id in enumerate(evaluator.params.catIds)
    }
    result["predictions_input"] = len(predictions)
    result["predictions_retained"] = len(detections)
    result["protocol"] = {
        "overlap": "Exact convex polygon IoU",
        "iou_thresholds": [round(float(value), 2) for value in IOU_THRESHOLDS],
        "recall_samples": len(RECALL_THRESHOLDS),
        "area_range": "all",
        "max_detections_per_image_across_categories": MAX_DETECTIONS,
        "units": "fraction",
    }
    return result


def _self_test():
    square = [[0, 0], [10, 0], [10, 10], [0, 10]]
    gt = {
        "images": [{"id": 5}],
        "categories": [{"id": 2, "name": "empty"}, {"id": 1, "name": "target"}],
        "annotations": [{"image_id": 5, "category_id": 1, "quad": square}],
    }
    target = {"image_id": 5, "category_id": 1, "quad": square, "score": 0.8}
    miss = {**target, "quad": [[20, 0], [30, 0], [30, 10], [20, 10]]}
    # Moving a 10-pixel square 2 pixels horizontally gives IoU 2/3.
    partial = {**target, "quad": [2, 0, 12, 0, 12, 10, 2, 10]}
    result = evaluate(gt, [target])
    empty = evaluate(gt, [])
    for metric in ("AP50_95", "AP50", "AP75", "AR300"):
        assert math.isclose(result[metric], 1.0, abs_tol=1e-12)
        assert math.isclose(result["per_class"]["target"][metric], 1.0, abs_tol=1e-12)
        assert result["per_class"]["empty"][metric] is None
        assert empty[metric] == 0.0
    result = evaluate(gt, [partial])
    assert math.isclose(result["AP50"], 1.0, abs_tol=1e-12)
    assert result["AP75"] == 0.0
    assert math.isclose(result["AP50_95"], 0.4, abs_tol=1e-12)
    # Rotating an identical square 45 degrees about its center gives IoU 1/sqrt(2).
    radius = math.sqrt(50)
    diamond = [[5, 5 - radius], [5 + radius, 5], [5, 5 + radius], [5 - radius, 5]]
    rotated = evaluate(gt, [{**target, "quad": diamond}])
    assert math.isclose(rotated["AP50_95"], 0.5, abs_tol=1e-12)
    assert rotated["AP75"] == 0.0
    assert math.isclose(evaluate(gt, [miss, target])["AP50"], 0.5, abs_tol=1e-12)
    assert math.isclose(evaluate(gt, [target, miss])["AP50"], 1.0, abs_tol=1e-12)
    decoys = [{**miss, "category_id": 2}] * MAX_DETECTIONS
    capped = evaluate(gt, decoys + [target])
    assert capped["predictions_retained"] == MAX_DETECTIONS
    assert capped["AR300"] == 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--gt", type=Path, help="COCO-like ground truth JSON")
    parser.add_argument("--pred", type=Path, help="prediction JSON list")
    parser.add_argument("--self-test", action="store_true", help="run synthetic checks")
    args = parser.parse_args()
    if args.self_test:
        if args.gt or args.pred:
            parser.error("--self-test cannot be combined with --gt or --pred")
        _self_test()
        print("Polygon evaluator self-test passed.")
        return
    if args.gt is None or args.pred is None:
        parser.error("--gt and --pred are both required")

    try:
        ground_truth = json.loads(args.gt.read_text(encoding="utf-8-sig"))
        predictions = json.loads(args.pred.read_text(encoding="utf-8-sig"))
        result = evaluate(ground_truth, predictions)
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
