"""Dataset and training registrations for the fixed fine-tuning protocol."""

import hashlib
import json
import math
from pathlib import Path

import torch
from mmengine.hooks import Hook
from mmengine.optim import AmpOptimWrapper
from mmengine.registry import HOOKS, OPTIM_WRAPPERS
from mmengine.runner import set_random_seed
from mmrotate.datasets.dota import DOTADataset
from mmrotate.registry import DATASETS

from configs.rtmdet_tiny_r import CLASSES


def relative_file(root, name, expected_sha256=None):
    relative = Path(name)
    if relative.is_absolute():
        raise ValueError(f"Manifest path must be relative: {name}")
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Manifest path must be a file inside the data root: {name}")
    if expected_sha256 and hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError(f"SHA256 mismatch: {name}")
    return path


def manifest_rows(path):
    rows = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list) or not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Expected a nonempty ordered JSON array of manifest rows.")
    return rows


@DATASETS.register_module()
class PaperDotaDataset(DOTADataset):
    """Load the ordered training subset from relative DOTA image and label paths."""

    def __init__(self, manifest_path, **kwargs):
        self.rows = manifest_rows(manifest_path)
        super().__init__(**kwargs)

    def load_data_list(self):
        root = Path(self.data_root).resolve(strict=True)
        class_index = {name: i for i, name in enumerate(CLASSES)}
        data = []
        seen = set()
        for row in self.rows:
            image = relative_file(root, row["image"], row.get("image_sha256"))
            label = relative_file(root, row["label"], row.get("label_sha256"))
            if image in seen:
                raise ValueError(f"Duplicate training image: {row['image']}")
            seen.add(image)
            instances = []
            for line in label.read_text(encoding="utf-8-sig").splitlines():
                fields = line.split()
                if not fields:
                    continue
                if len(fields) != 10 or fields[8] not in class_index or fields[9] != "0":
                    raise ValueError(f"Expected 8 pixel coordinates, a named class and difficulty 0: {row['label']}")
                quad = [float(x) for x in fields[:8]]
                if not all(math.isfinite(x) for x in quad):
                    raise ValueError(f"Nonfinite box coordinates: {row['label']}")
                instances.append(dict(bbox=quad, bbox_label=class_index[fields[8]], ignore_flag=0))
            data.append(dict(img_id=image.stem, file_name=image.name, img_path=str(image), instances=instances))
        return data


@OPTIM_WRAPPERS.register_module()
class FiniteAmpOptimWrapper(AmpOptimWrapper):
    def step(self, **kwargs):
        grads = [p.grad for group in self.optimizer.param_groups for p in group["params"] if p.grad is not None]
        if not grads or not torch.isfinite(torch.stack(torch._foreach_norm(grads))).all():
            raise RuntimeError("Gradient norm is missing or nonfinite.")
        return super().step(**kwargs)


@HOOKS.register_module()
class FixedStartHook(Hook):
    priority = "VERY_HIGH"

    def __init__(self, seed):
        self.seed = seed

    def before_train(self, runner):
        if runner.epoch != 0 or runner.optim_wrapper.optimizer.state:
            raise RuntimeError("Fixed supervised adaptation requires a new optimizer and epoch zero.")
        if len(runner.train_dataloader.dataset) != 2663:
            raise ValueError("The fixed paper protocol uses 2,663 training images.")
        initial = torch.load(runner.cfg.load_from, map_location="cpu", weights_only=True)["state_dict"]
        if not all(torch.equal(v.detach().cpu(), initial[k]) for k, v in runner.model.state_dict().items()):
            raise RuntimeError("Runner weights differ from the mapped initialization.")
        set_random_seed(self.seed, deterministic=False)

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        for name, value in (outputs or {}).items():
            if "loss" in name and not math.isfinite(float(value)):
                raise RuntimeError(f"Nonfinite training loss: {name}")
