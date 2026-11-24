"""Build, initialize, fine-tune, and infer with the paper's RTMDet variants."""

import argparse
import hashlib
import json
import math
import pickle
from pathlib import Path
from pprint import pformat

from configs.rtmdet_tiny_r import (
    CLASSES,
    INPUT_SIZE,
    PARAMETERS,
    STRIDES,
    fine_tune_config,
    inference_pipeline,
    model_config,
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_state(path, expected_sha256=None, trusted=False, epoch=None):
    import torch

    digest = sha256(path)
    if expected_sha256 is not None and digest != expected_sha256.lower():
        raise ValueError("Checkpoint SHA256 does not match the requested identity.")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as exc:
        if not trusted:
            raise ValueError(
                "Checkpoint requires legacy unpickling. Use a tensor-only export, "
                "or pass --trusted-checkpoint for an archive you trust."
            ) from exc
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a checkpoint dictionary or a raw model state dictionary.")
    meta = checkpoint.get("meta", {})
    saved_epoch = meta.get("epoch")
    if epoch is not None and saved_epoch is not None and saved_epoch != epoch:
        raise ValueError(f"Expected source epoch {epoch}, got {saved_epoch}.")
    saved_classes = meta.get("dataset_meta", {}).get("classes")
    if saved_classes is not None and tuple(saved_classes) != CLASSES:
        raise ValueError("Checkpoint class order differs from the four-class configuration.")
    # MMEngine EMAHook places deployment EMA in the main state_dict.
    state = checkpoint.get("state_dict", checkpoint)
    if not state or any(not torch.is_tensor(value) for value in state.values()):
        raise ValueError("Expected a nonempty model tensor state dictionary.")
    return state, digest


def build_model(variant):
    from mmengine.config import ConfigDict
    from mmrotate.registry import MODELS
    from mmrotate.utils import register_all_modules

    register_all_modules(init_default_scope=True)
    model = MODELS.build(ConfigDict(model_config(variant)))
    if sum(p.numel() for p in model.parameters()) != PARAMETERS[variant]:
        raise RuntimeError("Model parameter count differs from the archived configuration.")
    return model


def source_key(key, variant):
    parts = key.split(".")
    if variant == "two_scale" and parts[0] == "neck" and parts[1] in ("reduce_layers", "top_down_blocks"):
        parts[2] = str(int(parts[2]) + 1)
    return ".".join(parts)


def map_initial_state(source, target, variant):
    """Copy matching tensor roles or take the fixed prefix for the narrowed control."""
    state = {}
    for key, tensor in target.items():
        original = source_key(key, variant)
        if original not in source:
            raise ValueError(f"Missing source tensor for {key}: {original}")
        value = source[original]
        if value.shape != tensor.shape:
            if variant != "coarse_narrow" or value.ndim != tensor.ndim or any(
                old < new for old, new in zip(value.shape, tensor.shape)
            ):
                raise ValueError(f"Unsupported shape mapping for {key}: {value.shape} -> {tensor.shape}")
            value = value[tuple(slice(0, n) for n in tensor.shape)]
        if value.dtype != tensor.dtype:
            raise ValueError(f"Source tensor dtype differs from the initialized model: {key}")
        state[key] = value.detach().clone()
    return state


def write_initial(args, output):
    import torch

    source, digest = read_state(
        args.source_checkpoint, args.checkpoint_sha256, args.trusted_checkpoint, epoch=100
    )
    model = build_model(args.variant)
    state = map_initial_state(source, model.state_dict(), args.variant)
    model.load_state_dict(state, strict=True)
    loaded = model.state_dict()
    if not all(torch.equal(value, loaded[key]) for key, value in state.items()):
        raise RuntimeError("Mapped tensors disagree after loading shared model weights.")
    with output.open("xb") as stream:
        torch.save(dict(state_dict=state, meta=dict(dataset_meta=dict(classes=CLASSES),
                                                   source_checkpoint_sha256=digest)), stream)
    return digest


def make_config(args, initial):
    return fine_tune_config(
        args.variant, args.seed, args.epochs, args.data_root.resolve(),
        args.train_manifest.resolve(), initial.resolve(), args.work_dir.resolve(),
    )


def config_text(cfg):
    return "\n\n".join(f"{key} = {pformat(value, width=100)}" for key, value in cfg.items()) + "\n"


def print_config(args):
    content = config_text(make_config(args, args.initial_checkpoint))
    if args.output is None:
        print(content, end="")
    else:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(content)


def configure_torch():
    import cv2
    import torch

    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def train(args):
    import torch
    from mmengine.config import Config
    from mmengine.runner import Runner, set_random_seed
    from rtmdet_runtime import manifest_rows

    if len(manifest_rows(args.train_manifest)) != 2663:
        raise ValueError("The fixed paper protocol requires an ordered 2,663-image training manifest.")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Training requires a CUDA GPU with BF16 support.")
    configure_torch()
    set_random_seed(args.seed, deterministic=False)
    args.work_dir.mkdir(parents=True, exist_ok=False)
    initial = args.work_dir / "initial.pth"
    digest = write_initial(args, initial)
    cfg = Config(make_config(args, initial))
    config_path = args.work_dir / "resolved_config.py"
    cfg.dump(str(config_path))
    runner = Runner.from_cfg(cfg)
    runner.train()
    endpoint = args.work_dir / f"epoch_{args.epochs}.pth"
    if runner.epoch != args.epochs or not endpoint.is_file():
        raise RuntimeError("Fixed final epoch or checkpoint was not produced.")
    print(json.dumps(dict(
        variant=args.variant, seed=args.seed, epochs=args.epochs, checkpoint=str(endpoint),
        checkpoint_sha256=sha256(endpoint), source_checkpoint_sha256=digest,
        initial_checkpoint_sha256=sha256(initial), train_manifest_sha256=sha256(args.train_manifest),
        config=str(config_path), config_sha256=sha256(config_path),
    )))


def inspect_model(args):
    import torch

    configure_torch()
    model = build_model(args.variant)
    result = dict(variant=args.variant, parameters=PARAMETERS[args.variant], strides=STRIDES[args.variant])
    if args.forward:
        macs = 0

        def count_macs(layer, _inputs, output):
            nonlocal macs
            if isinstance(layer, torch.nn.Conv2d):
                macs += output.numel() * (layer.in_channels // layer.groups) * math.prod(layer.kernel_size)
            else:
                macs += output.numel() * layer.in_features

        handle_list = [layer.register_forward_hook(count_macs) for layer in model.modules()
                       if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear))]
        model.eval()
        try:
            with torch.inference_mode():
                levels = model.extract_feat(torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE))
                outputs = model.bbox_head(levels)
        finally:
            for handle in handle_list:
                handle.remove()
        expected = [(1, 96, INPUT_SIZE // s, INPUT_SIZE // s) for s in STRIDES[args.variant]]
        if [tuple(x.shape) for x in levels] != expected or not all(
            torch.isfinite(tensor).all() for branch in outputs for tensor in branch
        ):
            raise RuntimeError("Feature shapes or prediction outputs differ from the architecture contract.")
        result["feature_shapes"] = expected
        result["conv_linear_macs"] = macs
    print(json.dumps(result))


def rbox_to_quad(box):
    x, y, width, height, theta = box
    ux, uy = width * 0.5 * math.cos(theta), width * 0.5 * math.sin(theta)
    vx, vy = -height * 0.5 * math.sin(theta), height * 0.5 * math.cos(theta)
    return [[x - ux - vx, y - uy - vy], [x + ux - vx, y + uy - vy],
            [x + ux + vx, y + uy + vy], [x - ux + vx, y - uy + vy]]


def predict(args):
    import cv2
    import numpy as np
    import torch
    from mmcv.transforms import Compose
    from mmdet.apis import inference_detector
    from mmengine.config import Config
    from rtmdet_runtime import manifest_rows, relative_file

    if args.output.exists():
        raise FileExistsError(args.output)
    configure_torch()
    root = args.data_root.resolve(strict=True)
    rows = manifest_rows(args.val_manifest)
    ids = [entry["id"] for entry in rows]
    if any(type(x) is not int for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("Validation image IDs must be distinct integers.")
    state, _ = read_state(args.checkpoint, args.checkpoint_sha256, args.trusted_checkpoint)
    model = build_model(args.variant)
    model.load_state_dict(state, strict=True)
    model.dataset_meta = dict(classes=CLASSES)
    model.cfg = Config(dict(model=model_config(args.variant)))
    model = model.to(args.device).float().eval()
    pipeline = Compose(inference_pipeline())
    predictions = []
    with torch.inference_mode():
        for entry in rows:
            path = relative_file(root, entry["image"], entry.get("image_sha256"))
            image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (INPUT_SIZE, INPUT_SIZE):
                raise ValueError(f"Expected a valid 640×640 color tile: {entry['image']}")
            instances = inference_detector(model, image, test_pipeline=pipeline).pred_instances
            boxes = instances.bboxes
            if hasattr(boxes, "tensor"):
                boxes = boxes.tensor
            for box, label, score in zip(boxes.cpu().tolist(), instances.labels.cpu().tolist(),
                                         instances.scores.cpu().tolist()):
                predictions.append(dict(image_id=entry["id"], category_id=label,
                                        score=score, quad=rbox_to_quad(box)))
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(predictions, stream, allow_nan=False)
        stream.write("\n")


def add_variant(parser):
    parser.add_argument("--variant", choices=PARAMETERS, default="two_scale")


def add_source(parser, source_flag):
    parser.add_argument(source_flag, type=Path, required=True)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--trusted-checkpoint", action="store_true",
                        help="Allow legacy PyTorch unpickling for a checkpoint you trust.")


def add_schedule(parser):
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), default=42)
    parser.add_argument("--epochs", type=int, choices=(10, 30), default=30)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    inspect_cmd = sub.add_parser("inspect", help="Build the model on CPU and report parameter counts.")
    add_variant(inspect_cmd)
    inspect_cmd.add_argument("--forward", action="store_true",
                             help="Check CPU feature shapes and count convolutional and linear MACs.")
    inspect_cmd.set_defaults(handler=inspect_model)
    init_cmd = sub.add_parser("init", help="Map a source model to a fine-tuning initialization.")
    add_variant(init_cmd)
    add_source(init_cmd, "--source-checkpoint")
    init_cmd.add_argument("--output", type=Path, required=True)
    init_cmd.set_defaults(handler=lambda args: write_initial(args, args.output))
    config_cmd = sub.add_parser("config", help="Print or save the resolved fine-tuning configuration.")
    add_variant(config_cmd)
    add_schedule(config_cmd)
    config_cmd.add_argument("--initial-checkpoint", type=Path, required=True)
    config_cmd.add_argument("--output", type=Path)
    config_cmd.set_defaults(handler=print_config)
    train_cmd = sub.add_parser("train", help="Map the source and run fixed supervised fine-tuning.")
    add_variant(train_cmd)
    add_schedule(train_cmd)
    add_source(train_cmd, "--source-checkpoint")
    train_cmd.set_defaults(handler=train)
    predict_cmd = sub.add_parser("predict", help="Run native FP32 inference and write polygon predictions.")
    add_variant(predict_cmd)
    add_source(predict_cmd, "--checkpoint")
    predict_cmd.add_argument("--device", default="cuda:0")
    predict_cmd.add_argument("--data-root", type=Path, required=True)
    predict_cmd.add_argument("--val-manifest", type=Path, required=True)
    predict_cmd.add_argument("--output", type=Path, required=True)
    predict_cmd.set_defaults(handler=predict)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
