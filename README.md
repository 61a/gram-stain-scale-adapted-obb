# Scale-adapted rotated bacterial detection

Implementation for *Scale-adapted lightweight rotated detection of bacteria in Gram-stained microscopy images*. The detector predicts Gram-negative cocci, Gram-positive cocci, Gram-negative rods, and Gram-positive rods in 640 × 640 tiles.

| Variant | Feature strides | Parameters |
| --- | --- | ---: |
| Scale-adapted (`two_scale`) | 8, 16 | 1,403,934 |
| Standard capacity (`baseline`) | 8, 16, 32 | 4,874,343 |
| High-level channel reduction (`coarse_narrow`) | 8, 16, 32 | 3,824,007 |

The variants use the OpenMMLab CSPNeXt backbone, CSPNeXt-PAFPN neck, and rotated RTMDet prediction head. The scale-adapted variant retains the two finer feature levels. The high-level channel-reduction control retains all three levels and narrows the coarsest stage. The release includes checkpoint mapping, fixed-budget supervised adaptation, native rotated prediction, and exact polygon evaluation.

## Environment

The study used Linux under WSL, Python 3.10, PyTorch 2.7.1 with CUDA 12.8, MMEngine 0.10.7, MMDetection 3.2.0, MMCV 2.1.0 with compiled ops, and a local MMRotate 1.0.0rc1-compatible snapshot. The local MMRotate compatibility guard allowed MMDetection 3.2.0. Use a compatible MMRotate checkout and build MMCV operators against the selected PyTorch/CUDA environment.

The evaluator runs independently on CPU. Install its dependencies in any compatible Python 3.10 environment. With the model dependencies installed, the inspect command also reports parameters, feature-map shapes, and the executed convolutional and linear MACs for one tile.

```bash
python -m pip install -r requirements-eval.txt
python polygon_eval.py --self-test
python rtmdet.py --help
python rtmdet.py inspect --variant two_scale --forward
```

## Data and checkpoints

The [301Rose source dataset](https://doi.org/10.5281/zenodo.10526360) provides the original images and annotations. The paper used an ordered split of 2,663 training tiles and 870 validation tiles. Replaying the reported runs requires the same split, annotations, and source checkpoints. The repository contains code and configurations. Images, annotations, split files, predictions, and checkpoints are separate project inputs.

The training manifest is an ordered JSON array. Each entry names an image and a DOTA text annotation relative to the data root, as in `{"image":"images/a.jpg","label":"labels/a.txt"}`. A label line has eight corner coordinates in input pixels, one of `G_coccus`, `Gp_coccus`, `G_rod`, or `Gp_rod`, and difficulty `0`. Optional manifest fields `image_sha256` and `label_sha256` check file identity.

The validation manifest is an ordered JSON array of `{"id":1,"image":"images/b.jpg"}` entries. Each integer ID must match the corresponding ground-truth JSON image ID. The ground-truth JSON uses COCO-style `images`, `categories`, and `annotations` lists. Each annotation has `image_id`, `category_id`, and a pixel-space `quad`. Category IDs 0–3 follow the class order above.

## Matched 30-epoch comparison

Start each variant from the matching seed's epoch-100 rotated RTMDet-tiny deployment `state_dict`. The trainer maps the weights into the selected structure, resets the optimizer, and runs for the full fixed schedule. Use the final epoch's deployment `state_dict` for evaluation.

```bash
python rtmdet.py train --variant two_scale --seed 42 --epochs 30 \
  --data-root DATA --train-manifest train.json \
  --source-checkpoint source_epoch_100.pth --work-dir work/two_scale
python rtmdet.py predict --variant two_scale --checkpoint work/two_scale/epoch_30.pth \
  --data-root DATA --val-manifest val.json --output two_scale.predictions.json
python polygon_eval.py --gt val.groundtruth.json \
  --pred two_scale.predictions.json > two_scale.metrics.json
```

Inference uses FP32 with TF32 disabled, a score threshold of 0.01, up to 2,000 pre-NMS candidates per level, rotated NMS at IoU 0.7, and a cap of 300 detections per tile. The evaluator matches exact convex polygon IoU across thresholds 0.50–0.95 in steps of 0.05. It reports overall and per-class AP50:95, AP50, AP75, and AR300 as fractions from 0 to 1. Multiply by 100 for the paper's percentages.

Sources: [RTMDet](https://arxiv.org/abs/2212.07784), [MMDetection](https://github.com/open-mmlab/mmdetection), [MMRotate](https://github.com/open-mmlab/mmrotate), and [COCO evaluation](https://github.com/cocodataset/cocoapi).
