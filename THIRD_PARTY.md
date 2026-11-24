# Third-party attribution

The RTMDet configuration in `configs/rtmdet_tiny_r.py` is adapted from the
[OpenMMLab MMRotate](https://github.com/open-mmlab/mmrotate) rotated RTMDet
configuration. Copyright (c) OpenMMLab. All rights reserved. The study
changes the class count, feature levels and channel widths, and configures
the supervised adaptation and inference settings for 301Rose tiles. The
upstream-derived portions are covered by the Apache License, Version 2.0
([license text](LICENSE-APACHE-2.0.txt)).

MMRotate, MMDetection, MMCV, MMEngine, and pycocotools are separately installed
dependencies. Each is distributed under its own upstream license.
