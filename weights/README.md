# Model card: `yolo11s_uvh26`

| | |
|---|---|
| Files | `yolo11s_uvh26.pt` (PyTorch, 19 MB), `yolo11s_uvh26.onnx` (ONNX, dynamic input shape, FP32, 38 MB) |
| Architecture | YOLO11s, 9.4 M parameters, initialised from COCO-pretrained `yolo11s.pt` |
| Classes | `0 car`, `1 auto`, `2 two_wheeler`, `3 bus`, `4 truck`, `5 lcv` |
| Training data | 3,570 train + 630 val images from the official UVH-26 train split (seed 26), majority-vote labels, 14 classes merged into 6 (see the main README) |
| Training | imgsz 640, batch 8, workers 2, 32 epochs, AdamW lr 0.001 (auto) with cosine schedule, patience 10, mosaic (off for the last 5 epochs), mixup 0.05; best epoch 27; Apple M5 16 GB (MPS), 3.2 h. Full config: [`docs/metrics/train_args.yaml`](../docs/metrics/train_args.yaml) |
| Default inference size | **640** (better than 960 overall on the held-out test set; 960 is marginally better for two-wheelers and autos, equal for cars, and worse for bus, truck and LCV) |
| Held-out test (800 images from the official UVH-26 val split) | mAP@50 **0.814**, mAP@50-95 **0.678**, P 0.803, R 0.763 at 640 px (single-label NMS) |
| Per-class AP@50-95 at 640 | car 0.841, auto 0.778, two_wheeler 0.690, bus 0.701, truck 0.496, lcv 0.558 |
| Speed (1280x720 frame, Apple M5) | 12.2 ms on MPS (`.pt`), 23.5 ms with ONNX Runtime on CPU |
| Intended use | Counting and monitoring vehicles in fixed-camera traffic footage from Indian cities |
| Out of scope / known weaknesses | Night, rain, dash-cam, drone or non-Indian traffic (untested); truck vs LCV confusion; counting accuracy not benchmarked on real video |
| Licences | Fine-tuned from Ultralytics YOLO11 (AGPL-3.0). Training data UVH-26 by IISc AIM, CC-BY-4.0 |

`v1_legacy.onnx` is the original v1 model: a YOLOv8n with classes `Ambulance, Bus, Car, Motorcycle, Truck` (from its ONNX metadata), with a fixed 416x416 input. On the same test set, remapped to the 6 classes, it scores mAP@50 0.073 / mAP@50-95 0.041. It is kept only for comparison.
