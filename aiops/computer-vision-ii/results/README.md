# Computer Vision II: Train, Optimise, Serve and Monitor an Object Detector on CPU

Hands-on lab from the AIOps Additional Assessment Labs at Al Nafi.

## Scenario
A data centre wants an automated inspector that finds three kinds of objects in rack photos: `status_led`, `warning_sign`, and `cable_tag`. There is no GPU, no cloud account, and no labelled production data, so the whole pipeline runs on one CPU-only server with a generated, labelled dataset.

## What I did
1. **Dataset and QA:** generated a labelled YOLO-format dataset (300 train, 60 validation, 60 test images) and wrote a QA tool that checks class IDs, coordinate ranges, and missing labels. I injected three realistic annotation errors, confirmed the tool caught all three, then repaired the data by removing the bad line and quarantining the two untrusted images.
2. **IoU and NMS:** implemented vectorised IoU and Non-Maximum Suppression and checked them against torchvision. All tests passed.
3. **Pretrained baseline:** inspected a COCO-pretrained YOLOv8n (3,157,200 parameters, raw output `(1, 84, 8400)` at 640 px and `(1, 84, 2100)` at 320 px). It detected the bus and people in a sample image but found nothing on a rack image, because it only knows COCO classes.
4. **Fine-tuning:** fine-tuned YOLOv8n with transfer learning for 25 epochs on CPU (about 0.33 hours), using augmentation chosen for a fixed upright camera (no vertical flips or rotations).
5. **Evaluation and quality gate:** evaluated on the held-out test split, swept confidence thresholds with my own IoU matcher, and enforced an automated pass/fail quality gate.
6. **ONNX export:** exported to ONNX, built an inference engine using only NumPy, Pillow, and onnxruntime, and checked its output against PyTorch.
7. **Microservice:** packaged the engine as a FastAPI service in Docker (non-root user, health check, input validation, Prometheus metrics).
8. **Robustness and drift:** degraded the test images (low light, blur, noise), measured the accuracy loss, load-tested the service, and built a label-free drift monitor.

## Results

### Fine-tuned model on the test split (60 images, 225 objects)
| Class | Precision | Recall | AP50 | AP50-95 |
|-------|-----------|--------|------|---------|
| status_led | 0.998 | 1.000 | 0.995 | 0.967 |
| warning_sign | 0.997 | 1.000 | 0.995 | 0.983 |
| cable_tag | 0.999 | 1.000 | 0.995 | 0.983 |
| **All** | **0.998** | **1.000** | **0.995** | **0.977** |

**Important caveat:** the data is synthetic, drawn by a script with clean shapes and colours. Near-perfect scores are expected here and don't say how the model would do on real rack photos.

The confusion matrix had no misclassifications. In the threshold sweep, precision and recall were both 1.0 for every confidence threshold from 0.1 to 0.7, and recall only dropped at 0.8 (0.996) and 0.9 (0.951). The sweep picked 0.1 because F1 was tied from 0.1 to 0.7 and the lowest value won. In production I would choose a threshold with more margin, such as 0.5, since the tie shows the data can't distinguish them.

### ONNX engine versus PyTorch
| | Result |
|--|--------|
| Detections matched (same class, IoU of 0.8 or higher) | 71 of 71 (100%) |
| Largest confidence difference | 0.019 |
| Latency, PyTorch (end to end) | 37.5 ms per image |
| Latency, ONNX engine (end to end) | 34.2 ms per image |

### Service
- Docker image `rackvision:1.0` is 460 MB, running as a non-root user and reporting healthy.
- A bad upload returned HTTP 400, a wrong form field returned 422, and the Prometheus metrics (request counts, latency, detections by class, empty predictions, brightness histogram) were exposed.
- Load test, 120 requests each with 4 concurrent clients:

| | Clean images | Degraded images |
|--|--------------|-----------------|
| Errors | 0 | 0 |
| Throughput | 19.6 requests/s | 18.4 requests/s |
| Client p50 / p95 latency | 199.9 / 286.9 ms | 212.6 / 307.5 ms |
| Mean server inference time | 159.4 ms | 157.4 ms |
| Detections per image | 3.75 | 3.60 |
| Mean confidence | 0.960 | 0.685 |

### Robustness and drift
| Metric | Clean | Degraded | Change |
|--------|-------|----------|--------|
| Precision | 0.998 | 0.988 | -0.011 |
| Recall | 1.000 | 0.946 | -0.054 |
| mAP50 | 0.995 | 0.994 | -0.001 |
| mAP50-95 | 0.977 | 0.867 | -0.111 |

The lab expected the quality gate to fail on the degraded data, but it passed, because mAP50 barely moved. The damage showed up elsewhere: mAP50-95 fell by 0.111 (the boxes were placed less precisely), recall dropped, and the model's average confidence fell from 0.96 to 0.69. So a gate based only on mAP50 would have missed this degradation.

The drift monitor caught it without labels. The clean test set showed no drift. On the degraded set, mean brightness fell from a reference of 53.65 to 14.09, every image was an outlier, and the monitor exited with code 2 (drift detected).

### Validation
The lab's validation script passed all 18 checks.

## What I learned
- Annotation errors like a bad class ID or a missing label file are silently dropped by training frameworks, and a missing label file even turns a real frame into a "background" frame. A QA gate before training prevents that.
- Transfer learning (keeping the pretrained backbone and neck and re-initialising only the class head) gives an accurate detector in minutes on CPU.
- mAP50 can hide a real quality drop. mAP50-95, confidence, and input statistics showed the degradation that mAP50 missed.
- A label-free drift monitor on a simple input statistic like brightness can detect trouble before anyone notices missed detections.
- Perfect scores on a synthetic test set aren't evidence of real-world accuracy, and a tied F1 means the threshold choice needs a judgement call, not just the highest score.

## Repo contents
- `results/`: test and degraded-image metrics, the chosen operating point, load-test results, drift reports, and training and confusion-matrix plots

Course-provided scripts, the dataset generator, and the model files are not included.

## Tools
Python, PyTorch, Ultralytics YOLOv8, OpenCV, ONNX Runtime, FastAPI, Docker, Prometheus client, pandas, NumPy
