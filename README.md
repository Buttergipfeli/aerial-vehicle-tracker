# Aerial Vehicle Tracker

Fine-tuning YOLO on aerial drone imagery to detect and track vehicles. Two classification approaches are compared: multi-class (car, bus, truck) and binary (vehicle / no vehicle).

## Roadmap

### Multi-class (car / bus / truck)
- [x] Train YOLO26s
- [ ] Train YOLO26m
- [ ] Evaluate on test set

### Binary (vehicle / no vehicle)
- [ ] Train YOLO26s
- [ ] Train YOLO26m
- [ ] Evaluate on test set

### Detection & Tracking
- [x] Detection script (pretrained vs. fine-tuned comparison)
- [ ] Tracking script (image sequence)
- [ ] Tracking script (video)

### Analysis
- [ ] Notebook with results comparison (all models)

## Training Results

> These are validation metrics from training, not final test set evaluation.

| Model | Classes | Epochs | mAP50 | mAP50-95 | Precision | Recall | Time | Device |
|---|---|---|---|---|---|---|---|---|
| YOLO26s | car/bus/truck | 18 | 0.876 | 0.551 | 0.826 | 0.841 | 8.1h | MacBook Pro M1 Pro 32GB |

**Per class (YOLO26s, best epoch):**

| Class | mAP50 | mAP50-95 |
|---|---|---|
| car | 0.957 | 0.553 |
| bus | 0.855 | 0.595 |
| truck | 0.798 | 0.513 |

## Dataset

[Aerial Multi-Vehicle Detection Dataset](https://doi.org/10.5281/zenodo.7053442) by Makrigiorgis et al. (2022), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

> The dataset is not included in this repository. Download it from [zenodo.org/records/7053442](https://doi.org/10.5281/zenodo.7053442) and place it in `data/7053442/`.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Acknowledgements

This project uses [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), licensed under [AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0.html).

## Citation

```bibtex
@dataset{makrigiorgis2022aerial,
  author    = {Makrigiorgis, Rafael and Kolios, Panayiotis and Kyrkou, Christos},
  title     = {Aerial Multi-Vehicle Detection Dataset},
  year      = {2022},
  publisher = {Zenodo},
  version   = {1.0},
  doi       = {10.5281/zenodo.7053442},
  url       = {https://doi.org/10.5281/zenodo.7053442}
}
```
