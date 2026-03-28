# Aerial Vehicle Tracker

Fine-tuning YOLO on aerial drone imagery to detect and track vehicles (car, bus, truck).

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
