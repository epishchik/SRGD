# Веса моделей

## DVC Yandex S3

Веса (и другие данные, сохраненные на S3 с использованием [DVC](https://dvc.org/)) можно скачать следующей командой.

```bash
dvc pull
```

## Yandex Disk

### Pickle
Веса загружены на Yandex Disk и могут быть скачаны по [ссылке](https://disk.yandex.ru/d/P1w6Tis0cjoclQ) или скриптом.

```bash
cd utils
python3 download_weights.py \
  --download-source "https://disk.yandex.ru/d/P1w6Tis0cjoclQ"
```

### Onnx
Веса загружены на Yandex Disk и могут быть скачаны по [ссылке](https://disk.yandex.ru/d/csDaOUeCKxQk7g) или скриптом.

```bash
cd utils
python3 download_weights.py \
  --download-source "https://disk.yandex.ru/d/csDaOUeCKxQk7g"
```
