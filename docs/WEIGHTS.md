# Веса моделей

## DVC Yandex S3

Веса (и другие данные, сохраненные на S3 с использованием [DVC](https://dvc.org/)) можно скачать следующей командой.

```bash
dvc pull
```

## Yandex Disk

Веса загружены на Yandex Disk и могут быть скачаны по ссылке или скриптом.

```bash
cd utils
python3 download_weights.py
```
