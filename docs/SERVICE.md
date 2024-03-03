# Сервис

|UI                         | API                       |
|:-------------------------:|:-------------------------:|
|![UI](/images/ui.png)      |  ![API](/images/api.png)  |

## Deploy PyTorch API
```bash
docker build \
  --build-arg MODEL_FOLDER=model \
  -t srgb/api:0.0.0 \
  -f api/Dockerfile .
```

## Deploy Triton API
```bash
docker build \
  --build-arg MODEL_FOLDER=triton_model \
  -t srgb/api:0.0.0 \
  -f api/Dockerfile .
```

## Run API
```bash
docker run -d \
  -p 8000:8000 \
  --gpus 0 \
  --memory 16GB \
  --memory-swap 16GB \
  --restart=always \
  --name srgb-api \
  srgb/api:0.0.0
```

## Deploy PyTorch UI
```bash
docker build \
  --build-arg API_URL="http://torch_api:8000" \
  -t srgb/ui:0.0.0 \
  -f ui/Dockerfile .
```

## Deploy Triton UI
```bash
docker build \
  --build-arg API_URL="http://triton_api:8000" \
  -t srgb/ui:0.0.0 \
  -f ui/Dockerfile .
```

## Run UI
```bash
docker run -d \
  -p 8501:8501 \
  --cpus 1 \
  --memory 2GB \
  --memory-swap 2GB \
  --restart=always \
  --name srgb-ui \
  srgb/ui:0.0.0
```

## Deploy & Run PyTorch Full
```bash
docker compose --profile torch_backend build
docker compose --profile torch_backend up -d
```

Если контейнер с `Triton` работает с `GPU`, а `PyTorch` не находит `GPU`, то возможное решение (проверьте командой `nvidia-smi` внутри контейнера) - ["Failed to initialize NVML: Unknown Error"](https://bbs.archlinux.org/viewtopic.php?id=266915).

## Deploy & Run Triton Full
```bash
docker compose --profile triton_backend build
docker compose --profile triton_backend up -d
```
