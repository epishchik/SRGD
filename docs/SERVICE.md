# Сервис

|UI                         | API                       |
|:-------------------------:|:-------------------------:|
|![UI](/images/ui.png)      |  ![API](/images/api.png)  |

## Deploy API
```bash
docker build \
  -t srgb/api:0.0.0 \
  -f api/Dockerfile .
```
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

## Deploy UI
```bash
docker build \
  -t srgb/ui:0.0.0 \
  -f ui/Dockerfile .
```
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

## Deploy Full
```bash
docker compose build
docker compose up -d
```
