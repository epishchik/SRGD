# Сервис

|UI                         | API                       |
|:-------------------------:|:-------------------------:|
|![UI](/images/ui.png)      |  ![API](/images/api.png)  |

## Deploy API
```bash
docker build \
  -t super-resolution-benchmark/api:0.0.0 \
  -f api/Dockerfile .
```
```bash
docker run -d \
  -p 8000:8000 \
  --gpus 0 \
  --memory 16GB \
  --memory-swap 16GB \
  --restart=always \
  --name super-resolution-benchmark-api \
  super-resolution-benchmark/api:0.0.0
```

## Deploy UI
```bash
docker build \
  -t super-resolution-benchmark/ui:0.0.0 \
  -f ui/Dockerfile .
```
```bash
docker run -d \
  -p 8501:8501 \
  --cpus 1 \
  --memory 2GB \
  --memory-swap 2GB \
  --restart=always \
  --name super-resolution-benchmark-ui \
  super-resolution-benchmark/ui:0.0.0
```

## Deploy Full
```bash
docker compose build
docker compose up -d
```
