# Deploy

## Deploy API
```bash
sudo docker build \
  -t super-resolution-benchmark/api:0.0.0 \
  -f api/Dockerfile .
```
```bash
sudo docker run -d \
  -p 8000:8000 \
  --gpus 0 \
  --memory 16GB \
  --memory-swap 16GB \
  --restart=always \
  --name super-resolution-benchmark-api \
  super-resolution-benchmark/api:0.0.0
```

## Deploy Full
```bash
sudo docker compose build
sudo docker compose up
```
