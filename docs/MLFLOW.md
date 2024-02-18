# MLFlow

## Disclaimer
Don't push .env files to git, it's just an example without real credentials.

## Deploy Full

Compose Up.
```bash
docker compose build
docker compose up -d
```

Change MLFlow admin password.
```bash
curl --user "admin:password" \
    -X PATCH http://0.0.0.0:5000/api/2.0/mlflow/users/update-password \
    -H 'Content-Type: application/json' \
    -d '{"username":"admin","password":"new_password"}'
```

Create MLFlow user.
```bash
curl --user "admin:new_password" \
    -X POST http://0.0.0.0:5000/api/2.0/mlflow/users/create \
    -H 'Content-Type: application/json' \
    -d '{"username":"user","password":"password"}'
```

Check MLFlow user.

```bash
curl --user "admin:new_password" \
    -X GET http://0.0.0.0:5000/api/2.0/mlflow/users/get?username=user
```

Compose Down.
```bash
docker compose down --volumes
```

## Usage

Create AWS credentials to access MinIO.
```bash
mkdir -p ~/.aws
touch ~/.aws/credentials
```

AWS credentials example.
```text
[default]
aws_access_key_id = access_key_id
aws_secret_access_key = secret_access_key
```

Create MLFlow credentials.
```bash
mkdir -p ~/.mlflow
touch ~/.mlflow/credentials
```

MLFlow credentials example.
```text
[mlflow]
mlflow_tracking_username = user
mlflow_tracking_password = password
```

Set client MLFLOW_S3_ENDPOINT_URL environment variable.

Usage examples:
```bash
MLFLOW_S3_ENDPOINT_URL="http://0.0.0.0:9000" python3 real_esrgan.py \
  -opt configs/train/finetune_realesrgan_x4plus_game_engine.yaml
```

You can use only environment variables instead of credential files:
- MLFLOW_S3_ENDPOINT_URL.
- MLFLOW_TRACKING_USERNAME.
- MLFLOW_TRACKING_PASSWORD.
- AWS_ACCESS_KEY_ID.
- AWS_SECRET_ACCESS_KEY.
