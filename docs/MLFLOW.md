# MLFlow

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
