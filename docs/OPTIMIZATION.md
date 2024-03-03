# Оптимизация inference

## Конвертация в onnx

Пример конвертации.
```bash
cd optimization
python3 torch2onnx.py \
  --save-path "dvc_data/onnx/real-esrgan/RealESRGAN_x4plus.onnx" \
  --model-config "configs/model/pretrained/RealESRGAN_x4plus.yaml"
```

Сконвертированную модель можно визуализировать при помощи `netron`.
```bash
cd utils
python3 vizualize_net.py \
  --file ~/SRGB/dvc_data/onnx/real-esrgan/RealESRGAN_x4plus.onnx \
  --host 127.0.0.1 \
  --port 12651
```

Сам по себе `onnx` с `CUDAExecutionProvider` ускорения не дал, поэтому установим `TensorRT` для использования `TensorrtExecutionProvider` в `onnxruntime`.
[Инструкция по установке TensorRT.](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)

Для использования trtexec необходимо добавить `/usr/src/tensorrt/bin` в `PATH`.
```bash
export PATH="/usr/src/tensorrt/bin:$PATH"
```

## Конвертация в TensorRT

`TensorRT` для конвертации удобно использовать из официального образа [tensorrt](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorrt), однако если настроили его для `onnx`, то можно использовать локальный.

Воспользуемся `bash` скриптом для конвертации и положим в директорию `triton_models`, формат которой совпадает с форматом `--model-repository` для `triton`. Значение `maxr` необходимо выставлять максимально возможным, которое не будет приводить к превышению `VRAM` для вашей `GPU`.
```bash
cd optimization
bash onnx2trt.sh -o ~/SRGB/dvc_data/onnx/real-esrgan \
  -s ~/SRGB/triton_models \
  -m RealESRGAN_x4plus \
  -v 1 \
  --optr=270,480 \
  --maxr=540,960
```

## Deploy в Triton Inference Server

### Установка

- Установить [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- Скачать образ [tritonserver](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver).

### config.pbxt
Для каждой модели в `triton` необходимо задать конфигурацию. Примеры можно найти в папке [triton_models](/triton_models).

### Запуск сервера с моделью RealESRGAN_x4plus
```bash
docker run --rm \
    --gpus 0 \
    -p 8001:8000 \
    -p 8002:8001 \
    -p 8003:8002 \
    -v "$(realpath triton_models)":/models \
    --name srgb-local-triton \
    nvcr.io/nvidia/tritonserver:24.01-py3 \
    tritonserver --model-repository=/models \
    --model-control-mode=explicit \
    --load-model RealESRGAN_x4plus
```

Health check.
```bash
curl -v localhost:8001/v2/health/ready
```

Metrics.
```bash
curl localhost:8003/metrics
```

### Гайды
- [Deploy PyTorch](https://github.com/triton-inference-server/tutorials/tree/main/Quick_Deploy/PyTorch).
- [Deploy ONNX](https://github.com/triton-inference-server/tutorials/blob/main/Quick_Deploy/ONNX/README.md).
- [Deploy TensorRT](https://github.com/NVIDIA/TensorRT/tree/main/quickstart/deploy_to_triton).
- [Performance Analyzer](https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton-inference-server-2310/user-guide/docs/user_guide/perf_analyzer.html).

### Performance Analyzer
Запуск контейнера.
```bash
docker run --gpus 0 \
    --rm \
    -it \
    --network host \
    nvcr.io/nvidia/tritonserver:24.01-py3-sdk
```

Запуск измерения скорости `inference`.
```bash
perf_analyzer -m RealESRGAN_x4plus \
  -u "localhost:8001" \
  --shape lr:1,3,270,480 \
  --input-tensor-format "binary" \
  --output-tensor-format "binary" \
  --measurement-interval 5000 msec
```

## Время inference
Средние показатели `inference` для одного изображения (с учетом `preprocessing` и `postprocessing`) на `NVIDIA Geforce GTX 1660 Super 6Gb`.

### RealESRGAN_x4plus

|      TYPE       |  LR  |  HR   | SPEED, sec | INCREASE, % |
|:---------------:|:----:|:-----:|:----------:|:-----------:|
|  PyTorch CUDA   | 270p | 1080p |   1.070    |      -      |
|    ONNX CUDA    | 270p | 1080p |   1.072    |      0      |
|  ONNX TensorRT  | 270p | 1080p |   0.947    |     11      |
| Triton TensorRT | 270p | 1080p |   0.563    |     47      |
