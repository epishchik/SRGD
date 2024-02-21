# Оптимизация inference

## Конвертация в onnx

Все скрипты конвертации лежат в папке [optimization/onnx](/optimization/onnx).

Пример конвертации.
```bash
cd optimization/onnx
python3 real_esrgan.py \
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

Сам по себе onnx с CUDAExecutionProvider ускорения не дал, поэтому установим TensorRT для использования TensorrtExecutionProvider в onnxruntime.
[Инструкция по установке TensorRT.](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)

## Время inference
Средние показатели inference для одного изображения на NVIDIA Geforce GTX 1660 Super 6Gb.

### RealESRGAN_x4plus

|     TYPE      |  LR  |  HR   | SPEED, sec  | INCREASE, % |
|:-------------:|:----:|:-----:|:-----------:|:-----------:|
|     CUDA      | 270p | 1080p |     1.1     |      -      |
|   ONNX CUDA   | 270p | 1080p |     1.1     |      0      |
| ONNX TensorRT | 270p | 1080p |     0.9     |     18      |
