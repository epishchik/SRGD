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
