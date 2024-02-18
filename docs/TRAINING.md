# Обучение



## Датасет

### Варианты
- DataLoader через streaming API Huggingface.
- Локальный датасет скачанный с HuggingFace.

### Подготовка локального датасета
Можно подготовить два варианта локального датасета - GameEngineData и DownscaleData.
Подготовка датасета происходит для двух разрешений - низкого и высокого (например для 270p и 1080p).

```bash
cd utils
bash create_local_dataset.sh datasets GameEngineData 270p 1080p
```

Необходимо прописать пути к данному датасету и его train / val, lr / hr частям в выбранном конфиге для обучения.

## Обучение модели Real-ESRGAN

### RealESRGAN_x4plus

```bash
MLFLOW_S3_ENDPOINT_URL="http://0.0.0.0:9000" python3 real_esrgan.py \
  -opt configs/train/finetune_realesrgan_x4plus_game_engine.yaml
```

#### Пример графиков MLFlow

![real_esrgan_game_engine_data_steps=100k_bs=2_gt_crop=256_scale=4_losses](/images/train/real_esrgan_game_engine_data_steps=100k_bs=2_gt_crop=256_scale=4_losses.png)

![real_esrgan_game_engine_data_steps=100k_bs=2_gt_crop=256_scale=4_psnr](/images/train/real_esrgan_game_engine_data_steps=100k_bs=2_gt_crop=256_scale=4_psnr.png)

Графики остальных экспериментов можно найти в папке [images/train](/images/train).
