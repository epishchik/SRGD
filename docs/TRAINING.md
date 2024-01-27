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
