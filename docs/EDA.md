# EDA

## Данные
Данные представляют собой рендеры изображений напрямую с игровых движков в различных разрешениях: 270p, 360p, 540p и 1080p.

Важно получать именно данные с движка в разных разрешениях и с разным качеством, чтобы данные соответствовали реальному распределению.

Проблемы:
- Синхронизация кадров во времени.
- Особенности рендеринга (Motion Blur, Constrain Ratio).
- Баги игровых движков (различия физики, течения времени, теней между рендерами).

Игровые движки:
- Unreal Engine.
- Source.

## Типы данных
Данные представлены двух видов:
- Реальные: для каждого разрешения рендеринг происходит на движке, для более низких разрешений выставляются более низкие настройки графики.
- Синтетические: 1080p рендерится на движке, а 270p, 360p и 540p генерируются алгоритмом из статьи [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN). Скрипт генерации: [compress.py](/utils/compress.py).

## Unreal Engine

### Проекты
Было обработано 17 бесплатных проектов:
- [ActionRPG](https://www.unrealengine.com/marketplace/en-US/product/action-rpg).
- [ArchVizInterior](https://www.unrealengine.com/marketplace/en-US/product/archvis-interior-rendering).
- [ASCTeuthisan](https://www.unrealengine.com/marketplace/en-US/product/asc-teuthisan).
- [BroadcastSample](https://www.unrealengine.com/marketplace/en-US/product/broadcast-sample).
- [CitySample](https://www.unrealengine.com/marketplace/en-US/product/city-sample).
- [ElectricDreamsEnv](https://www.unrealengine.com/marketplace/en-US/product/electric-dreams-env).
- [ElementalDemo](https://www.unrealengine.com/marketplace/en-US/product/elemental-demo).
- [HillsideSample](https://www.unrealengine.com/marketplace/en-US/product/hillside-sample-project).
- [Matinee](https://www.unrealengine.com/marketplace/en-US/product/matinee).
- [MeerkatDemo](https://www.unrealengine.com/marketplace/en-US/product/meerkat-demo-05).
- [MLDeformerSample](https://www.unrealengine.com/marketplace/en-US/product/ml-deformer-sample).
- [ParticleEffects](https://www.unrealengine.com/marketplace/en-US/product/particle-effects).
- [RealisticRendering](https://www.unrealengine.com/marketplace/en-US/product/realistic-rendering).
- [SlayAnimationSample](https://www.unrealengine.com/marketplace/en-US/product/slay-animation-sample).
- [StylizedRendering](https://www.unrealengine.com/marketplace/en-US/product/stylized-rendering).
- [SubwaySequencer](https://www.unrealengine.com/marketplace/en-US/product/sequencer-subway).
- [SunTemple](https://www.unrealengine.com/marketplace/en-US/product/sun-temple).

### Конфигурация

Все проекты написаны в разных версиях, с разными конфигурациями, но получилось составить общий паттерн действий, чтобы избавиться от багов: блюра, смещения гаммы и т.д.

- В глобальных настройках проекта (Edit -> Project settings) найти параметр Motion Blur и отключить.
- Открыть Output Log (Window -> Dev Tools -> Output Log) и ввести в консоль два параметра: r.MotionBlur.Amount 0 и r.MotionBlur.Max 0.
- Выделить все камеры на сцене (поиск по слову camera в Outliner) и найти параметр Motion Blur в Details, ДАЖЕ ЕСЛИ ГАЛОЧКИ НЕ ВКЛЮЧЕНЫ их нужно включить и поставить Amount, Max в 0, а также найти параметр Constrain Ratio и выключить.
- Повторить предыдущий шаг для всех объектов Post Process на сцене.
- Записать / взять Level Sequence, открыть и проделать два предыдущих шага (это может и не понадобиться, если проект настроен определенным образом).
- Положить в любую папку конфиги для UE из [папки](/unreal).
- Открыть MRQ (Window -> Cinematics -> Movie Render Queue) и загрузить queue из папки с предыдущего шага.
- Запустить рендер и верить в успех.

### Объем данных

Общее количество изображений получилось ~ 72 тысячи или ~ 87 Gb, т.е. ~ 18 тысяч изображений в каждом из разрешений.

## Source

### Проекты
Был обработан 1 проект:
- [Dota 2](https://store.steampowered.com/app/570/Dota_2/).

### Конфигурация
В играх на движке Source есть встроенный механизм рендеринга через консоль.

- Открыть запись игры.
- Открыть консоль.
- Установить host_framerate 1.
- Запустить рендеринг startmovie name jpg jpeg_quality 100.
- Запустить воспроизведение записи игры.
- Выключить hud командой hud_toggle_visibility 0
- Закрыть консоль, чтобы ее не было видно на рендере.
- Остановить запись: endmovie.

### Объем данных

Общее количество изображений получилось ~ 28 тысяч или ~ 15 Gb, т.е. ~ 7 тысяч изображений в каждом из разрешений.

## Очистка данных

- Ручная: просмотр изображений и удаление рассинхрона, смещения, багов отрисовки, черных экранов.
- Синхронизация при помощи [align.py](/utils/align.py): удаляем баги из папки одного разрешения, например 270p, запускаем скрипт и он все файлы для которых нет пары удалит.

## Real-ESRGAN degradation

Адаптированная мной версия: [compress.py](/utils/compress.py).

Ниже приведена схема из оригинальной статьи [Real-ESRGAN](https://arxiv.org/abs/2107.10833).

![degradation](/pictures/real-esrgan-degradation.png)
