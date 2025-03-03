# Lection to Audio Converter

Система для конвертации текстовых лекций в аудио формат с использованием XTTS.

## Системные требования

- Docker и Docker Compose
- Минимум 8GB RAM (рекомендуется 16GB)
- NVIDIA GPU (опционально, для ускорения)
- 20GB свободного места на диске

## Быстрый старт

1. Клонируйте репозиторий:
```bash
git clone <repository-url>
cd lection_to_text
```

2. (Опционально) Если у вас есть GPU, установите NVIDIA Container Toolkit:
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

3. Запустите систему:
```bash
docker-compose up --build
```

## Структура проекта

- `scripts/` - скрипты для обработки
  - `lection_to_audio.py` - основной скрипт для конвертации
  - `setup_xtts.py` - скрипт настройки XTTS
  - `text_update_agent.py` - агент обновления текста
- `samples/` - аудио файлы и образцы голосов
- `lections_text_base/` - директория с исходными текстами
- `lections_text_mistral/` - директория с обработанными текстами
- `lections_audio/` - директория с готовыми аудио файлами

## Использование

1. Поместите текстовые файлы в директорию `lections_text_base/`
2. (Опционально) Добавьте образцы голосов в `samples/`
3. (Опционально) Добавьте фоновую музыку в `samples/`
4. Запустите контейнер:
```bash
docker-compose up
```

## Конфигурация

Основные настройки можно изменить через переменные окружения:

```bash
# .env
CUDA_VISIBLE_DEVICES=0  # Использовать конкретный GPU
```

## Устранение неполадок

1. Если возникает ошибка памяти:
   - Увеличьте размер swap
   - Уменьшите размер батча в настройках

2. Если возникают проблемы с GPU:
   - Убедитесь, что установлен NVIDIA Container Toolkit
   - Проверьте драйверы NVIDIA

## Лицензия

MIT 