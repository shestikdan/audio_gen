FROM python:3.10-slim

# Установка переменных среды
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV COQUI_TOS_AGREED=1
ENV NNPACK_IGNORE=1
ENV FORCE_CPU=1

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Создание рабочей директории
WORKDIR /app

# Копирование requirements и установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Установка TTS из GitHub для обеспечения совместимости
RUN pip install --no-cache-dir git+https://github.com/coqui-ai/TTS

# Создание необходимых директорий
RUN mkdir -p lections_text_base lections_text_mistral lections_audio samples model_cache

# Копирование всего проекта
COPY . .

# Настройка прав доступа
RUN chmod +x scripts/setup_xtts.py scripts/lection_to_audio.py

# Создание пользовательских директорий для кеша моделей
RUN mkdir -p /root/.cache/huggingface /root/.local/share/tts

# Установка контрольных точек для TTS
# Этот шаг будет пропущен если SKIP_XTTS_DOWNLOAD=1
RUN if [ "$SKIP_XTTS_DOWNLOAD" != "1" ]; then python scripts/setup_xtts.py; fi

# Выполнение скрипта по умолчанию
CMD ["python", "scripts/lection_to_audio.py"] 