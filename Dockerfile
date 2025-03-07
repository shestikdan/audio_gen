FROM python:3.10-slim

# Установка переменных среды
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV COQUI_TOS_AGREED=1
ENV NNPACK_IGNORE=1
ENV FORCE_CPU=1
# Предотвращаем загрузку модели во время сборки
ENV SKIP_XTTS_DOWNLOAD=1

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Создание рабочей директории
WORKDIR /app

# Оптимизация слоев кеша - разделяем установку зависимостей
COPY requirements.txt .

# Установка минимальных необходимых зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Установка TTS из GitHub для обеспечения совместимости
RUN pip install --no-cache-dir git+https://github.com/coqui-ai/TTS

# Создание необходимых директорий
RUN mkdir -p lections_text_base lections_text_mistral lections_audio samples model_cache

# Копирование всего проекта (за исключением больших файлов)
COPY scripts/ ./scripts/
COPY .env ./
COPY README.md ./

# Настройка прав доступа
RUN chmod +x scripts/setup_xtts.py scripts/lection_to_audio.py

# Создание пользовательских директорий для кеша моделей
RUN mkdir -p /root/.cache/huggingface /root/.local/share/tts

# Создаем точку входа для инициализации
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Используем entrypoint для инициализации при запуске
ENTRYPOINT ["/docker-entrypoint.sh"]

# Выполнение скрипта по умолчанию
CMD ["python", "scripts/lection_to_audio.py"] 