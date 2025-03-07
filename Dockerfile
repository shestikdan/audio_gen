FROM python:3.10-slim

# Установка переменных среды
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV COQUI_TOS_AGREED=1
ENV NNPACK_IGNORE=1
ENV FORCE_CPU=1
# Предотвращаем загрузку модели во время сборки
ENV SKIP_XTTS_DOWNLOAD=1

# Установка системных зависимостей с очисткой кэша в том же слое
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Создание рабочей директории
WORKDIR /app

# Оптимизация слоев кеша - разделяем установку зависимостей
COPY requirements.txt .

# Установка минимальных необходимых зависимостей с очисткой кэша
RUN pip install --no-cache-dir -r requirements.txt && \
    pip cache purge && \
    rm -rf /root/.cache/pip

# Установка TTS из GitHub для обеспечения совместимости
RUN pip install --no-cache-dir git+https://github.com/coqui-ai/TTS && \
    pip cache purge

# Создание необходимых директорий
RUN mkdir -p lections_text_base lections_text_mistral lections_audio samples model_cache

# Копирование проекта (за исключением больших файлов)
COPY scripts/ ./scripts/
COPY README.md ./
# Создаем пустой .env файл, если он не существует
RUN touch .env

# Настройка прав доступа
RUN chmod +x scripts/setup_xtts.py scripts/lection_to_audio.py

# Создание пользовательских директорий для кеша моделей
RUN mkdir -p /root/.cache/huggingface /root/.local/share/tts

# Создаем точку входа для инициализации
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Очистка временных файлов для уменьшения размера образа
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip cache purge && \
    rm -rf /tmp/* /var/tmp/*

# Используем entrypoint для инициализации при запуске
ENTRYPOINT ["/docker-entrypoint.sh"]

# Выполнение скрипта по умолчанию
CMD ["python", "scripts/lection_to_audio.py"] 