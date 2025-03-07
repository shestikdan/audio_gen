FROM python:3.10-slim

# Установка переменных среды
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV COQUI_TOS_AGREED=1
ENV NNPACK_IGNORE=1
ENV FORCE_CPU=1
# Предотвращаем загрузку модели во время сборки
ENV SKIP_XTTS_DOWNLOAD=1
# Важно! Обход проблемы совместимости PyTorch 2.6
ENV PYTORCH_WEIGHTS_ONLY=0

# Установка системных зависимостей с очисткой кэша в том же слое
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    wget \
    build-essential \
    python3-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Создание рабочей директории
WORKDIR /app

# Обновление pip для лучшей совместимости
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Предустановка NumPy (заранее для предотвращения проблем)
RUN pip install --no-cache-dir numpy==1.22.0

# Оптимизация слоев кеша - разделяем установку зависимостей
COPY requirements.txt .

# Установка минимальных необходимых зависимостей с очисткой кэша
RUN pip install --no-cache-dir -r requirements.txt && \
    pip cache purge && \
    rm -rf /root/.cache/pip

# Создаем патч для исправления проблемы с PyTorch
RUN echo 'import sys\nimport torch\n\n# Патч для функции torch.load\noriginal_torch_load = torch.load\ndef patched_torch_load(f, *args, **kwargs):\n    # Всегда устанавливаем weights_only=False\n    if "weights_only" not in kwargs:\n        kwargs["weights_only"] = False\n    return original_torch_load(f, *args, **kwargs)\n\n# Заменяем оригинальную функцию на нашу патченную версию\ntorch.load = patched_torch_load\n\ntry:\n    # Добавляем безопасные классы для десериализации\n    torch.serialization._get_safe_globals().add("TTS.tts.configs.xtts_config.XttsConfig")\n    print("✅ PyTorch патч для torch.load применен успешно")\nexcept Exception as e:\n    print(f"⚠️ Не удалось настроить безопасные глобалы: {e}, но продолжаем работу")' > /app/torch_patch.py

# Установка TTS через pip (более стабильный метод)
RUN pip install --no-cache-dir TTS==0.17.0 && \
    pip cache purge

# Создание необходимых директорий
RUN mkdir -p lections_text_base lections_text_mistral lections_audio samples model_cache

# Копирование проекта (за исключением больших файлов)
COPY scripts/ ./scripts/
COPY README.md ./
# Создаем пустой .env файл, если он не существует
RUN touch .env

# Патчим основной скрипт для применения исправления PyTorch
RUN sed -i '1s/^/import sys\nsys.path.insert(0, "\/app")\ntry:\n    import torch_patch  # применяем патч PyTorch\nexcept Exception as e:\n    print(f"Предупреждение: не удалось применить патч PyTorch: {e}")\n/' scripts/lection_to_audio.py

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