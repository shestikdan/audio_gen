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
# Используем только Silero для синтеза речи
ENV USE_SILERO_ONLY=1
ENV REAL_TTS_AVAILABLE=0

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
    procps \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Создание рабочей директории
WORKDIR /app

# Обновление pip для лучшей совместимости
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Модифицируем requirements.txt локально
COPY requirements.txt .
RUN sed -i '/^TTS/d' requirements.txt && \
    sed -i '/^# TTS/d' requirements.txt

# Предустановка NumPy (заранее для предотвращения проблем)
RUN pip install --no-cache-dir numpy==1.22.0

# Установка минимальных необходимых зависимостей с очисткой кэша
RUN pip install --no-cache-dir -r requirements.txt && \
    pip cache purge && \
    rm -rf /root/.cache/pip

# Установка зависимостей для Silero TTS
RUN pip install --no-cache-dir soundfile omegaconf && \
    pip cache purge

# Создаем патч для исправления проблемы с PyTorch
RUN echo 'import sys\nimport torch\n\n# Патч для функции torch.load\noriginal_torch_load = torch.load\ndef patched_torch_load(f, *args, **kwargs):\n    # Всегда устанавливаем weights_only=False\n    if "weights_only" not in kwargs:\n        kwargs["weights_only"] = False\n    return original_torch_load(f, *args, **kwargs)\n\n# Заменяем оригинальную функцию на нашу патченную версию\ntorch.load = patched_torch_load\n\ntry:\n    # Пробуем настроить безопасные классы (на случай если TTS будет установлен позже)\n    import importlib.util\n    if importlib.util.find_spec("TTS") is not None:\n        import torch.serialization\n        torch.serialization._get_safe_globals().add("TTS.tts.configs.xtts_config.XttsConfig")\n        print("✅ PyTorch патч для torch.load применен успешно")\nexcept Exception as e:\n    print(f"⚠️ Не удалось настроить безопасные глобалы: {e}, но продолжаем работу")' > /app/torch_patch.py

# Создаем патч для исправления проблемы с PyTorch
RUN echo 'import os\nimport sys\nimport warnings\nimport numpy as np\nfrom pathlib import Path\n\nclass FakeTTS:\n    def __init__(self, model_name=None, **kwargs):\n        self.model_name = model_name\n        self.device = "cpu"\n        print(f"⚠️ Fake TTS initialized with model: {model_name}")\n        warnings.warn("Using FakeTTS implementation. Real TTS not available.")\n    \n    def to(self, device):\n        """Имитирует перемещение модели на устройство (CPU/GPU)"""\n        print(f"⚠️ FakeTTS: имитация перемещения на устройство {device}")\n        self.device = device\n        return self\n    \n    def synthesize(self, text, **kwargs):\n        """Имитирует синтез речи, возвращает тишину"""\n        print(f"⚠️ FakeTTS: имитация синтеза: {text[:50]}...")\n        # Возвращает 1 секунду тишины (16кГц, моно)\n        return np.zeros(16000, dtype=np.float32)\n    \n    def tts(self, text, **kwargs):\n        """Имитирует синтез речи, возвращает тишину"""\n        print(f"⚠️ FakeTTS: имитация синтеза tts(): {text[:50]}...")\n        # Возвращает 1 секунду тишины (16кГц, моно)\n        return np.zeros(16000, dtype=np.float32)\n    \n    def tts_to_file(self, text, output_file, **kwargs):\n        """Имитирует сохранение аудио в файл"""\n        print(f"⚠️ FakeTTS: сохранение в файл: {output_file}")\n        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)\n        # Создаем минимальный WAV-файл\n        try:\n            import wave\n            import struct\n            # Создаем 1 секунду тишины 16кГц\n            duration = 1  # seconds\n            framerate = 16000  # Hz\n            amplitude = 0  # тишина\n            nframes = duration * framerate\n            with wave.open(output_file, "w") as wav_file:\n                wav_file.setparams((1, 2, framerate, nframes, "NONE", "not compressed"))\n                for i in range(nframes):\n                    value = int(amplitude)\n                    packed_value = struct.pack("<h", value)\n                    wav_file.writeframes(packed_value)\n            print(f"✅ Создан пустой WAV-файл: {output_file}")\n        except Exception as e:\n            print(f"❌ Ошибка при создании WAV: {e}")\n            # Запасной вариант, если не получилось создать WAV\n            with open(output_file, "wb") as f:\n                f.write(b"RIFF\\x24\\x00\\x00\\x00WAVEfmt \\x10\\x00\\x00\\x00\\x01\\x00\\x01\\x00\\x80>\\x00\\x00\\x00}\\x00\\x00\\x02\\x00\\x10\\x00data\\x00\\x00\\x00\\x00")\n        return output_file\n\n    # Добавляем другие методы, которые могут использоваться в скриптах\n    def __call__(self, *args, **kwargs):\n        return self.synthesize(*args, **kwargs)\n    \n    @staticmethod\n    def list_models():\n        return ["fake_tts_model"]\n    \n    def is_multi_speaker(self):\n        return False\n    \n    def get_speaker_ids(self):\n        return ["dummy_speaker"]\n    \n    def get_phonemizer(self):\n        return None\n\n# Если нет настоящего TTS, создаем фейковый модуль\nif not os.environ.get("REAL_TTS_AVAILABLE", ""):\n    print("Creating fake TTS API module...")\n    class FakeTTSModule:\n        TTS = FakeTTS\n        \n        @staticmethod\n        def list_models():\n            return ["tts_models/multilingual/multi-dataset/xtts_v2"]\n    \n    # Создаем fake TTS.api модуль\n    sys.modules["TTS"] = FakeTTSModule\n    sys.modules["TTS.api"] = FakeTTSModule\n' > /app/fake_tts.py

# Создание необходимых директорий
RUN mkdir -p lections_text_base lections_text_mistral lections_audio samples model_cache

# Копирование проекта (за исключением больших файлов)
COPY scripts/ ./scripts/
COPY README.md ./
# Создаем пустой .env файл, если он не существует
RUN touch .env

# Патчим основной скрипт для применения исправления PyTorch
RUN sed -i '1s/^/import sys\nsys.path.insert(0, "\/app")\ntry:\n    import torch_patch  # применяем патч PyTorch\nexcept Exception as e:\n    print(f"Предупреждение: не удалось применить патч: {e}")\n/' scripts/lection_to_audio.py && \
    sed -i '1s/^/import sys\nsys.path.insert(0, "\/app")\ntry:\n    import torch_patch  # применяем патч PyTorch\nexcept Exception as e:\n    print(f"Предупреждение: не удалось применить патч: {e}")\n/' scripts/text_update_agent.py

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