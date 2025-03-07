#!/bin/bash
set -e

echo "=== Начало инициализации контейнера ==="

# Создаем заглушку для distutils.msvccompiler для обхода ошибки
mkdir -p /tmp/msvccompiler_fix/distutils
echo "def get_build_version(): return ''" > /tmp/msvccompiler_fix/distutils/msvccompiler.py
echo "from distutils.msvccompiler import get_build_version  # заглушка" > /tmp/msvccompiler_fix/__init__.py
export PYTHONPATH=/tmp/msvccompiler_fix:$PYTHONPATH

# Проверка и обновление PIP и базовых пакетов
echo "* Проверка и обновление базовых пакетов..."
pip install --no-cache-dir --upgrade pip setuptools wheel

# Проверка и установка критических зависимостей
echo "* Проверка критических зависимостей..."
pip install --no-cache-dir numpy==1.22.0 torch==2.5.0 torchaudio==2.5.0 pyloudnorm

# Проверка переменной окружения для поддержки PyTorch
if [ "$PYTORCH_WEIGHTS_ONLY" != "0" ]; then
  echo "⚠️ ВНИМАНИЕ: PYTORCH_WEIGHTS_ONLY не установлен в 0. Устанавливаем принудительно."
  export PYTORCH_WEIGHTS_ONLY=0
fi

# Применяем патч к скрипту setup_xtts.py для совместимости с PyTorch
echo "* Применяем патч для setup_xtts.py..."
sed -i '1s/^/import sys\nsys.path.insert(0, "\/app")\ntry:\n    import torch_patch  # применяем патч PyTorch\n    import fake_tts  # загружаем заглушку TTS\nexcept Exception as e:\n    print(f"Предупреждение: не удалось применить патч: {e}")\n/' scripts/setup_xtts.py

# Устанавливаем необходимые переменные окружения
echo "* Настройка переменных окружения..."
export PYTORCH_WEIGHTS_ONLY=0
export TTS_CHECKPOINT_CONFIG_COMPAT=1

# Проверка доступной памяти
echo "* Проверка доступной памяти..."
free -h
echo "Рекомендуется минимум 4GB RAM для стабильной работы."

# Попытка загрузки силеро-модели
if [ -f "/app/silero_model.pt" ]; then
  echo "* ✅ Найдена модель Silero"
else
  echo "* ⚠️ Внимание: модель Silero не найдена в /app/silero_model.pt. Пробуем загрузить демо-модель..."
  mkdir -p /root/.cache/torch/hub/snakers4_silero-models_master/
  wget -q -O /app/silero_model.pt https://models.silero.ai/models/tts/ru/v4_ru.pt || echo "Не удалось загрузить модель Silero"
fi

# Попытка установки TTS из wheel
echo "* Попытка установки TTS из предварительно скомпилированного пакета..."
TTS_INSTALLED=0

# Сначала пробуем wheel для Linux
mkdir -p /tmp/wheels
cd /tmp/wheels
wget -q -O TTS-0.16.0-py3-none-any.whl https://files.pythonhosted.org/packages/8b/7f/cd31b87d57f6f7c17adf7bac96eedd099e2b71b42c7eece2fb12e2fcf607/TTS-0.16.0-py3-none-any.whl || true

if [ -f "TTS-0.16.0-py3-none-any.whl" ]; then
  echo "* Устанавливаем TTS из скачанного wheel-файла..."
  pip install --no-deps TTS-0.16.0-py3-none-any.whl && TTS_INSTALLED=1
  
  if [ $TTS_INSTALLED -eq 1 ]; then
    echo "* ✅ TTS установлен из wheel-файла"
    export REAL_TTS_AVAILABLE=1
  fi
fi

# Если не удалось, пробуем альтернативные методы
if [ $TTS_INSTALLED -eq 0 ]; then
  echo "* Пробуем установить только необходимые зависимости TTS..."
  pip install --no-cache-dir pydub scipy soundfile librosa unidic-lite phonemizer && {
    echo "* ✅ Базовые зависимости TTS установлены"
    
    # Создаем минимальную структуру папок для моделей
    mkdir -p /root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2
  }
fi

# Проверка переменных окружения для XTTS
if [ "$SKIP_XTTS_DOWNLOAD" = "0" ]; then
  echo "* Загрузка моделей XTTS..."
  
  # Пробуем установить pyloudnorm, если его нет
  if ! python -c "import pyloudnorm" 2>/dev/null; then
    echo "* Устанавливаем отсутствующую зависимость pyloudnorm..."
    pip install pyloudnorm
  fi
  
  # Проверяем наличие директории для моделей
  mkdir -p /root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2
  
  # Создаем скрипт для загрузки модели
  cat > /app/download_model.py << 'EOL'
#!/usr/bin/env python3
import os
import sys
import torch
import requests
from pathlib import Path
from tqdm import tqdm

# Патч для torch.load
original_torch_load = torch.load
torch.load = lambda f, *args, **kwargs: original_torch_load(f, weights_only=False, *args, **kwargs)

# Создаем нужные директории
os.makedirs("/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2", exist_ok=True)

print("✓ Начинаем ручную загрузку модели XTTS v2")

# Файлы для загрузки
model_files = [
    "config.json",
    "model_file.pth",
    "vocab.json",
    "speakers_map.json"
]

# Mirrors для моделей
mirrors = [
    "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/",
    "https://huggingface.co/coqui/XTTS-v2/resolve/main/",
    "https://github.com/coqui-ai/TTS/raw/main/models/multilingual/multi-dataset/xtts_v2/"
]

for file in model_files:
    target_path = f"/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/{file}"
    
    if os.path.exists(target_path):
        print(f"✓ Файл {file} уже существует, пропускаем")
        continue
    
    downloaded = False
    for mirror in mirrors:
        if downloaded:
            break
            
        print(f"⬇️ Загрузка {file} из {mirror}...")
        try:
            response = requests.get(f"{mirror}{file}", stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024
            
            with open(target_path, 'wb') as f, tqdm(
                    total=total_size, unit='B', unit_scale=True, unit_divisor=1024,
                    desc=file) as pbar:
                for data in response.iter_content(block_size):
                    f.write(data)
                    pbar.update(len(data))
                    
            print(f"✓ Файл {file} успешно загружен")
            downloaded = True
        except Exception as e:
            print(f"❌ Ошибка при загрузке {file} из {mirror}: {e}")
            continue
            
    if not downloaded:
        print(f"❌ Не удалось загрузить {file} ни из одного источника")

if all(os.path.exists(f"/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/{file}") for file in model_files):
    print("✓ Все файлы модели XTTS успешно загружены")
else:
    print("⚠️ Некоторые файлы модели XTTS не были загружены")
EOL
  
  # Запускаем скрипт загрузки модели
  python /app/download_model.py || {
    echo "⚠️ Не удалось загрузить модель XTTS вручную. Пробуем setup_xtts.py..."
    
    # Создаем фиктивный файл model_info.json, если модель не удалось скачать
    if [ ! -f "/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/model_file.pth" ]; then
      echo '{"description": "XTTS v2 dummy model", "language": ["ru"], "name": "xtts_v2"}' > "/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/model_info.json"
      touch "/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/model_file.pth"
      echo "⚠️ Создан фиктивный файл модели для обхода ошибок"
    fi
  }
else
  echo "* Пропуск загрузки моделей XTTS (SKIP_XTTS_DOWNLOAD=$SKIP_XTTS_DOWNLOAD)"
fi

echo "=== Инициализация контейнера завершена ==="

# Запуск переданной команды с правильными переменными окружения
export PYTORCH_WEIGHTS_ONLY=0
export TTS_CHECKPOINT_CONFIG_COMPAT=1
exec "$@" 