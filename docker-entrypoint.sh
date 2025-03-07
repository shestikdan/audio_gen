#!/bin/bash
set -e

echo "=== Начало инициализации контейнера ==="

# Проверка переменной окружения для поддержки PyTorch
if [ "$PYTORCH_WEIGHTS_ONLY" != "0" ]; then
  echo "⚠️ ВНИМАНИЕ: PYTORCH_WEIGHTS_ONLY не установлен в 0. Устанавливаем принудительно."
  export PYTORCH_WEIGHTS_ONLY=0
fi

# Применяем патч к скрипту setup_xtts.py для совместимости с PyTorch
echo "* Применяем патч для setup_xtts.py..."
sed -i '1s/^/import sys\nsys.path.insert(0, "\/app")\nimport torch_patch  # применяем патч PyTorch\n/' scripts/setup_xtts.py

# Устанавливаем необходимые переменные окружения
echo "* Настройка переменных окружения..."
export PYTORCH_WEIGHTS_ONLY=0
export TTS_CHECKPOINT_CONFIG_COMPAT=1

# Проверка доступной памяти
echo "* Проверка доступной памяти..."
free -h
echo "Рекомендуется минимум 4GB RAM для стабильной работы."

# Проверка переменных окружения
if [ "$SKIP_XTTS_DOWNLOAD" = "0" ]; then
  echo "* Загрузка моделей XTTS..."
  
  # Пробуем установить pyloudnorm, если его нет
  if ! python -c "import pyloudnorm" 2>/dev/null; then
    echo "* Устанавливаем отсутствующую зависимость pyloudnorm..."
    pip install pyloudnorm
  fi
  
  # Запускаем скрипт настройки с применением патча
  python scripts/setup_xtts.py || {
    echo "Предупреждение: загрузка моделей XTTS не удалась."
    echo "Пробуем альтернативный метод установки..."
    
    # Альтернативный метод загрузки модели
    python -c "
import torch, sys
from pathlib import Path

# Монкипатчим torch.load
original_torch_load = torch.load
torch.load = lambda f, *args, **kwargs: original_torch_load(f, weights_only=False, *args, **kwargs)
print('Применен монкипатч для torch.load')

try:
    from TTS.api import TTS
    print('Пытаемся загрузить модель XTTS...')
    tts = TTS('tts_models/multilingual/multi-dataset/xtts_v2', progress_bar=True)
    print('✅ Модель TTS успешно загружена и инициализирована!')
except Exception as e:
    print(f'❌ Не удалось загрузить модель: {e}')
    sys.exit(1)
" || echo "❌ Предупреждение: альтернативная загрузка также не удалась."
  }
else
  echo "* Пропуск загрузки моделей XTTS (SKIP_XTTS_DOWNLOAD=$SKIP_XTTS_DOWNLOAD)"
fi

# Проверка наличия моделей
if [ -f "/app/silero_model.pt" ]; then
  echo "* ✅ Найдена модель Silero"
else
  echo "* ⚠️ Внимание: модель Silero не найдена в /app/silero_model.pt"
fi

echo "=== Инициализация контейнера завершена ==="

# Запуск переданной команды с правильными переменными окружения
export PYTORCH_WEIGHTS_ONLY=0
export TTS_CHECKPOINT_CONFIG_COMPAT=1
exec "$@" 