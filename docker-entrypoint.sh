#!/bin/bash
set -e

echo "=== Начало инициализации контейнера ==="

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
sed -i '1s/^/import sys\nsys.path.insert(0, "\/app")\ntry:\n    import torch_patch  # применяем патч PyTorch\nexcept Exception as e:\n    print(f"Предупреждение: не удалось применить патч PyTorch: {e}")\n/' scripts/setup_xtts.py

# Устанавливаем необходимые переменные окружения
echo "* Настройка переменных окружения..."
export PYTORCH_WEIGHTS_ONLY=0
export TTS_CHECKPOINT_CONFIG_COMPAT=1

# Проверка доступной памяти
echo "* Проверка доступной памяти..."
free -h
echo "Рекомендуется минимум 4GB RAM для стабильной работы."

# Проверка наличия TTS и установка при необходимости
if ! python -c "import TTS" 2>/dev/null; then
  echo "* Устанавливаем TTS..."
  pip install --no-cache-dir TTS==0.17.0
fi

# Проверка переменных окружения для XTTS
if [ "$SKIP_XTTS_DOWNLOAD" = "0" ]; then
  echo "* Загрузка моделей XTTS..."
  
  # Пробуем установить pyloudnorm, если его нет
  if ! python -c "import pyloudnorm" 2>/dev/null; then
    echo "* Устанавливаем отсутствующую зависимость pyloudnorm..."
    pip install pyloudnorm
  fi
  
  # Создаем простой скрипт для загрузки модели
  echo "import torch, sys
try:
    from TTS.api import TTS
    print('Загрузка модели XTTS...')
    # Устанавливаем weights_only=False
    original_torch_load = torch.load
    torch.load = lambda f, *args, **kwargs: original_torch_load(f, weights_only=False, *args, **kwargs)
    
    # Загружаем модель
    tts = TTS('tts_models/multilingual/multi-dataset/xtts_v2')
    print('✅ Модель XTTS успешно загружена!')
except Exception as e:
    print(f'❌ Ошибка при загрузке модели: {e}')
    sys.exit(1)
" > /app/load_model.py
  
  # Запускаем скрипт загрузки модели
  python /app/load_model.py || {
    echo "⚠️ Не удалось загрузить модель XTTS. Пробуем альтернативный путь..."
    # Запускаем скрипт настройки
    python scripts/setup_xtts.py || echo "❌ Все попытки загрузки модели не удались."
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