#!/bin/bash
set -e

echo "=== Начало инициализации контейнера ==="

# Проверка переменных окружения
if [ "$SKIP_XTTS_DOWNLOAD" = "0" ]; then
  echo "* Загрузка моделей XTTS..."
  python scripts/setup_xtts.py || {
    echo "Предупреждение: загрузка моделей XTTS не удалась, но контейнер продолжит работу."
  }
else
  echo "* Пропуск загрузки моделей XTTS (SKIP_XTTS_DOWNLOAD=$SKIP_XTTS_DOWNLOAD)"
fi

# Проверка наличия моделей
if [ -f "/app/silero_model.pt" ]; then
  echo "* Найдена модель Silero"
else
  echo "* Внимание: модель Silero не найдена в /app/silero_model.pt"
fi

echo "=== Инициализация контейнера завершена ==="

# Запуск переданной команды
exec "$@" 