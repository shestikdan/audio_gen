#!/bin/bash

# Проверка зависимостей
check_dependencies() {
  echo "Проверка зависимостей..."
  
  # Проверка Docker
  if ! command -v docker &> /dev/null; then
    echo "❌ Docker не установлен. Установите Docker перед запуском."
    exit 1
  fi
  
  # Проверка Docker Compose
  if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose не установлен. Установите Docker Compose перед запуском."
    exit 1
  fi
  
  echo "✅ Все зависимости установлены."
}

# Проверка наличия необходимых файлов и директорий
check_files() {
  echo "Проверка файлов и директорий..."
  
  # Проверка Dockerfile
  if [ ! -f "Dockerfile" ]; then
    echo "❌ Dockerfile не найден."
    exit 1
  fi
  
  # Проверка docker-compose.yml
  if [ ! -f "docker-compose.yml" ]; then
    echo "❌ docker-compose.yml не найден."
    exit 1
  fi
  
  # Проверка модели Silero
  if [ ! -f "silero_model.pt" ]; then
    echo "⚠️ silero_model.pt не найден. Проект может работать некорректно без этого файла."
  fi
  
  # Создание необходимых директорий
  mkdir -p lections_text_base lections_text_mistral lections_audio samples model_cache tts_models
  
  echo "✅ Файлы и директории проверены."
}

# Очистка Docker кэша и контейнеров
clean_docker() {
  echo "Очистка Docker..."
  
  # Остановка и удаление контейнеров проекта
  docker-compose down -v
  
  # Удаление образа
  docker rmi $(docker images -q lection_to_text_lection_to_audio) 2>/dev/null || true
  
  # Общая очистка Docker
  docker system prune -f
  
  echo "✅ Docker очищен."
}

# Сборка и запуск
build_and_run() {
  echo "Сборка и запуск проекта..."
  
  # Сборка образа
  docker-compose build --no-cache
  
  # Запуск контейнера
  docker-compose up -d
  
  echo "✅ Проект запущен. Проверьте логи: docker-compose logs -f"
}

# Главная функция
main() {
  echo "=== Настройка и запуск проекта ==="
  
  check_dependencies
  check_files
  
  read -p "Хотите очистить Docker перед запуском? (y/n): " clean
  if [ "$clean" = "y" ]; then
    clean_docker
  fi
  
  build_and_run
  
  echo "=== Настройка завершена ==="
}

# Запуск скрипта
main 