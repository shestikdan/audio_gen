#!/bin/bash

# Очистка Docker
clean_docker() {
  echo "=== Очистка Docker ==="
  docker-compose down -v 2>/dev/null || true
  docker rmi $(docker images -q audio_gen_lection_to_audio) 2>/dev/null || true
  docker system prune -f
  echo "✅ Docker очищен"
}

# Клонирование или обновление репозитория
update_repo() {
  echo "=== Обновление репозитория ==="
  if [ -d ".git" ]; then
    git pull origin main
    echo "✅ Репозиторий обновлен"
  else
    echo "❌ Не найден .git каталог. Убедитесь, что вы в корне проекта."
    exit 1
  fi
}

# Создание необходимых директорий
setup_dirs() {
  echo "=== Создание директорий ==="
  mkdir -p lections_text_base lections_text_mistral lections_audio samples model_cache tts_models
  echo "✅ Директории созданы"
}

# Сборка образа (без запуска контейнера)
build_image() {
  echo "=== Сборка Docker образа ==="
  # Устанавливаем SKIP_XTTS_DOWNLOAD=1 для пропуска загрузки модели при сборке
  export SKIP_XTTS_DOWNLOAD=1
  docker-compose build --no-cache
  echo "✅ Образ собран"
}

# Запуск контейнера
run_container() {
  echo "=== Запуск контейнера ==="
  # Запускаем с SKIP_XTTS_DOWNLOAD=0 для загрузки модели при запуске
  docker-compose up -d
  echo "✅ Контейнер запущен"
}

# Отображение логов
show_logs() {
  echo "=== Отображение логов ==="
  docker-compose logs -f
}

# Очистка системы для освобождения места на диске
clean_system() {
  echo "=== Очистка системы для освобождения места ==="
  
  # Очистка apt кэша
  sudo apt-get clean
  sudo apt-get autoremove -y
  
  # Удаление старых логов
  sudo find /var/log -type f -name "*.gz" -delete 2>/dev/null || true
  sudo find /var/log -type f -name "*.1" -delete 2>/dev/null || true
  
  # Глубокая очистка Docker
  docker system prune -a --volumes -f
  
  echo "✅ Система очищена"
}

# Главное меню
menu() {
  clear
  echo "=== МЕНЮ ДЕПЛОЯ ==="
  echo "1. Обновить репозиторий (git pull)"
  echo "2. Очистить Docker (рекомендуется при проблемах)"
  echo "3. Создать нужные директории"
  echo "4. Собрать Docker образ (без запуска контейнера)"
  echo "5. Запустить контейнер"
  echo "6. Показать логи"
  echo "7. Выполнить полный процесс развертывания (шаги 1-5)"
  echo "8. Очистить систему (освободить место на диске)"
  echo "9. Выход"
  echo
  read -p "Выберите действие (1-9): " choice

  case $choice in
    1) update_repo; read -p "Нажмите Enter для продолжения..."; menu ;;
    2) clean_docker; read -p "Нажмите Enter для продолжения..."; menu ;;
    3) setup_dirs; read -p "Нажмите Enter для продолжения..."; menu ;;
    4) build_image; read -p "Нажмите Enter для продолжения..."; menu ;;
    5) run_container; read -p "Нажмите Enter для продолжения..."; menu ;;
    6) 
       show_logs
       # После выхода из логов (Ctrl+C) возвращаемся в меню
       read -p "Логи закрыты. Нажмите Enter для возврата в меню..."
       menu 
       ;;
    7) 
       update_repo
       clean_docker
       setup_dirs
       build_image
       run_container
       read -p "Нажмите Enter для продолжения..."; menu 
       ;;
    8) clean_system; read -p "Нажмите Enter для продолжения..."; menu ;;
    9) exit 0 ;;
    *) echo "Неверный выбор"; read -p "Нажмите Enter для продолжения..."; menu ;;
  esac
}

# Запуск меню
menu 