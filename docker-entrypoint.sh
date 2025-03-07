#!/bin/bash
set -e

echo "=== Начало инициализации контейнера ==="

# Устанавливаем флаг, что мы используем только Silero TTS
export USE_SILERO_ONLY=1
export REAL_TTS_AVAILABLE=0

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
sed -i '1s/^/import sys\nsys.path.insert(0, "\/app")\ntry:\n    import torch_patch  # применяем патч PyTorch\nexcept Exception as e:\n    print(f"Предупреждение: не удалось применить патч: {e}")\n/' scripts/setup_xtts.py

# Устанавливаем необходимые переменные окружения
echo "* Настройка переменных окружения..."
export PYTORCH_WEIGHTS_ONLY=0
export TTS_CHECKPOINT_CONFIG_COMPAT=1

# Проверка доступной памяти
echo "* Проверка доступной памяти..."
if command -v free &> /dev/null; then
  free -h
  echo "Рекомендуется минимум 4GB RAM для стабильной работы."
else
  echo "Команда 'free' недоступна. Пропускаем проверку памяти."
  echo "Рекомендуется минимум 4GB RAM для стабильной работы."
  
  # Альтернативная проверка памяти через /proc/meminfo
  if [ -f "/proc/meminfo" ]; then
    echo "Доступно приблизительно $(grep -i memtotal /proc/meminfo | awk '{print $2/1024/1024}') GB RAM"
  fi
fi

# Исправление проблемы с директорией silero_model.pt
if [ -d "/app/silero_model.pt" ]; then
  echo "* ⚠️ Обнаружена директория вместо файла: /app/silero_model.pt. Исправляем..."
  rm -rf /app/silero_model.pt
fi

# Установка и настройка Silero
echo "* Установка и настройка Silero модели..."
# Устанавливаем необходимые для Silero зависимости
pip install --no-cache-dir omegaconf

# Попытка загрузки силеро-модели
if [ -f "/app/silero_model.pt" ]; then
  echo "* ✅ Найдена модель Silero"
else
  echo "* Загрузка модели Silero..."
  mkdir -p /root/.cache/torch/hub/snakers4_silero-models_master/
  wget -q -O /app/silero_model.pt https://models.silero.ai/models/tts/ru/v4_ru.pt
  
  if [ -f "/app/silero_model.pt" ]; then
    echo "* ✅ Модель Silero успешно загружена"
  else
    echo "* ⚠️ Ошибка загрузки модели Silero. Пробуем альтернативный источник..."
    wget -q -O /app/silero_model.pt https://github.com/snakers4/silero-models/releases/download/v4_tts_models/ru_v4.pt
    
    if [ -f "/app/silero_model.pt" ]; then
      echo "* ✅ Модель Silero успешно загружена из альтернативного источника"
    else
      echo "* ❌ Не удалось загрузить модель Silero. Создаем заглушку..."
      echo "Заглушка Silero модели" > /app/silero_model.pt
    fi
  fi
fi

# Создаем обертку для использования Silero
echo "* Создание обертки для Silero TTS..."
cat > /app/silero_tts.py << 'EOL'
import os
import sys
import torch
import warnings
from pathlib import Path

print("🔄 Загрузка Silero TTS для синтеза речи...")

# Функция для инициализации модели Silero
def init_silero_model():
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_path = '/app/silero_model.pt'
        
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Модель не найдена: {model_path}")
            
        if os.path.getsize(model_path) < 1000000:  # Размер меньше 1MB? Вероятно, это не модель
            raise ValueError(f"Файл {model_path} слишком маленький для модели")
        
        print(f"🔄 Загрузка модели Silero из {model_path} на {device}...")
        model = torch.package.PackageImporter(model_path).load_pickle("tts_models", "model")
        model.to(device)
        print("✅ Модель Silero успешно загружена")
        return model, device
        
    except Exception as e:
        print(f"❌ Ошибка при загрузке модели Silero: {e}")
        return None, None

silero_model, silero_device = init_silero_model()

# Класс-обертка для Silero, совместимый с TTS API
class SileroTTSWrapper:
    def __init__(self, model_name=None, **kwargs):
        self.model = silero_model
        self.device = silero_device
        self.sample_rate = 48000  # Частота дискретизации для Silero TTS
        print(f"🔄 Инициализация Silero TTS")
        
        if self.model is None:
            warnings.warn("Silero модель не загружена. Используется заглушка.")
    
    def to(self, device):
        print(f"🔄 Перемещение модели на устройство: {device}")
        if self.model is not None:
            try:
                self.model.to(device)
                self.device = device
            except Exception as e:
                print(f"❌ Ошибка при перемещении модели: {e}")
        return self
    
    def tts_to_file(self, text, output_file, **kwargs):
        print(f"🔊 Синтез речи с Silero: {text[:50]}...")
        try:
            # Убедимся, что директория существует
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            
            if self.model is None:
                # Создаем пустой аудиофайл при ошибке
                import wave
                import struct
                
                duration = 1  # seconds
                framerate = 48000  # Hz
                with wave.open(output_file, "w") as wav_file:
                    wav_file.setparams((1, 2, framerate, framerate, "NONE", "not compressed"))
                    for i in range(framerate):
                        packed_value = struct.pack("<h", 0)
                        wav_file.writeframes(packed_value)
                print(f"⚠️ Создан пустой аудиофайл (Silero не загружен): {output_file}")
                return output_file
                
            # Получаем параметры из kwargs
            speaker = kwargs.get('speaker', 'xenia')
            sample_rate = kwargs.get('sample_rate', self.sample_rate)
            put_accent = kwargs.get('put_accent', True)
            put_yo = kwargs.get('put_yo', True)
            
            # Генерация речи с Silero
            audio = self.model.apply_tts(
                text=text,
                speaker=speaker,
                sample_rate=sample_rate,
                put_accent=put_accent,
                put_yo=put_yo
            )
            
            # Конвертация в NumPy и сохранение
            audio_np = audio.cpu().numpy()
            
            import soundfile as sf
            sf.write(output_file, audio_np, sample_rate)
            print(f"✅ Аудиофайл создан: {output_file}")
            
            return output_file
            
        except Exception as e:
            print(f"❌ Ошибка при синтезе речи с Silero: {e}")
            # Создаем пустой аудиофайл при ошибке
            import wave
            import struct
            
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            duration = 1  # seconds
            framerate = 48000  # Hz
            with wave.open(output_file, "w") as wav_file:
                wav_file.setparams((1, 2, framerate, framerate, "NONE", "not compressed"))
                for i in range(framerate):
                    packed_value = struct.pack("<h", 0)
                    wav_file.writeframes(packed_value)
            print(f"⚠️ Создан пустой аудиофайл из-за ошибки: {output_file}")
            return output_file
    
    # Дополнительные методы для совместимости
    @staticmethod
    def list_models():
        return ["silero_model"]
    
    def is_multi_speaker(self):
        return True
    
    def get_speaker_ids(self):
        return ["xenia", "baya", "kseniya", "eugene", "random"]

# Замещаем модуль TTS нашей реализацией с Silero
if "TTS" not in sys.modules:
    class SileroTTSModule:
        TTS = SileroTTSWrapper
        
        @staticmethod
        def list_models():
            return ["silero_tts"]
    
    # Регистрируем модули
    sys.modules["TTS"] = SileroTTSModule
    sys.modules["TTS.api"] = SileroTTSModule
EOL

# Применяем патч для скриптов
sed -i '1s/^/import sys\nsys.path.insert(0, "\/app")\ntry:\n    import silero_tts  # загружаем Silero TTS\nexcept Exception as e:\n    print(f"Предупреждение: не удалось загрузить Silero TTS: {e}")\n/' scripts/lection_to_audio.py
sed -i '1s/^/import sys\nsys.path.insert(0, "\/app")\ntry:\n    import silero_tts  # загружаем Silero TTS\nexcept Exception as e:\n    print(f"Предупреждение: не удалось загрузить Silero TTS: {e}")\n/' scripts/text_update_agent.py

echo "=== Инициализация контейнера завершена ==="

# Возвращаемся в рабочую директорию
cd /app

# Запуск переданной команды с правильными переменными окружения
export PYTORCH_WEIGHTS_ONLY=0
export TTS_CHECKPOINT_CONFIG_COMPAT=1
exec "$@" 