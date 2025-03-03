import os
import torch
import torchaudio
import numpy as np
import time
import argparse
import pyloudnorm as pyln
import warnings
from TTS.api import TTS
import subprocess
import re
from pathlib import Path

# Change to project root directory
os.chdir(Path(__file__).parent.parent)

# Подавляем предупреждения от NNPACK
os.environ["NNPACK_IGNORE"] = "1"

# Подавляем предупреждения от трансформеров о маске внимания
warnings.filterwarnings("ignore", message="The attention mask is not set")

# Глобальные переменные
tts_model = None
OUTPUT_DIR = "lections_audio"

# Настройки синтеза XTTS
XTTS_CONFIG = {
    "model_id": "tts_models/multilingual/multi-dataset/xtts_v2",
    "language": "ru",
    "speaker": "samples/combined.wav",
    "use_gpu": True
}

# Настройки текста
TEXT_CONFIG = {
    "max_chars_per_chunk": 500
}

# Конфигурация качества аудио
AUDIO_QUALITY = {
    "mp3_quality": "0",  # 0 - лучшее качество VBR
    "mp3_bitrate": "320k",  # максимальный битрейт для MP3
    "sample_rate": 44100,  # CD качество
    "normalize_audio": True,
    "target_loudness": -16.0  # LUFS
}

# Конфигурация для текста
TEXT_CONFIG = {
    "max_chars_per_chunk": 180  # для русского языка
}

# Объединенная конфигурация
config = {
    "xtts": XTTS_CONFIG,
    "audio": AUDIO_QUALITY,
    "text": TEXT_CONFIG
}

# Директория для выходных файлов
OUTPUT_DIR = "lections_audio"

# Глобальная переменная для хранения модели XTTS
tts_model = None

# Настройки фонового аудио
BACKGROUND_AUDIO = {
    "default_volume": 0.5,  # громкость фона по умолчанию (0.0 - 1.0)
    "fade_duration": 3,     # длительность затухания в начале и конце (в секундах)
    "sample_dir": "samples"  # директория с фоновыми аудиофайлами
}

def list_available_voices(voice_dir="samples"):
    """
    Возвращает список доступных голосов с метаданными.
    
    Args:
        voice_dir (str): Директория с голосами
        
    Returns:
        list: Список словарей с информацией о голосах
    """
    voices = []
    
    if not os.path.exists(voice_dir):
        os.makedirs(voice_dir, exist_ok=True)
        print(f"Создана новая директория для голосов: {voice_dir}")
        return voices
    
    # Рекурсивно обходим директорию с голосами
    for root, dirs, files in os.walk(voice_dir):
        for file in files:
            if file.endswith('.wav'):
                # Извлекаем метаданные из пути
                path_parts = os.path.relpath(root, voice_dir).split(os.sep)
                
                # Собираем метаданные
                voice_info = {
                    'path': os.path.join(root, file),
                    'filename': file,
                }
                
                # Добавляем метаданные, если путь подходит под структуру
                if len(path_parts) >= 1:
                    if path_parts[0] in ['male', 'female']:
                        voice_info.update({
                            'gender': path_parts[0],
                            'language': path_parts[1] if len(path_parts) > 1 else 'unknown',
                            'style': path_parts[2] if len(path_parts) > 2 else 'neutral',
                        })
                    elif path_parts[0] == 'russian':
                        voice_info.update({
                            'gender': 'male' if 'male' in file else 'female' if 'female' in file else 'unknown',
                            'language': 'ru',
                            'style': 'neutral',
                        })
                
                voice_info['name'] = os.path.splitext(file)[0]
                voices.append(voice_info)
    
    return voices

def select_voice_interactive():
    """
    Интерактивный выбор голоса из доступных.
    
    Returns:
        str: Путь к выбранному голосовому файлу или None, если голос не выбран
    """
    voices = list_available_voices()
    
    if not voices:
        print("Голоса не найдены.")
        print("Проверьте наличие директории 'samples' или создайте новую и добавьте в неё WAV-файлы.")
        return None
    
    print("\nДоступные голоса:")
    for i, voice in enumerate(voices):
        # Форматируем вывод в зависимости от доступных метаданных
        display_name = voice.get('name', 'Без имени')
        if 'gender' in voice and 'language' in voice:
            gender_str = voice.get('gender', 'unknown')
            lang_str = voice.get('language', 'unknown')
            style_str = voice.get('style', 'neutral')
            print(f"{i+1}. {display_name} ({gender_str}, {lang_str}, {style_str})")
        else:
            print(f"{i+1}. {display_name}")
        print(f"   Путь: {voice['path']}")
    
    print("\nВведите номер голоса или 0 для использования голоса по умолчанию:")
    try:
        choice = int(input("> "))
        if choice == 0:
            print("Будет использован голос по умолчанию.")
            return None
        
        selected_voice = voices[choice-1]['path']
        print(f"Выбран голос: {voices[choice-1].get('name', 'Без имени')}")
        print(f"Путь: {selected_voice}")
        
        print("\nПримечание: настройки высоты голоса и скорости отключены для предотвращения проблем с ускорением.")
        
        return selected_voice
    except (ValueError, IndexError):
        print("Неверный выбор. Будет использован голос по умолчанию.")
        return None

def get_tts_model():
    """
    Получает или инициализирует модель TTS.
    Максимально простая версия без дополнительных оптимизаций.
    
    Returns:
        TTS: Инициализированная модель TTS
    """
    global tts_model
    
    if tts_model is None:
        print("Загрузка модели XTTS...")
        
        # Определяем устройство для синтеза
        device = "cuda" if torch.cuda.is_available() and XTTS_CONFIG.get("use_gpu", True) else "cpu"
        print(f"Используется устройство: {device}")
        
        # Получаем путь к модели
        model_path = XTTS_CONFIG.get("model_id", "tts_models/multilingual/multi-dataset/xtts_v2")
        
        # Создаем модель без дополнительных настроек
        tts_model = TTS(model_path).to(device)
        print("Модель XTTS успешно загружена")
    
    return tts_model

def read_text_file(file_path: str):
    """Read text from file with proper error handling."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def enhance_audio(audio_data, sample_rate, volume=1.0):
    """
    Улучшает качество аудио с простыми настройками.
    
    Args:
        audio_data: Аудио данные (numpy array или torch tensor)
        sample_rate: Частота дискретизации
        volume: Множитель громкости (1.0 = без изменений)
        
    Returns:
        Улучшенные аудио данные
    """
    try:
        # Преобразуем в numpy array, если это torch tensor
        if hasattr(audio_data, 'numpy'):
            audio_data = audio_data.numpy()
            
        # Базовая нормализация громкости для улучшения слышимости
        if volume != 1.0:
            audio_data = audio_data * volume
            
        # Ограничиваем значения, чтобы предотвратить клиппинг
        audio_data = np.clip(audio_data, -1.0, 1.0)
        
        return audio_data
        
    except Exception as e:
        print(f"Ошибка при улучшении аудио: {e}")
        return audio_data  # Возвращаем исходные данные в случае ошибки

def split_text_into_chunks(text, max_chars=None):
    """
    Разделяет текст на фрагменты для синтеза.
    
    Args:
        text: Исходный текст
        max_chars: Максимальное количество символов в фрагменте
        
    Returns:
        list: Список фрагментов текста
    """
    # Если max_chars не указан, используем из конфигурации
    if max_chars is None:
        max_chars = TEXT_CONFIG.get("max_chars_per_chunk", 500)
        
    # Проверка наличия текста
    if not text:
        return []
    
    # Разбиваем текст на предложения
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # Если предложение слишком длинное, разбиваем его на части
        if len(sentence) > max_chars:
            # Если текущий фрагмент не пустой, добавляем его
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
                
            # Разбиваем длинное предложение на части по знакам препинания
            parts = re.split(r'(?<=[,:;])\s+', sentence)
            
            temp_chunk = ""
            for part in parts:
                if len(temp_chunk) + len(part) <= max_chars:
                    temp_chunk += part + " "
                else:
                    if temp_chunk:
                        chunks.append(temp_chunk.strip())
                    temp_chunk = part + " "
            
            if temp_chunk:
                current_chunk = temp_chunk.strip()
        
        # Если добавление предложения превысит лимит, создаем новый фрагмент
        elif len(current_chunk) + len(sentence) > max_chars:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
        else:
            current_chunk += sentence + " "
    
    # Добавляем последний фрагмент, если он не пустой
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def synthesize_speech(text, output_file, voice_file=None, language="ru"):
    """
    Синтезирует речь из текста, максимально простая версия без модификаций скорости и тона.
    
    Args:
        text: Текст для синтеза
        output_file: Путь к выходному файлу
        voice_file: Путь к файлу с образцом голоса (WAV)
        language: Язык синтеза (по умолчанию "ru")
    
    Returns:
        bool: True, если синтез успешен, иначе False
    """
    try:
        # Получаем модель TTS
        model = get_tts_model()
        
        # Используем указанный голос или голос по умолчанию
        speaker_wav = voice_file or XTTS_CONFIG.get("speaker")
        
        if not os.path.exists(speaker_wav):
            print(f"Ошибка: Файл с голосом {speaker_wav} не найден")
            return False
        
        print(f"Синтез речи для текста длиной {len(text)} символов...")
        print(f"Используется голос: {speaker_wav}")
        print(f"Язык: {language}")
        
        # Простой синтез без дополнительных преобразований
        model.tts_to_file(
            text=text,
            file_path=output_file,
            speaker_wav=speaker_wav,
            language=language
        )
        
        print(f"Синтез успешно завершен, файл сохранен: {output_file}")
        return True
    
    except Exception as e:
        print(f"Ошибка при синтезе речи: {e}")
        return False

def combine_audio_files(input_files, output_file):
    """
    Объединяет несколько аудиофайлов в один.
    """
    try:
        print(f"Объединение {len(input_files)} аудиофайлов в {output_file}...")
        
        # Используем FFmpeg для объединения файлов
        # Создаем временный файл со списком входных файлов
        list_file = "temp_file_list.txt"
        with open(list_file, "w") as f:
            for file in input_files:
                f.write(f"file '{file}'\n")
        
        # Объединяем файлы с помощью FFmpeg
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", 
            "-i", list_file, "-c", "copy", output_file
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Удаляем временный файл
        if os.path.exists(list_file):
            os.remove(list_file)
        
        if os.path.exists(output_file):
            print(f"Аудиофайлы успешно объединены: {output_file}")
            return True
        else:
            print(f"Ошибка: объединенный файл не был создан")
            return False
            
    except Exception as e:
        print(f"Ошибка при объединении аудиофайлов: {e}")
        return False

def add_background_audio(voice_file, background_file, output_file, bg_volume=None):
    """
    Накладывает фоновое аудио на голосовой файл.
    
    Args:
        voice_file: Путь к файлу с голосом
        background_file: Путь к файлу с фоновой музыкой
        output_file: Путь к выходному файлу
        bg_volume: Громкость фонового аудио (0.0 - 1.0)
        
    Returns:
        bool: True, если накладывание успешно, иначе False
    """
    try:
        # Используем значение громкости из параметров или из конфигурации
        if bg_volume is None:
            bg_volume = BACKGROUND_AUDIO.get("default_volume", 0.1)
            
        print(f"Накладывание фонового аудио на голосовой файл...")
        print(f"Голосовой файл: {voice_file}")
        print(f"Фоновое аудио: {background_file}")
        print(f"Громкость фона: {bg_volume}")
        
        # Проверяем наличие файлов
        if not os.path.exists(voice_file):
            print(f"Ошибка: Голосовой файл {voice_file} не найден")
            return False
            
        if not os.path.exists(background_file):
            print(f"Ошибка: Файл с фоновой музыкой {background_file} не найден")
            return False
        
        # Параметры для FFmpeg
        fade_duration = BACKGROUND_AUDIO.get("fade_duration", 3)
        
        # Используем FFmpeg для наложения фонового аудио
        cmd = [
            "ffmpeg", "-y",
            "-i", voice_file,
            "-i", background_file,
            "-filter_complex", 
            f"[1:a]volume={bg_volume},aloop=loop=-1:size=0,afade=t=in:st=0:d={fade_duration},afade=t=out:st=999999:d={fade_duration}[bg];"
            f"[0:a][bg]amix=inputs=2:duration=first[a]",
            "-map", "[a]",
            "-c:a", "pcm_s16le",  # WAV формат
            output_file
        ]
        
        # Выполняем команду
        process = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if os.path.exists(output_file):
            print(f"Фоновое аудио успешно наложено: {output_file}")
            return True
        else:
            print(f"Ошибка: выходной файл с фоновым аудио не был создан")
            return False
            
    except Exception as e:
        print(f"Ошибка при наложении фонового аудио: {e}")
        return False

def list_background_audio_files():
    """
    Возвращает список доступных файлов с фоновым аудио.
    
    Returns:
        list: Список путей к файлам с фоновым аудио
    """
    background_dir = BACKGROUND_AUDIO.get("sample_dir", "samples")
    
    # Создаем директорию, если она не существует
    if not os.path.exists(background_dir):
        os.makedirs(background_dir, exist_ok=True)
        print(f"Создана директория для фоновых аудио: {background_dir}")
        return []
    
    # Собираем все аудиофайлы из директории
    bg_files = []
    for root, dirs, files in os.walk(background_dir):
        for file in files:
            if file.endswith(('.wav', '.mp3', '.ogg')):
                bg_files.append(os.path.join(root, file))
    
    return bg_files

def create_podcast(text, output_file, voice_file=None, language="ru", background_audio=None, bg_volume=None):
    """
    Создает аудиофайл из текста.
    
    Args:
        text: Текст для синтеза или путь к файлу с текстом
        output_file: Путь к выходному WAV-файлу
        voice_file: Путь к файлу с образцом голоса (WAV)
        language: Язык синтеза (по умолчанию "ru")
        background_audio: Путь к файлу с фоновой музыкой
        bg_volume: Громкость фонового аудио (0.0 - 1.0)
        
    Returns:
        bool: True, если создание успешно, иначе False
    """
    try:
        print(f"\n=== Начало создания аудиофайла ===")
        
        # Проверяем, является ли text путем к файлу
        if os.path.exists(text) and os.path.isfile(text):
            print(f"Чтение текста из файла: {text}")
            with open(text, 'r', encoding='utf-8') as f:
                text = f.read()
        
        # Проверяем, что текст не пустой
        if not text:
            print("Ошибка: Пустой текст для синтеза")
            return False
            
        print(f"Длина текста для обработки: {len(text)} символов")
        
        # Создаем директорию для временных файлов
        temp_dir = os.path.join(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', "temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Разделяем текст на фрагменты
        chunks = split_text_into_chunks(text)
        print(f"Текст разделен на {len(chunks)} фрагментов")
        
        # Создаем временные аудиофайлы для каждого фрагмента
        audio_files = []
        for i, chunk in enumerate(chunks):
            print(f"\nОбработка фрагмента {i+1}/{len(chunks)}")
            
            # Путь к временному файлу
            temp_file = os.path.join(temp_dir, f"chunk_{i:03d}.wav")
            
            # Синтезируем речь для фрагмента
            success = synthesize_speech(
                text=chunk, 
                output_file=temp_file,
                voice_file=voice_file,
                language=language
            )
            
            if success:
                audio_files.append(temp_file)
            else:
                print(f"Ошибка при обработке фрагмента {i+1}")
        
        # Проверяем, что есть аудиофайлы для объединения
        if not audio_files:
            print("Ошибка: Не создано ни одного аудиофайла")
            return False
        
        # Имя файла для голоса без фона
        voice_only_file = os.path.join(temp_dir, "voice_only.wav")
        
        # Если создан только один файл, просто копируем его в voice_only_file
        if len(audio_files) == 1:
            print(f"Создан один аудиофайл, копируем его")
            try:
                import shutil
                shutil.copyfile(audio_files[0], voice_only_file)
            except Exception as e:
                print(f"Ошибка при копировании файла: {e}")
                return False
        else:
            # Объединяем все аудиофайлы в один файл для голоса
            print(f"Объединение {len(audio_files)} аудиофайлов")
            success = combine_audio_files(audio_files, voice_only_file)
            if not success:
                print("Ошибка при объединении аудиофайлов")
                return False
        
        # Если указан фоновый аудиофайл, накладываем его
        if background_audio and os.path.exists(background_audio):
            print(f"Добавление фонового аудио: {background_audio}")
            success = add_background_audio(
                voice_file=voice_only_file,
                background_file=background_audio,
                output_file=output_file,
                bg_volume=bg_volume
            )
            if not success:
                print("Ошибка при добавлении фонового аудио, используем только голос")
                import shutil
                shutil.copyfile(voice_only_file, output_file)
        else:
            # Если фоновое аудио не указано, просто копируем голосовой файл
            print("Фоновое аудио не указано, используем только голос")
            import shutil
            shutil.copyfile(voice_only_file, output_file)
            
        print(f"Аудиофайл успешно создан: {output_file}")
        
        # Удаляем временные файлы
        print("Удаление временных файлов")
        for file in audio_files:
            if os.path.exists(file):
                os.remove(file)
        
        if os.path.exists(voice_only_file):
            os.remove(voice_only_file)
            
        print(f"=== Создание аудиофайла завершено ===\n")
        return True
        
    except Exception as e:
        print(f"Ошибка при создании аудиофайла: {e}")
        return False

def process_all_text_files(directory="lections_text_mistral", output_dir="lections_audio", voice_file=None, 
                          language="ru", background_audio=None, bg_volume=None):
    """
    Обрабатывает все текстовые файлы в указанной директории и создает аудиофайлы.
    
    Args:
        directory: Директория с текстовыми файлами
        output_dir: Директория для выходных аудиофайлов
        voice_file: Путь к файлу с образцом голоса
        language: Язык текста
        background_audio: Путь к файлу с фоновой музыкой
        bg_volume: Громкость фонового аудио
        
    Returns:
        int: Количество успешно обработанных файлов
    """
    # Проверяем существование директорий
    if not os.path.exists(directory):
        print(f"Директория с текстами {directory} не найдена. Создаем...")
        os.makedirs(directory, exist_ok=True)
        print(f"Пожалуйста, добавьте текстовые файлы в директорию {directory} и запустите скрипт снова.")
        return 0
        
    # Создаем директорию для выходных файлов, если она не существует
    os.makedirs(output_dir, exist_ok=True)
    
    # Получаем список текстовых файлов
    text_files = [f for f in os.listdir(directory) if f.endswith(('.txt', '.md', '.text'))]
    
    if not text_files:
        print(f"В директории {directory} не найдено текстовых файлов.")
        return 0
        
    print(f"Найдено {len(text_files)} текстовых файлов для обработки.")
    
    # Обрабатываем каждый файл
    successful_count = 0
    for i, file_name in enumerate(text_files):
        # Полные пути к файлам
        input_file = os.path.join(directory, file_name)
        output_file = os.path.join(output_dir, os.path.splitext(file_name)[0] + '.wav')
        
        print(f"\n[{i+1}/{len(text_files)}] Обработка файла: {file_name}")
        print(f"Выходной файл: {output_file}")
        
        # Чтение текста из файла
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                text = f.read()
                
            # Создаем подкаст
            success = create_podcast(
                text=text,
                output_file=output_file,
                voice_file=voice_file,
                language=language,
                background_audio=background_audio,
                bg_volume=bg_volume
            )
            
            if success:
                successful_count += 1
                print(f"✅ Успешно создан аудиофайл: {output_file}")
            else:
                print(f"❌ Ошибка при создании аудиофайла: {output_file}")
                
        except Exception as e:
            print(f"❌ Ошибка при обработке файла {file_name}: {e}")
    
    return successful_count

def main():
    """
    Основная функция программы. Парсит аргументы командной строки и запускает создание подкаста.
    """
    parser = argparse.ArgumentParser(description='Создание аудио-подкаста из текста')
    
    # Основные параметры
    parser.add_argument('--input', '-i', default='lections_text_mistral', 
                        help='Путь к текстовому файлу или директории (по умолчанию: lections_text_mistral)')
    parser.add_argument('--output', '-o', default='lections_audio',
                        help='Путь к выходному аудиофайлу или директории (по умолчанию: lections_audio)')
    parser.add_argument('--voice', '-v', help='Путь к файлу с образцом голоса (WAV)')
    parser.add_argument('--language', '-l', default='ru', help='Язык текста (по умолчанию: ru)')
    parser.add_argument('--single', '-s', action='store_true', 
                       help='Обработать только один файл вместо всей директории')
    
    # Параметры фонового аудио
    parser.add_argument('--background', '-bg', default='samples/Calming Intros.mp3', 
                       help='Путь к файлу с фоновой музыкой (WAV, MP3, OGG) (по умолчанию: samples/Calming Intros.mp3)')
    parser.add_argument('--bg-volume', '-bv', type=float, default=BACKGROUND_AUDIO["default_volume"],
                        help=f'Громкость фонового аудио от 0.0 до 1.0 (по умолчанию: {BACKGROUND_AUDIO["default_volume"]})')
    parser.add_argument('--no-background', action='store_true', 
                       help='Не использовать фоновую музыку')
    
    # Парсим аргументы
    args = parser.parse_args()
    
    # Определение фонового аудио
    background_audio = None
    
    # Если не указано --no-background, используем фоновую музыку
    if not args.no_background:
        background_audio = args.background
        
        # Проверяем существование файла
        if background_audio and not os.path.exists(background_audio):
            print(f"Предупреждение: Файл с фоновой музыкой {background_audio} не найден")
            print("Аудио будет создано без фонового звука")
            background_audio = None
        elif background_audio:
            print(f"Используется фоновое аудио: {background_audio}")
            print(f"Громкость фона: {args.bg_volume}")
    else:
        print("Фоновая музыка отключена")
    
    # Обработка одного файла если указан флаг --single
    if args.single and not os.path.isdir(args.input):
        # Обработка одного файла
        print(f"\n=== Режим обработки одного файла ===")
        
        # Проверка наличия входных данных
        if not args.input:
            # Запрашиваем текст для конвертации
            text = input("\nВведите текст для преобразования в речь (или путь к файлу с текстом): ")
            
            # Проверяем, является ли ввод путем к файлу
            if os.path.exists(text):
                with open(text, 'r', encoding='utf-8') as f:
                    text = f.read()
        else:
            # Загружаем текст из файла
            try:
                # Проверяем существование файла
                if not os.path.exists(args.input):
                    print(f"Предупреждение: Файл {args.input} не найден.")
                    
                    # Проверяем существование директории
                    input_dir = os.path.dirname(args.input)
                    if input_dir and not os.path.exists(input_dir):
                        os.makedirs(input_dir, exist_ok=True)
                        print(f"Создана директория: {input_dir}")
                        
                    # Запрашиваем текст для сохранения
                    print(f"Введите текст для сохранения в файл {args.input} (для завершения введите пустую строку):")
                    lines = []
                    while True:
                        line = input()
                        if not line:
                            break
                        lines.append(line)
                    
                    # Сохраняем текст в файл
                    if lines:
                        with open(args.input, 'w', encoding='utf-8') as f:
                            f.write('\n'.join(lines))
                        print(f"Текст сохранен в файл: {args.input}")
                        text = '\n'.join(lines)
                    else:
                        print("Текст не введен. Используем пример текста.")
                        text = "Это пример текста для синтеза речи с использованием XTTS."
                else:
                    # Файл существует, читаем его
                    with open(args.input, 'r', encoding='utf-8') as f:
                        text = f.read()
                    print(f"Текст загружен из файла: {args.input}")
            except Exception as e:
                print(f"Ошибка при чтении файла {args.input}: {e}")
                return
        
        # Проверка выходного файла
        if not args.output:
            output_file = input("Введите путь для сохранения аудиофайла (по умолчанию: output.wav): ") or "output.wav"
        else:
            output_file = args.output
            
            # Проверяем расширение выходного файла
            if not output_file.lower().endswith('.wav'):
                print(f"Предупреждение: выходной файл не имеет расширения .wav")
                print(f"Аудио будет сохранено в формате WAV")
                output_file = os.path.splitext(output_file)[0] + '.wav'
                print(f"Новый путь к выходному файлу: {output_file}")
        
        # Проверка наличия директории для выходного файла
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        
        # Создаем подкаст
        create_podcast(
            text, 
            output_file, 
            voice_file=args.voice, 
            language=args.language,
            background_audio=background_audio,
            bg_volume=args.bg_volume
        )
        
        print(f"\nГотово! Аудиофайл сохранен: {output_file}")
    else:
        # По умолчанию - обработка всех файлов в директории
        # Определяем директорию с текстами
        text_dir = args.input if os.path.isdir(args.input) else 'lections_text_mistral'
        
        # Определяем директорию для выходных файлов
        output_dir = args.output if args.output and os.path.isdir(args.output) else 'lections_audio'
        
        # Запускаем пакетную обработку
        print(f"\n=== Запуск пакетной обработки файлов ===")
        print(f"Директория с текстами: {text_dir}")
        print(f"Директория для выходных файлов: {output_dir}")
        
        count = process_all_text_files(
            directory=text_dir,
            output_dir=output_dir,
            voice_file=args.voice,
            language=args.language,
            background_audio=background_audio,
            bg_volume=args.bg_volume
        )
        
        print(f"\n=== Пакетная обработка завершена ===")
        print(f"Успешно обработано файлов: {count}")


if __name__ == "__main__":
    # Создаем структуру директорий, если они отсутствуют
    for dir_path in ["lections_text_mistral", "lections_audio", "samples", BACKGROUND_AUDIO["sample_dir"]]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            print(f"Создана директория: {dir_path}")
    
    main()
