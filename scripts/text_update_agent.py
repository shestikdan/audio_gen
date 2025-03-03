import mistralai
from mistralai import Mistral
import tiktoken
from typing import List, Dict, Tuple, Optional, Union
import os
import json
import numpy as np
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import hashlib
import inspect
import re
from datetime import datetime
from pathlib import Path

# Change to project root directory
os.chdir(Path(__file__).parent.parent)

# Импортируем лингвистический процессор
try:
    from linguistic_processor import LinguisticProcessor
    LINGUISTIC_PROCESSOR_AVAILABLE = True
except ImportError:
    print("Warning: Linguistic processor not available. Using basic text processing.")
    LINGUISTIC_PROCESSOR_AVAILABLE = False

class TextUpdateAgent:
    def __init__(self, api_key: str = None, max_workers: int = 2, input_dir: str = "lections_text_base", output_dir: str = "lections_text_mistral"):
        """Initialize the agent with Mistral API key."""
        self.api_key = "WVDFzgzuQ0GlRuD6Q55yj9cKftjuA8q9"
        self.client = Mistral(api_key=self.api_key)
        self.max_workers = max_workers
        self.total_tokens_used = 0
        self.total_cost = 0
        self.price_per_1k_tokens = 0.003  # Mistral small-latest price per 1K tokens (output)
        self.quality_threshold = 0.65  # Минимальное сходство для принятия результата
        self.quality_results = []  # Сохраняем результаты оценки качества
        self.processed_chunks = []  # Список обработанных чанков
        self.autosave_interval = 5  # Автосохранение каждые X чанков
        
        # Директории для чтения и сохранения файлов
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # Проверяем и создаем выходную директорию, если она не существует
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Создана директория для выходных файлов: {self.output_dir}")
        
        # Инициализация лингвистического процессора
        self.linguistic_processor = None
        if LINGUISTIC_PROCESSOR_AVAILABLE:
            try:
                print("Initializing linguistic processor...")
                self.linguistic_processor = LinguisticProcessor()
                print("Linguistic processor initialized successfully")
            except Exception as e:
                print(f"Failed to initialize linguistic processor: {e}")

    def calculate_tokens_and_cost(self, text: str) -> Tuple[int, float]:
        """Calculate number of tokens and estimated cost."""
        # Примерная оценка токенов (4 символа на токен)
        tokens = len(text) // 4
        cost = (tokens / 1000) * self.price_per_1k_tokens
        return tokens, cost

    def get_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs, handling different line ending styles."""
        # Нормализуем переносы строк
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Удаляем специальные маркеры, которые могут нарушать разбивку
        text = text.replace('<CURRENT_CURSOR_POSITION>', '')
        
        print(f"Размер текста до разбивки: {len(text)} байт")
        
        # Сначала проверяем стандартное разделение на абзацы (пустая строка между абзацами)
        if '\n\n' in text:
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            if paragraphs:
                print(f"Текст разбит на {len(paragraphs)} абзацев по двойным переносам")
                for i, p in enumerate(paragraphs[:3]):  # Показываем первые 3 абзаца для отладки
                    print(f"Абзац {i+1}: {p[:50]}...")
                return paragraphs
        
        # Проверяем разделение по одиночным переносам строк
        if '\n' in text:
            paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
            if len(paragraphs) > 1:
                print(f"Текст разбит на {len(paragraphs)} абзацев по одиночным переносам")
                return paragraphs
        
        # Если обычные методы не сработали, разбиваем на предложения
        if len(text) > 100:
            try:
                from nltk.tokenize import sent_tokenize
                import nltk
                nltk.download('punkt', quiet=True)
                sentences = sent_tokenize(text)
                
                # Группируем предложения в абзацы по 3-5 предложений
                paragraphs = []
                current_paragraph = []
                
                for i, sentence in enumerate(sentences):
                    current_paragraph.append(sentence)
                    if len(current_paragraph) >= 4 or i == len(sentences) - 1:
                        paragraphs.append(' '.join(current_paragraph))
                        current_paragraph = []
                
                print(f"Текст разбит на {len(paragraphs)} абзацев по предложениям")
                return paragraphs
            except Exception as e:
                print(f"Ошибка при разбиении на предложения: {e}")
                # Если не получилось разбить на предложения, вернём весь текст как один абзац
                if text.strip():
                    return [text.strip()]
                return []
        
        # Проверяем, получили ли мы хоть что-то
        if not paragraphs and text.strip():
            print("Не удалось разбить текст на абзацы, используем весь текст как один абзац")
            return [text.strip()]
            
        print(f"Текст разбит на {len(paragraphs)} абзацев")
        return paragraphs

    def split_text_into_chunks(self, text: str, max_tokens: int = 2000) -> List[Dict]:
        """Split text into chunks with overlap for context preservation using paragraphs."""
        paragraphs = self.get_paragraphs(text)
        chunks = []
        current_chunk_paragraphs = []
        current_length = 0
        overlap_paragraphs = 2  # Количество абзацев для контекста

        for i, paragraph in enumerate(paragraphs):
            paragraph_tokens = len(paragraph)
            
            # Если текущий абзац слишком большой, разделим его на части
            if paragraph_tokens > max_tokens:
                if current_chunk_paragraphs:
                    # Сохраняем предыдущий чанк
                    chunk_text = " ".join(current_chunk_paragraphs)
                    context_before = " ".join(paragraphs[max(0, i-overlap_paragraphs):i]) if i > 0 else ""
                    context_after = " ".join(paragraphs[i:min(len(paragraphs), i+overlap_paragraphs)])
                    chunks.append({
                        "text": chunk_text,
                        "position": len(chunks),
                        "context_before": context_before,
                        "context_after": context_after
                    })
                    current_chunk_paragraphs = []
                    current_length = 0

                # Разделяем большой абзац на части
                words = paragraph.split()
                current_part = []
                current_part_tokens = 0
                
                for word in words:
                    word_tokens = len(word)
                    if current_part_tokens + word_tokens > max_tokens:
                        chunks.append({
                            "text": " ".join(current_part),
                            "position": len(chunks),
                            "context_before": context_before if current_part == words[:len(current_part)] else " ".join(words[:len(current_part)]),
                            "context_after": " ".join(words[len(current_part):min(len(words), len(current_part) + 50)])
                        })
                        current_part = [word]
                        current_part_tokens = word_tokens
                    else:
                        current_part.append(word)
                        current_part_tokens += word_tokens
                
                if current_part:
                    chunks.append({
                        "text": " ".join(current_part),
                        "position": len(chunks),
                        "context_before": " ".join(words[-50:]) if len(current_part) < len(words) else context_before,
                        "context_after": " ".join(words[len(current_part):min(len(words), len(current_part) + 50)])
                    })
                continue

            if current_length + paragraph_tokens > max_tokens and current_chunk_paragraphs:
                # Создаем новый чанк
                chunk_text = " ".join(current_chunk_paragraphs)
                context_before = " ".join(paragraphs[max(0, i-overlap_paragraphs):i]) if i > 0 else ""
                context_after = " ".join(paragraphs[i:min(len(paragraphs), i+overlap_paragraphs)])
                
                chunks.append({
                    "text": chunk_text,
                    "position": len(chunks),
                    "context_before": context_before,
                    "context_after": context_after
                })
                # Начинаем новый чанк с текущего абзаца
                current_chunk_paragraphs = [paragraph]
                current_length = paragraph_tokens
            else:
                current_chunk_paragraphs.append(paragraph)
                current_length += paragraph_tokens

        # Добавляем последний чанк
        if current_chunk_paragraphs:
            chunk_text = " ".join(current_chunk_paragraphs)
            context_before = " ".join(paragraphs[-overlap_paragraphs:]) if len(chunks) > 0 else ""
            chunks.append({
                "text": chunk_text,
                "position": len(chunks),
                "context_before": context_before,
                "context_after": ""
            })

        return chunks

    def process_chunk(self, chunk: Dict) -> Dict:
        """Process a single chunk of text using Mistral AI."""
        system_prompt = """Ты - редактор лекционных материалов. Твоя задача - сделать текст легким для слушания, запоминающимся и интересным, сохраняя основной смысл.

ПРИНЦИПЫ ОБРАБОТКИ:
- Делай предложения короче и чётче
- Упрощай формулировки, но не упрощай смысл
- Добавляй логические связки между идеями
- Структурируй текст в удобные для восприятия абзацы
- Убирай лишние повторы и отступления

ЧЕГО ИЗБЕГАТЬ:
- Не добавляй обращения "ты/вы" и личные вопросы
- Не превращай текст в популярную статью
- Не перегружай метафорами и аналогиями
- Не меняй порядок и логику исходных аргументов

Результат должен звучать естественно при чтении вслух и легко восприниматься на слух.

Исходный текст для преобразования
"""

        max_attempts = 3  # Максимальное количество попыток обработки одного чанка
        base_delay = 5  # начальная задержка в секундах
        
        # Храним все результаты обработки для выбора лучшего
        all_results = []
        
        # Примечание: Используем mistral-small-latest вместо mistral-large-latest для оптимизации 
        # соотношения цена/качество. Модель обеспечивает хорошую производительность при более низкой стоимости.

        for attempt in range(max_attempts):
            try:
                # Если это повторная попытка, добавим инструкцию об улучшении
                additional_instruction = ""
                if attempt > 0:
                    additional_instruction = f"""
                    Это повторная попытка обработки (попытка {attempt+1} из {max_attempts}).

                    Предыдущая версия нуждается в улучшении. Сделай текст:
                    1. Более четким и лаконичным
                    2. Лучше структурированным для восприятия на слух
                    3. Более связным между абзацами
                    4. Интереснее, но без потери смысла

                    Помни: текст должен быть легким для слушания и запоминания, но сохранять исходные идеи и их порядок.
                    """
                
                context_message = f"""
                Обработай этот фрагмент лекции, чтобы он стал легче для слушания и запоминания.

                Контекст перед отрывком:
                {chunk['context_before']}

                Текст для обработки:
                {chunk['text']}

                Контекст после отрывка:
                {chunk['context_after']}
                {additional_instruction}
                """
                
                messages = [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": context_message
                    }
                ]

                print(f"Обработка чанка {chunk['position']}, попытка {attempt+1} из {max_attempts}")

                response = self.client.chat.complete(
                    model="mistral-small-latest",
                    messages=messages,
                    temperature=0.7,
                    top_p=0.9,
                    max_tokens=2000
                )
                
                # Подсчет токенов и стоимости
                tokens_used = response.usage.total_tokens
                self.total_tokens_used += tokens_used
                self.total_cost += (tokens_used / 1000) * self.price_per_1k_tokens
                
                processed_text = response.choices[0].message.content
                
                # Применяем лингвистический процессор для дополнительной обработки
                if self.linguistic_processor:
                    try:
                        # Создаем две версии текста: с SSML и для Silero
                        ssml_text = self.linguistic_processor.process_text(processed_text, use_ssml=True)
                        silero_text = self.linguistic_processor.process_text_for_silero(processed_text)
                        
                        # Сохраняем обе версии
                        processed_text = {
                            "ssml": ssml_text,
                            "silero": silero_text
                        }
                        
                        print(f"Лингвистическая обработка текста для чанка {chunk['position']} завершена")
                    except Exception as e:
                        print(f"Ошибка лингвистической обработки: {e}")
                        # В случае ошибки используем исходный текст для обеих версий
                        processed_text = {
                            "ssml": processed_text,
                            "silero": processed_text
                        }
                else:
                    # Если лингвистический процессор недоступен, используем исходный текст
                    processed_text = {
                        "ssml": processed_text,
                        "silero": processed_text
                    }
                
                # Проверяем качество обработки (используем silero-версию для оценки)
                quality_score = self.validate_content_preservation(chunk["text"], processed_text["silero"])
                
                result = {
                    "position": chunk["position"],
                    "processed_text": processed_text,
                    "tokens_used": tokens_used,
                    "quality_score": quality_score,
                    "original_text": chunk["text"],
                    "attempt": attempt + 1
                }
                
                # Добавляем результат в список всех результатов
                all_results.append(result)
                
                print(f"Качество обработки чанка {chunk['position']}, попытка {attempt+1}: {quality_score:.2f}")
                
                # Если качество отличное (> 0.9), прекращаем попытки
                if quality_score > 0.9:
                    print(f"Достигнуто отличное качество ({quality_score:.2f}) на попытке {attempt+1}")
                    break
                
                # Если это не последняя попытка и качество не отличное, пробуем еще раз
                if attempt < max_attempts - 1:
                    print(f"Качество обработки чанка {chunk['position']} не отличное ({quality_score:.2f}), пробуем улучшить...")
                    # Добавляем небольшую задержку перед следующей попыткой
                    time.sleep(base_delay)
                    
            except Exception as e:
                error_str = str(e)
                if "rate_limit" in error_str.lower():
                    delay = base_delay * (2 ** attempt)
                    print(f"Достигнут лимит запросов, ожидаем {delay} секунд перед повторной попыткой...")
                    time.sleep(delay)
                    continue
                else:
                    print(f"Ошибка обработки чанка {chunk['position']} (попытка {attempt+1}): {e}")
                    # В случае ошибки создаем результат с исходным текстом
                    result = {
                        "position": chunk["position"],
                        "processed_text": {
                            "ssml": chunk["text"],
                            "silero": chunk["text"]
                        },
                        "tokens_used": 0,
                        "quality_score": 1.0,  # Исходный текст имеет 100% сходство с самим собой
                        "original_text": chunk["text"],
                        "attempt": attempt + 1,
                        "error": str(e)
                    }
                    all_results.append(result)
                    break
        
        # Если были попытки обработки, выбираем лучший результат по качеству
        if all_results:
            # Сортируем результаты по качеству от лучшего к худшему
            best_result = max(all_results, key=lambda x: x["quality_score"])
            print(f"Выбран лучший результат для чанка {chunk['position']} с качеством {best_result['quality_score']:.2f} (попытка {best_result['attempt']})")
            return best_result
        
        # Если не удалось получить результат (не должно происходить), возвращаем исходный текст
        print(f"ВНИМАНИЕ: Не удалось обработать чанк {chunk['position']} после {max_attempts} попыток")
        return {
            "position": chunk["position"],
            "processed_text": {
                "ssml": chunk["text"],
                "silero": chunk["text"]
            },
            "tokens_used": 0,
            "quality_score": 1.0,
            "original_text": chunk["text"],
            "attempt": max_attempts,
            "error": "Failed after all attempts"
        }

    def get_embedding(self, text: str) -> List[float]:
        """Получаем эмбеддинг текста."""
        try:
            # Используем Mistral для получения эмбеддингов
            # Проверяем сигнатуру метода
            sig = inspect.signature(self.client.embeddings.create)
            print(f"Сигнатура метода embeddings.create: {sig}")
            
            # Пробуем разные варианты вызова
            try:
                response = self.client.embeddings.create(
                    model="mistral-embed",
                    prompt=text
                )
            except Exception as e1:
                print(f"Первая попытка не удалась: {e1}")
                try:
                    response = self.client.embeddings.create(
                        model="mistral-embed",
                        text=text
                    )
                except Exception as e2:
                    print(f"Вторая попытка не удалась: {e2}")
                    # Последняя попытка - просто передаем текст
                    response = self.client.embeddings.create(text)
            
            # Отладочный вывод
            print(f"Тип ответа: {type(response)}")
            print(f"Ответ: {response}")
            
            # Пробуем извлечь эмбеддинг из разных форматов ответа
            if hasattr(response, 'data') and len(response.data) > 0:
                if hasattr(response.data[0], 'embedding'):
                    return response.data[0].embedding
                elif hasattr(response.data[0], 'embedding_vector'):
                    return response.data[0].embedding_vector
            
            # Если не удалось извлечь эмбеддинг стандартным способом,
            # пробуем другие варианты доступа к данным
            if isinstance(response, dict):
                if 'data' in response and len(response['data']) > 0:
                    if 'embedding' in response['data'][0]:
                        return response['data'][0]['embedding']
                    elif 'embedding_vector' in response['data'][0]:
                        return response['data'][0]['embedding_vector']
            
            print(f"Не удалось извлечь эмбеддинг из ответа: {response}")
            
        except Exception as e:
            print(f"Ошибка при получении эмбеддинга: {e}")
            
        # В случае ошибки возвращаем псевдо-эмбеддинг на основе хеша текста
        # Это запасной вариант, который работает хуже, но позволяет не прерывать процесс
        hash_obj = hashlib.md5(text.encode('utf-8'))
        hash_bytes = hash_obj.digest()
        pseudo_embedding = [float(byte) / 255.0 for byte in hash_bytes]
        # Нормализуем
        norm = np.sqrt(sum(x*x for x in pseudo_embedding))
        return [x/norm for x in pseudo_embedding]
            
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Расчет косинусного сходства между векторами."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm_a = np.sqrt(sum(a * a for a in vec1))
        norm_b = np.sqrt(sum(b * b for b in vec2))
        
        if norm_a == 0 or norm_b == 0:
            return 0
            
        return dot_product / (norm_a * norm_b)
        
    def validate_content_preservation(self, original_text: str, processed_text: str) -> float:
        """Проверяем, насколько хорошо сохранилась информация после обработки."""
        # Удаляем простые проверки и делаем более комплексную оценку
        
        # Проверка 1: Длина текста (слишком короткий или слишком длинный текст - плохо)
        length_ratio = len(processed_text) / max(1, len(original_text))
        if length_ratio < 0.5 or length_ratio > 2.0:
            length_score = 0.6
        elif length_ratio < 0.7 or length_ratio > 1.5:
            length_score = 0.8
        else:
            length_score = 1.0
            
        # Проверка 2: Наличие ключевых слов (существительные, глаголы, имена собственные)
        # Выделяем слова длиной более 5 символов как вероятные ключевые слова
        import re
        
        # Очищаем тексты от знаков препинания для сравнения
        original_clean = re.sub(r'[^\w\s]', '', original_text.lower())
        processed_clean = re.sub(r'[^\w\s]', '', processed_text.lower())
        
        # Находим ключевые слова (длиной > 5 символов)
        key_words = [word for word in set(original_clean.split()) if len(word) > 5]
        
        # Если ключевых слов меньше 3, используем все слова длиной > 3
        if len(key_words) < 3:
            key_words = [word for word in set(original_clean.split()) if len(word) > 3]
            
        # Ограничиваем количество проверяемых ключевых слов (максимум 10)
        key_words = key_words[:10]
        
        # Считаем, сколько ключевых слов содержится в обработанном тексте
        matches = sum(1 for word in key_words if word in processed_clean.split())
        keyword_score = matches / max(1, len(key_words))
        
        # Проверка 3: Сохранение числовых данных
        original_numbers = re.findall(r'\d+', original_text)
        processed_numbers = re.findall(r'\d+', processed_text)
        
        # Вычисляем балл для числовых данных (если они есть)
        if original_numbers:
            # Вычисляем пересечение множеств чисел
            common_numbers = set(original_numbers).intersection(set(processed_numbers))
            numbers_score = len(common_numbers) / len(original_numbers)
        else:
            numbers_score = 1.0  # Если чисел нет, максимальный балл
            
        # Комбинируем все оценки с разными весами
        # Ключевые слова имеют наибольший вес
        final_score = 0.2 * length_score + 0.6 * keyword_score + 0.2 * numbers_score
        
        # Применяем корректировки на основе эвристик:
        
        # 1. Если в обработанном тексте используется обращение на "ты", добавляем бонус
        if "ты" in processed_clean.split() or "тебя" in processed_clean.split() or "тебе" in processed_clean.split():
            final_score = min(1.0, final_score + 0.05)
            
        # 2. Если в обработанном тексте есть вопросительные предложения, добавляем бонус
        if "?" in processed_text:
            final_score = min(1.0, final_score + 0.05)
            
        # Для отладки выводим подробности вычисления
        print(f"Оценка качества: длина={length_score:.2f}, ключ.слова={keyword_score:.2f}, числа={numbers_score:.2f}, итог={final_score:.2f}")
        
        return final_score

    def _find_text_overlap(self, text1: str, text2: str, min_overlap: int = 20) -> str:
        """Находит перекрытие между концом первого текста и началом второго."""
        # Ограничиваем поиск для оптимизации
        end_of_text1 = text1[-200:] if len(text1) > 200 else text1
        start_of_text2 = text2[:200] if len(text2) > 200 else text2
        
        # Ищем максимальное перекрытие
        max_overlap = ""
        for i in range(min_overlap, len(end_of_text1) + 1):
            suffix = end_of_text1[-i:]
            if start_of_text2.startswith(suffix):
                max_overlap = suffix
                
        return max_overlap

    def remove_duplicate_sections(self, text: str) -> str:
        """
        Удаляет дублирующиеся разделы текста.
        
        Args:
            text: Исходный текст с возможными дубликатами
            
        Returns:
            Текст без дубликатов
        """
        print("Проверка и удаление дублирующихся разделов...")
        
        # Разбиваем текст на абзацы
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        
        if len(paragraphs) <= 1:
            print("Текст содержит всего один абзац, пропускаем проверку на дубликаты.")
            return text
        
        # Ищем разделы, начинающиеся с заголовков (### или другие маркеры)
        sections = []
        current_section = []
        
        for para in paragraphs:
            # Если это заголовок и у нас уже есть текущий раздел, сохраняем его
            if (para.startswith('###') or para.startswith('##') or para.startswith('#')) and current_section:
                sections.append('\n\n'.join(current_section))
                current_section = [para]
            else:
                current_section.append(para)
        
        # Добавляем последний раздел
        if current_section:
            sections.append('\n\n'.join(current_section))
        
        # Если нет разделов с заголовками, используем абзацы как отдельные разделы
        if not sections:
            sections = paragraphs
        
        # Удаляем дубликаты, сохраняя порядок
        unique_sections = []
        section_hashes = set()
        
        for section in sections:
            # Создаем хеш содержимого для сравнения
            section_hash = hashlib.md5(section.encode()).hexdigest()
            
            if section_hash not in section_hashes:
                section_hashes.add(section_hash)
                unique_sections.append(section)
            else:
                print(f"Обнаружен и удален дубликат раздела: {section[:50]}...")
        
        # Собираем текст обратно
        result = '\n\n'.join(unique_sections)
        
        # Выводим статистику
        removed_count = len(sections) - len(unique_sections)
        if removed_count > 0:
            print(f"Удалено дублирующихся разделов: {removed_count}")
        else:
            print("Дублирующихся разделов не обнаружено.")
        
        return result

    def remove_duplicate_paragraphs(self, text: str) -> str:
        """
        Удаляет дублирующиеся абзацы в тексте, сохраняя их порядок.
        
        Args:
            text: Исходный текст с возможными дубликатами абзацев
            
        Returns:
            Текст без дублирующихся абзацев
        """
        print("Проверка и удаление дублирующихся абзацев...")
        
        # Разбиваем текст на абзацы
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        
        if len(paragraphs) <= 1:
            print("Текст содержит всего один абзац, пропускаем проверку на дубликаты.")
            return text
        
        # Удаляем дубликаты, сохраняя порядок
        unique_paragraphs = []
        paragraph_hashes = set()
        
        for para in paragraphs:
            # Пропускаем слишком короткие абзацы (менее 15 символов) - они могут быть заголовками
            if len(para) < 15:
                unique_paragraphs.append(para)
                continue
                
            # Создаем хеш содержимого для сравнения
            para_hash = hashlib.md5(para.encode()).hexdigest()
            
            if para_hash not in paragraph_hashes:
                paragraph_hashes.add(para_hash)
                unique_paragraphs.append(para)
            else:
                print(f"Обнаружен и удален дубликат абзаца: {para[:50]}...")
        
        # Собираем текст обратно
        result = '\n\n'.join(unique_paragraphs)
        
        # Выводим статистику
        removed_count = len(paragraphs) - len(unique_paragraphs)
        if removed_count > 0:
            print(f"Удалено дублирующихся абзацев: {removed_count}")
        else:
            print("Дублирующихся абзацев не обнаружено.")
        
        return result

    def ensure_commas_in_output(self, filename, preserve_ellipsis=True):
        """
        Гарантированно заменяет все точки на запятые в указанном файле.
        Также удаляет символы ':' и '«'.
        Сохраняет многоточия, если preserve_ellipsis=True.
        
        Args:
            filename: Путь к файлу, в котором нужно заменить точки на запятые
            preserve_ellipsis: Сохранять ли многоточия (...)
        """
        if os.path.exists(filename):
            print(f"Проверка и замена точек на запятые в файле {filename}...")
            
            try:
                # Чтение файла
                with open(filename, "r", encoding="utf-8") as f:
                    text = f.read()
                
                # Проверяем исходное количество точек и других символов
                original_dot_count = text.count('.')
                original_colon_count = text.count(':')
                original_quote_count = text.count('«')
                
                if original_dot_count == 0 and original_colon_count == 0 and original_quote_count == 0:
                    print("В файле нет символов для замены.")
                    return
                
                print(f"Найдено {original_dot_count} точек, {original_colon_count} двоеточий и {original_quote_count} кавычек '«' в файле, выполняем замену...")
                
                # Создаем копию для проверки изменений
                original_text = text
                
                # Временные маркеры для сохранения многоточий
                if preserve_ellipsis:
                    # Маркируем многоточия для сохранения
                    markers = {}
                    ellipsis_pattern = r'\.{3,}'  # Три или более точек подряд
                    
                    for i, match in enumerate(re.finditer(ellipsis_pattern, text)):
                        marker = f"<ELLIPSIS_MARKER_{i}>"
                        markers[marker] = match.group()
                        text = text.replace(match.group(), marker, 1)
                
                # Заменяем все оставшиеся точки на запятые
                text = text.replace(".", ",")
                
                # Удаляем двоеточия и кавычки '«'
                text = text.replace(":", "")
                text = text.replace("«", "")
                
                # Восстанавливаем многоточия, если они были сохранены
                if preserve_ellipsis:
                    for marker, original in markers.items():
                        text = text.replace(marker, original)
                
                # Проверяем, были ли сделаны изменения
                dots_after = text.count('.')
                dots_replaced = original_dot_count - dots_after
                colons_removed = original_colon_count - text.count(':')
                quotes_removed = original_quote_count - text.count('«')
                
                # Запись обратно в файл
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(text)
                
                # Проверяем результат и выводим статистику
                print(f"Замена завершена:")
                print(f"- Заменено точек на запятые: {dots_replaced}")
                print(f"- Удалено двоеточий: {colons_removed}")
                print(f"- Удалено кавычек '«': {quotes_removed}")
                
                if dots_after > 0:
                    # Точки все еще остались, проверяем, являются ли они частью многоточий
                    ellipsis_count = text.count('...')
                    if dots_after != ellipsis_count * 3:
                        print(f"ВНИМАНИЕ: В файле все еще есть точки, которые не являются частью многоточий: {dots_after - ellipsis_count * 3}")
                    else:
                        print(f"Сохранено {ellipsis_count} многоточий.")
            
            except Exception as e:
                print(f"Ошибка при обработке файла: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"Файл {filename} не найден.")

    def analyze_and_improve_text(self, original_text: str, processed_text: str) -> str:
        """
        Анализирует обработанный текст, выявляет проблемы и генерирует улучшенную версию.
        
        Args:
            original_text: Исходный текст
            processed_text: Обработанный текст, который нужно проанализировать и улучшить
            
        Returns:
            Улучшенная версия текста
        """
        print("Выполняю повторный анализ и улучшение текста...")
        
        # Токены и стоимость
        total_tokens, cost = self.calculate_tokens_and_cost(original_text + processed_text)
        self.total_tokens_used += total_tokens
        self.total_cost += cost
        
        # Создаем промпт для анализа и улучшения
        analysis_prompt = f"""
        Ты - редактор лекционных материалов. Проанализируй текст и сделай его еще легче для слушания и запоминания.
        
        Твоя задача:
        1. Проанализировать переработанный текст
        2. Выявить проблемы, мешающие восприятию на слух
        3. Создать улучшенную версию, которая:
           - Сохраняет ключевую информацию оригинала
           - Легко воспринимается на слух
           - Хорошо запоминается
        
        КРИТЕРИИ КАЧЕСТВЕННОГО ТЕКСТА:
        1. Простые и четкие предложения (не более 15-20 слов)
        2. Логичная структура с понятными переходами между идеями
        3. Отсутствие повторов и лишних отступлений
        4. Хороший ритм - чередование коротких и средних предложений
        5. Ясность формулировок без потери глубины смысла
        
        ПРОБЛЕМЫ, КОТОРЫЕ НУЖНО ИСПРАВИТЬ:
        1. Длинные, сложные предложения
        2. Нелогичные переходы между абзацами
        3. Избыточные метафоры или аналогии
        4. Повторения одних и тех же идей разными словами
        5. Перегруженность терминами без пояснений
        
        ОРИГИНАЛЬНЫЙ ТЕКСТ:
        {original_text}
        
        ПЕРЕРАБОТАННЫЙ ТЕКСТ (с проблемами):
        {processed_text}
        
        Твой ответ должен включать:
        1. Анализ проблем: кратко перечисли 3-5 основных недостатков переработанного текста
        2. Улучшенная версия: создай текст, который легко слушать и легко запомнить
        
        ВАЖНО: В улучшенной версии сохрани всю ключевую информацию из оригинала. Сделай текст ясным и структурированным, но не изменяй его основной смысл.
        """
        
        try:
            # Отправляем запрос к Mistral API
            chat_response = self.client.chat.complete(
                model="mistral-medium",
                messages=[
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.7,
                top_p=0.9,
                max_tokens=2048,
            )
            
            response_text = chat_response.choices[0].message.content
            
            # Извлекаем улучшенную версию текста
            # Ищем улучшенную версию после анализа проблем
            improved_text = response_text
            
            # Паттерны, которые могут указывать на начало улучшенной версии
            patterns = [
                "Улучшенная версия:", 
                "УЛУЧШЕННАЯ ВЕРСИЯ:", 
                "Исправленный текст:", 
                "Вот улучшенная версия текста:"
            ]
            
            for pattern in patterns:
                if pattern in response_text:
                    improved_parts = response_text.split(pattern, 1)
                    if len(improved_parts) > 1:
                        improved_text = improved_parts[1].strip()
                        break
            
            # Если паттерны не найдены, берем всю вторую половину ответа
            if improved_text == response_text and len(response_text.split('\n\n')) > 2:
                parts = response_text.split('\n\n')
                middle_point = len(parts) // 2
                improved_text = '\n\n'.join(parts[middle_point:])
            
            print("Анализ и улучшение текста выполнены успешно")
            return improved_text
            
        except Exception as e:
            print(f"Ошибка при анализе и улучшении текста: {e}")
            return processed_text  # Возвращаем исходный обработанный текст в случае ошибки

    def process_text(self, text: str, output_base_filename: str, force_reprocess=True) -> Union[str, Dict[str, str]]:
        """Обрабатывает текст с разбивкой на параграфы для лучшего контекста.
        
        Args:
            text: Исходный текст для обработки
            output_base_filename: Базовое имя файла для сохранения результатов
            force_reprocess: Принудительно обрабатывать все чанки, даже если они уже были обработаны
        
        Returns:
            Обработанный текст в виде строки или словаря
        """
        # Формируем полные пути к файлам для данного обрабатываемого файла
        output_text_file = os.path.join(self.output_dir, output_base_filename)
        checkpoint_file = os.path.join(self.output_dir, f"{output_base_filename}.checkpoint.json")
        auto_checkpoint_file = os.path.join(self.output_dir, f"{output_base_filename}.auto_checkpoint.json")
        quality_report_file = os.path.join(self.output_dir, f"{output_base_filename}.quality_report.json")
        interrupted_file = os.path.join(self.output_dir, f"{output_base_filename}.interrupted_checkpoint.json")
        improved_text_file = os.path.join(self.output_dir, f"{output_base_filename}.improved.txt")
        
        # Фиксируем время начала обработки
        start_time = time.time()
        
        # Разделяем исходный текст на чанки для контекстной обработки
        chunks = self.split_text_into_chunks(text)
        
        # Подробная информация о чанках
        print(f"\nРазделение исходного текста:")
        print(f"Исходный текст: {len(text)} символов")
        print(f"Количество чанков: {len(chunks)}")
        for i, chunk in enumerate(chunks[:3]):  # Показываем первые 3 чанка для примера
            print(f"Чанк {i}: {len(chunk['text'])} символов, позиция {chunk['position']}")
        if len(chunks) > 3:
            print(f"... и ещё {len(chunks) - 3} чанков")
        
        # Проверяем наличие чекпоинта для этого файла
        if os.path.exists(checkpoint_file) and not force_reprocess:
            try:
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                    self.processed_chunks = checkpoint_data.get("chunks", [])
                    if self.processed_chunks:
                        print(f"Загружено {len(self.processed_chunks)} обработанных чанков из точки восстановления")
            except Exception as e:
                print(f"Ошибка загрузки чекпоинта: {e}")
                self.processed_chunks = []
        else:
            self.processed_chunks = []  
        
        # Используем локальную переменную для удобства
        processed_chunks = self.processed_chunks
        
        # Определяем, какие чанки уже обработаны
        if not force_reprocess:
            processed_positions = {chunk["position"] for chunk in processed_chunks}
            chunks_to_process = [chunk for chunk in chunks if chunk["position"] not in processed_positions]
        else:
            chunks_to_process = chunks
            print("Принудительная обработка всех чанков включена")
        
        print(f"Всего чанков: {len(chunks)}")
        print(f"Чанков для обработки: {len(chunks_to_process)}")
        if chunks:
            print(f"Содержимое первого чанка: {chunks[0]['text'][:100]}...")
        print(f"Оценка токенов в исходном тексте: {len(text) // 4}")
        
        # Статистика повторных попыток
        retry_stats = {
            "total_attempts": 0,
            "chunks_with_retries": 0,
            "excellent_on_first_try": 0,
            "excellent_after_retries": 0,
            "best_from_multiple_attempts": 0
        }
        
        if not chunks_to_process:
            print("Все чанки уже обработаны!")
        else:
            # Обрабатываем чанки параллельно с tqdm для наглядного отображения прогресса
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {}
                for chunk in chunks_to_process:
                    futures[executor.submit(self.process_chunk, chunk)] = chunk
                
                # Показываем прогресс-бар в процессе обработки
                for future in tqdm(as_completed(futures), total=len(futures), desc="Обработка текста"):
                    try:
                        result = future.result()
                        processed_chunks.append(result)
                        # Убираем двойное добавление результатов, оставляем только в один список
                        # self.processed_chunks.append(result)
                        
                        # Автосохранение прогресса каждые N чанков
                        if len(processed_chunks) % self.autosave_interval == 0:
                            try:
                                # Сохраняем текущий прогресс
                                # with open(auto_checkpoint_file, "w", encoding="utf-8") as f:
                                #     json.dump({"chunks": processed_chunks}, f, ensure_ascii=False, indent=2)
                                print(f"\nАвтосохранение: обработано {len(processed_chunks)} чанков")
                            except Exception as e:
                                print(f"\nОшибка автосохранения: {e}")
                        
                        # Обновляем статистику повторных попыток
                        retry_stats["total_attempts"] += result.get("attempt", 1)
                        if result.get("attempt", 1) > 1:
                            retry_stats["chunks_with_retries"] += 1
                        if result.get("quality_score", 0) > 0.9 and result.get("attempt", 1) == 1:
                            retry_stats["excellent_on_first_try"] += 1
                        if result.get("quality_score", 0) > 0.9 and result.get("attempt", 1) > 1:
                            retry_stats["excellent_after_retries"] += 1
                    except Exception as e:
                        print(f"Ошибка при обработке чанка: {e}")
                        # В случае ошибки, добавляем исходный чанк в список обработанных
                        chunk = futures[future]
                        result = {
                            "position": chunk["position"],
                            "processed_text": {
                                "ssml": chunk["text"],
                                "silero": chunk["text"]
                            },
                            "tokens_used": 0,
                            "quality_score": 1.0,  # Исходный текст имеет 100% сходство с самим собой
                            "original_text": chunk["text"],
                            "attempt": 1,
                            "error": str(e)
                        }
                        processed_chunks.append(result)
                        # Убираем двойное добавление и здесь
                        # self.processed_chunks.append(result)
        
        # Обновляем self.processed_chunks для сохранения в чекпоинтах
        self.processed_chunks = processed_chunks.copy()
        
        # Сортируем чанки по исходной позиции
        processed_chunks.sort(key=lambda x: x["position"])
        
        # Используем умное соединение чанков
        final_text = self.join_processed_chunks(processed_chunks)
        
        # Добавляем подробное логирование для отслеживания процесса
        print(f"\nРезультат объединения всех чанков:")
        print(f"Длина полного текста: {len(final_text['ssml'])} символов")
        newline_char = '\n\n'
        print(f"Количество абзацев в объединенном тексте: {final_text['ssml'].count(newline_char) + 1}")
        
        # Вычисляем общую оценку качества
        if processed_chunks:
            avg_quality = sum(result.get("quality_score", 0) for result in processed_chunks) / len(processed_chunks)
        else:
            print("Предупреждение: нет обработанных чанков для анализа качества")
            avg_quality = 0
        
        print(f"\nСтатистика обработки:")
        print(f"Использовано токенов: {self.total_tokens_used}")
        print(f"Примерная стоимость: ${self.total_cost:.2f}")
        print(f"Среднее качество сохранения информации: {avg_quality:.2f} из 1.0")
        
        # Статистика повторных обработок
        print("\nСтатистика повторных обработок:")
        print(f"Всего попыток обработки: {retry_stats['total_attempts']}")
        print(f"Чанков, потребовавших повторную обработку: {retry_stats['chunks_with_retries']}")
        print(f"Чанков с отличным качеством с первой попытки: {retry_stats['excellent_on_first_try']}")
        print(f"Чанков с отличным качеством после повторной обработки: {retry_stats['excellent_after_retries']}")
        print(f"Чанков, где лучшей оказалась не последняя попытка: {retry_stats['best_from_multiple_attempts']}")
        
        # Анализируем и выводим детальную статистику качества
        quality_stats = {
            "excellent": len([q for q in self.quality_results if q["quality_score"] > 0.9]),
            "good": len([q for q in self.quality_results if 0.8 < q["quality_score"] <= 0.9]),
            "acceptable": len([q for q in self.quality_results if 0.7 < q["quality_score"] <= 0.8]),
            "poor": len([q for q in self.quality_results if q["quality_score"] <= 0.7])
        }
        
        print("\nРаспределение качества по чанкам:")
        print(f"Отлично (>0.9): {quality_stats['excellent']} чанков")
        print(f"Хорошо (0.8-0.9): {quality_stats['good']} чанков")
        print(f"Приемлемо (0.7-0.8): {quality_stats['acceptable']} чанков")
        print(f"Низкое (<0.7): {quality_stats['poor']} чанков")
        
        # Создаем отчет о качестве в формате JSON
        quality_report = {
            "tokens_used": self.total_tokens_used,
            "cost": self.total_cost,
            "average_quality": avg_quality,
            "chunks_stats": quality_stats,
            "processing_time": time.time() - start_time,
            "date_processed": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Сохраняем отчет о качестве
        # with open(quality_report_file, "w", encoding="utf-8") as f:
        #     json.dump(quality_report, f, ensure_ascii=False, indent=4)
        
        # Определяем тип выходного текста
        processed_text = final_text
        
        # Сохраняем результат
        with open(output_text_file, "w", encoding="utf-8") as f:
            # Если текст пришел в виде словаря, берем первое значение
            if isinstance(processed_text, dict):
                output_text = list(processed_text.values())[0]
            else:
                output_text = processed_text
                
            # Преобразуем возможные переносы строк в нормализованный формат
            output_text = output_text.replace('\r\n', '\n')
            
            # Убираем запятые, которые были добавлены алгоритмом вместо переносов строк
            output_text = output_text.replace(', ,', ',')
            output_text = output_text.replace(' ,', ',')
            
            # Разделяем текст на абзацы
            text_blocks = []
            
            # Сначала попробуем разделить по двойным переносам
            if '\n\n' in output_text:
                text_blocks = [block.strip() for block in output_text.split('\n\n') if block.strip()]
            
            # Если не удалось разделить по двойным переносам, используем регулярные выражения
            if len(text_blocks) < 2:
                text_blocks = [block.strip() for block in re.split(r'\n\s*\n|(?<=[.?!])\s{2,}', output_text) if block.strip()]
            
            # Если всё ещё не удалось разбить на абзацы, используем поиск по знакам препинания
            if len(text_blocks) < 2:
                text_blocks = [block.strip() for block in re.split(r'(?<=[.?!])\s+(?=[А-ЯA-Z])', output_text) if block.strip()]
            
            # Удаляем все проверки, которые ограничивают количество абзацев до 6
            # и логику разделения/объединения для достижения этого количества
            
            print(f"Итоговое количество абзацев: {len(text_blocks)}")
            
            # Собираем текст с абзацами
            reconstructed_text = '\n\n'.join(text_blocks)
            
            # Удаляем дублирующиеся разделы
            cleaned_text = self.remove_duplicate_sections(reconstructed_text)
            
            # Удаляем дублирующиеся абзацы, если есть
            final_cleaned_text = self.remove_duplicate_paragraphs(cleaned_text)
            
            # Записываем финальный текст
            f.write(final_cleaned_text)
        
        # Заменяем все точки на запятые в выходном файле
        self.ensure_commas_in_output(output_text_file)
        
        print(f"\nВремя обработки: {time.time() - start_time:.2f} секунд")
        print(f"Результат сохранен в {output_text_file}")
        # print(f"Отчет о качестве сохранен в {quality_report_file}")
        
        # Выполняем повторный анализ и улучшение текста
        # try:
        #     print("\nНачинаю повторный анализ и улучшение текста...")
            
        #     # Читаем сохраненный результат
        #     with open(output_text_file, "r", encoding="utf-8") as f:
        #         processed_output = f.read()
            
        #     # Выполняем анализ и улучшение
        #     improved_text = self.analyze_and_improve_text(text, processed_output)
            
        #     # Сохраняем улучшенную версию
        #     with open(improved_text_file, "w", encoding="utf-8") as f:
        #         f.write(improved_text)
            
        #     # Заменяем все точки на запятые в улучшенном файле
        #     self.ensure_commas_in_output(improved_text_file)
            
        #     print(f"Улучшенная версия текста сохранена в {improved_text_file}")
            
        # except Exception as e:
        #     print(f"Ошибка при анализе и улучшении текста: {e}")
        
        # Сохраняем финальный чекпоинт
        # with open(checkpoint_file, "w", encoding="utf-8") as f:
        #     json.dump({"chunks": self.processed_chunks}, f, ensure_ascii=False, indent=2)
        
        return processed_text

    def join_processed_chunks(self, chunks: List[Dict]) -> Dict[str, str]:
        """Более умное соединение с учетом контекста."""
        processed_chunks = sorted(chunks, key=lambda x: x["position"])
        result_ssml = ""
        result_silero = ""
        
        for i, chunk in enumerate(processed_chunks):
            # Получаем SSML и Silero версии текста
            current_ssml = chunk["processed_text"]["ssml"] if isinstance(chunk["processed_text"], dict) else chunk["processed_text"]
            current_silero = chunk["processed_text"]["silero"] if isinstance(chunk["processed_text"], dict) else chunk["processed_text"]
            
            # Удаляем возможные дубликаты на стыках
            if i > 0:
                # SSML версия
                previous_ssml = processed_chunks[i-1]["processed_text"]["ssml"] if isinstance(processed_chunks[i-1]["processed_text"], dict) else processed_chunks[i-1]["processed_text"]
                overlap_ssml = self._find_text_overlap(previous_ssml, current_ssml)
                if overlap_ssml:
                    result_ssml += current_ssml[len(overlap_ssml):]
                else:
                    # Добавляем двойной перенос строки для разделения абзацев
                    result_ssml += "\n\n" + current_ssml
                
                # Silero версия
                previous_silero = processed_chunks[i-1]["processed_text"]["silero"] if isinstance(processed_chunks[i-1]["processed_text"], dict) else processed_chunks[i-1]["processed_text"]
                overlap_silero = self._find_text_overlap(previous_silero, current_silero)
                if overlap_silero:
                    result_silero += current_silero[len(overlap_silero):]
                else:
                    # Добавляем двойной перенос строки для разделения абзацев
                    result_silero += "\n\n" + current_silero
            else:
                result_ssml += current_ssml
                result_silero += current_silero
        
        # Применяем финальную лингвистическую обработку ко всему тексту
        if self.linguistic_processor:
            try:
                print("\nПрименяем финальную лингвистическую обработку ко всему тексту...")
                
                # Обрабатываем SSML-версию
                result_ssml = self.linguistic_processor.process_text(result_ssml, use_ssml=True)
                
                # Обрабатываем Silero-версию
                result_silero = self.linguistic_processor.process_text_for_silero(result_silero)
                
                print("Финальная лингвистическая обработка завершена успешно")
            except Exception as e:
                print(f"Ошибка финальной лингвистической обработки: {e}")
                
        return {
            "ssml": result_ssml,
            "silero": result_silero
        }

def main():
    agent = TextUpdateAgent()
    
    # Проверяем существование входной директории
    if not os.path.exists(agent.input_dir):
        print(f"Ошибка: Директория {agent.input_dir} не существует!")
        return
    
    # Получаем список всех файлов во входной директории
    input_files = [f for f in os.listdir(agent.input_dir) if os.path.isfile(os.path.join(agent.input_dir, f)) and f.endswith('.txt')]
    
    if not input_files:
        print(f"В директории {agent.input_dir} не найдено текстовых файлов (.txt)")
        return
    
    print(f"Найдено {len(input_files)} текстовых файлов для обработки:")
    for file in input_files:
        print(f"- {file}")
    
    # Обрабатываем каждый файл по очереди
    for file_name in input_files:
        print(f"\n{'='*60}")
        print(f"Обработка файла: {file_name}")
        print(f"{'='*60}")
        
        input_file = os.path.join(agent.input_dir, file_name)
        
        try:
            with open(input_file, "r", encoding="utf-8") as f:
                text = f.read()
            print(f"Загружен файл размером {len(text)} байт из {input_file}")
            print(f"Первые 100 символов: {text[:100]}")
        except Exception as e:
            print(f"Ошибка при загрузке файла {file_name}: {e}")
            continue
        
        try:
            # Обрабатываем текущий файл
            agent.process_text(text, file_name, force_reprocess=True)
            
        except KeyboardInterrupt:
            print(f"\n\nПрограмма прервана пользователем при обработке файла {file_name}.")
            choice = input("Хотите продолжить обработку других файлов? (y/n): ").strip().lower()
            if choice != 'y':
                print("Программа завершена по запросу пользователя.")
                break
        except Exception as e:
            print(f"Ошибка при обработке файла {file_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Очищаем список обработанных чанков перед обработкой следующего файла
        agent.processed_chunks = []
        agent.quality_results = []
    
    print("\nОбработка всех файлов завершена.")
    
    # Запускаем lection_to_audio.py для обработки всех файлов
    print("\nЗапуск конвертации текстов в аудио...")
    try:
        # Импортируем модуль lection_to_audio
        import importlib.util
        spec = importlib.util.spec_from_file_location("lection_to_audio", os.path.join("scripts", "lection_to_audio.py"))
        lection_to_audio = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lection_to_audio)
        
        # Запускаем обработку
        lection_to_audio.main()
        
    except Exception as e:
        print(f"Ошибка при запуске конвертации в аудио: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
