import random
import re
import time
import logging
from tts_helper import speak
from llama_cpp import Llama

logging.basicConfig(filename='interview_helper.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# НАСТРОЙКИ МОДЕЛИ
SYSTEM_PROMPT = (
    "Ты — HR-интервьюер. Задавай ровно один конкретный вопрос на русском языке, адаптированный к ответу кандидата, вакансии и истории диалога. "
    "Делай вопрос релевантным, уточняющим или углубляющим предыдущий ответ. "
    "НЕ давай списки, НЕ используй вступления, НЕ повторяй вопросы, НЕ добавляй заголовки вроде 'Примеры вопросов'. "
    "Обязательно учти предыдущий ответ кандидата для создания нового вопроса."
)

# Инициализация модели
try:
    llm = Llama(
        model_path="C:/Users/tttoli4/Desktop/Xakaton_1/models/llama-2-7b.Q4_K_M.gguf",
        n_ctx=2048,
        n_threads=6
    )
    logging.info("Модель LLaMA успешно загружена")
except Exception as e:
    logging.error(f"Ошибка загрузки модели LLaMA: {e}")

def normalize_question_text(text: str) -> str:
    """Нормализация текста вопроса"""
    text = text.strip()
    text = re.sub(r"^(Примеры вопросов|Вопрос:|Example questions:)\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^[\-\*\d\.\)]\s*", "", text)
    if "?" in text:
        text = text.split("?")[0] + "?"
    return text.strip()

def ai_generate_question(vacancy: dict, history: list, asked_questions: list, previous_answer: str = "") -> str:
    """Генерация адаптивного вопроса с учетом вакансии, истории и предыдущего ответа"""
    vacancy_info = (
        f"Вакансия: {vacancy.get('title', '')}\n"
        f"Требования: {', '.join(vacancy.get('requirements', []))}\n"
        f"Обязанности: {', '.join(vacancy.get('duties', []))}\n"
    )
    dialogue = "\n".join(history[-12:]) if history else "Диалог ещё не начат."
    prev_qs = "\n".join(asked_questions[-12:]) if asked_questions else "Нет"

    prompt = (
        SYSTEM_PROMPT + "\n\n" +
        vacancy_info + "\n" +
        "История диалога:\n" + dialogue + "\n\n" +
        f"Ранее заданные вопросы (не повторяй их):\n{prev_qs}\n\n" +
        f"Предыдущий ответ кандидата (учти его для адаптации): {previous_answer}\n\n" +
        "Сформулируй ровно ОДИН новый вопрос на русском языке. Только вопрос, без лишнего текста."
    )

    fallback_questions = vacancy.get('questions', [])  # Фоллбэк на вопросы из JSON

    for attempt in range(3):
        try:
            logging.info(f"Промпт для LLaMA: {prompt[:500]}")
            resp = llm(prompt, max_tokens=120, temperature=0.45,
                       stop=["HR:", "Кандидат:", "Candidate:"])
            raw = resp.get("choices", [{}])[0].get("text", "") if isinstance(resp, dict) else str(resp)
            logging.info(f"Сырой ответ LLaMA: {raw}")
            text = normalize_question_text(raw)

            if text and text not in asked_questions and len(text) > 5 and text.endswith("?"):
                asked_questions.append(text)
                logging.info(f"Сгенерирован вопрос: {text}")
                return text
            else:
                prompt += "\nТы уже задавал этот вопрос или он некорректен, придумай другой."
                logging.warning(f"Повтор вопроса или некорректный: {text}, попытка {attempt + 1}")
                continue
        except Exception as e:
            logging.error(f"Ошибка генерации вопроса: {e}")
            break

    # Фоллбэк
    fallback = random.choice([q for q in fallback_questions if q not in asked_questions] or fallback_questions)
    asked_questions.append(fallback)
    logging.info(f"Использован фоллбэк-вопрос: {fallback}")
    return fallback

def conduct_interview(vacancy: dict, log_callback, recognizer, max_q=3):
    """
    Основной цикл интервью.
    log_callback — функция для вывода лога в GUI.
    recognizer — объект распознавания речи.
    max_q — количество вопросов (фиксировано 3).
    """
    answers = []
    history = []
    asked_questions = []
    questions = vacancy.get("questions", [])

    if not questions:
        log_callback("Ошибка: в вакансии нет вопросов!")
        logging.error("Вакансия не содержит вопросов")
        return answers

    log_callback("Начинаем интервью...")

    # Первый вопрос — фиксированный из vacancies.json или сгенерированный
    q = questions[0] if questions else ai_generate_question(vacancy, history, asked_questions)
    asked_questions.append(q)

    for i in range(max_q):
        try:
            # Выводим и озвучиваем вопрос
            log_callback(f"Вопрос {i + 1}: {q}")
            try:
                speak(q)
            except Exception as e:
                log_callback(f"Ошибка озвучивания: {e}")
                logging.error(f"Ошибка озвучивания вопроса {i + 1}: {e}")

            # Активируем кнопку "Остановить запись"
            log_callback("[ENABLE_STOP]")

            # Слушаем ответ
            answer_text = ""
            duration = 0
            try:
                resp = recognizer.listen_and_transcribe(timeout=40, chunk_duration=5)
                answer_text = resp.get("text", "").strip()
                duration = resp.get("duration", 0)
                if resp.get("stopped_manually", False):
                    log_callback("Запись остановлена пользователем, переходим к следующему вопросу.")
                if answer_text:
                    log_callback(f"Ответ кандидата: {answer_text} (длительность: {duration:.1f}s)")
                else:
                    log_callback("Ответ не получен или пустой.")
            except Exception as e:
                log_callback(f"Ошибка распознавания: {e}")
                logging.error(f"Ошибка распознавания для вопроса {i + 1}: {e}")

            # Деактивируем кнопку "Остановить запись"
            log_callback("[DISABLE_STOP]")

            # Проверяем ключевые фразы для остановки записи (аналог кнопки)
            low = answer_text.lower()
            stop_phrases = ["всё, больше ничего", "закончил", "ничего больше", "все вопросы ответил", "всё", "все", "спасибо", "на этом все"]
            if any(phrase in low for phrase in stop_phrases):
                logging.info(f"Обнаружена фраза '{low}', переходим к следующему вопросу для вопроса {i + 1}")

            # Сохраняем результат
            answers.append({"question": q, "answer": answer_text, "duration": duration})
            logging.info(f"Сохранен ответ для вопроса {i + 1}: {answer_text}")

            # Генерация следующего вопроса на основе ответа
            if i < max_q - 1:
                q = ai_generate_question(vacancy, history, asked_questions, answer_text)
                history.append(f"HR: {q}")
                history.append(f"Кандидат: {answer_text}")

            time.sleep(1)
        except Exception as e:
            log_callback(f"Критическая ошибка в цикле интервью: {e}")
            logging.error(f"Критическая ошибка в цикле интервью для вопроса {i + 1}: {e}")
            answers.append({"question": q, "answer": "", "duration": 0})
            continue

    log_callback("Интервью завершено.")
    logging.info(f"Интервью завершено, собрано {len(answers)} ответов")
    return answers