import torch
import logging
from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger, Doc
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

logging.basicConfig(filename='analyzer.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Инициализация Natasha
segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)

# Модель для sentiment (русский)
try:
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="blanchefort/rubert-base-cased-sentiment"
    )
    logging.info("Модель sentiment-анализа загружена")
except Exception as e:
    logging.error(f"Ошибка загрузки модели sentiment: {e}")
    sentiment_analyzer = None

# Модель для семантического поиска (русский SBERT)
try:
    semantic_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    logging.info("Модель SBERT загружена")
except Exception as e:
    logging.error(f"Ошибка загрузки модели SBERT: {e}")
    semantic_model = None

def normalize_text(text: str):
    """Лемматизация текста"""
    try:
        text = text.lower().strip()
        doc = Doc(text)
        doc.segment(segmenter)
        doc.tag_morph(morph_tagger)
        lemmas = []
        for token in doc.tokens:
            token.lemmatize(morph_vocab)
            if token.lemma and (token.lemma.isalpha() or token.lemma in {'sql', 'python', 'it', 'osi', 'mikrotik', 'cisco', 'ssh', 'ubuntu'}):
                lemmas.append(token.lemma)
        return lemmas
    except Exception as e:
        logging.error(f"Ошибка нормализации текста: {e}")
        return []

def partial_match(req: str, resume_text: str) -> bool:
    """Проверка частичного совпадения текста"""
    try:
        req_words = set(req.lower().split())
        resume_words = set(resume_text.lower().split())
        return bool(req_words.intersection(resume_words))
    except Exception as e:
        logging.error(f"Ошибка в partial_match: {e}")
        return False

def semantic_match(req: str, text: str, threshold: float = 0.45) -> bool:
    """Семантическое сравнение текста"""
    try:
        if not semantic_model:
            logging.warning("Модель SBERT не загружена, семантический анализ невозможен")
            return False
        embeddings = semantic_model.encode([req, text], convert_to_tensor=True)
        similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
        logging.info(f"Семантическая похожесть: {similarity:.2f} (порог: {threshold})")
        return similarity >= threshold
    except Exception as e:
        logging.error(f"Ошибка в semantic_match: {e}")
        return False

def analyze_resume_vs_vacancy(resume_text: str, vacancy: dict) -> dict:
    """Анализ соответствия резюме вакансии"""
    try:
        resume_lemmas = set(normalize_text(resume_text))
        matched, missing = [], []

        for req in vacancy.get("requirements", []):
            req_lemmas = set(normalize_text(req))
            if resume_lemmas.intersection(req_lemmas) or partial_match(req, resume_text) or semantic_match(req, resume_text):
                matched.append(req)
            else:
                missing.append(req)

        score = round(len(matched) / len(vacancy["requirements"]) * 100, 1) if vacancy.get("requirements") else 0.0
        logging.info(f"Извлеченный текст резюме: {resume_text[:500]}")
        logging.info(f"Требования вакансии: {vacancy.get('requirements', [])}")
        logging.info(f"Анализ резюме: score={score}, matched={matched}, missing={missing}")
        return {
            "vacancy": vacancy.get("title", ""),
            "score": score,
            "matched": matched,
            "missing": missing
        }
    except Exception as e:
        logging.error(f"Ошибка в analyze_resume_vs_vacancy: {e}")
        return {
            "vacancy": vacancy.get("title", ""),
            "score": 0.0,
            "matched": [],
            "missing": vacancy.get("requirements", [])
        }

def analyze_interview(answers: list, vacancy: dict) -> dict:
    """Анализ ответов на интервью"""
    try:
        matched, strong_points, gaps = [], [], []
        score = 0
        total_questions = len(answers)
        vacancy_reqs = vacancy.get("requirements", [])

        for ans in answers:
            ans_text = ans.get("answer", "").strip()
            q = ans.get("question", "")
            ans_lemmas = set(normalize_text(ans_text))
            low = ans_text.lower()

            # Sentiment-анализ
            try:
                if sentiment_analyzer:
                    sentiment = sentiment_analyzer(ans_text)[0]
                    label = sentiment["label"]
                    sent_score = sentiment["score"]
                    logging.info(f"Sentiment: {label}, score={sent_score:.2f}")
                    if label == "POSITIVE" and sent_score > 0.7:
                        strong_points.append("Позитивный настрой в ответе")
                    elif label == "NEGATIVE" and sent_score > 0.7:
                        gaps.append("Негативный тон ответа")
                    elif sent_score < 0.4:
                        gaps.append("Неуверенный тон ответа")
                else:
                    logging.warning("Sentiment-анализ недоступен")
            except Exception as e:
                logging.error(f"Ошибка sentiment-анализа: {e}")

            # Совпадение с требованиями (вес 0.6)
            for req in vacancy_reqs:
                req_lemmas = set(normalize_text(req))
                if ans_lemmas.intersection(req_lemmas) or semantic_match(req, ans_text, 0.5):
                    if req not in matched:
                        matched.append(req)
                        score += 0.6
                        logging.info(f"Совпадение с требованием: {req}")

            # Релевантность к вопросу (вес 0.2)
            if semantic_match(q, ans_text, 0.5):
                strong_points.append("Ответ релевантен вопросу")
                score += 0.2
                logging.info("Ответ релевантен вопросу")
            else:
                gaps.append("Ответ не полностью релевантен вопросу")
                logging.info("Ответ не релевантен вопросу")

            # Конкретность ответа (вес 0.2)
            tech_terms = ["python", "crm", "ai", "модель", "беспилотник", "автоматизация"]
            if len(ans_text.split()) > 10 and (any(c.isdigit() for c in ans_text) or "пример" in low or "например" in low or any(term in low for term in tech_terms)):
                strong_points.append("Конкретный ответ с примерами")
                score += 0.2
                logging.info("Ответ конкретен")
            else:
                gaps.append("Ответ слишком общий или короткий")
                logging.info("Ответ неконкретен")

            # Проверка длины и длительности
            if len(ans_text.split()) < 3:
                gaps.append("Слишком короткий ответ")
                logging.info("Ответ слишком короткий")
            elif ans.get("duration", 0) > 60:
                strong_points.append("Хорошие коммуникативные навыки")
                logging.info("Хорошие коммуникативные навыки")

        max_possible = total_questions * (0.6 * len(set(vacancy_reqs)) + 0.2 + 0.2)
        interview_score = round((score / max_possible) * 100, 1) if max_possible else 0.0
        logging.info(f"score={score}, max_possible={max_possible}, interview_score={interview_score}")

        result = {
            "score": interview_score,
            "matched": matched,
            "missing": [req for req in vacancy_reqs if req not in set(matched)],
            "strong_points": list(set(strong_points)),
            "gaps": list(set(gaps))
        }
        logging.info(f"Анализ интервью: {result}")
        return result
    except Exception as e:
        logging.error(f"Критическая ошибка в analyze_interview: {e}")
        return {
            "score": 0.0,
            "matched": [],
            "missing": vacancy.get("requirements", []),
            "strong_points": [],
            "gaps": ["Ошибка анализа ответов"]
        }