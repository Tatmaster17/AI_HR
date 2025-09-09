import sys
import json
import logging
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog, QComboBox, QTextEdit, QMessageBox
)
from PySide6.QtCore import QThread, Signal
from resume_parser import extract_text
from vacancy_parser import extract_vacancy
from analyzer import analyze_resume_vs_vacancy, analyze_interview
from interview_helper import conduct_interview
from report_generator import generate_report
from db_helper import save_candidate
from tts_helper import speak
from stt_helper import SpeechRecognizer
import pyaudio

logging.basicConfig(filename='main.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = Path(__file__).parent
FILES_DIR = BASE_DIR / "files"
VACANCIES_JSON = BASE_DIR / "vacancies.json"

class InterviewThread(QThread):
    update_log = Signal(str)
    finished = Signal(dict)

    def __init__(self, vacancy, recognizer, parent=None):
        super().__init__(parent)
        self.vacancy = vacancy
        self.recognizer = recognizer

    def run(self):
        try:
            answers = conduct_interview(self.vacancy, self.update_log.emit, self.recognizer)
            logging.info(f"Interview completed: {answers}")
            self.finished.emit({"answers": answers})
        except Exception as e:
            self.update_log.emit(f"Критическая ошибка в интервью: {str(e)}")
            logging.error(f"InterviewThread error: {str(e)}")
            self.finished.emit({"answers": []})

class HRWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI HR Assistant")
        self.resize(600, 650)
        layout = QVBoxLayout()

        # ФИО
        layout.addWidget(QLabel("ФИО кандидата:"))
        self.fio_input = QLineEdit()
        layout.addWidget(self.fio_input)

        # Выбор резюме
        self.resume_btn = QPushButton("Выбрать резюме")
        layout.addWidget(self.resume_btn)
        self.resume_file = None

        # Выбор вакансии
        layout.addWidget(QLabel("Выберите вакансию:"))
        self.vacancy_combo = QComboBox()
        self.load_vacancies()
        layout.addWidget(self.vacancy_combo)

        # Кнопка старта
        self.start_btn = QPushButton("Начать анализ и интервью")
        layout.addWidget(self.start_btn)

        # Кнопка остановки записи
        self.stop_btn = QPushButton("Остановить запись")
        self.stop_btn.setEnabled(False)
        layout.addWidget(self.stop_btn)

        # Лог/результат
        self.result_box = QTextEdit()
        self.result_box.setReadOnly(True)
        layout.addWidget(self.result_box)

        self.setLayout(layout)

        #SpeechRecognizer
        self.recognizer = None
        try:
            self.recognizer = SpeechRecognizer(model_size="small", device="cpu")
            logging.info("SpeechRecognizer успешно инициализирован")
        except Exception as e:
            logging.error(f"Ошибка инициализации SpeechRecognizer: {e}")
            QMessageBox.critical(self, "Ошибка", f"Не удалось инициализировать распознаватель речи: {e}. Проверьте установку faster_whisper.")
            self.start_btn.setEnabled(False)
            return

        # Проверка микрофона
        try:
            p = pyaudio.PyAudio()
            p.get_default_input_device_info()
            p.terminate()
            logging.info("Микрофон доступен")
        except Exception as e:
            logging.warning(f"Микрофон недоступен: {e}")
            QMessageBox.warning(self, "Предупреждение", f"Микрофон недоступен: {e}. Интервью может не работать корректно.")

        # События
        self.resume_btn.clicked.connect(self.select_resume)
        self.start_btn.clicked.connect(self.start_process)
        self.stop_btn.clicked.connect(self.on_stop_clicked)

    def load_vacancies(self):
        if not VACANCIES_JSON.exists():
            QMessageBox.critical(self, "Ошибка", "vacancies.json не найден!")
            logging.error("vacancies.json не найден")
            return
        try:
            with open(VACANCIES_JSON, 'r', encoding='utf-8') as f:
                vacancies = json.load(f)
            for vac in vacancies:
                self.vacancy_combo.addItem(vac['title'], vac['id'])
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка загрузки вакансий: {e}")
            logging.error(f"Ошибка загрузки vacancies.json: {e}")

    def select_resume(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Выбрать резюме", str(FILES_DIR), "Документы (*.docx *.rtf *.pdf)"
        )
        if file:
            self.resume_file = Path(file)
            self.resume_btn.setText(f"Резюме: {self.resume_file.name}")

    def on_stop_clicked(self):
        """Остановить запись пользователем"""
        try:
            if self.recognizer:
                self.recognizer.stop_recording()
                self.stop_btn.setEnabled(False)
                self.result_box.append("Запись остановлена пользователем.")
                logging.info("Запись остановлена пользователем через GUI")
        except Exception as e:
            self.result_box.append(f"Ошибка при остановке записи: {e}")
            logging.error(f"Ошибка в on_stop_clicked: {e}")

    def handle_update_log(self, msg: str):
        """
        Перехватываем спец-сообщения от conduct_interview:
          - "[ENABLE_STOP]" -> включить кнопку остановки
          - "[DISABLE_STOP]" -> выключить кнопку
        Остальные сообщения отображаем, кроме отладочных.
        """
        try:
            if msg == "[ENABLE_STOP]":
                self.stop_btn.setEnabled(True)
                self.result_box.append("Нажмите 'Остановить запись', когда закончите отвечать...")
            elif msg == "[DISABLE_STOP]":
                self.stop_btn.setEnabled(False)
            elif not msg.startswith("Interview answers:"):
                self.result_box.append(msg)
            else:
                logging.info(msg)
        except Exception as e:
            logging.error(f"Ошибка в handle_update_log: {e}")

    def start_process(self):
        fio = self.fio_input.text().strip()
        if not fio or not self.resume_file or self.vacancy_combo.currentIndex() == -1:
            QMessageBox.warning(self, "Ошибка", "Заполните все поля!")
            logging.warning("Незаполнены поля для старта процесса")
            return
        if not self.recognizer:
            QMessageBox.critical(self, "Ошибка", "Распознаватель речи не инициализирован. Проверьте настройки.")
            logging.error("Попытка начать интервью без инициализированного SpeechRecognizer")
            return

        vac_id = self.vacancy_combo.currentData()
        self.result_box.clear()
        self.result_box.append("Начало анализа...")

        try:
            resume_text = extract_text(self.resume_file)
            vacancy = extract_vacancy(vac_id)
            resume_report = analyze_resume_vs_vacancy(resume_text, vacancy)

            self.result_box.append(f"Анализ резюме: {resume_report['score']}% соответствия.")
            self.result_box.append("Начало интервью...")
            self.start_btn.setEnabled(False)

            self.interview_thread = InterviewThread(vacancy, self.recognizer)
            self.interview_thread.update_log.connect(self.handle_update_log)
            self.interview_thread.finished.connect(
                lambda data: self.finish_process(data, fio, resume_text, vacancy, resume_report)
            )
            self.interview_thread.start()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))
            logging.error(f"Ошибка в start_process: {e}")
            self.start_btn.setEnabled(True)

    def finish_process(self, data, fio, resume_text, vacancy, resume_report):
        try:
            answers = data['answers']
            interview_report = analyze_interview(answers, vacancy)
            total_score = round(resume_report['score'] * 0.4 + interview_report['score'] * 0.6, 1)
            report = generate_report(
                total_score,
                resume_report['matched'] + interview_report['matched'],
                resume_report['missing'] + interview_report['missing'],
                interview_report.get('strong_points', []),
                interview_report.get('gaps', [])
            )

            self.result_box.append(f"Общий скоринг: {total_score}%")
            self.result_box.append(report)

            candidate_data = {
                'fio': fio,
                'resume_text': resume_text,
                'vacancy_id': vacancy['id'],
                'interview_json': json.dumps(answers, ensure_ascii=False),
                'score': total_score,
                'report_json': json.dumps(report, ensure_ascii=False)
            }
            save_candidate(candidate_data)
            self.result_box.append("Данные сохранены в БД.")
            speak("Интервью завершено.")
            self.start_btn.setEnabled(True)
        except Exception as e:
            self.result_box.append(f"Ошибка в обработке результатов: {e}")
            logging.error(f"Ошибка в finish_process: {e}")
            self.start_btn.setEnabled(True)

if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        win = HRWindow()
        win.show()
        sys.exit(app.exec())
    except Exception as e:
        logging.error(f"Критическая ошибка приложения: {e}")
        print(f"Критическая ошибка: {e}")