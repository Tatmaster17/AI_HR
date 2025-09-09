import pyaudio
import wave
import time
import logging
import threading
import io
from pathlib import Path
from faster_whisper import WhisperModel

logging.basicConfig(filename='stt_helper.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class SpeechRecognizer:
    def __init__(self, model_size="small", device="cpu"):
        try:
            self.model = WhisperModel(model_size, device=device, compute_type="int8")
            logging.info(f"Whisper модель '{model_size}' успешно загружена")
        except Exception as e:
            logging.error(f"Ошибка загрузки модели Whisper: {e}")
            raise ValueError(f"Не удалось загрузить модель Whisper: {e}")

        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.recording = False
        self.stopped_manually = False
        self.frames = []
        self.transcribed_text = []
        self._lock = threading.Lock()
        self.stream = None
        self.pyaudio_instance = None
        self._transcription_active = False  # Флаг для отслеживания активных транскрибаций

    def _transcribe_chunk(self, audio_bytes):
        """Фоновая транскрибация аудио с таймаутом"""
        try:
            # Проверяем, активна ли запись
            if not self.recording:
                logging.info("Транскрибация отменена: запись остановлена")
                return

            # Ограничиваем время выполнения транскрибации
            start_time = time.time()
            timeout = 5  # Таймаут 5 сек

            wav_io = io.BytesIO()
            with wave.open(wav_io, "wb") as wf:
                wf.setnchannels(self.CHANNELS)
                wf.setsampwidth(2)
                wf.setframerate(self.RATE)
                wf.writeframes(audio_bytes)

            wav_io.seek(0)
            if time.time() - start_time > timeout:
                logging.warning("Транскрибация куска превысила таймаут")
                return

            segments, _ = self.model.transcribe(wav_io, vad_filter=True, language="ru")
            text = " ".join([seg.text for seg in segments]).strip()

            if text and self.recording:  # Проверяем, что запись все еще активна
                with self._lock:
                    self.transcribed_text.append(text)
                logging.info(f"Промежуточный результат: {text}")
        except Exception as e:
            logging.error(f"Ошибка транскрибации куска: {e}")
        finally:
            self._transcription_active = False  # Сбрасываем флаг

    def start_recording(self):
        """Запуск записи"""
        with self._lock:
            self.frames = []
            self.transcribed_text = []
            self.recording = True
            self.stopped_manually = False
            self._transcription_active = False
            try:
                self.pyaudio_instance = pyaudio.PyAudio()
                self.stream = self.pyaudio_instance.open(
                    format=self.FORMAT,
                    channels=self.CHANNELS,
                    rate=self.RATE,
                    input=True,
                    frames_per_buffer=self.CHUNK
                )
                logging.info("Микрофон открыт, запись начата")
            except Exception as e:
                logging.error(f"Ошибка запуска записи: {e}")
                self.recording = False
                self.stream = None
                self.pyaudio_instance = None
                raise

    def stop_recording(self):
        """Остановка записи"""
        with self._lock:
            self.recording = False
            self.stopped_manually = True
            try:
                if self.stream is not None:
                    self.stream.stop_stream()
                    self.stream.close()
                    self.stream = None
                if self.pyaudio_instance is not None:
                    self.pyaudio_instance.terminate()
                    self.pyaudio_instance = None
                logging.info("Запись остановлена")
            except Exception as e:
                logging.error(f"Ошибка остановки записи: {e}")
                self.stream = None
                self.pyaudio_instance = None
            finally:
                self._transcription_active = False  # Сбрасываем флаг для всех транскрибаций

    def listen_and_transcribe(self, timeout=30, chunk_duration=5):
        """Потоковая запись и транскрибация"""
        try:
            self.start_recording()
            start_time = time.time()
            chunk_frames = []
            chunk_start = start_time

            while self.recording and (time.time() - start_time) < timeout:
                try:
                    with self._lock:
                        if self.stream is None or not self.recording:
                            logging.info("Чтение аудио прервано: поток закрыт или запись остановлена")
                            break
                    data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                    chunk_frames.append(data)
                    self.frames.append(data)

                    if (time.time() - chunk_start) >= chunk_duration and not self._transcription_active:
                        self._transcription_active = True
                        audio_bytes = b"".join(chunk_frames)
                        threading.Thread(target=self._transcribe_chunk, args=(audio_bytes,), daemon=True).start()
                        chunk_frames = []
                        chunk_start = time.time()
                except Exception as e:
                    logging.error(f"Ошибка чтения аудио: {e}")
                    break

            was_stopped_manually = self.stopped_manually
            self.stop_recording()

            final_text = ""
            if self.frames:
                try:
                    wav_path = Path("final_audio.wav")
                    with wave.open(str(wav_path), "wb") as wf:
                        wf.setnchannels(self.CHANNELS)
                        wf.setsampwidth(2)
                        wf.setframerate(self.RATE)
                        wf.writeframes(b"".join(self.frames))

                    segments, _ = self.model.transcribe(str(wav_path), vad_filter=True, language="ru")
                    final_text = " ".join([seg.text for seg in segments]).strip()
                except Exception as e:
                    logging.error(f"Ошибка финальной транскрибации: {e}")

            return {
                "text": final_text if final_text else " ".join(self.transcribed_text),
                "duration": time.time() - start_time,
                "stopped_manually": was_stopped_manually
            }
        except Exception as e:
            logging.error(f"Критическая ошибка в listen_and_transcribe: {e}")
            self.stop_recording()
            return {
                "text": " ".join(self.transcribed_text),
                "duration": time.time() - start_time,
                "stopped_manually": self.stopped_manually
            }