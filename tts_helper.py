# stt_helper.py
import threading
import wave
import time
from pathlib import Path
import pyttsx3

import pyaudio
from faster_whisper import WhisperModel

ASR_MODEL = "base"  # или "tiny" / "small"
_WHISPER = WhisperModel(ASR_MODEL, device="cpu", compute_type="int8")

_record_thread = None
_stop_event = None
_frames = []
_last_file = Path("temp_audio.wav")
_last_duration = 0.0
_lock = threading.Lock()

# Параметры записи
_RATE = 16000
_CHANNELS = 1
_CHUNK = 1024
_FORMAT = pyaudio.paInt16


def start_recording(filename: str = "temp_audio.wav"):
    """Запускает запись в фоновом потоке. Возвращает True если стартовали."""
    global _record_thread, _stop_event, _frames, _last_file, _last_duration

    if _record_thread and _record_thread.is_alive():
        return False

    _stop_event = threading.Event()
    _frames = []
    _last_file = Path(filename)
    _last_duration = 0.0

    def _record():
        global _frames, _last_duration
        pa = pyaudio.PyAudio()
        try:
            stream = pa.open(format=_FORMAT, channels=_CHANNELS, rate=_RATE,
                             input=True, frames_per_buffer=_CHUNK)
        except Exception as e:
            print("Ошибка открытия микрофона (record):", e)
            pa.terminate()
            return

        start = time.time()
        try:
            while not _stop_event.is_set():
                data = stream.read(_CHUNK, exception_on_overflow=False)
                _frames.append(data)
        except Exception as e:
            print("Ошибка в процессе записи:", e)
        finally:
            end = time.time()
            # закроем поток и запишем WAV
            try:
                stream.stop_stream()
                stream.close()
            except:
                pass
            sample_width = pa.get_sample_size(_FORMAT)
            pa.terminate()

            with wave.open(str(_last_file), "wb") as wf:
                wf.setnchannels(_CHANNELS)
                wf.setsampwidth(sample_width)
                wf.setframerate(_RATE)
                wf.writeframes(b"".join(_frames))

            _last_duration = end - start

    _record_thread = threading.Thread(target=_record, daemon=True)
    _record_thread.start()
    return True


def stop_recording():
    """Останавливает запись (просит фоновый поток завершиться)."""
    global _stop_event
    if _stop_event:
        _stop_event.set()


def is_recording() -> bool:
    """True если запись ещё идёт."""
    return _record_thread is not None and _record_thread.is_alive()


def wait_recording_finish(timeout: float = 5.0):
    """Дождаться окончания фонового потока записи (join)."""
    global _record_thread
    if _record_thread:
        _record_thread.join(timeout)


def transcribe_last(vad_filter: bool = True, language: str = "ru") -> dict:
    """
    Транскрибирует последний записанный файл и возвращает {"text": ..., "duration": ...}
    Подождёт окончания записи (join).
    """
    global _last_file, _last_duration

    wait_recording_finish(timeout=10.0)
    if not _last_file.exists():
        return {"text": "", "duration": 0.0}
    try:
        segments, _ = _WHISPER.transcribe(str(_last_file), vad_filter=vad_filter, language=language)
    except Exception as e:
        print("VAD/transcribe error (fallback):", e)
        segments, _ = _WHISPER.transcribe(str(_last_file), vad_filter=False, language=language)

    text = " ".join([seg.text for seg in segments]).strip()
    return {"text": text, "duration": _last_duration}


_engine = None

def get_engine():
    global _engine
    if _engine is None:
        _engine = pyttsx3.init()
        _engine.setProperty('rate', 160)

    return _engine

def speak(text: str):
    engine = get_engine()
    engine.say(text)
    engine.runAndWait()