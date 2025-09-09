import sqlite3
from pathlib import Path
import datetime

DB_PATH = Path(__file__).parent / "db" / "hr_assistant.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS candidates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fio TEXT,
        resume_text TEXT,
        vacancy_id TEXT,
        interview_json TEXT,
        score REAL,
        report_json TEXT,
        timestamp TEXT
    )
    """)
    conn.commit()
    conn.close()

def save_candidate(data: dict):
    init_db()  # На всякий случай
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    INSERT INTO candidates (fio, resume_text, vacancy_id, interview_json, score, report_json, timestamp)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (data['fio'], data['resume_text'], data['vacancy_id'], data['interview_json'],
          data['score'], data['report_json'], datetime.datetime.now().isoformat()))
    conn.commit()
    conn.close()