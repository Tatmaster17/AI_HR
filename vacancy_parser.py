import json
from pathlib import Path

BASE_DIR = Path(__file__).parent
VACANCIES_JSON = BASE_DIR / "vacancies.json"

def extract_vacancy(vac_id: str) -> dict:
    with open(VACANCIES_JSON, 'r', encoding='utf-8') as f:
        vacancies = json.load(f)
    for vac in vacancies:
        if vac['id'] == vac_id:
            return vac
    raise ValueError(f"Вакансия с ID {vac_id} не найдена")