from pathlib import Path
from docx import Document
from striprtf.striprtf import rtf_to_text
import PyPDF2

def extract_text(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix == ".docx":
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    elif suffix == ".rtf":
        raw_text = file_path.read_text(encoding="utf-8", errors="ignore")
        return rtf_to_text(raw_text)
    elif suffix == ".pdf":
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    else:
        raise ValueError(f"Формат {suffix} не поддерживается")