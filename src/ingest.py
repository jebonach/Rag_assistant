from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any


@dataclass(frozen=True)
class PageDoc:
    page: int
    text: str


def read_pdf_pages(pdf_path: Path) -> List[PageDoc]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    pages: List[PageDoc] = []

    for i, p in enumerate(reader.pages, start=1):
        txt = p.extract_text() or ""
        txt = normalize_text(txt)
        # Важно: сохраняем номер страницы в конце как в условии кейса
        if txt:
            txt = f"{txt}\n[ {i} ]"
        pages.append(PageDoc(page=i, text=txt))

    return pages


def normalize_text(s: str) -> str:
    # Минимальная нормализация: убираем лишние пробелы/пустые строки
    lines = [ln.strip() for ln in s.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def pages_to_rows(pages: List[PageDoc]) -> List[Dict[str, Any]]:
    return [{"page": p.page, "text": p.text} for p in pages]
