from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from src.chunking import validate_page_marker


PAGE_RE = re.compile(r"\[\s*(\d+)\s*\]\s*$", re.MULTILINE)


@dataclass(frozen=True)
class PageChunk:
    book_id: str
    page: int
    chunk_id: str
    text: str
    path: str


import re

PAGE_LINE_RE = re.compile(r"^\[\s*(\d+)\s*\]\s*$", re.MULTILINE)

def split_text_by_page_markers(text: str):
    # 1) Находим последний маркер страницы
    matches = list(PAGE_LINE_RE.finditer(text))
    if not matches:
        raise ValueError("Не найдено ни одного маркера страницы вида '[ N ]'.")

    last = matches[-1]
    tail = text[last.end():]
    if tail.strip():
        print(f"[WARN] Dropping tail after last page marker: {len(tail)} chars")

    text = text[:last.end()]


    pages: list[tuple[int, str]] = []
    seen = set()
    start = 0
    for m in matches:
        page_no = int(m.group(1))
        end = m.end()
        page_text = text[start:end].strip()
        if not page_text:
            raise ValueError(f"Пустая страница: [ {page_no} ].")
        if page_no in seen:
            raise ValueError(f"Найдены дубликаты номера страницы: {page_no}.")
        validate_page_marker(page_no, page_text)
        pages.append((page_no, page_text))
        seen.add(page_no)
        start = end

    tail = text[start:]
    if tail.strip():
        raise ValueError(
            "TXT/MD parsing: найден хвост без маркера страницы. "
            "Файл должен заканчиваться строкой вида '[ N ]'."
        )

    return pages


def write_pages(pages: list[tuple[int, str]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for page_no, page_text in pages:
        fp = out_dir / f"page_{page_no:04d}.txt"
        fp.write_text(page_text + "\n", encoding="utf-8")


def load_page_chunks(book_id: str, pages_dir: str | Path) -> list[PageChunk]:
    pages_dir = Path(pages_dir)
    files = sorted(pages_dir.glob("page_*.txt"))
    if not files:
        raise FileNotFoundError(f"Нет файлов page_*.txt в {pages_dir}")

    chunks: list[PageChunk] = []
    for fp in files:
        try:
            page = int(fp.stem.split("_")[-1])
        except ValueError as exc:
            raise ValueError(f"Некорректное имя файла страницы: {fp.name}") from exc
        text = fp.read_text(encoding="utf-8", errors="ignore").strip()
        validate_page_marker(page, text)
        chunk_id = f"{book_id}:p{page:04d}"
        chunks.append(
            PageChunk(
                book_id=book_id,
                page=page,
                chunk_id=chunk_id,
                text=text,
                path=str(fp),
            )
        )

    validate_page_chunks(chunks)
    return chunks


def validate_page_chunks(chunks: list[PageChunk]) -> None:
    if not chunks:
        raise ValueError("Пустой список страниц: нечего индексировать.")

    ids = [c.chunk_id for c in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("Найдены неуникальные chunk_id (дубликаты страниц).")

    pages = [c.page for c in chunks]
    if len(pages) != len(set(pages)):
        raise ValueError("Найдены дубликаты номеров страниц.")

    empty_pages = [c.page for c in chunks if not c.text.strip()]
    if empty_pages:
        raise ValueError(f"Найдены пустые страницы: {sorted(empty_pages)[:10]}...")
