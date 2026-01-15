from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional
from src.chunking import validate_page_marker

import regex as re


# Matches page marker like "[ 123 ]" possibly with spaces, at end of a line
_PAGE_RE = re.compile(r"^\s*\[\s*(\d+)\s*\]\s*$")


@dataclass(frozen=True)
class PageDoc:
    page: int
    text: str


def read_txt_pages(txt_path: Path, encoding: str = "utf-8") -> List[PageDoc]:
    """
    Reads a textbook from TXT and splits it into pages using markers like:
    [ 1 ]
    [ 2 ]
    ...
    The marker is expected to appear at the end of each page (often as its own line).
    """
    if not txt_path.exists():
        raise FileNotFoundError(f"TXT not found: {txt_path}")

    raw = txt_path.read_text(encoding=encoding, errors="replace")
    lines = raw.splitlines()

    pages: List[PageDoc] = []
    buf: List[str] = []
    current_page: Optional[int] = None

    def flush(page_num: int, buf_lines: List[str]) -> None:
        txt = normalize_text("\n".join(buf_lines))
        # сохраняем номер страницы в конце как в условии
        if txt:
            txt = f"{txt}\n[ {page_num} ]"
        pages.append(PageDoc(page=page_num, text=txt))

    for ln in lines:
        m = _PAGE_RE.match(ln)
        if m:
            page_num = int(m.group(1))
            # если встретили маркер — завершаем страницу
            if buf or current_page is not None:
                flush(page_num, buf)  # маркер относится к только что накопленному тексту
            else:
                # пустая страница: сохраняем как пустую
                pages.append(PageDoc(page=page_num, text=f"[ {page_num} ]"))
            buf = []
            current_page = page_num
        else:
            buf.append(ln)

    # Если файл не заканчивается маркером — это подозрительно.
    # Сохраняем хвост как page=-1, чтобы сразу видеть проблему в sanity-check.
    tail = normalize_text("\n".join(buf))
    if tail:
        pages.append(PageDoc(page=-1, text=tail))

    for p in pages:
        if p.page <= 0:
            raise ValueError(f"Invalid page number: {p.page}")
        validate_page_marker(p.page, p.text)

    validate_pages(pages)
    return pages


def normalize_text(s: str) -> str:
    lines = [ln.strip() for ln in s.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def validate_pages(pages: List[PageDoc]) -> None:
    # Если есть page=-1, значит последняя страница без маркера
    if any(p.page == -1 for p in pages):
        raise ValueError(
            "TXT parsing: file tail without page marker detected (page=-1). "
            "Ensure markers like '[ N ]' exist for every page and the file ends with a marker."
        )

    # Дубликаты страниц
    seen = set()
    dups = []
    for p in pages:
        if p.page in seen:
            dups.append(p.page)
        seen.add(p.page)
    if dups:
        raise ValueError(f"TXT parsing: duplicate page numbers detected: {sorted(set(dups))[:10]}...")


def pages_to_rows(pages: List[PageDoc]) -> List[Dict[str, Any]]:
    return [{"page": p.page, "text": p.text} for p in pages]
