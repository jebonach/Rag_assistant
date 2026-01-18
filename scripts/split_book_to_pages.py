from __future__ import annotations

from pathlib import Path


def main() -> None:
    import argparse
    import sys

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.chunking_pages import split_text_by_page_markers, write_pages

    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Путь к исходному TXT/MD (как текст).")
    ap.add_argument("--out", dest="out_dir", required=True, help="Папка для страниц.")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_dir = Path(args.out_dir)

    text = in_path.read_text(encoding="utf-8", errors="ignore")
    pages = split_text_by_page_markers(text)

    if len(pages) < 10:
        raise ValueError(
            f"Слишком мало страниц ({len(pages)}). Похоже, разметка [ XXX ] не распознана."
        )

    write_pages(pages, out_dir)
    print(f"OK: written {len(pages)} pages to {out_dir}")


if __name__ == "__main__":
    main()
