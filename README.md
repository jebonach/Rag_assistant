# RAG Assistant (prototype)

## Quickstart
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
# source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env

## Data format (page-based)
Textbooks are stored as page files:
`data/books/<book_id>/pages/page_0001.txt`, `page_0002.txt`, ...

Page boundaries come from markers like `[ 208 ]` in the source TXT/MD.
To split a raw book into pages:
```bash
python scripts/split_book_to_pages.py \
  --in data/books/devops_handbook/book.txt \
  --out data/books/devops_handbook/pages
```
