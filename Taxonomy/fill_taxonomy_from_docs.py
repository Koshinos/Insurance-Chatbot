
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from dotenv import load_dotenv
from pdf2image import convert_from_path

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - only hit if dependency is missing
    OpenAI = None  # type: ignore


STOPWORDS = {
    "a",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
    "under",
    "during",
    "due",
    "must",
    "not",
}

SYNONYMS = {
    "medical": ["hospital", "treatment", "doctor", "illness", "injury"],
    "baggage": ["luggage", "personal belongings"],
    "trip": ["journey", "travel"],
    "delay": ["delayed", "postponed"],
    "cancellation": ["cancel", "cancelled"],
    "hijacking": ["hijack"],
    "terrorism": ["terrorist", "terror"],
    "quarantine": ["isolation order", "isolation"],
    "covid": ["covid-19", "coronavirus"],
    "pre_existing": ["pre-existing", "existing medical condition"],
}


@dataclass
class DocText:
    path: Path
    product: str
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract policy wording text from PDFs/XLSX and fill taxonomy JSON."
    )
    parser.add_argument(
        "--taxonomy",
        default="Taxonomy/Taxonomy_Hackathon.json",
        help="Input taxonomy JSON path.",
    )
    parser.add_argument(
        "--docs-dir",
        default="Policy_Wordings",
        help="Directory containing policy PDFs/XLSX files.",
    )
    parser.add_argument(
        "--output",
        default="Taxonomy/taxonomy_filled.json",
        help="Output filled taxonomy JSON path.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model for OCR/transcription.",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable holding OpenAI API key.",
    )
    parser.add_argument(
        "--max-pages-per-pdf",
        type=int,
        default=80,
        help="Maximum number of PDF pages to OCR per document.",
    )
    parser.add_argument(
        "--product-map",
        action="append",
        default=[],
        help=(
            "Optional mapping rule in format 'Product Name:filename_keyword'. "
            "Can be provided multiple times."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process docs and show stats without writing output file.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_chunks(text: str, max_chars: int = 1200) -> List[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    current = ""
    for para in paras:
        if len(current) + len(para) + 2 <= max_chars:
            current = f"{current}\n\n{para}".strip()
        else:
            if current:
                chunks.append(current)
            current = para
    if current:
        chunks.append(current)
    return chunks


def _image_to_b64(img) -> str:
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        img.save(tmp.name, "JPEG", quality=90)
        tmp_path = tmp.name
    try:
        with open(tmp_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def transcribe_pdf_with_openai(
    client: OpenAI, pdf_path: Path, model: str, max_pages: int
) -> str:
    pages = convert_from_path(str(pdf_path), dpi=200, fmt="jpeg")
    if max_pages > 0:
        pages = pages[:max_pages]

    all_text: List[str] = []
    for idx, img in enumerate(pages, start=1):
        b64 = _image_to_b64(img)
        content = [
            {
                "type": "text",
                "text": (
                    "Transcribe this insurance policy page exactly into plain text. "
                    "Do not summarize. Preserve amounts, conditions, and exclusions."
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"},
            },
        ]
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[{"role": "user", "content": content}],
            max_tokens=4000,
        )
        page_text = (resp.choices[0].message.content or "").strip()
        all_text.append(f"[PAGE {idx}]\n{page_text}")
        print(f"[ocr] {pdf_path.name}: page {idx}/{len(pages)}")

    return normalize_text("\n\n".join(all_text))


def extract_xlsx_text(xlsx_path: Path) -> str:
    xls = pd.ExcelFile(xlsx_path)
    parts: List[str] = []
    for sheet in xls.sheet_names:
        df = pd.read_excel(xlsx_path, sheet_name=sheet, dtype=str).fillna("")
        sheet_text = [f"[SHEET] {sheet}"]
        for row in df.itertuples(index=False):
            values = [str(v).strip() for v in row if str(v).strip()]
            if values:
                sheet_text.append(" | ".join(values))
        parts.append("\n".join(sheet_text))
    return normalize_text("\n\n".join(parts))


def guess_product_from_filename(name: str, products: List[str]) -> str:
    n = name.lower()
    # Project-specific defaults (can be edited if your mapping differs)
    if "scoot" in n:
        return products[0]
    if "pre-ex" in n or "pre_ex" in n or "preex" in n:
        return products[2] if len(products) > 2 else products[-1]
    if "traveleasy" in n:
        return products[1] if len(products) > 1 else products[0]
    return products[0]


def parse_product_map(raw_rules: List[str], products: List[str]) -> List[Tuple[str, str]]:
    parsed: List[Tuple[str, str]] = []
    valid_products = set(products)
    for rule in raw_rules:
        if ":" not in rule:
            raise ValueError(
                f"Invalid --product-map '{rule}'. Expected format: Product Name:keyword"
            )
        product, keyword = rule.split(":", 1)
        product = product.strip()
        keyword = keyword.strip().lower()
        if product not in valid_products:
            raise ValueError(
                f"Invalid product in --product-map '{rule}'. Must be one of: {products}"
            )
        if not keyword:
            raise ValueError(f"Invalid --product-map '{rule}'. Keyword cannot be empty.")
        parsed.append((product, keyword))
    return parsed


def resolve_product_for_file(
    filename: str, products: List[str], mapping_rules: List[Tuple[str, str]]
) -> str:
    lower_name = filename.lower()
    for product, keyword in mapping_rules:
        if keyword in lower_name:
            return product
    return guess_product_from_filename(filename, products)


def build_query_tokens(entry: dict) -> List[str]:
    raw = " ".join(
        [
            str(entry.get("benefit_name", "")),
            str(entry.get("condition", "")),
            str(entry.get("condition_type", "")),
        ]
    )
    raw = raw.replace("-", " ").replace("/", " ").replace("_", " ").lower()
    words = [w for w in re.findall(r"[a-z0-9]+", raw) if w and w not in STOPWORDS]

    expanded: List[str] = []
    for w in words:
        expanded.append(w)
        if w in SYNONYMS:
            expanded.extend(SYNONYMS[w])
    return sorted(set(expanded))


def score_chunk(query_tokens: List[str], chunk: str) -> float:
    if not query_tokens:
        return 0.0
    c = chunk.lower()
    score = 0.0
    token_hits = 0
    for token in query_tokens:
        if len(token) <= 2:
            continue
        if re.search(rf"\b{re.escape(token)}\b", c):
            score += 2.0
            token_hits += 1
        elif token in c:
            score += 1.0
            token_hits += 1
    ratio = SequenceMatcher(None, " ".join(query_tokens), c[:700]).ratio()
    score += ratio * 2.5
    if token_hits < 2:
        score *= 0.6
    return score


def ensure_product_slot(entry: dict, product: str) -> dict:
    products_obj = entry.setdefault("products", {})
    slot = products_obj.get(product)
    if not isinstance(slot, dict):
        slot = {"condition_exist": False, "original_text": "", "parameters": {}}
        products_obj[product] = slot
    slot.setdefault("parameters", {})
    slot.setdefault("original_text", "")
    return slot


def fill_entry_for_product(entry: dict, product: str, chunks: List[str]) -> None:
    query_tokens = build_query_tokens(entry)
    best_score = -1.0
    best_chunk = ""

    for chunk in chunks:
        s = score_chunk(query_tokens, chunk)
        if s > best_score:
            best_score = s
            best_chunk = chunk

    slot = ensure_product_slot(entry, product)
    found = best_score >= 3.0 and bool(best_chunk.strip())
    slot["condition_exist"] = bool(found)
    slot["original_text"] = best_chunk[:1800] if found else ""


def load_and_extract_docs(
    docs_dir: Path,
    products: List[str],
    client: OpenAI,
    model: str,
    max_pages: int,
    mapping_rules: List[Tuple[str, str]],
) -> List[DocText]:
    docs: List[DocText] = []
    files = sorted(
        [
            p
            for p in docs_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".pdf", ".xlsx", ".xls"}
        ]
    )
    if not files:
        raise FileNotFoundError(f"No PDF/XLSX documents found in {docs_dir}")

    for f in files:
        product = resolve_product_for_file(f.name, products, mapping_rules)
        print(f"[doc] Processing {f.name} -> {product}")
        if f.suffix.lower() == ".pdf":
            text = transcribe_pdf_with_openai(
                client=client, pdf_path=f, model=model, max_pages=max_pages
            )
        else:
            text = extract_xlsx_text(f)
        docs.append(DocText(path=f, product=product, text=text))
        print(f"[doc] Extracted {len(text)} chars from {f.name}")
    return docs


def merge_docs_by_product(docs: Iterable[DocText], products: List[str]) -> Dict[str, str]:
    merged = {p: "" for p in products}
    for doc in docs:
        merged[doc.product] = (merged[doc.product] + "\n\n" + doc.text).strip()
    return merged


def fill_taxonomy(taxonomy: dict, product_texts: Dict[str, str]) -> Tuple[dict, dict]:
    stats = {"entries_total": 0, "product_hits": {p: 0 for p in product_texts.keys()}}
    layers = taxonomy.get("layers", {})
    products = taxonomy.get("products", [])

    product_chunks = {p: split_chunks(product_texts.get(p, "")) for p in products}

    for _, entries in layers.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            stats["entries_total"] += 1
            for product in products:
                chunks = product_chunks.get(product, [])
                fill_entry_for_product(entry, product, chunks)
                slot = ensure_product_slot(entry, product)
                if slot.get("condition_exist"):
                    stats["product_hits"][product] += 1
    return taxonomy, stats


def main() -> None:
    load_dotenv()
    args = parse_args()

    if OpenAI is None:
        sys.exit("Missing dependency: openai. Install with `pip install openai`.")

    taxonomy_path = Path(args.taxonomy)
    docs_dir = Path(args.docs_dir)
    output_path = Path(args.output)

    if not taxonomy_path.exists():
        sys.exit(f"Taxonomy file not found: {taxonomy_path}")
    if not docs_dir.exists():
        sys.exit(f"Docs directory not found: {docs_dir}")

    api_key = os.getenv(args.api_key_env, "").strip()
    if not api_key:
        sys.exit(
            f"Missing API key. Set {args.api_key_env} in your environment or .env file."
        )

    taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    products = taxonomy.get("products", [])
    if not isinstance(products, list) or not products:
        sys.exit("Invalid taxonomy: missing non-empty `products` list.")

    client = OpenAI(api_key=api_key)
    mapping_rules = parse_product_map(args.product_map, products)
    docs = load_and_extract_docs(
        docs_dir=docs_dir,
        products=products,
        client=client,
        model=args.model,
        max_pages=args.max_pages_per_pdf,
        mapping_rules=mapping_rules,
    )
    product_texts = merge_docs_by_product(docs, products)
    filled, stats = fill_taxonomy(taxonomy, product_texts)

    print("[stats] total taxonomy entries:", stats["entries_total"])
    for p, hits in stats["product_hits"].items():
        print(f"[stats] {p} matched entries: {hits}")

    if args.dry_run:
        print("[dry-run] No file written.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(filled, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] Wrote filled taxonomy to: {output_path}")


if __name__ == "__main__":
    main()