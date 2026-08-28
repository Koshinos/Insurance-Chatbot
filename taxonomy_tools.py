"""
taxonomy_tools.py

Data-access layer over the filled taxonomy JSON. Exposes exactly the two
things Block 2's spec calls for:

  - compare_policy_data(topic)  -> normalized/structured data across products
                                    (the "Policy Comparison Tool")
  - get_policy_wording(topic, product=None) -> original policy text
                                    (the "FAQ Tool")

Both search the taxonomy by loose token overlap against condition/benefit
names, since that's the identifier space the taxonomy uses (e.g.
"overseas_medical_expenses", "pre_existing_conditions").
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).parent / "data"
REAL_TAXONOMY_PATH = DATA_DIR / "taxonomy_filled.json"

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
    "does",
    "do",
    "what",
    "which",
    "how",
    "am",
    "i",
    "my",
    "me",
    "will",
    "policy",
    "plan",
    "cover",
    "coverage",
    "covered",
    "you",
    "your",
}


def _tokenize(text: str) -> set[str]:
    text = text.replace("-", " ").replace("_", " ").replace("/", " ").lower()
    words = re.findall(r"[a-z0-9]+", text)
    return {w for w in words if w and w not in STOPWORDS and len(w) > 2}


class TaxonomyStore:
    """Loads and indexes the taxonomy JSON once; answers lookups against it."""

    def __init__(self, path: Path | None = None):

        path = REAL_TAXONOMY_PATH
        self.path = path
        raw = json.loads(path.read_text(encoding="utf-8"))
        self.products: list[str] = raw.get("products", [])
        self.entries: list[dict[str, Any]] = []

        for layer_name, entries in raw.get("layers", {}).items():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                identifier = entry.get("condition") or entry.get("benefit_name", "")
                label_parts = [
                    entry.get("benefit_name", ""),
                    entry.get("condition", ""),
                ]
                label = " ".join(p for p in label_parts if p).strip()
                tokens = _tokenize(label)
                self.entries.append(
                    {
                        "layer": layer_name,
                        "identifier": identifier,
                        "label": label,
                        "condition_type": entry.get("condition_type", ""),
                        "tokens": tokens,
                        "raw": entry,
                    }
                )

    def _search(self, topic: str, top_k: int = 5) -> list[dict[str, Any]]:
        query_tokens = _tokenize(topic)
        if not query_tokens:
            return []
        scored = []
        for entry in self.entries:
            overlap = query_tokens & entry["tokens"]
            if not overlap:
                continue
            score = len(overlap) / max(len(query_tokens), 1)
            scored.append((score, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:top_k]]

    def compare_policy_data(self, topic: str) -> dict[str, Any]:
        """Structured, normalized comparison across products for entries
        matching `topic`. Deliberately excludes original_text — this is the
        normalized side of the system."""
        matches = self._search(topic)
        if not matches:
            return {"found": False, "topic": topic, "matches": []}

        results = []
        for m in matches:
            product_data = {}
            for product in self.products:
                slot = m["raw"].get("products", {}).get(product, {})
                product_data[product] = {
                    "exists": slot.get("condition_exist", False),
                    "parameters": slot.get("parameters", {}),
                }
            results.append(
                {
                    "layer": m["layer"],
                    "identifier": m["identifier"],
                    "condition_type": m["condition_type"],
                    "products": product_data,
                }
            )
        return {"found": True, "topic": topic, "matches": results}

    def get_policy_wording(
        self, topic: str, product: str | None = None
    ) -> dict[str, Any]:
        """Original policy text for entries matching `topic`, optionally
        scoped to one product. This is the FAQ side — exact wording only,
        no normalization."""
        matches = self._search(topic)
        if not matches:
            return {"found": False, "topic": topic, "matches": []}

        products_to_check = [product] if product else self.products
        results = []
        for m in matches:
            texts = {}
            for p in products_to_check:
                if p not in self.products:
                    continue
                slot = m["raw"].get("products", {}).get(p, {})
                if slot.get("condition_exist") and slot.get("original_text"):
                    texts[p] = slot["original_text"]
            if texts:
                results.append(
                    {
                        "layer": m["layer"],
                        "identifier": m["identifier"],
                        "condition_type": m["condition_type"],
                        "wording": texts,
                    }
                )
        return {"found": bool(results), "topic": topic, "matches": results}
