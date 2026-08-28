"""
claims_tools.py

Block 5: MSIG Proprietary Intelligence & Predictive Recommendations.

Loads a historical claims dataset from a CSV file keyed by destination
and benefit. Formats the data in memory to match the exact JSON structure
expected by quote_engine.py.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).parent / "data"
# Change this to match your actual CSV filename
CLAIMS_PATH = DATA_DIR / "claims_data.csv"


def _normalize_destination(destination: str) -> str:
    return re.sub(r"[^a-z ]", "", destination.lower().strip())


class ClaimsStore:
    def __init__(self, path: Path = CLAIMS_PATH):
        self.destinations: dict[str, dict[str, Any]] = {}
        self.default: dict[str, Any] = {
            "overseas_medical_expenses": {"p80_claim_sgd": 2500},
            "trip_cancellation": {"p80_claim_sgd": 1500},
            "delayed_baggage": {"p80_claim_sgd": 300},
        }

        # Load CSV into the nested dictionary structure
        if path.exists():
            with open(path, mode="r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Adjust these column names to match your actual CSV headers!
                    dest_raw = row.get("destination", "").strip()
                    benefit_raw = (
                        row.get("benefit", "").strip().lower().replace(" ", "_")
                    )

                    try:
                        p80_val = float(row.get("p80_claim_sgd", 0))
                    except ValueError:
                        p80_val = 0.0

                    if dest_raw and benefit_raw:
                        dest_key = _normalize_destination(dest_raw)

                        if dest_key not in self.destinations:
                            self.destinations[dest_key] = {}

                        self.destinations[dest_key][benefit_raw] = {
                            "p80_claim_sgd": p80_val
                        }
        else:
            print(
                f"[warning] Claims CSV not found at {path}. Using default mock values."
            )

    def get_claims_insight(
        self, destination: str, benefit: str | None = None
    ) -> dict[str, Any]:
        key = _normalize_destination(destination)
        matched = None

        # Search for destination match
        for dest_key in self.destinations:
            if dest_key in key or key in dest_key:
                matched = dest_key
                break

        stats = self.destinations[matched] if matched else self.default

        if benefit:
            benefit_key = benefit.lower().strip().replace(" ", "_")
            # Loose match against known benefit keys
            candidates = [k for k in stats if benefit_key in k or k in benefit_key]
            if candidates:
                stats = {candidates[0]: stats[candidates[0]]}

        return {
            "destination_matched": matched or "default (no destination-specific data)",
            "is_destination_specific": matched is not None,
            "stats": stats,
        }

    def get_claims_insight_for_chat(
        self, destination: str, benefit: str | None = None
    ) -> dict[str, Any]:
        """Same lookup as get_claims_insight, but reshaped for LLM consumption:
        no raw statistical field names (no 'p80', no 'percentile', no
        'frequency_pct') - just plain-language figures and a ready-made
        guidance sentence per benefit. This exists so the chatbot has
        nothing technical to echo back to the customer."""
        raw = self.get_claims_insight(destination, benefit)
        benefit_labels = {
            "overseas_medical_expenses": "medical expenses",
            "trip_cancellation": "trip cancellation",
            "delayed_baggage": "baggage delay",
        }

        benefits_out = {}
        for key, stat in raw["stats"].items():
            label = benefit_labels.get(key, key.replace("_", " "))

            # Use safe defaults (0) in case the CSV cell was completely empty
            typical = stat.get("average_claim_sgd", 0)
            ceiling = stat.get("p80_claim_sgd", 0)
            how_common = stat.get("claim_frequency_pct", 0)

            guidance = (
                f"Most {label} claims for this destination stay under about SGD {ceiling:,.0f} "
                f"(typical claim is around SGD {typical:,.0f}); it's worth choosing a coverage "
                f"limit above that."
                if ceiling and typical
                else f"Limited data available for {label} at this destination."
            )
            benefits_out[key] = {
                "typical_claim_sgd": typical,
                "suggested_minimum_coverage_sgd": ceiling,
                "how_common_pct_of_trips": how_common,
                "guidance": guidance,
            }

        return {
            "destination_matched": raw["destination_matched"],
            "is_destination_specific": raw["is_destination_specific"],
            "note": "Figures are illustrative estimates, not a live claims feed.",
            "benefits": benefits_out,
        }
