"""
quote_engine.py

Orchestrates Block 3 + Block 5 together, matching the spec's own framing
("alongside quote generation, the system simultaneously analyzes MSIG's
historical claims data..."):

  1. Validate the extracted trip data (dates make sense, duration computed).
  2. Pull normalized coverage data for the three core benefits across
     Product A/B/C from the taxonomy (reuses taxonomy_tools.store).
  3. Pull claims insight for the trip's destination from claims_tools.
  4. For each product, check whether its coverage limit for each benefit
     clears the destination's 80th-percentile claim amount (for medical /
     baggage) or the traveler's actual trip cost (for cancellation, since
     that benefit needs to cover the investment, not just typical claims).
  5. Recommend the cheapest product that clears every check, with the
     claims figures as rationale. Also computes an illustrative (not
     actuarial) premium per product.

No real MSIG pricing engine exists, so premiums here are a simple, clearly
labeled formula for demo purposes only.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

from claims_tools import ClaimsStore
from taxonomy_tools import TaxonomyStore
import json

# Load the normalized numerical limits database
try:
    with open("data/clean_limits.json", "r") as f:
        CLEAN_LIMITS = json.load(f)
except FileNotFoundError:
    CLEAN_LIMITS = {}
CORE_BENEFITS = ["overseas medical expenses", "trip cancellation", "delayed baggage"]
BENEFIT_KEYS = {
    "overseas medical expenses": "overseas_medical_expenses",
    "trip cancellation": "trip_cancellation",
    "delayed baggage": "delayed_baggage",
}

# Illustrative-only premium model: base rate per day + a percentage of trip
# cost (for cancellation risk) + a flat tier loading. NOT a real pricing engine.
TIER_DAILY_RATE = {"Product A": 1.20, "Product B": 1.60, "Product C": 2.10}
TIER_COST_PCT = {"Product A": 0.010, "Product B": 0.015, "Product C": 0.020}
TIER_FLAT_LOADING = {"Product A": 8, "Product B": 15, "Product C": 25}


def _parse_amount(text: str) -> float | None:
    """Pulls a numeric amount out of strings like 'SGD 1,000,000'."""
    if not text:
        return None
    match = re.search(r"[\d,]+(?:\.\d+)?", str(text))
    if not match:
        return None
    return float(match.group(0).replace(",", ""))


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


def validate_trip(trip_data: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Returns (enriched trip_data with trip_duration_days, list of warnings)."""
    warnings: list[str] = []
    dep = _parse_date(trip_data.get("departure_date"))
    ret = _parse_date(trip_data.get("return_date"))

    if dep and ret:
        if ret <= dep:
            warnings.append(
                "Return date is not after departure date - please double check the dates."
            )
            trip_data["trip_duration_days"] = None
        else:
            trip_data["trip_duration_days"] = (ret - dep).days
    else:
        warnings.append(
            "Could not determine trip duration - missing or unparseable dates."
        )
        trip_data["trip_duration_days"] = None

    if trip_data.get("trip_cost") is None:
        warnings.append(
            "No trip cost detected - cancellation-coverage adequacy can't be checked precisely."
        )

    if not trip_data.get("destination_country"):
        warnings.append(
            "No destination detected - using generic (non-destination-specific) claims data."
        )

    return trip_data, warnings


def _estimate_premium(product: str, duration_days: int, trip_cost: float) -> float:
    daily = TIER_DAILY_RATE[product] * duration_days
    cost_based = TIER_COST_PCT[product] * trip_cost
    return round(daily + cost_based + TIER_FLAT_LOADING[product], 2)


def build_quote(
    trip_data: dict[str, Any], taxonomy_store: TaxonomyStore, claims_store: ClaimsStore
) -> dict[str, Any]:
    trip_data, warnings = validate_trip(trip_data)
    destination = (
        trip_data.get("destination_country") or trip_data.get("destination_city") or ""
    )
    trip_cost = trip_data.get("trip_cost") or 0
    duration_days = (
        trip_data.get("trip_duration_days") or 7
    )  # fallback so premium calc still runs

    products = taxonomy_store.products
    benefit_analysis = []
    product_flags = {
        p: [] for p in products
    }  # list of bool "adequate" per benefit, per product

    for benefit_label in CORE_BENEFITS:
        comparison = taxonomy_store.compare_policy_data(benefit_label)
        claims = claims_store.get_claims_insight(destination, benefit_label)
        benefit_key = BENEFIT_KEYS[benefit_label]
        claim_stats = claims["stats"].get(benefit_key, {})
        p80 = claim_stats.get("p80_claim_sgd")

        # cancellation needs to cover the traveler's actual investment too
        threshold = p80
        if benefit_key == "trip_cancellation" and trip_cost:
            threshold = max(p80 or 0, trip_cost)

        # find the matching benefit-level entry (top of the matches list, layer_2)
        product_limits = {}
        for p in products:

            # 1. Grab the clean integer directly from our optimized database
            try:
                limit_value = CLEAN_LIMITS[benefit_key][p]
            except KeyError:
                limit_value = 0

            # 2. Format the integer into a nice string for the UI (e.g., "SGD 500,000")
            limit_text = f"SGD {limit_value:,.0f}" if limit_value > 0 else "Not covered"

            # 3. Check if the limit covers the claims threshold
            adequate = (
                limit_value > 0 and threshold is not None and limit_value >= threshold
            )

            # 4. Save the results
            product_limits[p] = {
                "coverage_limit": limit_text,
                "adequate": adequate if threshold is not None else None,
            }
            if threshold is not None:
                product_flags[p].append(adequate)

        benefit_analysis.append(
            {
                "benefit": benefit_label,
                "claims_destination_matched": claims["destination_matched"],
                "is_destination_specific": claims["is_destination_specific"],
                "claim_p80_sgd": p80,
                "threshold_used_sgd": threshold,
                "products": product_limits,
            }
        )

    # recommend cheapest product (in given order) that clears every check with a known threshold
    recommended = None
    for p in products:
        flags = product_flags[p]
        if flags and all(flags):
            recommended = p
            break
    if recommended is None:
        # nobody fully clears every threshold - pick the product with the most "adequate" flags
        recommended = max(products, key=lambda p: sum(1 for f in product_flags[p] if f))
        warnings.append(
            f"No single tier fully covers every benefit against the claims-based thresholds for this trip; "
            f"{recommended} covers the most of them. Consider riders for any gaps."
        )

    quotes = []
    for p in products:
        quotes.append(
            {
                "product": p,
                "estimated_premium_sgd": _estimate_premium(p, duration_days, trip_cost),
                "recommended": p == recommended,
            }
        )

    return {
        "trip_data": trip_data,
        "warnings": warnings,
        "benefit_analysis": benefit_analysis,
        "quotes": quotes,
        "recommended_product": recommended,
        "premium_disclaimer": "Estimated premiums are illustrative only (simple demo formula), not an actuarial quote.",
    }
