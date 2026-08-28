"""
document_extraction.py

Block 3: Document Intelligence & Auto-Extraction.
Uses OpenAI (gpt-4o) Vision capabilities to extract structured JSON
from uploaded travel documents (flight/hotel confirmations).

Now supports PDFs by converting them to images on the fly using PyMuPDF!
"""

from __future__ import annotations

import json
import os
import re
import base64
from typing import Any

import fitz  # PyMuPDF - for converting PDFs to images
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

MODEL = "gpt-4o"
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

EXTRACTION_PROMPT = """This is a travel booking document (flight confirmation, hotel booking, \
or itinerary). Extract the following fields and respond with ONLY a JSON object, no other text:

{
  "traveler_name": string or null,
  "departure_date": "YYYY-MM-DD" or null,
  "return_date": "YYYY-MM-DD" or null,
  "destination_city": string or null,
  "destination_country": string or null,
  "trip_cost": number or null (the total booking/ticket cost if shown),
  "currency": string or null,
  "cabin_class": string or null,
  "document_type": "flight" | "hotel" | "itinerary" | "other"
}

Use null for any field not present. However, if you find a destination_city but the country is not explicitly written, you MUST infer the destination_country (e.g., if city is "Abu Dhabi", infer "United Arab Emirates").
"""

SAMPLE_TRIP: dict[str, Any] = {
    "traveler_name": "John Tan",
    "departure_date": "2026-09-12",
    "return_date": "2026-09-19",
    "destination_city": "Tokyo",
    "destination_country": "Japan",
    "trip_cost": 3200,
    "currency": "SGD",
    "cabin_class": "Economy",
    "document_type": "flight",
    "_is_sample": True,
}

MIME_MAP = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "webp": "image/webp",
    "pdf": "application/pdf",
}


def _extract_json(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def extract_trip_data(file_bytes: bytes, file_ext: str) -> dict[str, Any]:
    mime_type = MIME_MAP.get(file_ext.lower().lstrip("."))

    if not mime_type:
        result = dict(SAMPLE_TRIP)
        result["_warning"] = (
            f"Unsupported file type '.{file_ext}' - showing a sample trip instead."
        )
        return result

    # 1. Prepare the content list for OpenAI
    content_list = [{"type": "text", "text": EXTRACTION_PROMPT}]

    # 2. Handle PDF vs Image
    if mime_type == "application/pdf":
        try:
            # Open the PDF from the raw bytes
            doc = fitz.open(stream=file_bytes, filetype="pdf")

            # Loop through pages (limit to first 3 pages to save OpenAI tokens/costs)
            for page_num in range(min(3, len(doc))):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(dpi=150)  # 150 DPI is clear enough for OCR
                img_bytes = pix.tobytes("png")
                b64 = base64.b64encode(img_bytes).decode("utf-8")

                content_list.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    }
                )
        except Exception as e:
            result = dict(SAMPLE_TRIP)
            result["_warning"] = (
                f"Failed to convert PDF to images: {str(e)}. Falling back to sample trip."
            )
            return result
    else:
        # It's already a standard image
        b64 = base64.b64encode(file_bytes).decode("utf-8")
        content_list.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{b64}"},
            }
        )

    # 3. Send it to OpenAI
    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=500,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": content_list,
                }
            ],
        )
        text = response.choices[0].message.content or ""
        parsed = _extract_json(text)
    except Exception as exc:
        parsed = None
        error_note = str(exc)
    else:
        error_note = None

    if not parsed:
        result = dict(SAMPLE_TRIP)
        result["_warning"] = (
            "Couldn't extract trip details from that document"
            + (f" ({error_note})" if error_note else "")
            + " - showing a sample trip instead."
        )
        return result

    # 4. Require dates and AT LEAST a city or a country
    has_dates = bool(parsed.get("departure_date") and parsed.get("return_date"))
    has_dest = bool(parsed.get("destination_country") or parsed.get("destination_city"))

    if not (has_dates and has_dest):
        result = dict(SAMPLE_TRIP)
        result["_warning"] = (
            "Extracted some fields but key trip details were missing "
            f"({parsed}) - showing a sample trip instead."
        )
        return result

    parsed["_is_sample"] = False
    parsed["_warning"] = None
    return parsed
