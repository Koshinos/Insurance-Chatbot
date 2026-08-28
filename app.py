"""
app.py

Flask entry point for the Block 2 & 3 chat demo.
"""

from __future__ import annotations

import os
import uuid
from flask import Flask, jsonify, redirect, render_template, request, session
from flask_cors import CORS

import chat_engine
import document_extraction
import quote_engine
import stripe

from pathlib import Path
from dotenv import load_dotenv  
load_dotenv()  # Load environment variables from .env file
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
app = Flask(__name__)
app.secret_key = "msig_demo_secret_key_123"  # fine for a local demo


# --- WEBPAGE ROUTES ---

CORS(app, supports_credentials=True)


@app.route("/")
def index():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return render_template("index.html")


# THIS IS THE MISSING PIECE! This serves the quote.html page when you click the tab.
@app.route("/quote", methods=["GET"])
def quote_page():
    return render_template("quote.html")


# --- API/DATA ROUTES ---


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(force=True)
    message = (data or {}).get("message", "").strip()
    if not message:
        return jsonify({"error": "empty message"}), 400

    session_id = session.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        session["session_id"] = session_id

    # 1. Open the backpack and grab BOTH dictionaries
    trip_data = session.get("trip_data", {})
    quote_data = session.get(
        "quote_data", {}
    )  # <--- This line prevents your NameError!

    try:
        # 2. Hand BOTH dictionaries to the chat engine
        result = chat_engine.chat(session_id, message, trip_data, quote_data)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify(result)


@app.route("/api/quote", methods=["POST"])
def api_quote():
    # 1. Handle file upload (Block 3 Document Extraction)
    if request.content_type and "multipart/form-data" in request.content_type:
        uploaded = request.files.get("file")
        if not uploaded or not uploaded.filename:
            return jsonify({"error": "No file uploaded."}), 400
        file_ext = (
            uploaded.filename.rsplit(".", 1)[-1] if "." in uploaded.filename else ""
        )
        file_bytes = uploaded.read()
        trip_data = document_extraction.extract_trip_data(file_bytes, file_ext)

    # 2. Handle sample data fallback
    else:
        data = request.get_json(force=True) or {}
        if not data.get("use_sample"):
            return jsonify({"error": "No file uploaded and use_sample not set."}), 400
        trip_data = dict(document_extraction.SAMPLE_TRIP)
        trip_data["_warning"] = "Using the built-in sample trip."
    # ---> NEW: STORE THE DATA IN THE SESSION BACKPACK <---
    session["trip_data"] = trip_data
    session.modified = True
    # 3. Build the quote (Block 5 Claims Intelligence)
    try:
        quote = quote_engine.build_quote(
            trip_data, chat_engine.store, chat_engine.claims_store
        )
        session["quote_data"] = quote  # Put the math in the backpack!
        session.modified = True
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify(quote)


@app.route("/create-checkout-session", methods=["POST"])
def create_checkout_session():
    # 1. Grab the exact plan name and price from the HTML button you clicked
    product_name = request.form.get("product_name", "Travel Insurance")
    amount_str = request.form.get("amount", "0.00")

    # 2. Stripe requires the price in cents (e.g., $43.36 becomes 4336)
    amount_in_cents = int(float(amount_str) * 100)

    try:
        # 3. Ask Stripe to generate a secure payment screen
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[
                {
                    "price_data": {
                        "currency": "sgd",
                        "product_data": {
                            "name": product_name,
                        },
                        "unit_amount": amount_in_cents,
                    },
                    "quantity": 1,
                }
            ],
            mode="payment",
            # Where to send the user after they pay (or if they click back)
            success_url="http://127.0.0.1:5000/",
            cancel_url="http://127.0.0.1:5000/quote",
        )

        # 4. Instantly redirect the user's browser to the Stripe screen!
        return redirect(checkout_session.url, code=303)

    except Exception as e:
        return str(e), 403


@app.route("/api/reset", methods=["POST"])
def api_reset():
    session_id = session.get("session_id")
    if session_id:
        chat_engine.reset_session(session_id)
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(debug=True, port=5000)


# 1. Point the Stripe SDK to your local stripe-mock server
stripe.api_key = os.getenv(
    "STRIPE_SECRET_KEY"
)  # You can use any fake key for stripe-mock
# ... your other routes ...
