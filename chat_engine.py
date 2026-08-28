"""
chat_engine.py

Implements Block 2's "Tools Layer" as OpenAI GPT tool-calling:
  - compare_policy_data  -> Policy Comparison Tool (normalized data)
  - get_policy_wording   -> FAQ Tool (original text)
  - get_claims_insight   -> Claims DB Tool (historical claims)

GPT decides which tool(s) a question needs. We track which tools fired
on the final turn so the UI can show a provenance tag ("Comparison" /
"Policy wording" / "Both") — that's the visible proof the system is
switching between normalized and original data rather than guessing.

Session memory: a plain in-memory dict keyed by session_id holds the
running message history, per Block 2's "Session Context" spec. This is a
demo, so it resets when the process restarts — swap for Redis/a DB for
anything long-lived.
"""

from __future__ import annotations

import json
import os
from typing import Any

from flask.cli import load_dotenv
from openai import OpenAI

from taxonomy_tools import TaxonomyStore
from claims_tools import ClaimsStore

MODEL = "gpt-5.6-luna"
load_dotenv()
client = OpenAI()
store = TaxonomyStore()
claims_store = ClaimsStore()

# session_id -> list of message dicts (OpenAI chat messages format)
SESSIONS: dict[str, list[dict[str, Any]]] = {}

# --- FLATTENED TOOLS ARRAY FOR RESPONSES API ---
TOOLS = [
    {
        "type": "function",
        "name": "compare_policy_data",
        "description": (
            "Get normalized, structured data comparing all three products for a given "
            "topic (a benefit or condition, e.g. 'medical expenses', 'baggage delay', "
            "'age eligibility', 'pre-existing conditions'). Returns whether each product "
            "has the condition/benefit and its parameters (limits, time windows, etc.) — "
            "no original wording. Use for comparison questions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The benefit or condition to compare, in plain words.",
                }
            },
            "required": ["topic"],
        },
    },
    {
        "type": "function",
        "name": "get_policy_wording",
        "description": (
            "Get the exact original policy wording for a topic (a benefit or condition), "
            "optionally scoped to one product. Use for explanation/eligibility questions "
            "where precise legal wording matters."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The benefit or condition to explain, in plain words.",
                },
                "product": {
                    "type": "string",
                    "description": "Optional: restrict to one product ('Scootsurance A', 'TravelEasy B', or 'TravelEasy C').",
                },
            },
            "required": ["topic"],
        },
    },
    {
        "type": "function",
        "name": "get_claims_insight",
        "description": (
            "Get historical claims statistics (frequency, average claim amount, 80th "
            "percentile claim amount) for a destination and benefit. Synthetic/illustrative "
            "data standing in for MSIG's real claims database. Use for value-assessment or "
            "'why this tier' questions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "destination": {
                    "type": "string",
                    "description": "Destination country or city.",
                },
                "benefit": {
                    "type": "string",
                    "description": "Optional: restrict to one benefit, e.g. 'medical expenses', 'trip cancellation', 'baggage delay'.",
                },
            },
            "required": ["destination"],
        },
    },
]


def _run_tool(name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    if name == "compare_policy_data":
        return store.compare_policy_data(tool_input["topic"])
    if name == "get_policy_wording":
        return store.get_policy_wording(tool_input["topic"], tool_input.get("product"))
    if name == "get_claims_insight":
        return claims_store.get_claims_insight(
            tool_input["destination"], tool_input.get("benefit")
        )
    return {"error": f"unknown tool {name}"}


def chat(
    session_id: str, user_message: str, trip_data: dict = None, quote_data: dict = None
):
    """Runs one turn of the conversation, returns the assistant text plus
    which tool(s) were used (for the UI's provenance tag)."""

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY in environment.")

    if trip_data is None:
        trip_data = {}

    dest = (
        trip_data.get("destination_city")
        or trip_data.get("destination_country")
        or "your destination"
    )
    dep_date = trip_data.get("departure_date", "your upcoming departure date")
    ret_date = trip_data.get("return_date", "your return date")
    traveler = trip_data.get("traveler_name") or "there"

    pricing_context = "The user has not generated a quote yet."
    if quote_data and "quotes" in quote_data:
        price_list = []
        for q in quote_data["quotes"]:
            product_name = q.get("product", "Unknown Plan")
            price = q.get("estimated_premium_sgd", "Unknown")
            price_list.append(f"{product_name} (SGD ${price})")
        pricing_context = (
            f"The actual quoted premiums for this trip are: {', '.join(price_list)}."
        )

    SYSTEM_PROMPT = f"""
You are a warm, sharp, and highly conversational travel insurance concierge for MSIG Singapore. 
You are currently assisting {traveler} with their upcoming trip to {dest} from {dep_date} to {ret_date}.
They are evaluating their options, and your job is to help them compare coverage across Product A (Standard), Product B (Elite), and Product C (Premier).

{pricing_context} 
CRITICAL RULE: When summarizing or recommending a plan, you MUST include the exact SGD prices listed above. Do not tell the user to check the website for pricing.

YOUR TOOLS & WHEN TO USE THEM:
- compare_policy_data: Use this for comparison questions ("which plan has better X", "what's the medical limit on each plan"). It returns normalized, structured data across the tiers.
- get_policy_wording: Use for explanation questions ("what exactly does X mean", "what counts as a covered reason") where precise policy language or eligibility conditions matter.
- get_claims_insight: Use for "why should I pick X", value-assessment questions, or whenever {dest} is mentioned alongside a coverage question. It returns historical claim frequency/amounts to show if a limit is actually adequate for that specific trip. 
  *CRITICAL:* State clearly that this uses illustrative/synthetic claims data, not real MSIG claims history.

For scenario questions ("what happens if I break my leg skiing"), call multiple tools as needed to cover every relevant benefit and condition, then walk through the coverage step by step, comparing how the different products handle the scenario.

YOUR PERSONALITY VIBE:
1. Be conversational, empathetic, and relatable. Do not sound like a robotic legal document. 
2. Use short paragraphs, bullet points, and emojis to keep the chat visually light and engaging.
3. Be fiercely protective of their wallet. Highlight what is a good deal, where they might be over-insuring, and explain coverage differences in plain English.

THE TRAVEL GUIDE PROTOCOL (PROACTIVE FRIENDLINESS):
- You are a well-traveled friend. Because you know they are going to {dest}, or if they mention a specific activity (e.g., "skiing", "food", "driving"), seamlessly weave in 1 quick, highly specific travel recommendation (e.g., a local dish, a hidden gem, or a travel tip).
- Tie the recommendation back to their insurance smoothly (e.g., "Make sure you try the Tonkotsu ramen! Also, since you'll be hitting the slopes, let's compare the winter sports coverage across the tiers...").

STRICT FACTUAL BOUNDARIES:
- Answer ONLY from tool results. 
- If a tool returns 'found: false', say plainly that nothing in the taxonomy matches, rather than guessing from general knowledge of insurance.
"""

    history = SESSIONS.setdefault(session_id, [])
    history.append({"role": "user", "content": user_message})

    tools_used: set[str] = set()

    # 4. Tool-use loop: keep calling GPT until it stops requesting tools.
    for _ in range(6):  # hard cap to avoid runaway loops
        response = client.responses.create(
            model=MODEL,
            max_output_tokens=1200,
            tools=TOOLS,
            tool_choice="auto",
            reasoning={"effort": "low"},
            input=[{"role": "system", "content": SYSTEM_PROMPT}, *history],
        )

        # In the new Responses API, we look for items of type "function_call" in the output array
        tool_calls = [item for item in response.output if item.type == "function_call"]

        if not tool_calls:
            # Final answer: The Responses API provides a convenient output_text property
            final_text = response.output_text or ""

            # Save the final text output back to the session history
            history.extend(response.output)

            return {
                "reply": final_text,
                "sources": sorted(tools_used),
            }

        # We must append the model's current output items (which includes the function_call items)
        # to the history so it knows WHAT it called
        history.extend(response.output)

        for call in tool_calls:
            name = call.name
            tools_used.add(name)
            try:
                tool_input = json.loads(call.arguments or "{}")
            except json.JSONDecodeError:
                tool_input = {}

            # Execute the tool
            result = _run_tool(name, tool_input)

            # Send the result back as a strict "function_call_output" item mapped by call_id
            history.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps(result, ensure_ascii=False),
                }
            )

    return {
        "reply": "I wasn't able to settle on an answer — could you rephrase the question?",
        "sources": sorted(tools_used),
    }


def reset_session(session_id: str) -> None:
    SESSIONS.pop(session_id, None)
