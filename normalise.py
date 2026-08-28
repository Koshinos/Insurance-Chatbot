"""
normalise.py

An ETL pipeline script to clean unstructured policy text into a numerical database.
Includes a pre-filter to avoid OpenAI token limits!
"""

import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

print("Loading massive JSON file...")

# 1. Load the messy database as a Python Dictionary
try:
    with open("data/taxonomy_filled.json", "r") as f:
        raw_data = json.load(f)
except FileNotFoundError:
    with open("taxonomy_filled.json", "r") as f:
        raw_data = json.load(f)

# 2. PRE-FILTER: Shrink the data to beat the token limit
# We only want to send the specific layer that has our benefits
target_benefits = ["overseas_medical_expenses", "trip_cancellation", "delayed_baggage"]
shrunken_data = []

layers = raw_data.get("layers", {})
layer_2 = layers.get("layer_2_benefits", [])

for item in layer_2:
    if item.get("benefit_name") in target_benefits:
        shrunken_data.append(item)

# Convert our tiny filtered list back into a text string for the AI
mini_json_string = json.dumps(shrunken_data)

print("File successfully shrunk! Analyzing the specific benefits with AI...")

# 3. Instruct OpenAI to act as a data parser
prompt = """
You are a data extraction tool. I will provide a tiny JSON snippet containing travel insurance policy text for three specific benefits. 
Look through the text (specifically inside the "original_text" fields) to find the maximum coverage limits (in SGD) for each benefit, 
across all three products (Product A, Product B, Product C).

If the text mentions different tiers (Standard, Elite, Premier), map them logically to Product A, Product B, and Product C.
Output ONLY a raw, valid JSON object with the following structure, containing pure integers (no commas, no $).
If a limit is not covered or says "Not covered", output 0.

{
  "overseas_medical_expenses": {
    "Product A": 0,
    "Product B": 0,
    "Product C": 0
  },
  "trip_cancellation": {
    "Product A": 0,
    "Product B": 0,
    "Product C": 0
  },
  "delayed_baggage": {
    "Product A": 0,
    "Product B": 0,
    "Product C": 0
  }
}
"""

# 4. Process the data
try:
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": mini_json_string,
            },  # <-- Sending the tiny string!
        ],
    )

    clean_json = response.choices[0].message.content
    parsed_limits = json.loads(clean_json)

    # 5. Save the clean database
    os.makedirs("data", exist_ok=True)
    with open("data/clean_limits.json", "w") as f:
        json.dump(parsed_limits, f, indent=2)

    print("\n✅ Success! Normalized data saved to data/clean_limits.json")
    print(json.dumps(parsed_limits, indent=2))

except Exception as e:
    print(f"❌ Error during normalization: {e}")
