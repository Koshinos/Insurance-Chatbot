import json

# The exact keys your ETL pipeline and quote engine need to function
REQUIRED_KEYS = ["overseas_medical_expenses", "trip_cancellation", "delayed_baggage"]

print("🔍 Running Mismatch Checker...\n")

# 1. Load the database safely
try:
    with open("data/taxonomy_filled.json", "r") as f:
        data = json.load(f)
except FileNotFoundError:
    try:
        with open("taxonomy_filled.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("❌ Could not find taxonomy_filled.json in the current or data/ folder!")
        exit()

# 2. Navigate the specific architecture we discovered
layers = data.get("layers", {})
layer_2 = layers.get("layer_2_benefits", [])

# Extract all the benefit names found in layer 2
found_benefits = []
for item in layer_2:
    name = item.get("benefit_name")
    if name and name not in found_benefits:  # avoid printing duplicates
        found_benefits.append(name)

if not found_benefits:
    print(
        "❌ Could not find any benefits in 'layer_2_benefits'. Check if the JSON structure changed."
    )
    exit()

# 3. Print what is actually in the file
print("=== BENEFITS FOUND IN YOUR JSON ===")
for b in found_benefits:
    print(f"• {b}")

# 4. Check for our required keys
print("\n=== CHECKING REQUIRED QUOTE ENGINE KEYS ===")
missing_keys = []
for req in REQUIRED_KEYS:
    if req in found_benefits:
        print(f"✅ MATCH: '{req}'")
    else:
        print(f"❌ MISMATCH: Your JSON is missing '{req}'")
        missing_keys.append(req)

# 5. Final Report
print("\n=== SUMMARY ===")
if not missing_keys:
    print("🎉 All required keys match! Your data pipeline is good to go.")
else:
    print(
        f"⚠️ Missing {len(missing_keys)} keys. Check the exact spelling in your JSON!"
    )
