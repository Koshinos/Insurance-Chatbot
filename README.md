# Policy Assistant — Block 2 Chat Demo

A working chat prototype for Block 2 (Conversational FAQ & Comparison Engine).
Implements the spec's "Tools Layer" as literal Claude tool-use:

- **`compare_policy_data`** — the *Policy Comparison Tool*. Returns normalized,
  structured data (does the condition/benefit exist, what are its parameters)
  across Product A/B/C. No original wording.
- **`get_policy_wording`** — the *FAQ Tool*. Returns exact original policy text
  for a topic, optionally scoped to one product.

Claude decides which tool(s) to call per question. The UI shows a colored tag
on every answer ("Normalized data" / "Policy wording" / both) so you can see
which data source actually answered the question — that's the point of the
block: seamlessly switching between the two.

Session memory is a plain in-memory dict (`chat_engine.SESSIONS`), matching
the spec's "Session Context" — good enough for a demo, not for production
(swap in Redis/a DB if this needs to survive restarts or multiple workers).

## Setup

```bash
cd block2_chat_demo
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
python app.py
```

Open http://localhost:5000

## Using your real taxonomy data

You said you already have a filled taxonomy. Drop it in as:

```
block2_chat_demo/data/taxonomy_filled.json
```

`taxonomy_tools.py` checks for that file first and only falls back to
`data/sample_taxonomy_filled.json` (mock data, clearly labeled in the UI) if
it's missing. No code changes needed — just make sure your file follows the
same schema as `Taxonomy_Hackathon.json`:

```json
{
  "products": ["Product A", "Product B", "Product C"],
  "layers": {
    "layer_1_general_conditions": [ ... ],
    "layer_2_benefits": [ ... ],
    "layer_3_benefit_specific_conditions": [ ... ]
  }
}
```

Each entry needs `condition` or `benefit_name` (used for topic search),
and a `products` dict with `condition_exist`, `original_text`, and
`parameters` per product.

## Try asking

- "Which plan has better medical coverage?" → comparison tool
- "What exactly counts as a covered cancellation reason?" → FAQ tool
- "Am I covered for pre-existing conditions?" → both
- "What happens if I break my leg skiing in Japan?" → both, multi-benefit

## Known limitations (it's a demo)

- Topic search is token-overlap based, not embeddings — very different
  phrasing from the taxonomy's condition/benefit names may not match. If
  this becomes a real issue, add a synonym table (like the one in
  `fill_taxonomy_from_docs.py`) or swap in embedding-based retrieval.
- Session memory is per-process and in-memory — restarting the server
  clears all conversations.
- No auth/rate-limiting — fine for a local hackathon demo, not for
  anything public-facing.
