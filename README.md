# Agentic SAM3 + LLM System (Hugging Face + Gemini, API-only)

## Summary

This repo implements an agentic system where a Gemini LLM drives a conversational agent that interprets user prompts and delegates visual perception tasks to a SAM-style vision model served through the Hugging Face Inference API. No local model downloads required; uses free-tier API tokens. The document contains architecture, usage notes, and the full reference implementation. The runnable code files are maintained alongside this document.

Key constraints and choices made per user request:

* LLM: Google Gemini via google-generativeai (API key required). Use GEMINI_MODEL env var to pick model.

* Vision: Hugging Face Inference API for a promptable segmentation model (SAM-style). Use HF Inference endpoints; free-tier tokens apply.

* No local model downloads. All model inference is remote.

* Text/document retrieval (RAG) support: optional sized document store passed to LLM; keep size limits and explicit user-provided docs only.

* Code is modular; examples for image + video processing included.

---

## Goals

1. Use Gemini (LLM) for natural-language understanding, planning, and tool orchestration.
2. Use a SAM agent (Hugging Face Inference API hosting of Segment Anything family) for segmentation and visual grounding.
3. Implement a minimal agent loop: LLM receives user prompt -> decides whether to call SAM -> sends instruction (image URL + prompt/point/box) -> SAM returns mask(s), bbox/shape metadata -> LLM composes final answer.
4. Provide clear Python example code, documentation, and tests so you can run on free-tier accounts.

---

## Repo layout (single-file example plus extras)

```
agentic-ai-sam-gemini/
├─ README.md                # this document
├─ requirements.txt
├─ .env.example             # env var names/format
├─ app/
│  ├─ agent_llm.py          # LLM orchestration logic (Gemini client wrapper)
│  ├─ sam_agent.py          # SAM wrapper (Hugging Face Inference API client)
│  ├─ utils.py              # helpers (image download, mask -> bbox conversion)
│  └─ run_example.py        # example end-to-end flow
├─ tests/
│  └─ test_end_to_end.py    # simple smoke test with public image
└─ LICENSE
```

---

## Environment

Set these environment variables before running:

* **GEMINI_API_KEY** — API key for Google Gemini (generativeai).

* **GEMINI_MODEL** — Gemini model name (e.g., gemini-1.5-mini or whichever free-tier model is available).

* **HF_API_TOKEN** — Hugging Face Inference API token.

* **HF_SAM_MODEL** — Hugging Face model repo path or inference endpoint (e.g., facebook/sam-vit-large or a community SAM-variant). Use the model name as documented by Hugging Face.

Optionally: DOC_TEXT_LIMIT — max number of characters from uploaded/attached text to include in LLM prompt (default 2000).

## Quick setup

1. Create accounts and tokens (free tiers available):
   - Hugging Face account -> `HF_API_KEY` (use Inference API). You can host a SAM model or use the HF hosted model (e.g. `facebook/sam-vit-huge` or `facebook/sam2.1-hiera-large`).
   - Google Cloud / Vertex AI -> `GEMINI_API_KEY` (Gemini). The free quotas vary by region; use your developer project. Alternatively you can use `google-generativeai` library with an API key.

2. Copy `.env.example` to `.env` and set the two environment variables.

3. Install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

4. Run the example:

```bash
python app/run_example.py --image https://example.com/myphoto.jpg --question "Find the red bicycle in the image and give its bounding box"
```

---

## Design and architecture

### Agents

**LLM agent (agent_llm.py)**
- Receives user prompt (text + optional image URL).
- Performs intent classification: determine whether segmentation/vision tool is needed.
- If tool required, prepares a structured call (tool name `sam.segment`, arguments: `image_url`, `prompt_type`, `points` or `box` or `text_prompt`).
- Sends the tool call to the SAM agent and awaits its response.
- Integrates SAM response (masks, bboxes, confidence) into final textual output.

**SAM agent (sam_agent.py)**
- Uses Hugging Face Inference API endpoints to call a Segment Anything model.
- Accepts image URL and structured prompts (points, boxes, or text label).
- Returns mask(s) as PNG (or RLE), polygon coordinates, bounding boxes, and a small visualization image (mask overlay).

### Communication format

Tool call (from LLM to SAM):
```json
{
  "tool": "sam.segment",
  "args": {
    "image_url": "https://...",
    "prompt_type": "box",        
    "prompt": ["x1","y1","x2","y2"] 
  }
}
```

SAM response (to LLM):
```json
{
  "masks": ["base64_png_1"],
  "polygons": [["x","y"],"..."],
  "bboxes": [["x1","y1","x2","y2"], "..."],
  "scores": [0.98, "..."]
}
```

---

## Example code (key files)

### `requirements.txt`

```
requests
python-dotenv
Pillow
numpy
opencv-python
google-generativeai>=0.2.0
```

### `.env.example`

```
HF_API_KEY=hf_...
GEMINI_API_KEY=...
HF_SAM_MODEL=facebook/sam-vit-huge
GEMINI_MODEL=gemini-1.5-pro
```