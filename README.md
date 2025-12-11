# Agentic SAM3 + LLM System (Hugging Face + Gemini, API-only)

## Summary

This repo implements an agentic system where a Gemini LLM drives a conversational agent that interprets user prompts and delegates visual perception tasks to a SAM-style vision model served through the Hugging Face Inference API. No local model downloads required; uses free-tier API tokens. The document contains architecture, usage notes, and the full reference implementation. The runnable code files are maintained alongside this document.

Key constraints and choices made per user request:

* LLM: Google Gemini via google-generativeai (API key required). Use GEMINI_MODEL env var to pick model.

* Vision: Hugging Face Inference API for a promptable segmentation model (SAM-style). Use HF Inference endpoints; free-tier tokens apply.

* No local model downloads. All model inference is remote.

* Text/document retrieval (RAG) support: optional sized document store passed to LLM; keep size limits and explicit user-provided docs only.

* Code is modular; examples for image + video processing included.

## Environment

Set these environment variables before running:

* **GEMINI_API_KEY** — API key for Google Gemini (generativeai).

* **GEMINI_MODEL** — Gemini model name (e.g., gemini-1.5-mini or whichever free-tier model is available).

* **HF_API_TOKEN** — Hugging Face Inference API token.

* **HF_SAM_MODEL** — Hugging Face model repo path or inference endpoint (e.g., facebook/sam-vit-large or a community SAM-variant). Use the model name as documented by Hugging Face.

Optionally: DOC_TEXT_LIMIT — max number of characters from uploaded/attached text to include in LLM prompt (default 2000).

## Files (implemented in repository)

requirements.txt

config.py

agents/llm_agent.py

agents/sam_agent.py

utils/image_utils.py

main.py

README.md

## Implementation notes

All HTTP calls use REST to Gemini/Hugging Face; no managed SDKs required except google-generativeai for convenience.

The LLM parsing step returns strict JSON; a fallback heuristic exists if parsing fails.

The SAM agent expects the HF inference response to return a per-instance mask or COCO-style polygons; small adapters convert HF output into binary masks for contour extraction.

Keep requests small to respect free-tier usage; batch frames when processing video and respect frame sampling.

