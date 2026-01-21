"""
Python 3 - Verify Gemini token connectivity using the official google-genai SDK.

This script:
1) Lists available models for YOUR API key (source of truth).
2) Picks a model that supports generateContent (if metadata is available).
3) Calls generate_content with the selected model to confirm connectivity.
"""

from __future__ import annotations

import os
from dotenv import load_dotenv
from google import genai


def main() -> None:
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY (or GOOGLE_API_KEY) in your environment/.env")

    client = genai.Client(api_key=api_key)

    # 1) LIST MODELS (this answers the 404)
    models = list(client.models.list())
    if not models:
        raise RuntimeError("No models returned. This usually indicates account/project access issues.")

    print("=== Available models (first 30) ===")
    for m in models[:30]:
        # name often looks like: models/xxxx
        name = getattr(m, "name", str(m))
        # Some SDK versions expose supported methods/inputs; print if present.
        methods = getattr(m, "supported_generation_methods", None)
        print("-", name, "| methods:", methods)

    # 2) PICK A MODEL THAT SUPPORTS generateContent (if the field exists)
    chosen = None
    for m in models:
        name = getattr(m, "name", "")
        methods = getattr(m, "supported_generation_methods", None)
        if methods and "generateContent" in methods:
            chosen = name
            break

    # If metadata isn't present, fall back to a reasonable guess from the list:
    if not chosen:
        # Pick the first model that looks like Gemini and is not embeddings-only
        for m in models:
            name = getattr(m, "name", "")
            if "gemini" in name.lower():
                chosen = name
                break

    if not chosen:
        raise RuntimeError("Could not auto-select a usable model. Check the printed model list.")

    print(f"\nChosen model for test: {chosen}")

    # 3) MAKE A MINIMAL CALL (the real connectivity test)
    resp = client.models.generate_content(
        model=chosen,  # use the exact model name from ListModels
        contents="Reply ONLY with: CONNECTED"
    )

    print("\n=== Gemini response ===")
    print(resp.text)


if __name__ == "__main__":
    main()
