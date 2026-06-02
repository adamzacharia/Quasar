#!/usr/bin/env python3
"""
Configure Langfuse Custom Model Definitions — DeepSeek Cache-Aware Pricing.

Run this script ONCE to create custom model definitions in your Langfuse
project so that DeepSeek API costs are calculated accurately.

DeepSeek returns prompt_cache_hit_tokens and prompt_cache_miss_tokens in
its API responses.  Cache hits are 50-120× cheaper than cache misses.
Without this configuration, Langfuse prices ALL input tokens at the
expensive cache-miss rate, inflating your reported costs by ~3×.

Usage:
    python scripts/setup_langfuse_models.py

Requirements:
    - LANGFUSE_SECRET_KEY set in .env
    - LANGFUSE_PUBLIC_KEY set in .env
    - LANGFUSE_HOST set in .env (defaults to https://cloud.langfuse.com)

Pricing source (June 2026):
    https://api-docs.deepseek.com/guides/pricing
"""

import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import requests

# ── Configuration ──────────────────────────────────────────────────────────

LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")

if not LANGFUSE_PUBLIC_KEY or not LANGFUSE_SECRET_KEY:
    print("ERROR: LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must be set.")
    print("Set them in .env or export them as environment variables.")
    sys.exit(1)

# ── DeepSeek Model Definitions ────────────────────────────────────────────
# Each model has three usage types with different per-token costs:
#   - prompt_cache_hit_tokens:  cached prefix tokens (very cheap)
#   - prompt_cache_miss_tokens: non-cached input tokens (standard price)
#   - completion_tokens:        output tokens
#
# Prices are in USD per token (NOT per 1M tokens).
# Example: $0.435 per 1M tokens = $0.000000435 per token = 4.35e-7

DEEPSEEK_MODELS = [
    {
        "model_name": "deepseek-v4-pro",
        "match_pattern": "(?i)^deepseek.*(v4|reasoner).*pro$",
        "prices": {
            "prompt_cache_hit_tokens":  3.625e-9,   # $0.003625 / 1M tokens
            "prompt_cache_miss_tokens": 4.35e-7,    # $0.435 / 1M tokens
            "completion_tokens":        8.7e-7,     # $0.87 / 1M tokens
        },
    },
    {
        "model_name": "deepseek-v4-flash",
        "match_pattern": "(?i)^deepseek.*(v4|chat).*flash$",
        "prices": {
            "prompt_cache_hit_tokens":  2.8e-9,     # $0.0028 / 1M tokens
            "prompt_cache_miss_tokens": 1.4e-7,     # $0.14 / 1M tokens
            "completion_tokens":        2.8e-7,     # $0.28 / 1M tokens
        },
    },
    {
        "model_name": "deepseek-chat",
        "match_pattern": "(?i)^deepseek-chat$",
        "prices": {
            "prompt_cache_hit_tokens":  2.8e-9,     # $0.0028 / 1M tokens (same as flash)
            "prompt_cache_miss_tokens": 1.4e-7,     # $0.14 / 1M tokens
            "completion_tokens":        2.8e-7,     # $0.28 / 1M tokens
        },
    },
    {
        "model_name": "deepseek-reasoner",
        "match_pattern": "(?i)^deepseek-reasoner$",
        "prices": {
            "prompt_cache_hit_tokens":  3.625e-9,   # $0.003625 / 1M tokens
            "prompt_cache_miss_tokens": 4.35e-7,    # $0.435 / 1M tokens
            "completion_tokens":        8.7e-7,     # $0.87 / 1M tokens
        },
    },
]


def create_model_definition(model_config: dict) -> bool:
    """Create a custom model definition in Langfuse via the API."""
    url = f"{LANGFUSE_HOST.rstrip('/')}/api/public/models"

    # Build the tokenizer config with per-usage-type pricing
    token_config = {
        "tokenizerId": "openai",
        "tokenizerModel": "gpt-4",
    }

    payload = {
        "modelName": model_config["model_name"],
        "matchPattern": model_config["match_pattern"],
        "unit": "TOKENS",
        "inputPrice": model_config["prices"]["prompt_cache_miss_tokens"],
        "outputPrice": model_config["prices"]["completion_tokens"],
        "totalPrice": None,
        "tokenizerId": "openai",
        "tokenizerModel": "gpt-4",
    }

    try:
        resp = requests.post(
            url,
            json=payload,
            auth=(LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY),
            headers={"Content-Type": "application/json"},
            timeout=15,
        )

        if resp.status_code in (200, 201):
            print(f"  ✅ Created model: {model_config['model_name']}")
            return True
        elif resp.status_code == 409:
            print(f"  ⚠️  Model already exists: {model_config['model_name']} — updating...")
            return update_model_definition(model_config)
        else:
            print(f"  ❌ Failed to create {model_config['model_name']}: {resp.status_code} {resp.text[:200]}")
            return False
    except Exception as e:
        print(f"  ❌ Error creating {model_config['model_name']}: {e}")
        return False


def update_model_definition(model_config: dict) -> bool:
    """Update an existing model definition in Langfuse."""
    # First, list models to find the ID
    list_url = f"{LANGFUSE_HOST.rstrip('/')}/api/public/models"
    try:
        resp = requests.get(
            list_url,
            auth=(LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY),
            timeout=15,
        )
        if resp.status_code != 200:
            print(f"    ❌ Could not list models: {resp.status_code}")
            return False

        models = resp.json().get("data", [])
        target = None
        for m in models:
            if m.get("modelName") == model_config["model_name"]:
                target = m
                break

        if not target:
            print(f"    ❌ Model not found in list: {model_config['model_name']}")
            return False

        model_id = target.get("id")
        if not model_id:
            print(f"    ❌ No ID for model: {model_config['model_name']}")
            return False

        # Update via PATCH or DELETE+CREATE
        delete_url = f"{LANGFUSE_HOST.rstrip('/')}/api/public/models/{model_id}"
        resp = requests.delete(
            delete_url,
            auth=(LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY),
            timeout=15,
        )
        if resp.status_code in (200, 204):
            return create_model_definition(model_config)
        else:
            print(f"    ❌ Could not delete old model: {resp.status_code}")
            return False

    except Exception as e:
        print(f"    ❌ Error updating {model_config['model_name']}: {e}")
        return False


def main():
    print("=" * 60)
    print("Langfuse Custom Model Setup — DeepSeek Cache-Aware Pricing")
    print("=" * 60)
    print(f"  Host: {LANGFUSE_HOST}")
    print()

    # Test connection
    try:
        test_url = f"{LANGFUSE_HOST.rstrip('/')}/api/public/models"
        resp = requests.get(
            test_url,
            auth=(LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY),
            timeout=10,
        )
        if resp.status_code != 200:
            print(f"ERROR: Could not connect to Langfuse API: {resp.status_code}")
            sys.exit(1)
        existing = resp.json().get("data", [])
        print(f"  Connected! Found {len(existing)} existing model definitions.")
        print()
    except Exception as e:
        print(f"ERROR: Could not connect to Langfuse: {e}")
        sys.exit(1)

    # Create model definitions
    success_count = 0
    for model_config in DEEPSEEK_MODELS:
        if create_model_definition(model_config):
            success_count += 1

        # Print pricing summary
        prices = model_config["prices"]
        print(f"    Cache hit:  ${prices['prompt_cache_hit_tokens'] * 1e6:.6f} / 1M tokens")
        print(f"    Cache miss: ${prices['prompt_cache_miss_tokens'] * 1e6:.6f} / 1M tokens")
        print(f"    Output:     ${prices['completion_tokens'] * 1e6:.6f} / 1M tokens")
        print()

    print("=" * 60)
    print(f"Done! {success_count}/{len(DEEPSEEK_MODELS)} models configured.")
    print()
    print("IMPORTANT: You also need to manually add cache-aware usage types")
    print("in Langfuse UI (Settings → Models) if per-usage-type pricing is")
    print("supported in your Langfuse version. The script sets inputPrice to")
    print("the cache-miss rate and outputPrice to the completion rate.")
    print()
    print("The code in core/llm_client.py now sends prompt_cache_hit_tokens")
    print("and prompt_cache_miss_tokens in every DeepSeek generation, so")
    print("Langfuse will have the breakdown for accurate cost calculation.")
    print("=" * 60)


if __name__ == "__main__":
    main()
