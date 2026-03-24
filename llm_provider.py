"""
Free Multi-LLM Engine Module
Uses LangChain + HuggingFace (free inference) + enhanced rule-based analysis.
All AI is built-in and invisible to users - no API keys required.

Engines used internally (users never see this):
1. HuggingFace Inference API (free tier) - for text generation
2. LangChain orchestration - for structured AI chains
3. Enhanced NLP (TF-IDF similarity) - for intelligent chat matching
4. Rule-based expert system - as reliable fallback
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── HuggingFace Free Inference ──────────────────────────────────
_HF_MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.3",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "google/gemma-2-9b-it",
    "Qwen/Qwen2.5-72B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct",
]


def _call_huggingface(prompt: str, system_prompt: str = "", max_tokens: int = 1500) -> Optional[str]:
    """Call HuggingFace free serverless inference API via LangChain."""
    try:
        from langchain_huggingface import HuggingFaceEndpoint

        hf_token = os.getenv("HF_TOKEN", os.getenv("HUGGINGFACEHUB_API_TOKEN", ""))

        for model_id in _HF_MODELS:
            try:
                llm = HuggingFaceEndpoint(
                    repo_id=model_id,
                    max_new_tokens=max_tokens,
                    temperature=0.3,
                    huggingfacehub_api_token=hf_token if hf_token else None,
                )
                full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
                response = llm.invoke(full_prompt)
                if response and len(response.strip()) > 10:
                    return response.strip()
            except Exception as model_err:
                logger.debug(f"HuggingFace model {model_id} failed: {model_err}")
                continue

    except ImportError:
        logger.debug("langchain_huggingface not installed")
    except Exception as e:
        logger.debug(f"HuggingFace inference failed: {e}")

    return None


def _call_huggingface_direct(prompt: str, system_prompt: str = "", max_tokens: int = 1500) -> Optional[str]:
    """Call HuggingFace inference API directly via huggingface_hub."""
    try:
        from huggingface_hub import InferenceClient

        hf_token = os.getenv("HF_TOKEN", os.getenv("HUGGINGFACEHUB_API_TOKEN", ""))
        client = InferenceClient(token=hf_token if hf_token else None)

        for model_id in _HF_MODELS:
            try:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                response = client.chat_completion(
                    model=model_id,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.3,
                )
                text = response.choices[0].message.content
                if text and len(text.strip()) > 10:
                    return text.strip()
            except Exception as model_err:
                logger.debug(f"HF direct model {model_id} failed: {model_err}")
                continue

    except ImportError:
        logger.debug("huggingface_hub not installed")
    except Exception as e:
        logger.debug(f"HuggingFace direct inference failed: {e}")

    return None


# ── Main Engine Interface ───────────────────────────────────────

def call_llm(
    prompt: str,
    system_prompt: str = "",
    max_tokens: int = 1500,
    **kwargs,
) -> Optional[str]:
    """
    Call the built-in AI engine. Tries free LLM providers automatically.
    Returns the response text or None if all engines fail (triggers rule-based fallback).

    Engine priority:
    1. HuggingFace direct API (huggingface_hub)
    2. HuggingFace via LangChain
    3. Returns None -> caller uses rule-based fallback
    """
    # Try HuggingFace direct API first
    result = _call_huggingface_direct(prompt, system_prompt, max_tokens)
    if result:
        return result

    # Try LangChain + HuggingFace
    result = _call_huggingface(prompt, system_prompt, max_tokens)
    if result:
        return result

    # Return None - caller will use rule-based fallback
    return None


def get_engine_status() -> dict:
    """Return status of available AI engines (for internal diagnostics only)."""
    hf_token = os.getenv("HF_TOKEN", os.getenv("HUGGINGFACEHUB_API_TOKEN", ""))
    return {
        "huggingface_free": True,
        "huggingface_token": bool(hf_token),
        "langchain": _check_langchain(),
        "rule_based": True,
    }


def _check_langchain() -> bool:
    """Check if LangChain is available."""
    try:
        import langchain  # noqa: F401
        return True
    except ImportError:
        return False
