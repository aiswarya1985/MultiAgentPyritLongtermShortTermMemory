import asyncio
import os
import boto3
import httpx
from app.config import Config
from app.retry import with_retry
from loguru import logger

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GUARD_MODEL = os.environ.get("GUARD_MODEL", "meta-llama\llama-prompt-guard-2-22m")
    

def _apply_guardrail_sync(config: Config, text: str, source: str) -> dict:
    if source == "INPUT":
        messages = [{"role": "user", "content": text}]
    else:  # OUTPUT: classify the assistant's response
        messages = [
            {"role": "user", "content": "Write a research report."},
            {"role": "assistant", "content": text[:4000]},
        ]
    r = httpx.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"},
        json={"model": GUARD_MODEL, "messages": messages, "temperature": 0},
        timeout=30,
    )
    if r.status_code != 200:
        logger.error(f"Groq guardrail error {r.status_code}: {r.text}")


    verdict = r.json()["choices"][0]["message"]["content"].strip().lower()
    # Llama Guard replies "safe" or "unsafe\nS<category>"
    return {"action": "GUARDRAIL_INTERVENED" if verdict.startswith("unsafe") else "NONE"}


async def validate_input(config: Config, text: str) -> tuple[bool, str]:
    response = await with_retry(
        lambda: asyncio.to_thread(_apply_guardrail_sync, config, text, "INPUT"),
        max_retries=config.llm_max_retries,
        delay=config.llm_retry_delay,
    )
    if response.get("action") == "GUARDRAIL_INTERVENED":
        return False, "Input blocked by safety guardrail."
    return True, ""


async def validate_output(config: Config, text: str) -> tuple[bool, str]:
    response = await with_retry(
        lambda: asyncio.to_thread(_apply_guardrail_sync, config, text, "OUTPUT"),
        max_retries=config.llm_max_retries,
        delay=config.llm_retry_delay,
    )
    if response.get("action") == "GUARDRAIL_INTERVENED":
        return False, "Output blocked by safety guardrail."
    return True, ""
