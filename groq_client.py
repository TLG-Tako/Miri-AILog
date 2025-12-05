# groq_client.py
import os
import httpx
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("groq_client")

GROQ_KEY = os.getenv("GROQ_KEY", "")
GROQ_ENDPOINT = os.getenv("GROQ_ENDPOINT", "https://api.groq.com/openai/v1/chat/completions")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

class GroqClient:
    def __init__(self):
        self.api_key = GROQ_KEY
        self.endpoint = GROQ_ENDPOINT
        self.model = GROQ_MODEL
        self._configured = bool(self.api_key)

    @property
    def is_configured(self) -> bool:
        return self._configured

    async def chat(self, system_prompt: str, user_message: str, max_tokens: int = 400) -> Dict[str, Any]:
        if not self.is_configured:
            raise RuntimeError("Groq client not configured")
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "max_tokens": max_tokens
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(self.endpoint, headers=headers, json=body)
            r.raise_for_status()
            return r.json()
