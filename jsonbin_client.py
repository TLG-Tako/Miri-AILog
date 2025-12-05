# jsonbin_client.py
import os
import httpx
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("jsonbin_client")

JSONBIN_MASTER_KEY = os.getenv("JSONBIN_MASTER_KEY", "")
MAIN_BIN_URL = os.getenv("MAIN_BIN_URL", "")
PRIORITY_BIN_URL = os.getenv("PRIORITY_BIN_URL", "")

class JsonBinClient:
    def __init__(self):
        self.master_key = JSONBIN_MASTER_KEY
        self.main_url = MAIN_BIN_URL
        self.priority_url = PRIORITY_BIN_URL
        self._configured = bool(self.master_key and self.main_url and self.priority_url)

    @property
    def is_configured(self) -> bool:
        return self._configured

    async def _get(self, url: str) -> Dict[str, Any]:
        headers = {"X-Master-Key": self.master_key}
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(url, headers=headers)
            r.raise_for_status()
            return r.json()

    async def _put(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = {"X-Master-Key": self.master_key, "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.put(url, headers=headers, json=payload)
            r.raise_for_status()
            return r.json()

    async def get_main_record(self) -> Dict[str, Any]:
        if not self.is_configured:
            raise RuntimeError("JsonBin not configured")
        return await self._get(self.main_url)

    async def put_main_record(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_configured:
            raise RuntimeError("JsonBin not configured")
        return await self._put(self.main_url, payload)

    async def get_priority_record(self) -> Dict[str, Any]:
        if not self.is_configured:
            raise RuntimeError("JsonBin not configured")
        return await self._get(self.priority_url)

    async def put_priority_record(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_configured:
            raise RuntimeError("JsonBin not configured")
        return await self._put(self.priority_url, payload)
