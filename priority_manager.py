# priority_manager.py
import logging
from typing import List, Dict, Any
import json
import ast

logger = logging.getLogger("priority_manager")

class PriorityManager:
    """
    Normalizes PriorityLog from JSONBin, safely appends entries and exposes parsed list for prompt builder.
    """

    def __init__(self, jsonbin=None):
        self._jsonbin = jsonbin
        self._raw_priority = None
        self._parsed: List[Dict[str, str]] = []

    def load_priority_record(self, raw: Dict[str, Any]):
        self._raw_priority = raw or {}
        self._parsed = self._parse_priority_log(self._raw_priority.get("PriorityLog") if isinstance(self._raw_priority, dict) else self._raw_priority)

    async def append_priority_entries(self, new_entries: List[Dict[str, str]]):
        if not self._jsonbin:
            raise RuntimeError("JsonBin client required")
        # fetch latest
        pr = await self._jsonbin.get_priority_record()
        record = pr.get("record", pr) if isinstance(pr, dict) else pr
        existing = record.get("PriorityLog")
        parsed_existing = self._parse_priority_log(existing)
        for e in new_entries:
            if isinstance(e, dict):
                parsed_existing.append({"type": str(e.get("type","")), "text": str(e.get("text",""))})
        record["PriorityLog"] = parsed_existing
        await self._jsonbin.put_priority_record(record)

    def get_parsed_priority_items(self) -> List[Dict[str, str]]:
        return self._parsed

    def _parse_priority_log(self, raw_priority: Any) -> List[Dict[str, str]]:
        if raw_priority is None:
            return []
        if isinstance(raw_priority, list):
            out = []
            for it in raw_priority:
                if isinstance(it, dict):
                    out.append({"type": str(it.get("type", "")), "text": str(it.get("text", ""))})
            return out
        if isinstance(raw_priority, dict):
            vals = []
            for v in raw_priority.values():
                if isinstance(v, list):
                    vals = v
                    break
            return self._parse_priority_log(vals)
        if isinstance(raw_priority, str):
            s = raw_priority.strip()
            # try JSON
            try:
                parsed = json.loads(s)
                return self._parse_priority_log(parsed)
            except Exception:
                pass
            # try ast literal
            try:
                parsed = ast.literal_eval(s)
                return self._parse_priority_log(parsed)
            except Exception:
                pass
            # try simple replacement
            try:
                repaired = s.replace("'", '"')
                parsed = json.loads(repaired)
                return self._parse_priority_log(parsed)
            except Exception:
                logger.debug("PriorityLog parsing yielded nothing")
                return []
        return []
