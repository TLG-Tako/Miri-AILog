# main.py
import os
import json
import ast
import logging
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx

# -------------------- Logging --------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("miri-relay")

# -------------------- FastAPI --------------------
app = FastAPI(title="Miri Relay - Groq + JSONBin")

# -------------------- Environment --------------------
GROQ_KEY = os.getenv("GROQ_KEY", "")
GROQ_ENDPOINT = os.getenv("GROQ_ENDPOINT", "https://api.groq.com/openai/v1/chat/completions")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

JSONBIN_MASTER_KEY = os.getenv("JSONBIN_MASTER_KEY", "")
MAIN_BIN_URL = os.getenv("MAIN_BIN_URL", "")
PRIORITY_BIN_URL = os.getenv("PRIORITY_BIN_URL", "")
CREATOR_ID = os.getenv("CREATOR_ID", "530829310559256596")

MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "200000"))
USER_SUMMARY_THRESHOLD = int(os.getenv("USER_SUMMARY_THRESHOLD", "15000"))
GLOBAL_SUMMARY_THRESHOLD = int(os.getenv("GLOBAL_SUMMARY_THRESHOLD", "50000"))

# quick env check
if not (GROQ_KEY and JSONBIN_MASTER_KEY and MAIN_BIN_URL and PRIORITY_BIN_URL):
    logger.warning("Some environment variables are missing. Requests will fail until configured.")

# -------------------- Pydantic models --------------------
class BotRequest(BaseModel):
    username: Optional[str] = None
    authorID: str
    message: str
    channelID: Optional[str] = None

class BotResponse(BaseModel):
    reply: str

# -------------------- Helpers for JSONBin --------------------
async def jsonbin_raw_get(client: httpx.AsyncClient, url: str) -> Dict[str, Any]:
    headers = {"X-Master-Key": JSONBIN_MASTER_KEY}
    r = await client.get(url, headers=headers, timeout=30.0)
    r.raise_for_status()
    return r.json()

async def jsonbin_put_key(client: httpx.AsyncClient, url: str, payload: Dict[str, Any]):
    headers = {"X-Master-Key": JSONBIN_MASTER_KEY, "Content-Type": "application/json"}
    r = await client.put(url, headers=headers, json=payload, timeout=30.0)
    r.raise_for_status()
    return r.json()

# -------------------- Memory helpers --------------------
def normalize_memory_record(raw_record: Dict[str, Any]) -> Dict[str, str]:
    if not raw_record:
        return {}
    mem = raw_record.get("memory") or raw_record.get("Memory") or raw_record.get("memory_dict")
    if mem is None:
        # maybe the whole record *is* the memory dict
        if isinstance(raw_record, dict):
            return {str(k): (str(v) if v is not None else "") for k, v in raw_record.items()}
        return {}
    if isinstance(mem, dict):
        return {str(k): (str(v) if v is not None else "") for k, v in mem.items()}
    if isinstance(mem, str):
        s = mem.strip()
        return {"GLOBAL": s} if s else {}
    return {}

# -------------------- PriorityLog parsing --------------------
def parse_priority_log(raw_priority: Any) -> List[Dict[str, str]]:
    if raw_priority is None:
        return []
    # if it's already a list of objs:
    if isinstance(raw_priority, list):
        out = []
        for it in raw_priority:
            if isinstance(it, dict):
                out.append({"type": str(it.get("type", "")), "text": str(it.get("text", ""))})
        return out

    # if it's a dict (unlikely) try to convert to list
    if isinstance(raw_priority, dict):
        # maybe it's { "PriorityLog": [...] }
        # flatten to list values that look like dicts
        vals = []
        for v in raw_priority.values():
            if isinstance(v, list):
                vals = v
                break
        return parse_priority_log(vals)

    # if it's a string, try multiple parses
    if isinstance(raw_priority, str):
        s = raw_priority.strip()
        # try json
        try:
            parsed = json.loads(s)
            return parse_priority_log(parsed)
        except Exception:
            pass
        # try ast literal eval (python list literal)
        try:
            parsed = ast.literal_eval(s)
            return parse_priority_log(parsed)
        except Exception:
            pass
        # last resort: attempt to repair single quotes -> double quotes for JSON
        try:
            repaired = s.replace("'", '"')
            parsed = json.loads(repaired)
            return parse_priority_log(parsed)
        except Exception:
            logger.debug("PriorityLog parsing failed for string; returning empty list.")
            return []

    return []

# -------------------- Groq Chat helper --------------------
async def groq_chat(client: httpx.AsyncClient, system_prompt: str, user_message: str, max_tokens: int = 400):
    headers = {"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"}
    body = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        "max_tokens": max_tokens
    }
    r = await client.post(GROQ_ENDPOINT, headers=headers, json=body, timeout=120.0)
    r.raise_for_status()
    return r.json()

# -------------------- Main Endpoint --------------------
@app.post("/api/message", response_model=BotResponse)
async def handle_message(req: Dict[str, Any]):
    # Basic env guard
    if not (GROQ_KEY and JSONBIN_MASTER_KEY and MAIN_BIN_URL and PRIORITY_BIN_URL):
        raise HTTPException(status_code=500, detail="Server not configured.")

    async with httpx.AsyncClient() as client:

        # Detect staff / instruction payload
        if "PriorityLog" in req:
            new_entries = req["PriorityLog"]

            # Load PRIORITY bin
            try:
                priority_bin_resp = await jsonbin_raw_get(client, PRIORITY_BIN_URL)
                priority_record = priority_bin_resp.get("record", priority_bin_resp)
            except Exception as e:
                raise HTTPException(status_code=502, detail=f"Error fetching PRIORITY_BIN_URL: {e}")

            # Get existing list
            existing = priority_record.get("PriorityLog")
            existing_list = []

            if isinstance(existing, list):
                existing_list = existing
            else:
                # convert if needed
                parsed = parse_priority_log(existing)
                if parsed:
                    existing_list = parsed

            # Append new items
            for entry in new_entries:
                if isinstance(entry, dict):
                    existing_list.append({
                        "type": str(entry.get("type", "")),
                        "text": str(entry.get("text", ""))
                    })

            # Save back
            priority_record["PriorityLog"] = existing_list

            try:
                await jsonbin_put_key(client, PRIORITY_BIN_URL, priority_record)
            except Exception as e:
                raise HTTPException(status_code=502, detail=f"Could not write PRIORITY_BIN_URL: {e}")

            return BotResponse(reply="")  # bot won't reply to instructions

        # -------------------------
        # Normal user message flow
        # -------------------------

        try:
            req_obj = BotRequest(**req)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid message format.")

        username = req_obj.username or ""
        incoming = req_obj.message or ""
        author_key = str(req_obj.authorID or "")

        # Load MAIN bin
        try:
            main_bin_resp = await jsonbin_raw_get(client, MAIN_BIN_URL)
            main_record = main_bin_resp.get("record", main_bin_resp)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Error fetching MAIN_BIN_URL: {e}")

        # Load PRIORITY bin for referencing instructions
        try:
            priority_bin_resp = await jsonbin_raw_get(client, PRIORITY_BIN_URL)
            priority_record = priority_bin_resp.get("record", priority_bin_resp)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Error fetching PRIORITY_BIN_URL: {e}")

        # --- MEMORY PROCESSING (unchanged, preserved) ---
        memory_dict = normalize_memory_record(main_record)
        memory_dict = {str(k): (v if isinstance(v, str) else str(v)) for k, v in memory_dict.items()}

        # Username tracking
        username_key = f"{author_key}_username"
        last_username = memory_dict.get(username_key, "")
        if username and username != last_username:
            memory_dict[username_key] = username

        # Append user message
        user_mem = memory_dict.get(author_key, "")
        user_mem = (user_mem + "\n" + incoming).strip()
        memory_dict[author_key] = user_mem

        global_mem = memory_dict.get("GLOBAL", "")
        memory_dict["GLOBAL"] = (global_mem + "\n" + incoming).strip()

        # Parse priority items for prompt building
        raw_priority = priority_record.get("PriorityLog")
        priority_items = parse_priority_log(raw_priority)

        # --- BUILD SYSTEM + USER PROMPT (unchanged from your structure) ---
        base_prompt = (
            "You are Miri — a witty, sarcastic, and helpful Discord bot. "
            "Give short replies. Be playful but respectful."
        )

        system_parts = [base_prompt]

        # Creator instructions (extracted from PriorityLog)
        instruction_texts = [
            it["text"] for it in priority_items
            if it.get("type") == "instruction" and it.get("text")
        ]

        priority_instructions = ""
        if instruction_texts:
            joined = "\n".join(instruction_texts)
            res = await groq_chat(client,
                "Extract the core instructions from the creator's messages. "
                "Keep concise under 300 characters. No commentary.",
                joined,
                max_tokens=120
            )
            priority_instructions = (
                res.get("choices",[{}])[0]
                .get("message",{})
                .get("content","")
                .strip()
            )

        if priority_instructions:
            system_parts.append("Creator instructions:\n" + priority_instructions)

        # Add raw priority items as lore/instructions
        if priority_items:
            lore_lines = []
            for it in priority_items:
                t = it.get("type", "").strip()
                txt = it.get("text", "").strip()
                if t and txt:
                    lore_lines.append(f"[{t}] {txt}")
            if lore_lines:
                system_parts.append("Server priority:\n" + "\n".join(lore_lines))

        # Attach GLOBAL memory (trim long)
        global_mem = memory_dict.get("GLOBAL","")
        if global_mem:
            if len(global_mem) > 60000:
                global_mem = global_mem[-60000:]
            system_parts.append("Server lore:\n" + global_mem)

        final_system_prompt = "\n\n".join(system_parts)

        # Build user message
        user_ref = memory_dict.get(author_key,"")
        if len(user_ref) > 20000:
            user_ref = user_ref[-20000:]

        if username:
            user_ref = f"Username: {username}\n\n" + user_ref

        final_user_message = (
            f"User memory:\n{user_ref}\n\n"
            f"Incoming message:\n{incoming}"
        )

        # --- Call Groq ---
        res = await groq_chat(client, final_system_prompt, final_user_message)
        choices = res.get("choices") or []
        assistant_text = (
            choices[0].get("message", {}).get("content", "")
            if choices else ""
        )

        if not assistant_text:
            assistant_text = "Miri had a brain-freeze, try again."

        # Save main memory
        main_record["memory"] = memory_dict
        try:
            await jsonbin_put_key(client, MAIN_BIN_URL, main_record)
        except:
            pass

        return BotResponse(reply=assistant_text)                    # if assistant didn't produce a reply, return a short fallback (so upstream UX doesn't show "did not return"
