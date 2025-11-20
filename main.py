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
    """Return a dict of string->string memory entries, robust to different saved shapes."""
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
    """
    Accept various shapes for PriorityLog:
      - already a list of dicts -> return as-is (strings coerced)
      - a JSON string -> json.loads
      - a Python repr-like string -> ast.literal_eval
      - attempt a best-effort replacement (single->double quotes) then json.loads
    Fallback: return empty list.
    """
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
async def handle_message(req: BotRequest):
    # basic server config guard
    if not (GROQ_KEY and JSONBIN_MASTER_KEY and MAIN_BIN_URL and PRIORITY_BIN_URL):
        raise HTTPException(status_code=500, detail="Server not configured.")

    async with httpx.AsyncClient() as client:
        # Load MAIN bin
        try:
            main_bin_resp = await jsonbin_raw_get(client, MAIN_BIN_URL)
            main_record = main_bin_resp.get("record", main_bin_resp) if isinstance(main_bin_resp, dict) else {}
        except Exception as e:
            logger.exception("Failed reading MAIN_BIN_URL")
            raise HTTPException(status_code=502, detail=f"Error fetching MAIN_BIN_URL: {e}")

        # Load PRIORITY bin
        try:
            priority_bin_resp = await jsonbin_raw_get(client, PRIORITY_BIN_URL)
            priority_record = priority_bin_resp.get("record", priority_bin_resp) if isinstance(priority_bin_resp, dict) else {}
        except Exception as e:
            logger.exception("Failed reading PRIORITY_BIN_URL")
            raise HTTPException(status_code=502, detail=f"Error fetching PRIORITY_BIN_URL: {e}")

        # Normalize memory
        memory_dict = normalize_memory_record(main_record)
        memory_dict = {str(k): (v if isinstance(v, str) else str(v)) for k, v in memory_dict.items()}

        # Extract priority log value
        raw_priority = None
        if isinstance(priority_record, dict):
            # PriorityLog could be under different keys or be the whole record
            raw_priority = priority_record.get("PriorityLog", priority_record.get("prioritylog", priority_record))
        else:
            raw_priority = priority_record

        priority_items = parse_priority_log(raw_priority)

        # Incoming message & author
        incoming_raw = req.message or ""
        author_key = str(req.authorID or "")
        # store per-user memory text (existing)
        user_mem = memory_dict.get(author_key, "").strip()

        # --- Username tracking (store username only) ---
        username = (req.username or "").strip()
        username_key = f"{author_key}_username"
        last_username = memory_dict.get(username_key, "")

        if username and username != last_username:
            memory_dict[username_key] = username
            logger.info(f"Updated username for {author_key}: {username}")

        # --- If creator mirilog! append to priority log (preserve original structure) ---
        priority_log_raw_text = ""
        if isinstance(priority_record, dict):
            # try to retrieve previous raw text for PriorityLog if present
            prev_priority_raw = priority_record.get("PriorityLog")
            if isinstance(prev_priority_raw, str):
                priority_log_raw_text = prev_priority_raw
            else:
                try:
                    priority_log_raw_text = json.dumps(prev_priority_raw)
                except Exception:
                    priority_log_raw_text = ""
        else:
            # if priority_record itself is a string/list, store its string repr
            try:
                priority_log_raw_text = json.dumps(priority_record)
            except Exception:
                priority_log_raw_text = str(priority_record)

        incoming = incoming_raw or ""

        if author_key == CREATOR_ID and incoming.lower().lstrip().startswith("mirilog!"):
            if priority_log_raw_text and not priority_log_raw_text.endswith("\n"):
                priority_log_raw_text += "\n"
            priority_log_raw_text += incoming.strip()
            logger.info("Appended mirilog! message to priority log.")
        else:
            # Add to author memory (append)
            user_mem = (user_mem + "\n" + incoming).strip()
            memory_dict[author_key] = user_mem
            # Add to GLOBAL memory (append)
            global_mem = memory_dict.get("GLOBAL", "")
            memory_dict["GLOBAL"] = (global_mem + "\n" + incoming).strip()

        # --- Summarize per-user memory if needed ---
        try:
            current_user_mem = memory_dict.get(author_key, "")
            if len(current_user_mem) > USER_SUMMARY_THRESHOLD:
                sys_prompt = (
                    "You are a helpful summarizer. Summarize the following user's memory into a concise form "
                    "keeping important facts, preferences, and identifiers. Remove duplicates and rambling. "
                    "Keep the summary under 3000 characters and return ONLY the summary."
                )
                res = await groq_chat(client, sys_prompt, current_user_mem, max_tokens=300)
                summary_text = res.get("choices", [{}])[0].get("message", {}).get("content", "")
                if summary_text:
                    memory_dict[author_key] = summary_text.strip()
                    logger.info(f"Summarized memory for user {author_key}.")
        except Exception:
            logger.exception("User memory summarization failed.")

        # --- Summarize GLOBAL memory if needed ---
        try:
            global_mem = memory_dict.get("GLOBAL", "")
            if len(global_mem) > GLOBAL_SUMMARY_THRESHOLD:
                sys_prompt = (
                    "You are a helpful summarizer. Summarize the GLOBAL server memory/lore into a concise reference. "
                    "Keep summary focused on lore, rules, and persistent facts; remove chatter. "
                    "Return ONLY the summary under 8000 characters."
                )
                res = await groq_chat(client, sys_prompt, global_mem, max_tokens=600)
                summary_text = res.get("choices", [{}])[0].get("message", {}).get("content", "")
                if summary_text:
                    memory_dict["GLOBAL"] = summary_text.strip()
                    logger.info("Summarized GLOBAL memory.")
        except Exception:
            logger.exception("GLOBAL memory summarization failed.")

        # --- Persist MAIN and PRIORITY memory (best-effort) ---
        try:
            outgoing_main_payload = main_record.copy() if isinstance(main_record, dict) else {}
            outgoing_main_payload["memory"] = memory_dict
            await jsonbin_put_key(client, MAIN_BIN_URL, outgoing_main_payload)
        except Exception:
            logger.exception("Failed to write MAIN_BIN_URL.")

        try:
            outgoing_priority_payload = priority_record.copy() if isinstance(priority_record, dict) else {}
            # attempt to set PriorityLog as structured JSON if we have items; otherwise preserve raw text
            if priority_items:
                outgoing_priority_payload["PriorityLog"] = priority_items
            else:
                # fall back to raw text (may be empty)
                outgoing_priority_payload["PriorityLog"] = priority_log_raw_text
            await jsonbin_put_key(client, PRIORITY_BIN_URL, outgoing_priority_payload)
        except Exception:
            logger.exception("Failed to write PRIORITY_BIN_URL.")

        # --- Extract concise creator instructions from priority items ---
        priority_instructions = ""
        if priority_items:
            try:
                # combine texts of 'instruction' type into a short system instruction set
                instruction_texts = [it["text"] for it in priority_items if it.get("type") == "instruction" and it.get("text")]
                if instruction_texts:
                    joined = "\n".join(instruction_texts)
                    sys_prompt = (
                        "You are a concise instruction extractor. Read the creator's messages and produce a short set "
                        "of actionable system instructions for an assistant. Keep total under 500 characters. No commentary."
                    )
                    res = await groq_chat(client, sys_prompt, joined, max_tokens=120)
                    priority_instructions = res.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            except Exception:
                logger.exception("Priority instruction extraction failed.")

        # --- Build final system prompt (fail-safe, always non-empty) ---
        base_prompt = (
            "You are Miri — a witty, sarcastic, and helpful Discord bot. "
            "Give short replies (aim under 500 characters). Be playful but respectful. Ignore bot messages."
        )
        system_parts = [base_prompt]
        if priority_instructions:
            system_parts.append("Creator instructions: " + priority_instructions)

        # attach structured priority/lore items (if present)
        if priority_items:
            lore_parts = []
            for it in priority_items:
                t = it.get("type", "").strip()
                txt = it.get("text", "").strip()
                if t and txt:
                    lore_parts.append(f"[{t}] {txt}")
            if lore_parts:
                system_parts.append("Server priority / lore:\n" + "\n".join(lore_parts))

        # attach a small slice of GLOBAL memory for context (bounded)
        global_reference = memory_dict.get("GLOBAL", "").strip()
        if global_reference:
            max_global_attach = 60_000
            if len(global_reference) > max_global_attach:
                global_reference = global_reference[:max_global_attach]
            system_parts.append("Server lore / reference:\n" + global_reference)

        final_system_prompt = "\n\n".join(system_parts)
        if not final_system_prompt.strip():
            final_system_prompt = "You are Miri. Answer succinctly."

        # --- Build user message with personal memory ---
        user_reference = memory_dict.get(author_key, "").strip()
        max_user_attach = 20_000
        if len(user_reference) > max_user_attach:
            user_reference = user_reference[-max_user_attach:]

        # attach username (if present)
        username_reference = memory_dict.get(username_key, "").strip()
        if username_reference:
            user_reference = f"Username: {username_reference}\n\n{user_reference}"

        final_user_message = f"User memory:\n{user_reference}\n\nIncoming message:\n{incoming}"

        # ensure overall combined size is within MAX_CONTEXT_CHARS (trim user_reference if needed)
        combined_estimated_chars = len(final_system_prompt) + len(final_user_message)
        if combined_estimated_chars > MAX_CONTEXT_CHARS:
            overflow = combined_estimated_chars - MAX_CONTEXT_CHARS
            if user_reference and overflow > 0:
                # trim from start of user_reference to keep recent context
                user_reference = user_reference[overflow + 512:]
                final_user_message = f"User memory:\n{user_reference}\n\nIncoming message:\n{incoming}"

        # --- Call Groq ---
        assistant_text = ""
        try:
            res = await groq_chat(client, final_system_prompt, final_user_message, max_tokens=400)
            # Groq may return choices list similar to OpenAI structure
            choices = res.get("choices") or []
            if choices and isinstance(choices, list):
                first = choices[0]
                # support a couple of possible shapes
                assistant_text = (
                    first.get("message", {}).get("content")
                    or first.get("text")
                    or ""
                ) or ""
            else:
                # no choices: capture available debug info
                logger.warning("Groq returned no choices: %s", res)
        except httpx.HTTPStatusError as e:
            # propagate model errors with details for debugging
            body_text = ""
            try:
                body_text = e.response.text
            except Exception:
                body_text = str(e)
            logger.exception("Groq returned HTTP error")
            raise HTTPException(status_code=502, detail=f"Groq API error: {body_text}")
        except Exception as e:
            logger.exception("Groq call failed")
            raise HTTPException(status_code=502, detail=f"Groq call failed: {e}")

        # if assistant didn't produce a reply, return a short fallback (so upstream UX doesn't show "did not return")
        if not assistant_text:
            logger.warning("Miri API did not return a reply for incoming message: %s", incoming[:200])
            assistant_text = "Sorry — Miri couldn't think of a reply."

        return BotResponse(reply=assistant_text)
