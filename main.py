# main.py
import os
import asyncio
import logging
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("miri-relay")

app = FastAPI(title="Miri Relay - Groq + JSONBin")

@app.get("/")
async def root():
    return {"status": "alive", "message": "Miri Relay API is running."}

# --- ENV / CONFIG ---
GROQ_KEY = os.getenv("GROQ_KEY", "")
GROQ_ENDPOINT = os.getenv("GROQ_ENDPOINT", "https://api.groq.com/openai/v1/chat/completions")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

JSONBIN_MASTER_KEY = os.getenv("JSONBIN_MASTER_KEY", "")
MAIN_BIN_URL = os.getenv("MAIN_BIN_URL", "")        # main memory bin (will become dict of users)
PRIORITY_BIN_URL = os.getenv("PRIORITY_BIN_URL", "")# priority instructions bin (global)
CREATOR_ID = os.getenv("CREATOR_ID", "530829310559256596")

# thresholds and sizes
# conservative character caps: Llama-3.1-8b-instant ~131k tokens -> ~524k chars; use safe caps
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "200000"))  # hard safety clamp per combined prompt
USER_SUMMARY_THRESHOLD = int(os.getenv("USER_SUMMARY_THRESHOLD", "15000"))  # chars before per-user summarise
GLOBAL_SUMMARY_THRESHOLD = int(os.getenv("GLOBAL_SUMMARY_THRESHOLD", "50000"))  # chars before summarise GLOBAL

# Basic checks: we allow import in dev even if not configured; raise on request if missing
if not (GROQ_KEY and JSONBIN_MASTER_KEY and MAIN_BIN_URL and PRIORITY_BIN_URL):
    logger.warning("Some environment variables are missing. The server will raise on requests until configured.")


# ---------- Pydantic models ----------
class BotRequest(BaseModel):
    authorID: str
    message: str
    channelID: Optional[str] = None

class BotResponse(BaseModel):
    reply: str


# ---------- JSONBin helpers ----------
async def jsonbin_raw_get(client: httpx.AsyncClient, url: str) -> Dict[str, Any]:
    """
    Return the parsed JSON response from JSONBin's GET.
    """
    headers = {"X-Master-Key": JSONBIN_MASTER_KEY}
    r = await client.get(url, headers=headers, timeout=30.0)
    r.raise_for_status()
    return r.json()


async def jsonbin_put_key(client: httpx.AsyncClient, url: str, payload: Dict[str, Any]):
    """
    PUT a full JSON body into the bin. payload is a dict representing the content.
    """
    headers = {"X-Master-Key": JSONBIN_MASTER_KEY, "Content-Type": "application/json"}
    r = await client.put(url, headers=headers, json=payload, timeout=30.0)
    r.raise_for_status()
    return r.json()


# ---------- Memory helpers (per-user migration & access) ----------
def normalize_memory_record(raw_record: Dict[str, Any]) -> Dict[str, str]:
    """
    Given the 'record' object returned by JSONBin, extract/normalize the 'memory' into a dict:
      { "GLOBAL": "...", "12345": "...", ... }
    Cases handled:
      - record has "memory" as a string -> migrate to {"GLOBAL": str}
      - record has "memory" as an object/dict -> return it (cast values to str)
      - record already missing -> return empty dict
    """
    if not raw_record:
        return {}

    mem = raw_record.get("memory")
    if mem is None:
        # maybe the whole record is already the dict of memories
        # attempt to detect direct structure: { "memory": { ... } } handled above; otherwise return {}
        return {}

    # If memory is a dict already, coerce to str values
    if isinstance(mem, dict):
        return {str(k): (str(v) if v is not None else "") for k, v in mem.items()}

    # If memory is a raw string, migrate to GLOBAL key
    if isinstance(mem, str):
        s = mem.strip()
        return {"GLOBAL": s} if s else {}

    # Fallback: unknown type -> empty
    return {}


def build_memory_payload(existing_record: Dict[str, Any], memory_dict: Dict[str, str]) -> Dict[str, Any]:
    """
    Build the JSONBin payload preserving other keys in record but replacing/setting 'memory' to memory_dict.
    existing_record is the full JSON that came from jsonbin. We want to preserve other fields (if present)
    while writing back a consistent structure.
    """
    record = existing_record.copy() if isinstance(existing_record, dict) else {}
    # Ensure we write under 'memory' as an object/dict
    record["memory"] = memory_dict
    return record


# ---------- Groq Chat helper ----------
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


# ---------- Main endpoint ----------
@app.post("/api/message", response_model=BotResponse)
async def handle_message(req: BotRequest):
    # config check
    if not (GROQ_KEY and JSONBIN_MASTER_KEY and MAIN_BIN_URL and PRIORITY_BIN_URL):
        raise HTTPException(status_code=500, detail="Server not configured. Set GROQ_KEY, JSONBIN_MASTER_KEY, MAIN_BIN_URL, PRIORITY_BIN_URL.")

    async with httpx.AsyncClient() as client:
        # 1) Read current MAIN bin and PRIORITY bin
        try:
            main_bin_resp = await jsonbin_raw_get(client, MAIN_BIN_URL)
            # main_bin_resp expected shape: {'record': {...}} or direct; JSONBin usually wraps in top-level 'record'
            main_record = main_bin_resp.get("record", main_bin_resp) if isinstance(main_bin_resp, dict) else {}
        except Exception as e:
            logger.exception("Failed reading MAIN_BIN_URL")
            raise HTTPException(status_code=502, detail=f"Error fetching MAIN_BIN_URL: {e}")

        try:
            priority_bin_resp = await jsonbin_raw_get(client, PRIORITY_BIN_URL)
            priority_record = priority_bin_resp.get("record", priority_bin_resp) if isinstance(priority_bin_resp, dict) else {}
        except Exception as e:
            logger.exception("Failed reading PRIORITY_BIN_URL")
            raise HTTPException(status_code=502, detail=f"Error fetching PRIORITY_BIN_URL: {e}")

        # Normalize memory dict from main_record
        memory_dict = normalize_memory_record(main_record)  # dict str->str
        # Prioritize preserving any pre-existing memory_dict content
        # If migration required, memory_dict will include GLOBAL key

        # Extract priority log string (raw)
        priority_log_raw = ""
        # priority_record may already contain 'PriorityLog' key or 'priority' or similar; try common keys
        if isinstance(priority_record, dict):
            if "PriorityLog" in priority_record:
                priority_log_raw = str(priority_record.get("PriorityLog") or "")
            elif "priority" in priority_record:
                priority_log_raw = str(priority_record.get("priority") or "")
            elif "record" in priority_record and isinstance(priority_record["record"], dict) and "PriorityLog" in priority_record["record"]:
                priority_log_raw = str(priority_record["record"].get("PriorityLog") or "")
            else:
                # fallback - try common keys/strings
                # if entire record is just a string under some other key, try join
                for v in priority_record.values():
                    if isinstance(v, str):
                        priority_log_raw += v + "\n"

        # Defensive: ensure keys are strings
        memory_dict = {str(k): (v if isinstance(v, str) else str(v)) for k, v in memory_dict.items()}

        # 2) Append incoming message to the author's personal memory (per Option B)
        incoming = req.message or ""
        author = req.authorID or ""
        author_key = str(author)

        # Ensure author exists in dict
        user_mem = memory_dict.get(author_key, "").strip()
        # Use '↑' delimiter to maintain compatibility with previous format
        if author == CREATOR_ID:
            # If creator and message begins with 'mirilog!' we want to put it into priority_log_raw
            if incoming.lower().lstrip().startswith("mirilog!"):
                # append to priority log with delimiter
                if priority_log_raw and not priority_log_raw.endswith("\n"):
                    priority_log_raw += "\n"
                priority_log_raw += incoming.strip()
                logger.info("Appended a mirilog! message to priority log.")
            else:
                # creator non-priority content goes into GLOBAL and into creator's personal mem
                # We'll add to both GLOBAL (so other users + GLOBAL context can see) and the creator user key
                global_mem = memory_dict.get("GLOBAL", "")
                global_mem = (global_mem + "\n" + f"(Creator) {incoming}").strip()
                memory_dict["GLOBAL"] = global_mem
                user_mem = (user_mem + "\n" + f"(Creator) {incoming}").strip()
                memory_dict[author_key] = user_mem
        else:
            # normal user message appended to their personal memory and also appended to GLOBAL (option 2)
            user_mem = (user_mem + "\n" + incoming).strip()
            memory_dict[author_key] = user_mem

            global_mem = memory_dict.get("GLOBAL", "")
            global_mem = (global_mem + "\n" + incoming).strip()
            memory_dict["GLOBAL"] = global_mem

        # 3) Summarize per-user memory if too long
        # Only summarize the specific user's memory to keep per-user context small
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
            logger.exception("User memory summarization failed; keeping original user memory.")

        # 4) Optionally summarise GLOBAL memory if it grows too large
        try:
            global_mem = memory_dict.get("GLOBAL", "")
            if len(global_mem) > GLOBAL_SUMMARY_THRESHOLD:
                sys_prompt = (
                    "You are a helpful summarizer. Summarize the GLOBAL server memory/lore into a concise reference. "
                    "Keep the summary focused on lore, rules, and persistent facts; remove chatter. "
                    "Return ONLY the summary, under 8000 characters."
                )
                res = await groq_chat(client, sys_prompt, global_mem, max_tokens=600)
                summary_text = res.get("choices", [{}])[0].get("message", {}).get("content", "")
                if summary_text:
                    memory_dict["GLOBAL"] = summary_text.strip()
                    logger.info("Summarized GLOBAL memory.")
        except Exception:
            logger.exception("GLOBAL memory summarization failed; keeping original GLOBAL memory.")

        # 5) Persist updated memory_dict back to MAIN_BIN_URL and updated priority_log to PRIORITY_BIN_URL
        try:
            # We want to preserve any other keys that were originally in main_record.
            # Build payload using main_record as base and replace 'memory' key with memory_dict
            outgoing_main_payload = main_record.copy() if isinstance(main_record, dict) else {}
            outgoing_main_payload["memory"] = memory_dict
            await jsonbin_put_key(client, MAIN_BIN_URL, outgoing_main_payload)
        except Exception:
            logger.exception("Failed to write MAIN_BIN_URL; continuing without failing the request.")

        try:
            outgoing_priority_payload = priority_record.copy() if isinstance(priority_record, dict) else {}
            outgoing_priority_payload["PriorityLog"] = priority_log_raw
            await jsonbin_put_key(client, PRIORITY_BIN_URL, outgoing_priority_payload)
        except Exception:
            logger.exception("Failed to write PRIORITY_BIN_URL; continuing without failing the request.")

        # 6) Extract concise creator instructions from priority_log_raw for system prompt
        priority_instructions = ""
        if priority_log_raw.strip():
            try:
                sys_prompt = (
                    "You are a concise instruction extractor. Read the creator's messages and produce a short set "
                    "of actionable system instructions for an assistant. Keep total under 500 characters. No commentary."
                )
                res = await groq_chat(client, sys_prompt, priority_log_raw, max_tokens=120)
                priority_instructions = res.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            except Exception:
                logger.exception("Priority instruction extraction failed.")
                priority_instructions = ""

        # 7) Build the final system prompt; include GLOBAL lore + priority instructions + small policy
        base_prompt = (
            "You are Miri — a witty, sarcastic, and helpful Discord bot. "
            "You have personality and opinions. Give short to moderate-length helpful replies (aim under 500 characters). "
            "Be playful and slightly mocking when appropriate, but keep respect and avoid insults. Ignore bot messages. "
        )

        # Attach GLOBAL memory as lore reference (if exists) and attach user's personal memory as context (but not flooded)
        global_reference = memory_dict.get("GLOBAL", "").strip()
        user_reference = memory_dict.get(author_key, "").strip()

        # Compose system prompt parts carefully to avoid exceeding model context; prefer small headers
        system_parts = [base_prompt]

        if priority_instructions:
            system_parts.append("Creator instructions: " + priority_instructions)

        if global_reference:
            # Put as Lore / Server Reference
            # Truncate to reasonable length if extremely long
            max_global_attach = 60_000
            if len(global_reference) > max_global_attach:
                global_reference = global_reference[:max_global_attach]
            system_parts.append("Server lore / reference:\n" + global_reference)

        # We'll send user's memory as part of the user prompt rather than the system prompt to keep roles clear.
        final_system_prompt = "\n\n".join(system_parts)

        # 8) Build user message that includes the user's memory (short) + incoming message
        # Keep user memory attached but truncated to a safe size
        user_memory_attach = user_reference
        max_user_attach = 20_000
        if len(user_memory_attach) > max_user_attach:
            user_memory_attach = user_memory_attach[-max_user_attach:]  # keep recent tail

        # Compose the final user content
        user_content_parts = []
        if user_memory_attach:
            user_content_parts.append("User memory:\n" + user_memory_attach)
        user_content_parts.append("Incoming message:\n" + incoming)
        final_user_message = "\n\n".join(user_content_parts)

        # Hard safety: ensure combined prompt size not insane (characters)
        combined_estimated_chars = len(final_system_prompt) + len(final_user_message)
        if combined_estimated_chars > MAX_CONTEXT_CHARS:
            # Trim user memory first
            overflow = combined_estimated_chars - MAX_CONTEXT_CHARS
            # trim from user_memory_attach
            if user_memory_attach and overflow > 0:
                trim_amount = min(len(user_memory_attach), overflow + 1000)
                user_memory_attach = user_memory_attach[trim_amount:]
                user_content_parts[0] = "User memory:\n" + user_memory_attach
                final_user_message = "\n\n".join(user_content_parts)
                combined_estimated_chars = len(final_system_prompt) + len(final_user_message)
            # If still overflowing, trim global_reference in the system prompt
            if combined_estimated_chars > MAX_CONTEXT_CHARS and global_reference:
                trim_amount = combined_estimated_chars - MAX_CONTEXT_CHARS + 1000
                global_reference = global_reference[trim_amount:]
                # rebuild system prompt
                system_parts = [base_prompt]
                if priority_instructions:
                    system_parts.append("Creator instructions: " + priority_instructions)
                system_parts.append("Server lore / reference:\n" + global_reference)
                final_system_prompt = "\n\n".join(system_parts)
                combined_estimated_chars = len(final_system_prompt) + len(final_user_message)

        # 9) Final model call
        try:
            res = await groq_chat(client, final_system_prompt, final_user_message, max_tokens=400)
            assistant_text = res.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        except httpx.HTTPStatusError as e:
            logger.exception("Groq returned HTTP error")
            raise HTTPException(status_code=502, detail=f"Groq API error: {e.response.text}")
        except Exception as e:
            logger.exception("Groq call failed")
            raise HTTPException(status_code=502, detail=f"Groq call failed: {e}")

        # 10) Return assistant text
        return BotResponse(reply=assistant_text)
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        "max_tokens": max_tokens
    r = await client.post(GROQ_ENDPOINT, headers=headers, json=body, timeout=60.0)
    r.raise_for_status()
    return r.json()

# Main endpoint
@app.post("/api/message", response_model=BotResponse)
async def handle_message(req: BotRequest):
    if not (GROQ_KEY and JSONBIN_MASTER_KEY and MAIN_BIN_URL and PRIORITY_BIN_URL):
        raise HTTPException(status_code=500, detail="Server not configured. Set GROQ_KEY, JSONBIN_MASTER_KEY, MAIN_BIN_URL, PRIORITY_BIN_URL.")

    async with httpx.AsyncClient() as client:
        # 1. load main memory and priority log
        try:
            main_memory = await jsonbin_get(client, MAIN_BIN_URL)
            priority_log = await jsonbin_get(client, PRIORITY_BIN_URL)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Error fetching JSONBin: {e}")

        # 2. append incoming message correctly
        incoming = req.message or ""
        author = req.authorID or ""
        # detect creator messages
        if author == CREATOR_ID:
            if "mirilog!" in incoming.lower():
                priority_log = (priority_log + "↑" + incoming).strip("↑")
            else:
                main_memory = (main_memory + "↑(Creator)|" + incoming).strip("↑")
        else:
            main_memory = (main_memory + "↑" + incoming).strip("↑")

        # 3. escape fix - preserve quotes (not strictly necessary for JSON)
        # already used as JSON fields, so no special replace needed

        # 4. summarise main memory if too long
        if len(main_memory) > MAIN_MEMORY_THRESHOLD:
            try:
                sys_prompt = "You are a helpful summarizer. Return ONLY a concise summary of the memory. Remove duplicates, sensitive info, and rambling. Keep it under 5000 characters."
                j = await groq_chat(client, sys_prompt, main_memory, max_tokens=300)
                summary = j.get("choices", [{}])[0].get("message", {}).get("content", "")
                if summary:
                    main_memory = summary
            except Exception as e:
                # keep existing memory if summariser fails
                pass

        # 5. persist memory back to JSONBin (both)
        try:
            await jsonbin_put(client, MAIN_BIN_URL, "memory", main_memory)
            await jsonbin_put(client, PRIORITY_BIN_URL, "PriorityLog", priority_log)
        except Exception as e:
            # Not fatal — continue but warn in logs
            pass

        # 6. summarise priority log into short instruction (<=150 chars)
        priority_instructions = ""
        if priority_log.strip():
            try:
                sys_prompt = "You are a concise instruction extractor. Read the creator's priority messages and produce a short set of actionable system instructions for an assistant. Keep total under 500 characters. No commentary."
                j = await groq_chat(client, sys_prompt, priority_log, max_tokens=120)
                priority_instructions = j.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            except Exception:
                priority_instructions = ""

        # 7. build base system prompt
        base_prompt = "You are Miri, a witty, sarcastic, but helpful and friendly Discord bot. You are your own person with your own personality and opinions. Short to moderate length replies. Slightly mocking when appropriate. Ignore bots. Keep replies under 500 characters."

        if priority_instructions:
            final_system_prompt = f"{base_prompt}\nCreator instructions: {priority_instructions}"
        else:
            final_system_prompt = base_prompt

        # 8. final assistant call
        try:
            j = await groq_chat(client, final_system_prompt, incoming, max_tokens=400)
            assistant_text = j.get("choices", [{}])[0].get("message", {}).get("content", "")
            if assistant_text is None:
                assistant_text = ""
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=502, detail=f"Groq API error: {e.response.text}")
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Groq call failed: {e}")

        # 9. return the assistant text
        return BotResponse(reply=assistant_text)
