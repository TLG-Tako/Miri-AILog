# main.py
import os
import logging
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx

# -------------------- Logging --------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("miri-relay")

# -------------------- FastAPI --------------------
app = FastAPI(title="Miri Relay - Groq + JSONBin")

@app.get("/")
async def root():
    return {"status": "alive", "message": "Miri Relay API is running."}

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

if not (GROQ_KEY and JSONBIN_MASTER_KEY and MAIN_BIN_URL and PRIORITY_BIN_URL):
    logger.warning("Some environment variables are missing. Requests will fail until configured.")

# -------------------- Pydantic models --------------------
class BotRequest(BaseModel):
    authorID: str
    message: str
    channelID: Optional[str] = None

class BotResponse(BaseModel):
    reply: str

# -------------------- JSONBin helpers --------------------
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
    mem = raw_record.get("memory")
    if mem is None:
        return {}
    if isinstance(mem, dict):
        return {str(k): (str(v) if v is not None else "") for k, v in mem.items()}
    if isinstance(mem, str):
        s = mem.strip()
        return {"GLOBAL": s} if s else {}
    return {}

# -------------------- Groq Chat Helper --------------------
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
    if not (GROQ_KEY and JSONBIN_MASTER_KEY and MAIN_BIN_URL and PRIORITY_BIN_URL):
        raise HTTPException(status_code=500, detail="Server not configured.")

    async with httpx.AsyncClient() as client:
        # --- Load MAIN and PRIORITY bins ---
        try:
            main_bin_resp = await jsonbin_raw_get(client, MAIN_BIN_URL)
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

        # --- Normalize memory ---
        memory_dict = normalize_memory_record(main_record)
        memory_dict = {str(k): (v if isinstance(v, str) else str(v)) for k, v in memory_dict.items()}

        # --- Extract priority log ---
        priority_log_raw = ""
        if isinstance(priority_record, dict):
            if "PriorityLog" in priority_record:
                priority_log_raw = str(priority_record.get("PriorityLog") or "")
            elif "record" in priority_record and isinstance(priority_record["record"], dict):
                priority_log_raw = str(priority_record["record"].get("PriorityLog") or "")

        # --- Append incoming message ---
        incoming = req.message or ""
        author_key = str(req.authorID or "")
        user_mem = memory_dict.get(author_key, "").strip()

        if author_key == CREATOR_ID and incoming.lower().lstrip().startswith("mirilog!"):
            if priority_log_raw and not priority_log_raw.endswith("\n"):
                priority_log_raw += "\n"
            priority_log_raw += incoming.strip()
            logger.info("Appended mirilog! message to priority log.")
        else:
            # Add to author memory
            user_mem = (user_mem + "\n" + incoming).strip()
            memory_dict[author_key] = user_mem
            # Add to GLOBAL
            global_mem = memory_dict.get("GLOBAL", "")
            global_mem = (global_mem + "\n" + incoming).strip()
            memory_dict["GLOBAL"] = global_mem

        # --- Summarize per-user memory ---
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

        # --- Summarize GLOBAL memory ---
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

        # --- Persist MAIN and PRIORITY memory ---
        try:
            outgoing_main_payload = main_record.copy() if isinstance(main_record, dict) else {}
            outgoing_main_payload["memory"] = memory_dict
            await jsonbin_put_key(client, MAIN_BIN_URL, outgoing_main_payload)
        except Exception:
            logger.exception("Failed to write MAIN_BIN_URL.")

        try:
            outgoing_priority_payload = priority_record.copy() if isinstance(priority_record, dict) else {}
            outgoing_priority_payload["PriorityLog"] = priority_log_raw
            await jsonbin_put_key(client, PRIORITY_BIN_URL, outgoing_priority_payload)
        except Exception:
            logger.exception("Failed to write PRIORITY_BIN_URL.")

        # --- Extract concise creator instructions ---
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

        # --- Build final system prompt ---
        base_prompt = (
            "You are Miri — a witty, sarcastic, and helpful Discord bot. "
            "Give short replies (aim under 500 characters). Be playful but respectful. Ignore bot messages."
        )
        system_parts = [base_prompt]
        if priority_instructions:
            system_parts.append("Creator instructions: " + priority_instructions)
        global_reference = memory_dict.get("GLOBAL", "").strip()
        if global_reference:
            max_global_attach = 60_000
            if len(global_reference) > max_global_attach:
                global_reference = global_reference[:max_global_attach]
            system_parts.append("Server lore / reference:\n" + global_reference)
        final_system_prompt = "\n\n".join(system_parts)

        # --- Build user message with personal memory ---
        user_reference = memory_dict.get(author_key, "").strip()
        max_user_attach = 20_000
        if len(user_reference) > max_user_attach:
            user_reference = user_reference[-max_user_attach:]
        user_content_parts = []
        if user_reference:
            user_content_parts.append("User memory:\n" + user_reference)
        user_content_parts.append("Incoming message:\n" + incoming)
        final_user_message = "\n\n".join(user_content_parts)

        # --- Ensure combined prompt size within MAX_CONTEXT_CHARS ---
        combined_estimated_chars = len(final_system_prompt) + len(final_user_message)
        if combined_estimated_chars > MAX_CONTEXT_CHARS:
            overflow = combined_estimated_chars - MAX_CONTEXT_CHARS
            if user_reference and overflow > 0:
                trim_amount = min(len(user_reference), overflow + 1000)
                user_reference = user_reference[trim_amount:]
                user_content_parts[0] = "User memory:\n" + user_reference
                final_user_message = "\n\n".join(user_content_parts)

        # --- Final Groq call ---
        try:
            res = await groq_chat(client, final_system_prompt, final_user_message, max_tokens=400)
            assistant_text = res.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        except httpx.HTTPStatusError as e:
            logger.exception("Groq returned HTTP error")
            raise HTTPException(status_code=502, detail=f"Groq API error: {e.response.text}")
        except Exception as e:
            logger.exception("Groq call failed")
            raise HTTPException(status_code=502, detail=f"Groq call failed: {e}")

        # --- Return the assistant text ---
        return BotResponse(reply=assistant_text)
