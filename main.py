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
    username: Optional[str] = None
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

# -------------------- Groq Helper --------------------
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

        # --- Load MAIN BIN ---
        try:
            main_bin_resp = await jsonbin_raw_get(client, MAIN_BIN_URL)
            main_record = main_bin_resp.get("record", main_bin_resp) if isinstance(main_bin_resp, dict) else {}
        except Exception as e:
            logger.exception("Failed reading MAIN_BIN_URL")
            raise HTTPException(status_code=502, detail=f"Error fetching MAIN_BIN_URL: {e}")

        # --- Load PRIORITY BIN ---
        try:
            priority_bin_resp = await jsonbin_raw_get(client, PRIORITY_BIN_URL)
            priority_record = priority_bin_resp.get("record", priority_bin_resp) if isinstance(priority_bin_resp, dict) else {}
        except Exception as e:
            logger.exception("Failed reading PRIORITY_BIN_URL")
            raise HTTPException(status_code=502, detail=f"Error fetching PRIORITY_BIN_URL: {e}")

        # --- Normalize memory ---
        memory_dict = normalize_memory_record(main_record)
        memory_dict = {str(k): (v if isinstance(v, str) else str(v)) for k, v in memory_dict.items()}

        # Extract priority log
        priority_log_raw = ""
        if isinstance(priority_record, dict):
            priority_log_raw = str(priority_record.get("PriorityLog", ""))

        incoming = req.message or ""
        author_key = req.authorID
        user_mem = memory_dict.get(author_key, "").strip()

        # -------------------- Username tracking --------------------
        username = (req.username or "").strip()
        username_key = f"{author_key}_username"
        last_username = memory_dict.get(username_key, "")

        if username and username != last_username:
            memory_dict[username_key] = username
            logger.info(f"Updated username for {author_key}: {username}")

        # -------------------- Logging / memory saving --------------------
        if author_key == CREATOR_ID and incoming.lower().startswith("mirilog!"):
            if priority_log_raw and not priority_log_raw.endswith("\n"):
                priority_log_raw += "\n"
            priority_log_raw += incoming
        else:
            user_mem = (user_mem + "\n" + incoming).strip()
            memory_dict[author_key] = user_mem
            global_mem = memory_dict.get("GLOBAL", "")
            memory_dict["GLOBAL"] = (global_mem + "\n" + incoming).strip()

        # --- Summaries ---
        try:
            current_user_mem = memory_dict.get(author_key, "")
            if len(current_user_mem) > USER_SUMMARY_THRESHOLD:
                sys_prompt = (
                    "Summarize this user's memory. Keep important details, shorten rambling. "
                    "Return only the summary under 3000 characters."
                )
                res = await groq_chat(client, sys_prompt, current_user_mem, max_tokens=300)
                summary_text = res.get("choices", [{}])[0].get("message", {}).get("content", "")
                if summary_text:
                    memory_dict[author_key] = summary_text.strip()
        except Exception:
            logger.exception("User summary failed.")

        try:
            global_mem = memory_dict.get("GLOBAL", "")
            if len(global_mem) > GLOBAL_SUMMARY_THRESHOLD:
                sys_prompt = (
                    "Summarize server lore + persistent facts. "
                    "Remove chatter. Under 8000 characters."
                )
                res = await groq_chat(client, sys_prompt, global_mem, max_tokens=600)
                summary_text = res.get("choices", [{}])[0].get("message", {}).get("content", "")
                if summary_text:
                    memory_dict["GLOBAL"] = summary_text.strip()
        except Exception:
            logger.exception("Global summary failed.")

        # --- Save updated bins ---
        try:
            main_payload = main_record.copy()
            main_payload["memory"] = memory_dict
            await jsonbin_put_key(client, MAIN_BIN_URL, main_payload)
        except Exception:
            logger.exception("Saving MAIN_BIN failed.")

        try:
            priority_payload = priority_record.copy()
            priority_payload["PriorityLog"] = priority_log_raw
            await jsonbin_put_key(client, PRIORITY_BIN_URL, priority_payload)
        except Exception:
            logger.exception("Saving PRIORITY_BIN failed.")

        # --- Extract creator instructions ---
        priority_instructions = ""
        if priority_log_raw.strip():
            try:
                sys_prompt = (
                    "Extract concise creator system rules. Under 500 characters. No commentary."
                )
                res = await groq_chat(client, sys_prompt, priority_log_raw, max_tokens=120)
                priority_instructions = res.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            except Exception:
                logger.exception("Instruction extraction failed.")

        # -------------------- Build final system prompt --------------------
        base_prompt = (
            "You are Miri — a witty, sarcastic Discord bot. "
            "Short replies under 500 characters. Ignore bot messages."
        )

        system_parts = [base_prompt]

        if priority_instructions:
            system_parts.append("Creator instructions: " + priority_instructions)

        global_ref = memory_dict.get("GLOBAL", "").strip()
        if global_ref:
            if len(global_ref) > 60000:
                global_ref = global_ref[:60000]
            system_parts.append("Server lore:\n" + global_ref)

        final_system_prompt = "\n\n".join(system_parts)

        # -------------------- Build user message --------------------
        user_reference = memory_dict.get(author_key, "").strip()

        username_value = memory_dict.get(username_key, "")
        if username_value:
            user_reference = f"Username: {username_value}\n\n{user_reference}"

        if len(user_reference) > 20000:
            user_reference = user_reference[-20000:]

        final_user_message = (
            f"User memory:\n{user_reference}\n\nIncoming message:\n{incoming}"
        )

        # -------------------- Groq call --------------------
        try:
            res = await groq_chat(client, final_system_prompt, final_user_message, max_tokens=400)
            assistant_text = res.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        except Exception as e:
            logger.exception("Groq error")
            raise HTTPException(status_code=502, detail=f"Groq API error: {e}")

        return BotResponse(reply=assistant_text)
