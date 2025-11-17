# main.py
import os
import asyncio
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx

app = FastAPI(title="Miri Relay - Groq + JSONBin")
@app.get("/")
async def root():
    return {"status": "alive", "message": "Miri Relay API is running."}

# Config from env
GROQ_KEY = os.getenv("GROQ_KEY", "")
GROQ_ENDPOINT = os.getenv("GROQ_ENDPOINT", "https://api.groq.com/openai/v1/chat/completions")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

JSONBIN_MASTER_KEY = os.getenv("JSONBIN_MASTER_KEY", "")
MAIN_BIN_URL = os.getenv("MAIN_BIN_URL", "")        # e.g. https://api.jsonbin.io/v3/b/69195a32ae596e708f5c1946
PRIORITY_BIN_URL = os.getenv("PRIORITY_BIN_URL", "")# e.g. https://api.jsonbin.io/v3/b/691a417943b1c97be9b1b422

CREATOR_ID = os.getenv("CREATOR_ID", "530829310559256596")
MAIN_MEMORY_THRESHOLD = int(os.getenv("MAIN_MEMORY_THRESHOLD", "1000"))

# Basic checks
if not (GROQ_KEY and JSONBIN_MASTER_KEY and MAIN_BIN_URL and PRIORITY_BIN_URL):
    # don't crash on import in dev; raise on request instead
    pass

# Models
class BotRequest(BaseModel):
    authorID: str
    message: str
    channelID: Optional[str] = None

class BotResponse(BaseModel):
    reply: str

# Helpers
async def jsonbin_get(client: httpx.AsyncClient, url: str) -> str:
    headers = {"X-Master-Key": JSONBIN_MASTER_KEY}
    r = await client.get(url, headers=headers, timeout=30.0)
    r.raise_for_status()
    j = r.json()
    # Try to find memory key either "record" or direct
    # If bin used earlier, memory stored at record.memory or record.PriorityLog
    record = j.get("record", {})
    # try several keys
    if "memory" in record:
        return record.get("memory", "") or ""
    if "PriorityLog" in record:
        return record.get("PriorityLog", "") or ""
    # fallback: return empty string
    return ""

async def jsonbin_put(client: httpx.AsyncClient, url: str, payload_key: str, content: str):
    headers = {"X-Master-Key": JSONBIN_MASTER_KEY, "Content-Type": "application/json"}
    body = {payload_key: content}
    r = await client.put(url, headers=headers, json=body, timeout=30.0)
    r.raise_for_status()
    return r.json()

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
                sys_prompt = "You are a helpful summarizer. Return ONLY a concise summary of the memory. Remove duplicates, sensitive info, and rambling. Keep it under 800 characters."
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
        base_prompt = "You are Miri, a witty, sarcastic, helpful Discord bot. You are your own person with your own personality and opinions. Short to moderate length replies. Slightly mocking. Ignore bots. Keep replies under 500 characters."

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
