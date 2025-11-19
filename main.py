import os
import json
import logging
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("miri-relay")

app = FastAPI(title="Miri Relay - Groq + JSONBin")

# ---------------------------
# JSONBin Setup
# ---------------------------
JSONBIN_URL = os.getenv("JSONBIN_URL")
JSONBIN_KEY = os.getenv("JSONBIN_KEY")
GROQ_KEY = os.getenv("GROQ_KEY")

headers_bin = {
    "Content-Type": "application/json",
    "X-Master-Key": JSONBIN_KEY
}

# ---------------------------
# Request Model
# ---------------------------
class BotRequest(BaseModel):
    username: Optional[str] = None
    authorID: str
    message: str
    channelID: Optional[str] = None


# ---------------------------
# Utility: Escape unsafe message content
# ---------------------------
def escape_message(text: str) -> str:
    """
    Escapes user input safely so JSON never breaks.
    Removes the surrounding quotes json.dumps() adds.
    """
    return json.dumps(text)[1:-1]


# ---------------------------
# JSONBin Load & Save Helpers
# ---------------------------
async def load_bin():
    async with httpx.AsyncClient() as client:
        res = await client.get(JSONBIN_URL, headers=headers_bin)
        return res.json()

async def save_bin(data):
    async with httpx.AsyncClient() as client:
        await client.put(JSONBIN_URL, headers=headers_bin, json=data)

# ---------------------------
# Root
# ---------------------------
@app.get("/")
async def root():
    return {"status": "running"}

# ---------------------------
# Main Chat Endpoint
# ---------------------------
@app.post("/relay")
async def relay(req: BotRequest):
    try:
        # Load memory storage
        memory_data = await load_bin()

        memory_dict: Dict[str, Any] = memory_data.get("Memory", {})
        priority_log = memory_data.get("PriorityLog", "[]")

        # Convert PriorityLog safely
        try:
            priority_items = json.loads(priority_log)
        except:
            priority_items = []

        # ---------------------------
        # Username tracking
        # ---------------------------
        user_key = str(req.authorID)

        if req.username:
            safe_user = escape_message(req.username.strip())
            stored_username = memory_dict.get(f"{user_key}_username", "")

            if stored_username != safe_user:
                memory_dict[f"{user_key}_username"] = safe_user
                logger.info(f"Updated username for {user_key}: {safe_user}")

        # ---------------------------
        # Escape user message safely
        # ---------------------------
        safe_msg = escape_message(req.message)

        # Fetch user memory block
        user_memory = memory_dict.get(user_key, "")

        # Fetch username for injection
        user_name = memory_dict.get(f"{user_key}_username", "")

        # ---------------------------
        # Build System Prompt
        # ---------------------------
        system_prompt = "These are persistent instructions and world data:\n"

        for item in priority_items:
            if isinstance(item, dict) and "type" in item and "text" in item:
                system_prompt += f"- [{item['type']}] {item['text']}\n"

        # Inject user memory
        if user_name:
            system_prompt += f"\nCurrent Username: {user_name}\n"

        if user_memory.strip():
            system_prompt += f"\nUser Memory:\n{user_memory.strip()}\n"

        # ---------------------------
        # Build Request for Groq
        # ---------------------------
        payload = {
            "model": "mixtral-8x7b-32768",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": safe_msg}
            ],
            "temperature": 0.5,
            "max_tokens": 300,
        }

        # ---------------------------
        # Call Groq API
        # ---------------------------
        async with httpx.AsyncClient() as client:
            r = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_KEY}",
                         "Content-Type": "application/json"},
                json=payload
            )

        result = r.json()

        if "choices" not in result:
            raise HTTPException(status_code=500, detail="Groq API returned invalid response.")

        reply = result["choices"][0]["message"]["content"]

        # ---------------------------
        # Save memory dictionary if changed
        # ---------------------------
        memory_data["Memory"] = memory_dict
        await save_bin(memory_data)

        return {"response": reply}

    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
