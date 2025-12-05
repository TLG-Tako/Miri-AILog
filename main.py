# main.py
from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import JSONResponse
import logging
import os
from typing import Dict, Any

from models import BotRequest, BotResponse
from jsonbin_client import JsonBinClient
from groq_client import GroqClient
from memory_manager import MemoryManager
from priority_manager import PriorityManager
from prompt_builder import PromptBuilder

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger("miri-relay-enterprise")

# create app
app = FastAPI(title="Miri Relay — Enterprise (Option 3)")

# instantiate shared clients at startup (DI-friendly)
@app.on_event("startup")
async def startup_event():
    logger.info("Starting up — creating service clients")
    # clients created lazily via singletons in their modules or created here
    # nothing to do because clients are created by their factories when requested


# Dependency providers
async def get_jsonbin_client() -> JsonBinClient:
    client = JsonBinClient()
    return client

async def get_groq_client() -> GroqClient:
    client = GroqClient()
    return client

async def get_memory_manager(jsonbin: JsonBinClient = Depends(get_jsonbin_client)) -> MemoryManager:
    return MemoryManager(jsonbin=jsonbin)

async def get_priority_manager(jsonbin: JsonBinClient = Depends(get_jsonbin_client)) -> PriorityManager:
    return PriorityManager(jsonbin=jsonbin)

async def get_prompt_builder(
    memory: MemoryManager = Depends(get_memory_manager),
    priority: PriorityManager = Depends(get_priority_manager),
    groq: GroqClient = Depends(get_groq_client),
) -> PromptBuilder:
    return PromptBuilder(memory_manager=memory, priority_manager=priority, groq_client=groq)


@app.post("/api/message", response_model=BotResponse)
async def handle_message(
    payload: Dict[str, Any],
    jsonbin: JsonBinClient = Depends(get_jsonbin_client),
    groq: GroqClient = Depends(get_groq_client),
    memory_mgr: MemoryManager = Depends(get_memory_manager),
    priority_mgr: PriorityManager = Depends(get_priority_manager),
    prompt_builder: PromptBuilder = Depends(get_prompt_builder),
):
    """
    Main message endpoint.
    Accepts either normal BotRequest payloads or a PriorityLog instruction payload.
    """
    # Quick config guard
    if not jsonbin.is_configured or not groq.is_configured:
        logger.error("Required environment variables are missing")
        raise HTTPException(status_code=500, detail="Server not configured.")

    # If this is a priority push, delegate
    if "PriorityLog" in payload:
        try:
            await priority_mgr.append_priority_entries(payload["PriorityLog"])
            return BotResponse(reply="")
        except Exception as exc:
            logger.exception("Failed to append priority entries")
            raise HTTPException(status_code=502, detail=str(exc))

    # Validate input
    try:
        req = BotRequest(**payload)
    except Exception as exc:
        logger.exception("Invalid request format")
        raise HTTPException(status_code=400, detail="Invalid request format")

    # Standard message flow
    try:
        # load memory + priority (may be async)
        main_record = await jsonbin.get_main_record()
        priority_record = await jsonbin.get_priority_record()

        # normalize memory and update with incoming message
        memory_mgr.load_main_record(main_record)
        priority_mgr.load_priority_record(priority_record)

        # update in-memory memory with incoming
        memory_mgr.append_user_message(str(req.authorID), req.username or "", req.message)

        # build prompts (may call groq for instruction condense)
        system_prompt = await prompt_builder.build_system_prompt()
        user_prompt = prompt_builder.build_user_prompt(str(req.authorID), req.username, req.message)

        # ask model
        groq_res = await groq.chat(system_prompt, user_prompt)
        assistant_text = prompt_builder.extract_response_text(groq_res)

        # save memory back (best-effort)
        try:
            await jsonbin.put_main_record(memory_mgr.get_main_record_for_storage())
        except Exception:
            logger.exception("Failed to save main record (non-fatal)")

        return BotResponse(reply=assistant_text or "Miri had a brain-freeze, try again.")
    except Exception as exc:
        logger.exception("Unhandled error during message handling")
        raise HTTPException(status_code=500, detail="Internal server error")
