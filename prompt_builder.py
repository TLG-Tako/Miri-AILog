# prompt_builder.py
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger("prompt_builder")

class PromptBuilder:
    def __init__(self, memory_manager, priority_manager, groq_client):
        self.memory_manager = memory_manager
        self.priority_manager = priority_manager
        self.groq_client = groq_client

    async def build_system_prompt(self) -> str:
        """
        Build the system prompt using:
         - base persona
         - condensed creator instructions (using Groq if necessary)
         - raw priority items as extra context
         - global memory (trimmed)
        """
        base_prompt = (
            "You are Miri — a witty, sarcastic, and helpful Discord bot. "
            "Give short replies. Be playful but respectful."
        )
        system_parts = [base_prompt]

        # condense instruction items via groq if any
        priority_items = self.priority_manager.get_parsed_priority_items()
        instruction_texts = [it["text"] for it in priority_items if it.get("type") == "instruction" and it.get("text")]
        if instruction_texts and self.groq_client and self.groq_client.is_configured:
            joined = "\n".join(instruction_texts)
            try:
                res = await self.groq_client.chat(
                    "Extract the core instructions from the creator's messages. Keep concise under 300 characters. No commentary.",
                    joined,
                    max_tokens=120
                )
                condensed = res.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                if condensed:
                    system_parts.append("Creator instructions:\n" + condensed)
            except Exception:
                logger.exception("Groq condensation failed; falling back to raw instructions")
                system_parts.append("Creator instructions:\n" + "\n".join(instruction_texts[:10]))

        # add raw priority items as context
        if priority_items:
            lore_lines = []
            for it in priority_items:
                t = it.get("type","").strip()
                txt = it.get("text","").strip()
                if t and txt:
                    lore_lines.append(f"[{t}] {txt}")
            if lore_lines:
                system_parts.append("Server priority:\n" + "\n".join(lore_lines))

        # attach global memory
        main_record_mem = (self.memory_manager.get_main_record_for_storage()).get("memory", {}) if hasattr(self.memory_manager, "get_main_record_for_storage") else {}
        global_mem = main_record_mem.get("GLOBAL","")
        if global_mem:
            if len(global_mem) > 60000:
                global_mem = global_mem[-60000:]
            system_parts.append("Server lore:\n" + global_mem)

        return "\n\n".join(system_parts)

    def build_user_prompt(self, author_key: str, username: Optional[str], incoming: str) -> str:
        """
        Build the user prompt which includes user memory + incoming message.
        """
        memory_dict = (self.memory_manager.get_main_record_for_storage()).get("memory", {}) if hasattr(self.memory_manager, "get_main_record_for_storage") else {}
        user_ref = memory_dict.get(author_key, "")
        if len(user_ref) > 20000:
            user_ref = user_ref[-20000:]
        if username:
            user_ref = f"Username: {username}\n\n" + user_ref
        return f"User memory:\n{user_ref}\n\nIncoming message:\n{incoming}"

    def extract_response_text(self, groq_response: Dict[str, Any]) -> str:
        choices = groq_response.get("choices") or []
        assistant_text = ""
        if choices:
            assistant_text = choices[0].get("message", {}).get("content", "") or ""
        return assistant_text.strip()
