# models.py
from pydantic import BaseModel, Field
from typing import Optional

class BotRequest(BaseModel):
    username: Optional[str] = None
    authorID: str = Field(..., description="Discord user ID that sent the message")
    message: str = Field(..., description="Incoming message text")
    channelID: Optional[str] = None

class BotResponse(BaseModel):
    reply: str
