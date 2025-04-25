from datetime import datetime
from typing import Literal

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


class StateManagerError(Exception):
    """Base exception for state manager errors"""
    pass


@dataclass
class Message:
    role: Literal["user", "assistant", "system"]
    content: str
    sources: List[Dict[str, Any]]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class AppState:
    messages: List[Message] = field(default_factory=list)
    rag_engine: Optional[Any] = None
    
    def add_message(self, role: str, content: str, sources: List[Dict] = None) -> None:
        if not content.strip():
            raise StateManagerError("Message content cannot be empty")
        if role not in ["user", "assistant", "system"]:
            raise StateManagerError(f"Invalid role: {role}")
        message = Message(role=role, content=content, sources=sources or [])
        self.messages.append(message)