from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["user", "assistant"] = Field(
        ..., description="Who sent this message"
    )
    content: str = Field(..., min_length=1, description="Message text")


class ChatRequest(BaseModel):
    messages: list[Message] = Field(
        ..., min_length=1, description="Conversation history"
    )


class Recommendation(BaseModel):
    name: str = Field(..., description="Assessment name")
    url: str = Field(..., description="SHL product catalog URL")
    test_type: str = Field(
        ...,
        description=(
            "Abbreviated type codes: K=Knowledge & Skills, "
            "P=Personality & Behavior, A=Ability & Aptitude, "
            "S=Simulations, B=Biodata & Situational Judgment, "
            "C=Competencies, D=Development & 360"
        ),
    )


class ChatResponse(BaseModel):
    reply: str = Field(..., description="Agent's text reply")
    recommendations: list[Recommendation] = Field(
        default_factory=list,
        description="1-10 recommended assessments (empty when clarifying/refusing)",
    )
    end_of_conversation: bool = Field(
        default=False,
        description="True when conversation is complete or turn cap reached",
    )
