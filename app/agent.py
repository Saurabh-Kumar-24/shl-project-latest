from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx
from groq import Groq

from app.catalog import Assessment, Catalog
from app.prompts import SYSTEM_PROMPT
from app.retriever import HybridRetriever
from app.schemas import ChatResponse, Message, Recommendation

logger = logging.getLogger(__name__)

MODEL = "llama-3.3-70b-versatile"
MAX_TURNS = 8
LLM_TIMEOUT = 25


class ConversationAgent:
    def __init__(self, catalog: Catalog, retriever: HybridRetriever) -> None:
        self.catalog = catalog
        self.retriever = retriever
        ssl_cert = os.environ.get("GROQ_CA_BUNDLE", "")
        http_client = httpx.Client(verify=ssl_cert) if ssl_cert and os.path.exists(ssl_cert) else None
        self._client = Groq(
            api_key=os.environ.get("GROQ_API_KEY", ""),
            timeout=LLM_TIMEOUT,
            http_client=http_client,
        )

    def handle_conversation(self, messages: list[Message]) -> ChatResponse:
        turn_count = len(messages)
        force_end = turn_count >= MAX_TURNS

        catalog_context = self._build_catalog_context(messages)

        system_content = SYSTEM_PROMPT.format(catalog_context=catalog_context)

        llm_messages: list[dict[str, str]] = [{"role": "system", "content": system_content}]
        for m in messages:
            llm_messages.append({"role": m.role, "content": m.content})

        if force_end:
            llm_messages.append({
                "role": "system",
                "content": (
                    "IMPORTANT: This is the last turn. You MUST set intent to 'recommend' "
                    "and provide final recommendations if possible. The conversation ends after this."
                ),
            })

        raw = self._call_llm(llm_messages)
        parsed = self._parse_response(raw)

        intent = parsed.get("intent", "clarify")
        reply = parsed.get("reply", "I can help you find the right SHL assessments.")
        search_query = parsed.get("search_query")
        filters = parsed.get("filters") or {}
        selected_ids = parsed.get("selected_ids")

        recommendations: list[Recommendation] = []

        if intent in ("recommend", "refine"):
            recommendations = self._build_recommendations(
                search_query=search_query,
                filters=filters,
                selected_ids=selected_ids,
                messages=messages,
            )

        if force_end and not recommendations:
            last_user_msg = ""
            for m in reversed(messages):
                if m.role == "user":
                    last_user_msg = m.content
                    break
            if last_user_msg:
                candidates = self.retriever.search(last_user_msg)
                recommendations = [self._to_recommendation(a) for a in candidates[:5]]

        return ChatResponse(
            reply=reply,
            recommendations=recommendations,
            end_of_conversation=force_end or intent == "refuse" and not recommendations,
        )

    def _build_catalog_context(self, messages: list[Message]) -> str:
        all_user_text = " ".join(m.content for m in messages if m.role == "user")
        if not all_user_text.strip():
            return self._full_catalog_summary()

        candidates = self.retriever.search(all_user_text, top_k=30)
        if not candidates:
            return self._full_catalog_summary()

        lines: list[str] = []
        for a in candidates:
            lines.append(
                f"[ID:{a.entity_id}] {a.name} | Types: {a.type_codes} | "
                f"Keys: {', '.join(a.keys)} | Levels: {', '.join(a.job_levels)} | "
                f"Duration: {a.duration or 'N/A'} | Remote: {a.remote} | "
                f"URL: {a.link}\n  {a.description[:200]}"
            )
        return "\n".join(lines)

    def _full_catalog_summary(self) -> str:
        lines: list[str] = []
        for a in self.catalog.assessments[:100]:
            lines.append(
                f"[ID:{a.entity_id}] {a.name} | Types: {a.type_codes} | "
                f"Levels: {', '.join(a.job_levels[:3])} | Duration: {a.duration or 'N/A'} | "
                f"URL: {a.link}"
            )
        return "\n".join(lines)

    def _call_llm(self, messages: list[dict[str, str]]) -> str:
        try:
            response = self._client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.3,
                max_tokens=2000,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            logger.exception("LLM call failed")
            error_str = str(exc).lower()
            if "rate_limit" in error_str or "429" in error_str:
                reply = (
                    "The service is temporarily at capacity. "
                    "Please wait a minute and try again."
                )
            elif "timeout" in error_str:
                reply = (
                    "The request timed out. Please try again."
                )
            else:
                reply = (
                    "I'm having trouble processing your request. "
                    "Could you rephrase?"
                )
            return json.dumps({
                "intent": "clarify",
                "reply": reply,
                "search_query": None,
                "filters": {},
                "selected_ids": None,
            })

    def _parse_response(self, raw: str) -> dict[str, Any]:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1:
                try:
                    return json.loads(cleaned[start : end + 1])
                except json.JSONDecodeError:
                    pass
            logger.warning("Failed to parse LLM response: %s", raw[:200])
            return {
                "intent": "clarify",
                "reply": raw if raw else "Could you tell me more about what you're looking for?",
                "search_query": None,
                "filters": {},
                "selected_ids": None,
            }

    def _build_recommendations(
        self,
        search_query: str | None,
        filters: dict[str, Any],
        selected_ids: list[str] | None,
        messages: list[Message],
    ) -> list[Recommendation]:
        recommendations: list[Recommendation] = []
        seen: set[str] = set()

        if selected_ids:
            for eid in selected_ids:
                a = self.catalog.by_id.get(str(eid))
                if a and a.entity_id not in seen:
                    seen.add(a.entity_id)
                    recommendations.append(self._to_recommendation(a))

        if len(recommendations) < 10 and search_query:
            candidates = self.retriever.search(
                query=search_query,
                job_level=filters.get("job_level"),
                test_type=filters.get("test_type"),
                max_duration=filters.get("max_duration"),
                remote_only=filters.get("remote_only", False),
            )
            for a in candidates:
                if a.entity_id not in seen:
                    seen.add(a.entity_id)
                    recommendations.append(self._to_recommendation(a))
                if len(recommendations) >= 10:
                    break

        if not recommendations:
            all_user_text = " ".join(m.content for m in messages if m.role == "user")
            if all_user_text.strip():
                candidates = self.retriever.search(all_user_text)
                for a in candidates:
                    if a.entity_id not in seen:
                        seen.add(a.entity_id)
                        recommendations.append(self._to_recommendation(a))
                    if len(recommendations) >= 10:
                        break

        validated = [r for r in recommendations if r.url in self.catalog.by_url]
        return validated[:10]

    def _to_recommendation(self, a: Assessment) -> Recommendation:
        return Recommendation(
            name=a.name,
            url=a.link,
            test_type=a.type_codes,
        )
