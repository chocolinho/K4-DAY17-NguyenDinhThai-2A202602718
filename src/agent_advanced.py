from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import re

from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent with short-term thread memory, persistent profile, and compaction."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(self.config.compact_threshold_tokens, self.config.compact_keep_messages)
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.langchain_agent = None if force_offline else self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent is not None:
            from langchain_core.messages import HumanMessage, SystemMessage
            facts = self.profile_store.read_text(user_id)
            context = self.compact_memory.context(thread_id)
            prior = "\n".join(m["content"] for m in context["messages"])
            result = self.langchain_agent.invoke(
                [SystemMessage(content=f"User profile memory:\n{facts}\nThread summary:\n{context['summary']}\nRecent thread:\n{prior}"), HumanMessage(content=message)]
            )
            answer = result.content if isinstance(result.content, str) else str(result.content)
            for key, value in extract_profile_updates(message).items():
                self.profile_store.upsert_fact(user_id, key, value)
            self.compact_memory.append(thread_id, "user", message)
            self.compact_memory.append(thread_id, "assistant", answer)
            self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + estimate_tokens(answer)
            self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + self._estimate_prompt_context_tokens(user_id, thread_id)
            return {"response": answer, "token_usage": self.thread_tokens[thread_id], "prompt_tokens_processed": self.thread_prompt_tokens[thread_id]}
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        for key, value in extract_profile_updates(message).items():
            self.profile_store.upsert_fact(user_id, key, value)
        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        answer = self._offline_response(user_id, thread_id, message)
        self.compact_memory.append(thread_id, "assistant", answer)
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + estimate_tokens(answer)
        return {"response": answer, "token_usage": self.thread_tokens[thread_id], "prompt_tokens_processed": self.thread_prompt_tokens[thread_id]}

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        context = self.compact_memory.context(thread_id)
        text = self.profile_store.read_text(user_id) + "\n" + str(context["summary"])
        text += "\n" + "\n".join(m["content"] for m in context["messages"])
        return estimate_tokens(text)

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        facts = self.profile_store.facts(user_id)
        low = message.lower()
        if any(x in low for x in ("tên gì", "tên mình", "tên của mình")):
            values = [facts[k] for k in ("name", "favorite_drink", "favorite_food", "interests") if k in facts]
            return "Mình nhớ: " + "; ".join(values) + "." if values else "Trong hồ sơ chưa có tên của bạn."
        if "nghề" in low or "công việc hiện tại" in low:
            values = [facts[k] for k in ("profession", "location") if k in facts]
            return "Hiện tại: " + "; ".join(values) + "." if values else "Mình chưa có thông tin nghề nghiệp."
        if "đồ uống" in low or "món ăn" in low or "nuôi con gì" in low:
            values = [facts[k] for k in ("favorite_drink", "favorite_food", "pet") if k in facts]
            return "Mình nhớ: " + "; ".join(values) + "." if values else "Mình chưa có thông tin sở thích đó."
        if any(x in low for x in ("tên gì", "tên mình", "tên của mình")):
            return f"Tên bạn là {facts.get('name', 'mình chưa được biết')}."
        if any(x in low for x in ("nghề", "làm nghề", "công việc hiện tại")):
            return f"Hiện tại bạn làm {facts.get('profession', 'mình chưa có thông tin nghề nghiệp')}."
        if any(x in low for x in ("ở đâu", "nơi ở", "đang ở đâu")):
            return f"Nơi ở hiện tại của bạn là {facts.get('location', 'mình chưa có thông tin')}."
        if "style" in low or "kiểu trả lời" in low or "trả lời mình thích" in low:
            return f"Bạn thích câu trả lời {facts.get('response_style', 'ngắn gọn, rõ ý')}."
        # Recall composite requests from any known profile facts.
        if any(x in low for x in ("nhắc lại", "sang thread mới", "nhớ lại", "mình là ai")):
            chosen = [facts[k] for k in ("name", "profession", "location", "response_style", "favorite_drink", "favorite_food", "interests", "pet") if k in facts]
            return "Mình nhớ: " + "; ".join(chosen) + "." if chosen else "Mình chưa có thông tin hồ sơ để nhắc lại."
        # For content-specific follow-up, surface the compact summary when available.
        summary = str(self.compact_memory.context(thread_id)["summary"])
        if summary and any(x in low for x in ("tin", "stress", "ý chính", "trade-off", "news")):
            snippets = re.sub(r"\s+", " ", summary).strip()
            return "Tóm tắt ngữ cảnh đã giữ: " + snippets[-500:]
        return "Mình đã ghi nhận thông tin bạn vừa chia sẻ."

    def _maybe_build_langchain_agent(self):
        # Provider wiring is optional; use a live model when dependencies are installed.
        try:
            return build_chat_model(self.config.model)
        except Exception:
            return None
