from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Baseline with full in-thread history and no persistent user memory."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None if force_offline else self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent is None:
            return self._reply_offline(thread_id, message)
        state = self.sessions.setdefault(thread_id, SessionState())
        from langchain_core.messages import HumanMessage
        response = self.langchain_agent.invoke({"messages": [HumanMessage(content=message)]}, config={"configurable": {"thread_id": thread_id}})
        answer = response["messages"][-1].content
        state.prompt_tokens_processed += estimate_tokens("\n".join(m["content"] for m in state.messages) + message)
        state.messages.extend([{"role": "user", "content": message}, {"role": "assistant", "content": answer}])
        state.token_usage += estimate_tokens(answer)
        return {"response": answer, "token_usage": state.token_usage, "prompt_tokens_processed": state.prompt_tokens_processed}

    def token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        state = self.sessions.setdefault(thread_id, SessionState())
        prompt = "\n".join(item["content"] for item in state.messages) + "\n" + message
        state.prompt_tokens_processed += estimate_tokens(prompt)
        # Recall only explicit facts available in this thread.
        answer = "Mình đã ghi nhận thông tin bạn vừa chia sẻ."
        current = [m["content"] for m in state.messages if m["role"] == "user"] + [message]
        joined = "\n".join(current)
        if any(q in message.lower() for q in ("tên gì", "tên mình", "tên của mình")):
            import re
            match = re.search(r"(?:mình|tôi) tên là\s+([^,.!?]+)", joined, re.I)
            answer = f"Tên bạn là {match.group(1).strip()}." if match else "Trong thread này mình chưa biết tên bạn."
        elif any(q in message.lower() for q in ("đồ uống", "uống yêu thích")):
            import re
            match = re.search(r"(?:đồ uống yêu thích là|thích uống)\s+([^,.!?]+)", joined, re.I)
            answer = f"Đồ uống yêu thích của bạn là {match.group(1).strip()}." if match else "Trong thread này mình chưa có thông tin đó."
        elif any(q in message.lower() for q in ("ở đâu", "nơi ở")):
            import re
            matches = re.findall(r"(?:mình|tôi) (?:đang )?(?:ở|sống tại)\s+([^,.!?]+)", joined, re.I)
            answer = f"Bạn đang ở {matches[-1].strip()}." if matches else "Trong thread này mình chưa biết nơi ở của bạn."
        state.messages.extend([{"role": "user", "content": message}, {"role": "assistant", "content": answer}])
        state.token_usage += estimate_tokens(answer)
        return {"response": answer, "token_usage": state.token_usage, "prompt_tokens_processed": state.prompt_tokens_processed}

    def _maybe_build_langchain_agent(self):
        try:
            from langchain.agents import create_agent
            from langgraph.checkpoint.memory import InMemorySaver
            model = build_chat_model(self.config.model)
            return create_agent(model, checkpointer=InMemorySaver())
        except (ImportError, ValueError, TypeError):
            return None
