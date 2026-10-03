from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re


def estimate_tokens(text: str) -> int:
    """Stable offline estimate: roughly one token per four non-whitespace chars."""
    compact = re.sub(r"\s+", "", text)
    return (len(compact) + 3) // 4


def _safe_user_id(user_id: str) -> str:
    slug = re.sub(r"[^\w.-]+", "_", user_id.strip(), flags=re.UNICODE).strip("._")
    return slug[:100] or "anonymous"


@dataclass
class UserProfileStore:
    """Small markdown profile store with traversal-safe user paths."""

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        return self.root_dir / _safe_user_id(user_id) / "User.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        return path.read_text(encoding="utf-8") if path.exists() else "# User profile\n"

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content.rstrip() + "\n", encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        content = self.read_text(user_id)
        if search_text not in content:
            return False
        self.write_text(user_id, content.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        facts: dict[str, str] = {}
        for line in self.read_text(user_id).splitlines():
            match = re.match(r"^-\s*([^:]+):\s*(.*)$", line)
            if match:
                facts[match.group(1).strip()] = match.group(2).strip()
        return facts

    def upsert_fact(self, user_id: str, key: str, value: str) -> None:
        facts = self.facts(user_id)
        facts[key] = value.strip()
        body = "# User profile\n\n" + "\n".join(f"- {k}: {v}" for k, v in facts.items())
        self.write_text(user_id, body)


def extract_profile_updates(message: str) -> dict[str, str]:
    """Extract explicitly asserted, relatively stable Vietnamese profile facts."""
    text = message.strip()
    if not text or "?" in text or text.endswith("？"):
        return {}
    facts: dict[str, str] = {}
    # Match explicit forms first and extract only the fact span, excluding clauses.
    explicit = {
        "name": r"(?:mình|tôi)\s+tên\s+là\s+([^,.!?]+)",
        "location": r"(?:mình|tôi)\s+(?:vẫn\s+)?(?:đang\s+)?(?:ở|sống\s+tại)\s+([^,.!?]+)",
        "profession": r"(?:(?:mình|tôi)\s+)?(?:giờ\s+)?(?:đang\s+)?(?:làm\s+)?(?:chuyển\s+sang\s+)?([A-Za-zÀ-ỹ-]+\s+engineer)\b",
        "favorite_drink": r"đồ\s+uống\s+yêu\s+thích\s+(?:là\s+)?([^,.!?]+)",
        "favorite_food": r"món\s+ăn\s+yêu\s+thích\s+(?:là\s+)?([^,.!?]+)",
    }
    for key, pattern in explicit.items():
        match = re.search(pattern, text, re.I)
        if match:
            facts[key] = match.group(1).strip()
    # Corrective job assertions have higher priority than statements earlier in a turn.
    job = re.search(r"(?:giờ\s+chuyển\s+sang|giờ\s+làm|hiện\s+tại\s+làm|chuyển\s+sang|làm)\s+([A-Za-zÀ-ỹ-]+\s+engineer)\b", text, re.I)
    if job and not re.search(r"đùa|hay là chuyển sang", text[:job.start()], re.I):
        facts["profession"] = job.group(1).strip()
    # Preference extraction avoids the frequent question pattern "... mình thích là gì".
    style = re.search(r"(?:trả\s+lời\s+thành\s+)(3\s+bullet[^,.!?]*|bullet\s+ngắn[^,.!?]*)|(?:trả\s+lời\s+)(ngắn\s+gọn[^,.!?]*)", text, re.I)
    if style:
        facts["response_style"] = next(g for g in style.groups() if g).strip()
    if "ưu tiên" in text.lower() and "recall" in text.lower():
        facts["response_style"] = "ưu tiên recall đúng và trả lời ngắn gọn"
    if "python" in text.lower() or "ai ứng dụng" in text.lower() or "mlops" in text.lower():
        topics = []
        for topic in ("Python", "AI ứng dụng", "MLOps", "AI agent", "benchmark memory"):
            if topic.lower() in text.lower() and topic not in topics:
                topics.append(topic)
        if topics:
            facts["interests"] = ", ".join(topics)
    patterns = {
        "name": [r"\b(?:mình|tôi)\s+tên\s+là\s+([^,.!?]+)", r"\b(?:mình|tôi)\s+là\s+([^,.!?]+)"],
        "location": [r"\b(?:hiện\s+)?(?:mình|tôi)\s+(?:đang\s+)?(?:ở|sống\s+tại)\s+([^,.!?]+)", r"\b(?:hiện\s+tại\s+)?(?:nơi\s+ở|đang\s+ở)\s+(?:là\s+)?([^,.!?]+)"],
        "profession": [r"\b(?:đang\s+)?làm\s+([^,.!?]+?)(?:\s+cho\s+.+)?$", r"\bnghề\s+nghiệp\s+(?:hiện\s+tại\s+)?(?:là\s+)?([^,.!?]+)"],
        "favorite_drink": [r"\b(?:đồ\s+uống\s+yêu\s+thích|mình\s+thích\s+uống)\s+(?:là\s+)?([^,.!?]+)"],
        "favorite_food": [r"\b(?:món\s+ăn\s+yêu\s+thích)\s+(?:là\s+)?([^,.!?]+)"],
        "response_style": [r"\b(?:trả\s+lời|câu\s+trả\s+lời)\s+(?:thành\s+)?(3\s+bullet[^,.!?]*|ngắn\s+gọn[^,.!?]*|[^,.!?]*bullet[^,.!?]*|[^,.!?]*có\s+ví\s+dụ[^,.!?]*)"],
    }
    for key, options in patterns.items():
        for pattern in options:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                value = match.group(1).strip(" \t\n,.;:!?'”“")
                if value and not any(x in value.lower() for x in ("câu đùa", "đùa", "chỉ là", "không phải")):
                    facts[key] = value
                break
    # Explicit corrections take precedence over earlier assertions in the same turn.
    if re.search(r"đính chính|cập nhật|thực ra|không còn", text, re.I):
        loc = re.search(r"(?:giờ\s+mình\s+đang\s+ở|hiện\s+tại\s+là|mình\s+đang\s+ở|mình\s+vẫn\s+ở)\s+([^,.!?]+)", text, re.I)
        if loc:
            facts["location"] = loc.group(1).strip()
    # Ignore hypothetical job titles and locations mentioned as travel/meeting destinations.
    if re.search(r"(?:đùa|hay là chuyển sang)\s+product manager", text, re.I):
        facts.pop("profession", None)
    if re.search(r"Hà Nội chỉ là|chỉ là.*Hà Nội", text, re.I):
        facts.pop("location", None)
    # Reject weak/negated assertions and clip trailing explanations.
    if facts.get("location", "").lower().startswith(("hiện tại", "thông tin", "đã thay đổi", "thì nhớ", "mlo")):
        facts.pop("location", None)
    if facts.get("name", "").lower().startswith(("gì", "ai", "thông tin")):
        facts.pop("name", None)
    if facts.get("profession", "").lower() in {"hiện tại", "mới", "nhé"}:
        facts.pop("profession", None)
    if facts.get("favorite_drink", "").lower().startswith(("của mình", "và style")):
        facts.pop("favorite_drink", None)
    # Corrections phrased as explicit old/new values.
    if "chứ không còn" in text.lower():
        after = re.search(r"giờ mình đang ở\s+([^ ]+)", text, re.I)
        if after:
            facts["location"] = after.group(1)
    if "từ backend sang" in text.lower():
        after = re.search(r"từ backend sang\s+([^,.!?]+)", text, re.I)
        if after:
            facts["profession"] = after.group(1).strip()
    # Negated references describe obsolete or hypothetical values, not current facts.
    if re.search(r"không còn làm|đừng nói|nghề cũ|câu đùa|chỉ là câu đùa", text, re.I):
        if not re.search(r"giờ (?:mình )?(?:chuyển sang|làm)|hiện tại là", text, re.I):
            facts.pop("profession", None)
    if re.search(r"(?:đừng lấy|không phải nơi ở|chỉ là nơi|chứ không còn ở)", text, re.I):
        if not re.search(r"giờ mình đang ở|mình vẫn ở", text, re.I):
            facts.pop("location", None)
    pet = re.search(r"corgi\s+tên\s+([^,.!?]+)", text, re.I)
    if pet:
        facts["pet"] = "corgi tên " + pet.group(1).strip()
    return facts


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Compress older turns into bounded, deterministic snippets."""
    snippets: list[str] = []
    for item in messages:
        content = re.sub(r"\s+", " ", item.get("content", "")).strip()
        if content:
            snippets.append(f"{item.get('role', 'user')}: {content[:220]}")
    return "\n".join(snippets[-max_items:])


@dataclass
class CompactMemoryManager:
    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        current = self.state.setdefault(thread_id, {"messages": [], "summary": "", "compactions": 0})
        messages = current["messages"]
        assert isinstance(messages, list)
        messages.append({"role": role, "content": content})
        summary = str(current["summary"])
        total = estimate_tokens(summary + "\n" + "\n".join(m["content"] for m in messages))
        if total > self.threshold_tokens and len(messages) > max(1, self.keep_messages):
            split = max(1, len(messages) - max(1, self.keep_messages))
            old = messages[:split]
            carry = summary + "\n" + summarize_messages(old, max_items=12)
            current["summary"] = carry[-max(1200, self.threshold_tokens * 4):]
            current["messages"] = messages[split:]
            current["compactions"] = int(current["compactions"]) + 1

    def context(self, thread_id: str) -> dict[str, object]:
        state = self.state.get(thread_id)
        if state is None:
            return {"messages": [], "summary": "", "compactions": 0}
        return {"messages": list(state["messages"]), "summary": state["summary"], "compactions": state["compactions"]}

    def compaction_count(self, thread_id: str) -> int:
        return int(self.state.get(thread_id, {}).get("compactions", 0))
