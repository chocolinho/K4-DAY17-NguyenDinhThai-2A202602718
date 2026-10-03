from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig
from memory_store import CompactMemoryManager, UserProfileStore
from model_provider import ProviderConfig


def make_config(tmp_path: Path):
    model = ProviderConfig("openai", "offline", 0)
    return LabConfig(tmp_path, tmp_path / "data", tmp_path / "state", 80, 2, model, model)


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    assert store.read_text("user") == "# User profile\n"
    path = store.write_text("user", "# User profile\n- name: Linh")
    assert path.exists() and store.file_size("user") > 0
    assert store.edit_text("user", "Linh", "Mai")
    assert "Mai" in store.read_text("user")
    assert not store.edit_text("user", "missing", "value")


def test_compact_trigger(tmp_path: Path) -> None:
    manager = CompactMemoryManager(threshold_tokens=20, keep_messages=2)
    for i in range(8):
        manager.append("thread", "user", f"Message {i}: " + "detail " * 12)
    assert manager.compaction_count("thread") > 0
    assert len(manager.context("thread")["messages"]) <= 2


def test_cross_session_recall(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    advanced = AdvancedAgent(cfg, force_offline=True)
    baseline = BaselineAgent(cfg, force_offline=True)
    advanced.reply("u", "old", "Mình tên là Linh.")
    baseline.reply("u", "old", "Mình tên là Linh.")
    assert "Linh" in advanced.reply("u", "new", "Mình tên gì?")["response"]
    assert "chưa biết tên" in baseline.reply("u", "new", "Mình tên gì?")["response"]


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    baseline = BaselineAgent(cfg, force_offline=True)
    advanced = AdvancedAgent(cfg, force_offline=True)
    for i in range(24):
        text = f"Turn {i} has a repeated context: " + "Vietnamese long context detail " * 10
        baseline.reply("u", "base", text)
        advanced.reply("u", "adv", text)
    assert advanced.compaction_count("adv") > 0
    assert advanced.prompt_token_usage("adv") < baseline.prompt_token_usage("base")
