from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    """Student TODO: define the shared configuration for the lab.

    Hints:
    - Keep paths for the repo root, dataset directory, and state directory.
    - Add compact-memory settings such as threshold and number of messages to keep.
    - Add provider settings for `openai`, `custom`, `gemini`, `anthropic`, `ollama`, and `openrouter`.
    """

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load optional .env settings and return a complete lab configuration.

    No provider or API key is needed for the deterministic offline mode. The
    provider settings are only used if an agent's live model path is enabled.
    """
    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    try:
        from dotenv import load_dotenv

        load_dotenv(root / ".env", override=False)
    except ImportError:
        pass

    provider = normalize_provider(os.getenv("LLM_PROVIDER", "openai"))
    model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0"))
    keys = {
        "openai": os.getenv("OPENAI_API_KEY"),
        "custom": os.getenv("CUSTOM_API_KEY"),
        "gemini": os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
        "anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "ollama": None,
        "openrouter": os.getenv("OPENROUTER_API_KEY"),
    }
    urls = {
        "openai": os.getenv("OPENAI_BASE_URL"),
        "custom": os.getenv("CUSTOM_BASE_URL"),
        "gemini": None,
        "anthropic": None,
        "ollama": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "openrouter": os.getenv("OPENROUTER_BASE_URL"),
    }
    model = ProviderConfig(provider, model_name, temperature, keys[provider], urls[provider])

    judge_provider = normalize_provider(os.getenv("JUDGE_PROVIDER", provider))
    judge_model = ProviderConfig(
        provider=judge_provider,
        model_name=os.getenv("JUDGE_MODEL", model_name),
        temperature=float(os.getenv("JUDGE_TEMPERATURE", "0")),
        api_key=keys[judge_provider],
        base_url=urls[judge_provider],
    )

    state_dir = Path(os.getenv("STATE_DIR", str(root / "state"))).expanduser()
    if not state_dir.is_absolute():
        state_dir = root / state_dir
    state_dir.mkdir(parents=True, exist_ok=True)
    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir.resolve(),
        compact_threshold_tokens=int(os.getenv("COMPACT_THRESHOLD_TOKENS", "1800")),
        compact_keep_messages=int(os.getenv("COMPACT_KEEP_MESSAGES", "6")),
        model=model,
        judge_model=judge_model,
    )
