from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return payload


def recall_points(answer: str, expected: list[str]) -> float:
    if not expected:
        return 1.0
    hits = sum(value.casefold() in answer.casefold() for value in expected)
    ratio = hits / len(expected)
    return 1.0 if ratio == 1 else (0.5 if ratio > 0 else 0.0)


def heuristic_quality(answer: str, expected: list[str]) -> float:
    recall = recall_points(answer, expected)
    return round(recall * 0.8 + (0.2 if answer.strip() else 0), 2)


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    recall_scores: list[float] = []
    quality_scores: list[float] = []
    token_total = prompt_total = compactions = 0
    memory_start = sum(p.stat().st_size for p in (config.state_dir / "profiles").rglob("User.md")) if (config.state_dir / "profiles").exists() else 0
    for conv in conversations:
        user_id = conv["user_id"]
        thread_id = f"{agent_name}-{conv['id']}"
        for turn in conv["turns"]:
            agent.reply(user_id, thread_id, turn)
        token_total += agent.token_usage(thread_id)
        prompt_total += agent.prompt_token_usage(thread_id)
        compactions += agent.compaction_count(thread_id)
        for index, question in enumerate(conv.get("recall_questions", [])):
            result = agent.reply(user_id, f"{agent_name}-{conv['id']}-recall-{index}", question["question"])
            answer = str(result["response"])
            expected = question.get("expected_contains", [])
            recall_scores.append(recall_points(answer, expected))
            quality_scores.append(heuristic_quality(answer, expected))
            token_total += agent.token_usage(f"{agent_name}-{conv['id']}-recall-{index}")
            prompt_total += agent.prompt_token_usage(f"{agent_name}-{conv['id']}-recall-{index}")
    memory_end = sum(p.stat().st_size for p in (config.state_dir / "profiles").rglob("User.md")) if (config.state_dir / "profiles").exists() else 0
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
    return BenchmarkRow(agent_name, token_total, prompt_total, avg_recall, avg_quality, max(0, memory_end - memory_start), compactions)


def format_rows(rows: list[BenchmarkRow]) -> str:
    header = "| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |"
    sep = "|---|---:|---:|---:|---:|---:|---:|"
    lines = [header, sep]
    for row in rows:
        lines.append(f"| {row.agent_name} | {row.agent_tokens_only} | {row.prompt_tokens_processed} | {row.recall_score:.2f} | {row.response_quality:.2f} | {row.memory_growth_bytes} | {row.compactions} |")
    return "\n".join(lines)


def main() -> None:
    config = load_config(Path(__file__).resolve().parent.parent)
    suites = [("Standard Benchmark", "conversations.json"), ("Long-Context Stress Benchmark", "advanced_long_context.json")]
    for title, filename in suites:
        conversations = load_conversations(config.data_dir / filename)
        # Use distinct run folders so reruns have stable memory-growth measurements.
        for name, cls in (("Baseline", BaselineAgent), ("Advanced", AdvancedAgent)):
            agent_config = type(config)(**{**config.__dict__, "state_dir": config.state_dir / title.lower().replace(" ", "_") / name.lower()})
            agent_config.state_dir.mkdir(parents=True, exist_ok=True)
            agent = cls(config=agent_config, force_offline=True)
            if name == "Baseline":
                baseline = run_agent_benchmark(name, agent, conversations, agent_config)
            else:
                advanced = run_agent_benchmark(name, agent, conversations, agent_config)
        print(f"\n## {title}\n\n{format_rows([baseline, advanced])}")


if __name__ == "__main__":
    main()
