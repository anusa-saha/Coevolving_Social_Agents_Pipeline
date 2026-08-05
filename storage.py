import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

import config

_counter_lock = threading.Lock()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def next_scenario_id() -> str:
    path = Path(config.OUTPUT_DIR, ".scenario_counter")
    path.parent.mkdir(parents=True, exist_ok=True)
    with _counter_lock:
        n = int(path.read_text().strip()) if path.exists() else 0
        n += 1
        path.write_text(str(n))
    return f"scenario_{n}"


def _write_json(path: Path, data) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, path)
    return path


def _append_jsonl(path: Path, entry: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")
    return path


def read_jsonl(path) -> list:
    path = Path(path)
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def scenario_dir(scenario_id: str) -> Path:
    return Path(config.OUTPUT_DIR, "scenarios", scenario_id)


def round_stage_dir(scenario_id: str, round_num: int, stage: str) -> Path:
    return scenario_dir(scenario_id) / f"round_{round_num}" / stage


def save_stage_result(scenario_id: str, round_num: int, stage: str, scenario: dict,
                       result: dict, feedback: dict, passed: bool) -> Path:
    stage_dir = round_stage_dir(scenario_id, round_num, stage)

    _write_json(stage_dir / "scenario.json", scenario)
    _write_json(stage_dir / "result.json", result)
    _write_json(stage_dir / "feedback.json", feedback)
    _write_json(stage_dir / "outcome.json", {"passed": passed, "timestamp": now_iso()})

    for i, rollout in enumerate(result.get("rollouts", [])):
        _write_json(stage_dir / f"rollout_{i}.json", rollout)

    entry = {
        "timestamp": now_iso(),
        "scenario_id": scenario_id,
        "round": round_num,
        "stage": stage,
        "passed": passed,
        "reject_tag": feedback.get("reject_tag"),
        "diagnosis": feedback.get("diagnosis"),
        "fix_instructions": feedback.get("fix_instructions"),
    }
    _append_jsonl(Path(config.OUTPUT_DIR, "all_iterations.jsonl"), entry)
    _append_jsonl(
        Path(config.OUTPUT_DIR, "accepted.jsonl" if passed else "rejected.jsonl"), entry
    )
    return stage_dir


def save_scenario_outcome(scenario_id: str, status: str, scenario: dict, rounds_taken: int,
                           exhausted_stage, stage_failure_counts: dict) -> Path:
    data = {
        "scenario_id": scenario_id,
        "status": status,
        "rounds_taken": rounds_taken,
        "exhausted_stage": exhausted_stage,
        "stage_failure_counts": stage_failure_counts,
        "scenario": scenario,
        "timestamp": now_iso(),
    }
    return _write_json(scenario_dir(scenario_id) / "outcome.json", data)


def load_json_array(path) -> list:
    with open(path) as f:
        return json.load(f)


def write_json_array(path, data) -> Path:
    return _write_json(Path(path), data)
