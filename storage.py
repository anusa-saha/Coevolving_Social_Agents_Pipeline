"""
Storage: append-one-at-a-time persistence for the pipeline's outputs.

A JSON array has to stay syntactically valid as a whole file, so "add one entry
at a time" here means: read the current array, append the new item, write the
whole array back out atomically. The important part is *when* this happens --
append_json_array() is called immediately after each stage's result is known,
not batched up and written once at the very end. If the pipeline crashes
mid-run, everything processed so far is already safely on disk.

Files produced under output_dir:
  - accepted.json        every scenario that cleared all three gates
  - rejected.json         every rejected attempt, tagged with its reject tag
  - all_iterations.json   every single attempt at every stage (superset of the above)
  - transcripts/<scenario_id>/round_<n>_<arm>_rollout_<i>.json
        full transcript/output for every weak-arm and strong-arm rollout (constraint #5)
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from config import CONFIG


def _path(filename: str) -> Path:
    p = Path(CONFIG.output_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p / filename


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_json_array(filename: str, item: dict) -> Path:
    """Read-modify-write append of a single item into a JSON array file."""
    path = _path(filename)
    if path.exists():
        with open(path, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    data.append(item)

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp_path, path)  # atomic on POSIX, avoids a half-written file on crash
    return path


def record_iteration(entry: dict) -> Path:
    """Every single attempt at any stage: verifier reject, weak-arm reject, strong-arm reject, or accept."""
    entry = {"timestamp": now_iso(), **entry}
    return append_json_array("all_iterations.json", entry)


def record_rejected(entry: dict) -> Path:
    """Rejected attempts only, still carrying their reject_tag (MALFORMED / LEAKED / UNCOORDINATED)."""
    entry = {"timestamp": now_iso(), **entry}
    return append_json_array("rejected.json", entry)


def record_accepted(entry: dict) -> Path:
    entry = {"timestamp": now_iso(), **entry}
    return append_json_array("accepted.json", entry)


def save_transcript(scenario_id: str, round_num: int, arm: str, rollout_idx: int, data: dict) -> Path:
    """Persist one rollout's full transcript/output (constraint #5)."""
    scenario_dir = Path(CONFIG.output_dir, "transcripts", scenario_id)
    scenario_dir.mkdir(parents=True, exist_ok=True)
    path = scenario_dir / f"round_{round_num}_{arm}_rollout_{rollout_idx}.json"
    with open(path, "w") as f:
        json.dump({"timestamp": now_iso(), **data}, f, indent=2, default=str)
    return path
