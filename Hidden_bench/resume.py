"""resume.py - crash-safe persistence shared by main.py and run.py.

No torch import: run.py reads checkpoint progress through this file without touching CUDA.

The one rule everything here enforces: a checkpoint is either complete or invisible. A
directory is written under `<name>.tmp`, fsynced, and only then swapped into place, so a
kill at any instant leaves `<name>` or `<name>.old` whole. Append-only logs record their
byte size at every commit and are truncated back to it on resume, so rows written after
the last checkpoint are never duplicated.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil

# run.py never retries this exit code: bad config, bad data, a failed preflight, a
# checkpoint from a different experiment. Restarting cannot fix any of those.
EXIT_FATAL = 3


def read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def write_json_atomic(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def fresh_dir(path):
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)


def fsync_tree(root):
    for d, _dirs, files in os.walk(root):
        for name in files:
            with open(os.path.join(d, name), "rb") as f:
                os.fsync(f.fileno())


def commit_dir(tmp, final):
    """Swap the fully written `tmp` into `final`. At every instant either `final` or
    `final.old` is a complete copy - `final` only ever comes into existence by renaming a
    finished `tmp`, so it is safe to drop `old` whenever `final` exists."""
    old = final + ".old"
    if os.path.isdir(final):
        shutil.rmtree(old, ignore_errors=True)
        os.rename(final, old)
    os.rename(tmp, final)
    shutil.rmtree(old, ignore_errors=True)


def committed_dir(final, marker="state.json"):
    """The complete copy of a commit_dir target, or None. `marker` is written last."""
    for d in (final, final + ".old"):
        if os.path.isfile(os.path.join(d, marker)):
            return d
    return None


def sizes(files):
    """{name: open file} -> {name: bytes on disk}, after flushing and fsyncing each."""
    out = {}
    for name, f in files.items():
        f.flush()
        os.fsync(f.fileno())
        out[name] = os.fstat(f.fileno()).st_size
    return out


def truncate(path, size):
    if os.path.exists(path) and os.path.getsize(path) > size:
        with open(path, "r+b") as f:
            f.truncate(size)


def sha1_adapter(path):
    """Content hash of a saved LoRA adapter's weight files, to tell a stale eval apart from
    one run against the current checkpoint."""
    h = hashlib.sha1()
    names = sorted(n for n in os.listdir(path) if n.startswith("adapter_model"))
    for n in names:
        h.update(n.encode("utf-8"))
        with open(os.path.join(path, n), "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
    return h.hexdigest() if names else None
