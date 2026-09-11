"""Where this baseline reads the dataset from and writes its outputs to.

The dataset and the discovery logic are shared (csa_core.paths); data/, logs/ and ckpt/
are local to this package so two arms never write into one another's records.

The sys.path insert is a fallback for running the scripts straight out of a fresh clone
(`cd sotopia_rl && python selftest.py`). `pip install -e .` at the repo root makes it
unnecessary, and is the documented route.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from csa_core.paths import (                            # noqa: E402,F401
    ALL_HUB_DOMAINS, DOMAINS, HF_REPO, HF_REPO_TYPE, PUBLISHED_DOMAINS,
    PUBLISHED_SPLIT, RAW_CACHE, SCENARIOS_PER_DOMAIN,
    download_raw, find_raw, find_reference, is_published_config, workspace)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, LOGS, CKPT = workspace(__file__)
