"""Where this baseline reads the dataset from and writes its outputs to.

Everything this arm needs lives in this folder. `_paths.py` is a vendored copy of the
shared discovery logic; data/, logs/ and ckpt/ sit beside this file.
"""
import os

from _paths import (                                    # noqa: F401
    ALL_HUB_DOMAINS, DOMAINS, HF_REPO, HF_REPO_TYPE, PUBLISHED_DOMAINS,
    PUBLISHED_SPLIT, RAW_CACHE, SCENARIOS_PER_DOMAIN,
    download_raw, find_raw, find_reference, is_published_config, workspace)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, LOGS, CKPT = workspace(__file__)
