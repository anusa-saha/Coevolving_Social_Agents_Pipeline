"""Where this baseline reads the dataset from and writes its outputs to.

The dataset and the discovery logic are shared (csa_core.paths); data/, logs/ and ckpt/
are local to this package so two arms never write into one another's records.

One extra thing lives here that the other arms do not need: `find_ppdpp()`. DAT steers a
frozen LM through a prefix and changes NOTHING about the prompt, so its chair prompt has
to be the *same* chair prompt PPDPP and EPO use with no planner act attached
(`CSAMessages(case, 'system', conv, action=None)` -- the no-planner control). Borrowing
that builder rather than reimplementing it is what lets the DAT rows be read against the
other arms at all; selftest.py asserts the rendered prompt is byte-identical to it.

The sys.path insert is a fallback for running the scripts straight out of a fresh clone
(`cd dat && python selftest.py`). `pip install -e .` at the repo root makes it
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


def find_ppdpp():
    """Locate ppdpp/ so the shared chair-prompt builder can be imported.

    Same discovery rule EPO uses (config._find_ppdpp_csa): an explicit override first,
    then a walk up the tree looking for the file actually read. Fails with a usable
    message rather than a ModuleNotFoundError three imports later.
    """
    def ok(d):
        return d and os.path.isfile(os.path.join(d, 'prompt.py'))

    env = os.environ.get('CSA_PPDPP_DIR')
    if env:
        if not ok(os.path.abspath(env)):
            raise SystemExit('CSA_PPDPP_DIR=%r does not look like ppdpp/ '
                             '(needs prompt.py)' % env)
        return os.path.abspath(env)

    seen, node = [], HERE
    for _ in range(6):
        for cand in (os.path.join(node, 'ppdpp'),
                     os.path.join(node, 'baselines', 'ppdpp')):
            seen.append(os.path.abspath(cand))
            if ok(cand):
                return os.path.abspath(cand)
        parent = os.path.dirname(node)
        if parent == node:
            break
        node = parent

    raise SystemExit(
        'cannot find ppdpp/. This package borrows its chair-prompt builder so the\n'
        'unsteered condition is byte-identical to the other arms.\nLooked in:\n  %s\n\n'
        'Fix: point CSA_PPDPP_DIR at it, e.g.\n'
        '  export CSA_PPDPP_DIR=/path/to/baselines/ppdpp'
        % '\n  '.join(dict.fromkeys(seen)))


PPDPP = find_ppdpp()
if PPDPP not in sys.path:            # ppdpp is a flat package; it imports by bare name
    sys.path.insert(0, PPDPP)
