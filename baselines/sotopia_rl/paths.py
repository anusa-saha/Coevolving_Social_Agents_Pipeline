"""Where things live. No code is imported from any other baseline.

This package is self-contained: it reads the raw scenario JSON and rebuilds the
train/valid/test split itself, rather than importing ppdpp_csa's loader or verifier.
The only thing shared with the other baselines is the *data*, and even that is
re-derived rather than read, so the split cannot silently drift.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

DATA = os.path.join(HERE, 'data')
LOGS = os.path.join(HERE, 'logs')
CKPT = os.path.join(HERE, 'ckpt')
for _d in (DATA, LOGS, CKPT):
    os.makedirs(_d, exist_ok=True)

# THREE domains, deliberately, even though the Hub repo ships eleven.
#
# The published 99/9/42 split was built from healthcare + defense + software_technology
# only. Adding the other eight would change every bucket in the stratified split, so the
# scenarios this arm trains and evaluates on would no longer be the ones PPDPP and EPO
# used, and the three baselines would stop being comparable. selftest.py asserts the
# resulting split still matches the published one exactly, and would fail loudly here.
DOMAINS = ('healthcare', 'defense', 'software_technology')

# Every domain the Hub repo carries, for reference. Extending DOMAINS to this list is a
# deliberate benchmark change, not a configuration tweak -- it invalidates the existing
# results and needs its own re-run of all three arms.
ALL_HUB_DOMAINS = ('bargaining', 'defense', 'education', 'entertainment',
                   'family_friends_informal', 'finance', 'healthcare', 'legal',
                   'manufacturing', 'software_technology', 'workplace_interpersonal')

# Fetched from the Hub the same way the model is, so a fresh clone needs no manual data
# copying. Files live under data/ in the repo, not at its root.
HF_REPO = os.environ.get('CSA_HF_REPO', 'anusasaha/Coevolving_Social_Agents')
HF_REPO_TYPE = os.environ.get('CSA_HF_REPO_TYPE', 'dataset')
HF_PATH_PREFIX = os.environ.get('CSA_HF_PREFIX', 'data/')
RAW_CACHE = os.path.join(DATA, 'raw')


def _has_scenarios(d):
    return bool(d) and all(os.path.isfile(os.path.join(d, '%s_scenarios.json' % x))
                           for x in DOMAINS)


def download_raw(repo=None, repo_type=None, dest=None):
    """Fetch the three <domain>_scenarios.json from the Hub into a local cache.

    Mirrors how the model arrives: transformers pulls the checkpoint on first use, so the
    data should not be the one thing a fresh clone has to be handed by hand. Returns the
    directory, or None if no repo is configured or the download fails.
    """
    repo = repo or HF_REPO
    repo_type = repo_type or HF_REPO_TYPE
    dest = dest or RAW_CACHE
    if not repo:
        return None
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print('[paths] huggingface_hub not installed; cannot download the scenarios')
        return None

    os.makedirs(dest, exist_ok=True)
    got = 0
    for dom in DOMAINS:
        name = '%s_scenarios.json' % dom
        target = os.path.join(dest, name)
        if os.path.isfile(target):
            got += 1
            continue
        try:
            src = hf_hub_download(repo_id=repo, repo_type=repo_type,
                                  filename=HF_PATH_PREFIX + name)
            # copy rather than symlink: the hub cache is shared and may be pruned
            import shutil
            shutil.copyfile(src, target)
            got += 1
            print('[paths] downloaded %s from %s' % (name, repo))
        except Exception as e:                       # noqa: BLE001
            print('[paths] could not fetch %s from %s: %s' % (name, repo, e))
    return dest if got == len(DOMAINS) else None


def find_raw(allow_download=True):
    """Locate the directory holding <domain>_scenarios.json.

    Order: explicit override, then a walk up the tree, then the Hub. Discovered rather
    than hardcoded because these folders get moved, and downloadable so a fresh clone is
    not blocked on someone copying three JSON files by hand.
    """
    env = os.environ.get('CSA_RAW_DIR')
    if env:
        if not _has_scenarios(os.path.abspath(env)):
            raise SystemExit('CSA_RAW_DIR=%r has no <domain>_scenarios.json' % env)
        return os.path.abspath(env)

    seen, node = [], HERE
    for _ in range(6):
        for cand in (RAW_CACHE, os.path.join(node, 'raw'),
                     os.path.join(node, 'ppdpp', 'raw'),
                     os.path.join(node, 'baselines', 'ppdpp', 'raw')):
            seen.append(os.path.abspath(cand))
            if _has_scenarios(cand):
                return os.path.abspath(cand)
        parent = os.path.dirname(node)
        if parent == node:
            break
        node = parent

    if allow_download:
        got = download_raw()
        if _has_scenarios(got):
            return os.path.abspath(got)

    raise SystemExit(
        'cannot find the raw scenario JSON.\nLooked in:\n  %s\n\n'
        'Fix, in order of preference:\n'
        '  export CSA_HF_REPO=<org>/<dataset>     # fetch it like the model\n'
        '  export CSA_RAW_DIR=/path/to/raw        # point at a local copy'
        % '\n  '.join(dict.fromkeys(seen)))


def find_reference(name):
    """Optional: a file from another baseline, used ONLY by selftest to verify that
    this package's from-scratch split and verifier agree with the published ones.
    Returns None if absent; nothing in the training path depends on it."""
    seen, node = [], HERE
    for _ in range(6):
        for stem in ('ppdpp_csa', os.path.join('ppdpp', 'ppdpp_csa'),
                     os.path.join('baselines', 'ppdpp', 'ppdpp_csa')):
            cand = os.path.join(node, stem, name)
            seen.append(cand)
            if os.path.isfile(cand):
                return cand
        parent = os.path.dirname(node)
        if parent == node:
            break
        node = parent
    return None
