"""Where the data lives, and where each baseline keeps its own outputs.

Two different things, deliberately kept apart:

  * the DATASET is shared. One copy at the repo root under data/raw/, discovered by
    walking up the tree, and downloadable from the Hub so a fresh clone is not blocked
    on someone copying eleven JSON files by hand.
  * the OUTPUTS are per-baseline. Each package gets its own data/, logs/ and ckpt/
    beside its own source, because two arms writing into one logs/ would overwrite each
    other's records. `workspace(__file__)` builds them.

Reference artifacts (records, checkpoints, manufactured corpora) are NOT in the repo.
They are large and they are results. Point CSA_ARTIFACTS_DIR at wherever they were
archived and the selftests will validate against them; without it those specific checks
skip and say so.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                     # the repo root, one level up

# ALL ELEVEN domains, 50 scenarios each = 550.
#
# This is a deliberate benchmark change from the original configuration, which was three
# domains x 50 = 150. Breadth was preferred over depth: eleven domains at 50 exercise far
# more of the distribution than three at 100, for a comparable total.
#
# It INVALIDATES every result produced under the old configuration. The split is rebuilt
# from scratch, so the scenarios an arm trains and evaluates on are not the ones the
# archived records used. Every arm needs re-running before its numbers mean anything.
#
# Names here are FILENAME STEMS, which is what the Hub serves. Two are worth knowing
# about: `informal_commerce_bargaining` (not `bargaining`), and
# `family_friends_informal`, whose file carries an internal `domain` field spelled
# `friends_family_informal` -- transposed. data_csa normalises the field to the stem so
# the domain list, the uids and the filenames all agree.
DOMAINS = ('defense', 'education', 'entertainment', 'family_friends_informal',
           'finance', 'healthcare', 'informal_commerce_bargaining', 'legal',
           'manufacturing', 'software_technology', 'workplace_interpersonal')

# Bound before the override below, so it stays the full eleven whatever DOMAINS is set to.
ALL_HUB_DOMAINS = DOMAINS

# How many scenarios to take per domain, lowest scenario_id first. The Hub ships 100 per
# domain; 50 is the configured subset. Taking the head rather than a sample is what makes
# the three original domains bit-identical to the scenarios the earlier runs used.
SCENARIOS_PER_DOMAIN = int(os.environ.get('CSA_SCENARIOS_PER_DOMAIN', '50'))

# The original configuration, kept so selftests can recognise it and check themselves
# against the published 99/9/42 split. Set CSA_DOMAINS=published to restore it.
PUBLISHED_DOMAINS = ('healthcare', 'defense', 'software_technology')
PUBLISHED_SPLIT = (99, 9, 42)

if os.environ.get('CSA_DOMAINS', '').strip().lower() == 'published':
    DOMAINS = PUBLISHED_DOMAINS


def is_published_config():
    """True when the loader is set up exactly as the archived runs were.

    The explicit split files describe a different dataset, so they rule this out however
    DOMAINS is configured -- otherwise selftests would check the 1100-scenario split
    against the published 99/9/42 shape and fail for the wrong reason.
    """
    return (find_splits_dir() is None
            and tuple(sorted(DOMAINS)) == tuple(sorted(PUBLISHED_DOMAINS))
            and SCENARIOS_PER_DOMAIN == 50)


# Fetched from the Hub the same way the model is, so a fresh clone needs no manual data
# copying. Files live under data/ in the repo, not at its root.
# The annotator that labels chair turns (PPDPP's four acts) and writes EPO's strategy
# targets. ONE setting, because both arms' supervision comes from it -- when they used
# different annotators (Gemma 3 for EPO, Gemma 4 for PPDPP) the two arms were trained
# against differently-labelled data, which is a confound rather than a difference in
# method.
#
# google/gemma-4-31b-it is the stronger dense alternative; the 26b-a4b is a mixture with
# ~4B active, so it is faster and has a free tier. Changing this means RE-ANNOTATING both
# arms, not just the one you are working on.
ANNOTATOR_MODEL = os.environ.get('CSA_ANNOTATOR_MODEL',
                                 'google/gemma-4-26b-a4b-it:free')
ANNOTATOR_BASE_URL = os.environ.get('CSA_ANNOTATOR_BASE_URL',
                                    'https://openrouter.ai/api/v1')

HF_REPO = os.environ.get('CSA_HF_REPO', 'anusasaha/Coevolving_Social_Agents')
HF_REPO_TYPE = os.environ.get('CSA_HF_REPO_TYPE', 'dataset')
HF_PATH_PREFIX = os.environ.get('CSA_HF_PREFIX', 'data/')
RAW_CACHE = os.path.join(ROOT, 'data', 'raw')


def workspace(pkg_file):
    """(data, logs, ckpt) beside the calling package, created if absent.

    Takes the caller's __file__ so each baseline gets its own directories without
    hardcoding a path, and without two arms sharing one logs/.
    """
    base = os.path.dirname(os.path.abspath(pkg_file))
    out = tuple(os.path.join(base, x) for x in ('data', 'logs', 'ckpt'))
    for d in out:
        os.makedirs(d, exist_ok=True)
    return out


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


# The scenario set as ONE file: all eleven domains x 50, lowest scenario_id first. It is
# the source of truth whenever it can be found, and data/raw/ is only the fallback.
#
# Same 550 uids as the head-50 of data/raw/, so the split does not move. Seven scenarios
# differ in content (defense 3/8/45, education 4/6, workplace_interpersonal 12/17): each
# gains model-facing guidance fields under interaction_config or an agent entry. No arm
# reads those fields today, but they are why the two sources must not be mixed.
SCENARIOS_FILENAME = 'first_50.json'

# Internal `domain` spellings that differ from the filename stems in DOMAINS. The
# combined file has no filename to fall back on, so the stem is restored from this.
# Without it the uid, the bucket sort order and therefore the whole split change.
DOMAIN_ALIASES = {'friends_family_informal': 'family_friends_informal'}


def find_scenarios_file():
    """The combined scenario file, or None when the per-domain files should be used.

    CSA_SCENARIOS_FILE names it explicitly ('' turns it off). CSA_RAW_DIR, when set,
    asks for the per-domain files and wins. Otherwise walk up the tree, the same way
    find_raw() does, so the vendored copy in roundtable/ finds the repo-root file too.
    """
    env = os.environ.get('CSA_SCENARIOS_FILE')
    if env is not None:
        if not env.strip():
            return None
        if not os.path.isfile(env):
            raise SystemExit('CSA_SCENARIOS_FILE=%r does not exist' % env)
        return os.path.abspath(env)
    if os.environ.get('CSA_RAW_DIR'):
        return None

    node = HERE
    for _ in range(6):
        cand = os.path.join(node, SCENARIOS_FILENAME)
        if os.path.isfile(cand):
            return os.path.abspath(cand)
        parent = os.path.dirname(node)
        if parent == node:
            break
        node = parent
    return None


# ------------------------------------------------------------------ explicit splits
# The 1100-scenario benchmark ships its split as FILES rather than as a procedure: one
# train file and three test files, prepared upstream. When they are present they are the
# source of truth and nothing is re-derived -- data_csa reads them verbatim.
#
# This supersedes the from-scratch split over first_50.json. The two describe different
# datasets (945 distinct scenarios across 11 domains here, 550 there), so results from
# one configuration say nothing about the other.
#
#   train.json        720  the training pool; data_csa carves `valid` out of it
#   test_all.json     380  the evaluation set == test_seen + test_unseen
#   test_seen.json    180  the nine domains train also covers
#   test_unseen.json  200  family_friends_informal + informal_commerce_bargaining,
#                          held out of training entirely
SPLITS_DIRNAME = os.path.join('data', 'splits')

# Which file `test` resolves to. test_all is the whole evaluation set; point this at
# test_unseen.json to make every arm's headline number a generalisation number.
TEST_SPLIT_FILE = os.environ.get('CSA_TEST_FILE', 'test_all.json')

SPLIT_FILES = {'train': 'train.json',
               'test': TEST_SPLIT_FILE,
               'test_seen': 'test_seen.json',
               'test_unseen': 'test_unseen.json'}

# Every file that must exist for a directory to count as a split directory. `test` is
# excluded because it is an alias for one of these.
SPLIT_REQUIRED = ('train.json', 'test_all.json', 'test_seen.json', 'test_unseen.json')


def _has_splits(d):
    return bool(d) and all(os.path.isfile(os.path.join(d, x)) for x in SPLIT_REQUIRED)


def find_splits_dir():
    """The directory holding train.json and the three test files, or None.

    None means "fall back to the derived split", so the older configuration still runs.
    CSA_SPLITS_DIR names the directory explicitly and '' turns the files off. Asking for
    the derived split by pointing at raw scenarios (CSA_RAW_DIR / CSA_SCENARIOS_FILE)
    also wins, so the two never fight over which dataset is in play.
    """
    env = os.environ.get('CSA_SPLITS_DIR')
    if env is not None:
        if not env.strip():
            return None
        if not _has_splits(os.path.abspath(env)):
            raise SystemExit('CSA_SPLITS_DIR=%r is missing one of %s'
                             % (env, ', '.join(SPLIT_REQUIRED)))
        return os.path.abspath(env)
    if os.environ.get('CSA_RAW_DIR') or os.environ.get('CSA_SCENARIOS_FILE'):
        return None

    node = HERE
    for _ in range(6):
        cand = os.path.join(node, SPLITS_DIRNAME)
        if _has_splits(cand):
            return os.path.abspath(cand)
        parent = os.path.dirname(node)
        if parent == node:
            break
        node = parent
    return None


def find_raw(allow_download=True):
    """Locate the directory holding <domain>_scenarios.json.

    Order: explicit override, then the repo-root cache, then a walk up the tree, then the
    Hub. Discovered rather than hardcoded because these folders get moved, and
    downloadable so a fresh clone is not blocked on copying three JSON files by hand.
    """
    env = os.environ.get('CSA_RAW_DIR')
    if env:
        if not _has_scenarios(os.path.abspath(env)):
            raise SystemExit('CSA_RAW_DIR=%r has no <domain>_scenarios.json' % env)
        return os.path.abspath(env)

    seen, node = [], HERE
    for _ in range(6):
        for cand in (RAW_CACHE,
                     os.path.join(node, 'data', 'raw'),
                     os.path.join(node, 'raw'),
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
    """An archived artifact, used ONLY by selftests to check this code against the
    published run.

    Records, manufactured corpora and checkpoints are results, so they are not in the
    repo. Set CSA_ARTIFACTS_DIR to wherever they were archived to re-enable the checks
    that need them. Returns None when absent -- nothing in any training path depends on
    this, and every caller skips cleanly and says it skipped.
    """
    env = os.environ.get('CSA_ARTIFACTS_DIR')
    roots = []
    if env:
        roots += [os.path.join(env, 'ppdpp'), env]

    node = HERE
    for _ in range(6):
        roots += [os.path.join(node, 'ppdpp'),
                  os.path.join(node, 'baselines', 'ppdpp'),
                  os.path.join(node, 'csa-artifacts', 'ppdpp')]
        parent = os.path.dirname(node)
        if parent == node:
            break
        node = parent

    for r in roots:
        cand = os.path.join(r, name)
        if os.path.isfile(cand):
            return cand
    return None
