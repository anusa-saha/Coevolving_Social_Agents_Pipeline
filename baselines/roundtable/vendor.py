"""Refresh this folder's vendored copies of the shared modules from csa_core.

roundtable/ is standalone on purpose: it imports nothing from the rest of the repo, so
the folder can be lifted out and dropped into another project. The price is that its copy
of the scoring contract can drift from the one the other arms use -- and drift would not
crash anything, it would quietly make this arm's numbers incomparable.

So the copies are generated, never hand-edited, and selftest.py fails if they stop
matching. To change the contract: edit csa_core/, then run this.

    python vendor.py            # refresh
    python vendor.py --check    # report drift without writing (exit 1 if any)
"""
import argparse
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.join(os.path.dirname(HERE), 'csa_core')

BANNER = '''# ---------------------------------------------------------------------------
# VENDORED from csa_core/%s. Do not edit this copy.
#
# roundtable/ is deliberately standalone -- it imports nothing from the rest of the
# repo, so the folder can be lifted out on its own. The price is that this file can
# drift from the shared one and silently make this arm's numbers incomparable, so
# selftest.py compares it against csa_core whenever csa_core is reachable.
#
# To change the scoring contract: edit csa_core/%s, then re-run
#     python roundtable/vendor.py
# ---------------------------------------------------------------------------
'''

# Functions that are meaningless once the folder stands alone, cut from the vendored
# copy. assert_matches_ppdpp() compares csa_core's detector against ppdpp/env.py -- there
# is no ppdpp here, and leaving it in would both fail confusingly if called and make the
# copy look like it has an unresolvable `import env`.
DROP_FUNCS = {'_detectors.py': ('assert_matches_ppdpp',)}

# source name -> (vendored name, [(find, replace), ...])
PLAN = [
    ('detectors.py', '_detectors.py', []),
    ('compat.py', '_compat.py', []),
    ('verifier.py', '_verifier.py',
     [('from csa_core.detectors import', 'from _detectors import')]),
    ('data_csa.py', '_data_csa.py',
     [('from csa_core import paths', 'import _paths as paths')]),
    ('paths.py', '_paths.py', [
        ("ROOT = os.path.dirname(HERE)                     # the repo root, one level up",
         "# Standalone: the cache sits inside this folder, not at a repo root that may\n"
         "# not exist. CSA_RAW_DIR still overrides, and the walk up the tree still finds\n"
         "# a sibling data/raw/ when this folder is used inside the baselines repo.\n"
         "ROOT = HERE"),
    ]),
]


def _drop(src, names):
    """Remove top-level `def name(...)` blocks, by indentation."""
    lines = src.splitlines(True)
    out, i = [], 0
    while i < len(lines):
        stripped = lines[i].lstrip()
        if (lines[i][:1] not in (' ', '\t')
                and any(stripped.startswith('def %s(' % n) for n in names)):
            i += 1
            while i < len(lines) and (not lines[i].strip()
                                      or lines[i][:1] in (' ', '\t')):
                i += 1
            while out and not out[-1].strip():        # trim blank lines before it
                out.pop()
            out.append('\n')
            continue
        out.append(lines[i])
        i += 1
    return ''.join(out)


def render(src_name, rewrites):
    src = io.open(os.path.join(CORE, src_name), encoding='utf-8').read()
    for a, b in rewrites:
        if a not in src:
            raise SystemExit('vendor: anchor missing in csa_core/%s:\n  %r'
                             % (src_name, a[:70]))
        src = src.replace(a, b)

    out_name = next((o for s_, o, _ in PLAN if s_ == src_name), None)
    drop = DROP_FUNCS.get(out_name, ())
    for name in drop:
        if ('def %s(' % name) not in src:
            raise SystemExit('vendor: csa_core/%s no longer defines %s(); update '
                             'DROP_FUNCS' % (src_name, name))
    if drop:
        src = _drop(src, drop)
        for name in drop:
            assert ('def %s(' % name) not in src, 'failed to drop %s' % name

    return BANNER % (src_name, src_name) + src


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--check', action='store_true',
                   help='report drift without writing; exit 1 if any')
    cli = p.parse_args()

    if not os.path.isdir(CORE):
        raise SystemExit('csa_core/ is not reachable from %s. This folder is standalone, '
                         'so there is nothing to refresh from.' % HERE)

    drift = 0
    for src_name, out_name, rewrites in PLAN:
        want = render(src_name, rewrites)
        out = os.path.join(HERE, out_name)
        have = io.open(out, encoding='utf-8').read() if os.path.isfile(out) else None
        if have == want:
            continue
        drift += 1
        if cli.check:
            print('  DRIFT  %s differs from csa_core/%s' % (out_name, src_name))
        else:
            io.open(out, 'w', encoding='utf-8', newline='\n').write(want)
            print('  refreshed %s from csa_core/%s' % (out_name, src_name))

    if not drift:
        print('  all %d vendored copies match csa_core' % len(PLAN))
    return 1 if (drift and cli.check) else 0


if __name__ == '__main__':
    sys.exit(main())
