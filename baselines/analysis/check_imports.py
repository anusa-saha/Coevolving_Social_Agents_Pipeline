"""Resolve every import in the repo without executing anything.

Most of this repo cannot be imported on a machine without a GPU stack: `import torch` at
module scope means the training and environment files never run under the selftests. That
left a whole class of bug invisible -- when the shared modules moved into csa_core, every
`import data_csa  # noqa: E402` kept pointing at a file that no longer existed, and
nothing noticed, because nothing imports those files here.

This walks the AST instead. For each module it asks: does every bare local import name a
file that actually sits beside it, or a real csa_core submodule? No execution, no torch,
runs in under a second.

    python analysis/check_imports.py

Exits non-zero when something cannot resolve, so it works as a CI gate.
"""
import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Third-party and stdlib names are not our problem; requirements.txt covers those.
EXTERNAL = {
    'os', 'sys', 're', 'json', 'ast', 'math', 'time', 'glob', 'random', 'argparse',
    'collections', 'statistics', 'itertools', 'functools', 'io', 'shutil', 'copy',
    'hashlib', 'pathlib', 'typing', 'warnings', 'unicodedata', 'textwrap', 'string',
    'datetime', 'subprocess', 'traceback', 'tempfile', 'contextlib', 'dataclasses',
    'logging', 'pickle', 'types', 'inspect', 'abc', 'enum', 'csv', 'gzip', 'operator',
    'importlib', 'signal',
    'torch', 'transformers', 'peft', 'accelerate', 'numpy', 'nltk', 'openai',
    'huggingface_hub', 'reportlab', 'matplotlib', 'tqdm', 'datasets', 'safetensors',
    'sklearn', 'scipy', 'pandas', 'fastchat', 'pypdf', 'PyPDF2', 'tomllib', 'setuptools',
    'tensorboardX', 'pytorch_transformers',
}

# Imports that resolve only after a deliberate sys.path insert at runtime. Listed with
# the reason, so an unexplained one still fails the check.
DYNAMIC = {
    ('csa_core/detectors.py', 'env'):
        'assert_matches_ppdpp() puts ppdpp/ on sys.path, then imports its env.py to '
        'compare the two copies of the overlap rule',
    ('epo/prompt_epo.py', 'prompt'):
        'EPO renders LLM_d with ppdpp/prompt.py on purpose, so the dialogue agent is '
        'identical across the two arms; epo/config.py does the sys.path insert',
}

CORE = os.path.join(ROOT, 'csa_core')
CORE_MODULES = {f[:-3] for f in os.listdir(CORE) if f.endswith('.py')} if \
    os.path.isdir(CORE) else set()


def local_names(d):
    return {f[:-3] for f in os.listdir(d) if f.endswith('.py')}


def check(path, siblings, rel):
    """-> list of (lineno, message)."""
    try:
        tree = ast.parse(open(path, encoding='utf-8').read())
    except SyntaxError as e:
        return [(e.lineno or 0, 'SYNTAX ERROR: %s' % e.msg)]

    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                top = a.name.split('.')[0]
                if (top in EXTERNAL or top in siblings
                        or (rel, top) in DYNAMIC):
                    continue
                if top == 'csa_core':
                    sub = a.name.split('.')[1] if '.' in a.name else None
                    if sub and sub not in CORE_MODULES:
                        bad.append((node.lineno, 'csa_core has no module %r' % sub))
                    continue
                bad.append((node.lineno, 'cannot resolve `import %s`' % a.name))
        elif isinstance(node, ast.ImportFrom):
            if node.level:                       # explicit relative import
                continue
            if not node.module:
                continue
            top = node.module.split('.')[0]
            if top in EXTERNAL or top in siblings or (rel, top) in DYNAMIC:
                continue
            if top == 'csa_core':
                parts = node.module.split('.')
                if len(parts) > 1 and parts[1] not in CORE_MODULES:
                    bad.append((node.lineno, 'csa_core has no module %r' % parts[1]))
                elif len(parts) == 1:
                    for a in node.names:
                        if a.name not in CORE_MODULES:
                            # could be a symbol re-exported by csa_core/__init__
                            pass
                continue
            bad.append((node.lineno, 'cannot resolve `from %s import ...`' % node.module))
    return bad


def main():
    problems = 0
    checked = 0
    for d, dirs, files in os.walk(ROOT):
        dirs[:] = [x for x in dirs if x not in ('__pycache__', '.git', 'data')]
        pys = [f for f in files if f.endswith('.py')]
        if not pys:
            continue
        siblings = local_names(d)
        for fn in sorted(pys):
            p = os.path.join(d, fn)
            rel = os.path.relpath(p, ROOT).replace('\\', '/')
            checked += 1
            for line, msg in check(p, siblings, rel):
                print('  %s:%d  %s' % (rel, line, msg))
                problems += 1

    print('\nchecked %d modules, %d unresolved import%s'
          % (checked, problems, '' if problems == 1 else 's'))
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
