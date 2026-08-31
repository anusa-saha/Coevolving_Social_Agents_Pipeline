"""The lexical disclosure detector, copied verbatim from ppdpp_csa/env.py.

Copied rather than imported because ppdpp_csa/env.py pulls in torch, transformers and
nltk at module level, and stage 1 (manufacturing) plus every unit test here must run
without a GPU stack.

Copying invites drift, so assert_identical_to_ppdpp() re-imports the originals and
compares them token-for-token. run_epo.py calls it at startup, where torch is present.
Every disclosure figure in the comparison depends on these three objects being the same
in both packages.
"""
import re

# ---- verbatim from ppdpp_csa/env.py:36 ---------------------------------------
_STOP = set('a an the is are was were be been being of to in on at for with and or but '
            'that this these those it its as by from we you i they he she them us our your '
            'has have had do does did not no yes will would can could should may might'.split())


def _content_tokens(text):
    return [t for t in re.findall(r"[a-z0-9%$./-]+", text.lower())
            if t not in _STOP and len(t) > 1]


def _overlap(fact_text, utterance):
    ftok = set(_content_tokens(fact_text))
    if not ftok:
        return 0.0
    return len(ftok & set(_content_tokens(utterance))) / len(ftok)
# ---- end verbatim ------------------------------------------------------------


def assert_identical_to_ppdpp():
    """Fail loudly if the copy above has drifted from ppdpp_csa/env.py.

    Only callable where torch/transformers are importable. Returns True on success,
    raises AssertionError on drift, and returns None if the original cannot be loaded
    (so a CPU-only box can still run stage 1).
    """
    try:
        import env as ppdpp_env                      # ppdpp_csa/env.py, via config path
    except Exception as e:                           # noqa: BLE001
        print('[detectors] skipped drift check: %s' % e)
        return None

    assert _STOP == ppdpp_env._STOP, 'stopword list has drifted from ppdpp_csa'
    probes = [
        ('Lane 2 has a current floor-load certification for the 1,800-kilogram pallet.',
         'Morgan says only Lane 2 is certified to 1800 kilogram floor-load.'),
        ('The tilt-bed requires 4.0-meter turning clearance available only at Lane 3.',
         'No constraints from movement control.'),
        ('', 'anything at all'),
    ]
    for fact, utt in probes:
        a, b = _overlap(fact, utt), ppdpp_env._overlap(fact, utt)
        assert abs(a - b) < 1e-12, 'overlap drifted on %r: %s vs %s' % (fact[:40], a, b)
        assert _content_tokens(utt) == ppdpp_env._content_tokens(utt), \
            'tokeniser drifted on %r' % utt[:40]
    print('[detectors] identical to ppdpp_csa/env.py')
    return True


if __name__ == '__main__':
    import config                                    # noqa: F401  (sys.path shim)
    assert_identical_to_ppdpp()
