"""Version shims for transformers / peft / accelerate.

The API surface this package touches moved several times across releases, and the box
this runs on will not always match whatever the requirements file was pinned against.
Each shim tries the modern spelling first and falls back, rather than pinning a version.

What actually differs, and where:

  from_pretrained(dtype=)      renamed from torch_dtype= in transformers 4.56.
                               Older releases TypeError on dtype=.
  device_map=                  needs accelerate installed. Without it, load on CPU and
                               .to(device) instead.
  apply_chat_template          added in 4.34; Qwen2 architecture support landed in 4.37.
                               That is the hard floor -- below it Qwen2.5 will not load
                               at all, and no shim can help.
  enable_thinking=             Qwen3 only. Qwen2.5 tokenizers TypeError on it.
  get_linear_schedule_with_warmup   moved between transformers.optimization and the
                               top-level namespace.
  disable_adapter()            peft context manager for reference-model KL. Missing or
                               differently named in old peft.

Run `python compat.py` to print what this box has and whether it will work.
"""
import contextlib

MIN_TRANSFORMERS = (4, 37)          # Qwen2 architecture support
MIN_PEFT = (0, 6)                   # LoraConfig + PeftModel.from_pretrained(is_trainable)


def _ver(mod):
    try:
        return tuple(int(x) for x in mod.__version__.split('.')[:2])
    except Exception:                                # noqa: BLE001
        return (0, 0)


def versions():
    out = {}
    for name in ('torch', 'transformers', 'peft', 'accelerate', 'openai', 'nltk',
                 'numpy'):
        try:
            m = __import__(name)
            out[name] = getattr(m, '__version__', '?')
        except Exception:                            # noqa: BLE001
            out[name] = None
    return out


def problems():
    """Blocking issues, as actionable strings. Empty list means good to go."""
    v = versions()
    out = []
    if v['torch'] is None:
        out.append('torch is not installed:  pip install torch')
    if v['transformers'] is None:
        out.append('transformers is not installed:  pip install "transformers>=4.37"')
    else:
        import transformers
        if _ver(transformers) < MIN_TRANSFORMERS:
            out.append('transformers %s is too old for Qwen2.5 (need >= %d.%d):  '
                       'pip install -U "transformers>=4.37"'
                       % (v['transformers'], *MIN_TRANSFORMERS))
    if v['peft'] is None:
        out.append('peft is not installed (needed to train LLM_s):  pip install "peft>=0.6"')
    else:
        import peft
        if _ver(peft) < MIN_PEFT:
            out.append('peft %s is too old (need >= %d.%d):  pip install -U "peft>=0.6"'
                       % (v['peft'], *MIN_PEFT))
    return out


def warnings_():
    """Non-blocking, but worth knowing."""
    v = versions()
    out = []
    if v['accelerate'] is None:
        out.append('accelerate missing: device_map is unavailable, so models load on CPU '
                   'then move to the device. Works, but slower to start and needs the '
                   'full model to fit one device. pip install accelerate')
    if v['nltk'] is None:
        out.append('nltk missing: sentence trimming falls back to raw text, which differs '
                   'from ppdpp_csa behaviour. pip install nltk, then download punkt.')
    if v['openai'] is None:
        out.append('openai missing: stage 1 needs --fallback_only, and --prm judge is '
                   'unavailable. pip install "openai>=1.0"')
    return out


# ------------------------------------------------------------------ loading
def load_tokenizer(model_id):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if getattr(tok, 'chat_template', None) is None:
        raise SystemExit(
            '%s has no chat template. This package builds every prompt through '
            'apply_chat_template; a base (non-instruct) model will not work. Use an '
            '-Instruct checkpoint.' % model_id)
    return tok


def load_causal_lm(model_id, dtype='bfloat16', device='cuda:0', trainable=False):
    """from_pretrained across the dtype rename and the accelerate dependency."""
    import torch
    from transformers import AutoModelForCausalLM

    td = getattr(torch, dtype) if isinstance(dtype, str) else dtype

    def _try(**kw):
        return AutoModelForCausalLM.from_pretrained(model_id, **kw)

    last = None
    # (dtype spelling) x (device_map or not)
    for dt_kw in ({'dtype': td}, {'torch_dtype': td}):
        for dev_kw in ({'device_map': device}, {}):
            try:
                model = _try(**dt_kw, **dev_kw)
            except TypeError as e:                   # wrong dtype kwarg for this version
                last = e
                break                                # try the other spelling
            except (ImportError, ValueError) as e:   # accelerate missing / bad device_map
                last = e
                continue

            # Some releases swallow an unknown kwarg into the config instead of
            # raising, which would silently load a 7B in fp32 (~28 GiB). Check rather
            # than trust, and retry with the other spelling if it was ignored.
            got = next(model.parameters()).dtype
            if got != td:
                if dt_kw == {'dtype': td}:
                    del model
                    last = RuntimeError('dtype= was ignored (loaded %s, wanted %s)'
                                        % (got, td))
                    break
                model = model.to(td)                 # last resort: cast after loading
                print('[compat] %s loaded as %s, cast to %s' % (model_id, got, td))

            if not dev_kw:                           # no device_map: move it ourselves
                model = model.to(device)
            if not trainable:
                model.eval()
            return model
    raise RuntimeError('could not load %s: %s' % (model_id, last))


def render_chat(tokenizer, messages):
    """apply_chat_template, tolerating tokenizers without enable_thinking."""
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)


def linear_schedule(optimizer, warmup_steps, total_steps):
    try:
        from transformers import get_linear_schedule_with_warmup
    except ImportError:
        from transformers.optimization import get_linear_schedule_with_warmup
    return get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)


@contextlib.contextmanager
def adapter_disabled(model):
    """Reference model for the KL term, by turning the LoRA adapters off.

    Older peft lacks the context manager. Rather than silently computing a KL against
    the policy itself (which is identically zero and would hide the bug), this yields
    False so the caller can skip the term.
    """
    fn = getattr(model, 'disable_adapter', None)
    if fn is None:
        yield False
        return
    try:
        with fn():
            yield True
    except TypeError:                                # not a context manager in this peft
        yield False


def report():
    v = versions()
    print('environment')
    for k in ('torch', 'transformers', 'peft', 'accelerate', 'openai', 'nltk', 'numpy'):
        print('  %-14s %s' % (k, v[k] or 'MISSING'))
    try:
        import torch
        print('  %-14s %s (%d device(s))'
              % ('cuda', torch.cuda.is_available(), torch.cuda.device_count()))
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print('      cuda:%d  %s  %.0f GiB' % (i, p.name, p.total_memory / 2**30))
    except Exception:                                # noqa: BLE001
        pass

    probs, warns = problems(), warnings_()
    if warns:
        print('\nwarnings')
        for w in warns:
            print('  - %s' % w)
    if probs:
        print('\nBLOCKING')
        for p in probs:
            print('  - %s' % p)
        return 1
    print('\nno blocking problems')
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(report())
