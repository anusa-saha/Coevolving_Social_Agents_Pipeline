"""Version shims for transformers / peft / accelerate.

The API surface this package touches moved several times across releases, and the box
this runs on will not always match whatever the requirements file was pinned against.
Each shim tries the modern spelling first and falls back, rather than pinning a version.

What actually differs, and where:

  from_pretrained(dtype=)      renamed from torch_dtype= in transformers 4.56.
                               Older releases TypeError on dtype=.
  device_map=                  needs accelerate installed. Without it, load on CPU and
                               .to(device) instead.
  qwen3_5 architecture         landed in 5.2. That is the hard floor -- below it
                               Qwen3.5-9B will not load at all, and no shim can help.
                               AutoModelForCausalLM maps its multimodal checkpoint to the
                               text-only Qwen3_5ForCausalLM, dropping the vision tower and
                               the MTP head.
  enable_thinking=             Qwen3 / Qwen3.5 only; Qwen3.5 thinks unless told not to.
                               Qwen2.5 tokenizers TypeError on it.
  get_linear_schedule_with_warmup   moved between transformers.optimization and the
                               top-level namespace.
  disable_adapter()            peft context manager for reference-model KL. Missing or
                               differently named in old peft.

Run `python compat.py` to print what this box has and whether it will work.
"""
import contextlib

MIN_TRANSFORMERS = (5, 2)           # Qwen3.5 (qwen3_5) architecture support
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
        out.append('transformers is not installed:  pip install "transformers>=5.2"')
    else:
        import transformers
        if _ver(transformers) < MIN_TRANSFORMERS:
            out.append('transformers %s is too old for Qwen3.5 (need >= %d.%d):  '
                       'pip install -U "transformers>=5.2"'
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
                   'the other baselines. pip install nltk, then download punkt.')
    if v['openai'] is None:
        out.append('openai missing: only the API-backed paths need it -- EPO stage 1 '
                   'without --fallback_only, EPO --prm judge, and Omega corpus C. '
                   'pip install "openai>=1.0"')
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
            'apply_chat_template; a base model will not work. Use a chat checkpoint '
            '(Qwen/Qwen3.5-9B, not -Base).' % model_id)
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


# ------------------------------------------------------------------ memory
def logits_tail(model, input_ids, keep):
    """Logits for the last `keep` positions only.

    Qwen3.5's vocabulary is 248k tokens, so the LM head over a 1536-token sequence is
    ~0.7 GiB in bf16 and ~1.4 GiB once a loss upcasts it, held again for the backward.
    Every loss in this repo reads only the completion at the END of the sequence, so on a
    24 GB card the rest is memory spent on nothing. `logits_to_keep` applies the head to
    the tail alone; a release without it falls back to slicing the full logits, which
    gives the same numbers at the old memory cost.
    """
    keep = max(1, min(int(keep), input_ids.shape[-1]))
    try:
        return model(input_ids=input_ids, logits_to_keep=keep).logits
    except TypeError:
        return model(input_ids=input_ids).logits[:, -keep:]


def completion_nll(model, input_ids, labels, ignore_index=-100):
    """`model(input_ids, labels=labels).loss`, computed from the completion's logits only.

    The same number whenever the labelled positions are a suffix of the sequence, which
    every SFT example here is: prompt masked, completion last. When they are not, this
    falls back to the model's own loss rather than returning a different number.
    parallel/preflight.py --probe checks the two agree on the real model.
    """
    import torch.nn.functional as F
    mask = labels != ignore_index
    if input_ids.shape[0] != 1 or not bool(mask.any()):
        return model(input_ids=input_ids, labels=labels).loss
    first = int(mask[0].nonzero()[0])
    if first < 1 or not bool(mask[0, first:].all()):
        return model(input_ids=input_ids, labels=labels).loss
    logits = logits_tail(model, input_ids, input_ids.shape[-1] - first + 1)[:, :-1]
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                           labels[:, first:].reshape(-1), ignore_index=ignore_index)


@contextlib.contextmanager
def cached_generation(model):
    """Let a no-grad generate use its KV cache on a model that trains with checkpointing.

    transformers switches the cache off whenever gradient checkpointing is on and the
    model is in train mode, so every sampled token re-runs the whole prompt. Checkpointing
    only matters for a backward pass, so it is lifted for the generate and restored after;
    dropout and every other train-mode behaviour are left as they were.
    """
    on = bool(getattr(model, 'is_gradient_checkpointing', False))
    if on:
        model.gradient_checkpointing_disable()
    try:
        yield
    finally:
        if on:
            model.gradient_checkpointing_enable()


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
