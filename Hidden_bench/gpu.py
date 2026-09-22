"""gpu.py - where the models go and how much memory they may use. The ONLY file to edit
per machine; main.py (training) and inf.py (evaluation) both read it.

None of this changes the method. The VRAM cap, the eval batch and gradient checkpointing
trade memory for speed only: env.run_batch is verified bit-identical at every chunk size,
and training still runs SCENARIOS_PER_STEP x GROUP_SIZE = 32 episodes per generate().

THE 60 GiB BUDGET (one card, both models on it)

    Qwen3-4B router weights  bf16            ~8.0 GiB
    Qwen3-8B agent  weights  bf16           ~16.4 GiB
    two LoRAs + their AdamW moments          ~1.0 GiB
    ------------------------------------------------
    resident before a single episode runs   ~25.4 GiB
    left for KV cache / update activations  ~34.6 GiB

Both models are Qwen3 with the same KV geometry (36 layers, 8 KV heads, head_dim 128):
2 * 36 * 8 * 128 * 2 B = 0.141 MiB per token, so a full 4096-token episode costs ~0.56 GiB
of KV per model. At 32 episodes in lockstep that is ~18 GiB if only one model's cache is
live and ~36 GiB if both are, which is why 32 - not more - is the default here.

These are arithmetic, not measurements. Watch `peak_gib` in results_train/train_steps.csv
for the first few steps. Neither number is a cliff: env.run_batch halves the rollout batch
and retries on OOM, and main._update_with_retry frees the cache and retries an update, so
a too-large value costs time rather than the run.
"""

from __future__ import annotations

GPU = "cuda:1"      # "cuda:0" or "cuda:1": the ONE card both models (router + agent) use
VRAM_GIB = 60       # hard cap for this process on that card

# OFF: spend the VRAM on activations and update faster (identical gradients either way).
# The 8B agent's backward over one ~3-4k token span is the largest single allocation in
# the run; it fits in the ~34 GiB left above, which is why this can stay off at 60 GiB.
# Turn it ON if an update OOMs twice (the retry is only good for one).
GRAD_CHECKPOINT = False

# Episodes in lockstep per generate() at EVAL (inf.py). Eval takes no backward
# pass, so the whole ~34.6 GiB is KV cache: 32 episodes is ~18-36 GiB depending on how
# much of both models' cache is live at once. This is the first knob to raise if
# nvidia-smi shows headroom, and OOM halves and retries.
EVAL_BATCH = 32


def device():
    """GPU if that card is visible, else cuda:0. Under SLURM (or any CUDA_VISIBLE_DEVICES
    mask) the allocated card is always cuda:0 inside the job, whatever its physical index,
    so asking for cuda:1 there would crash - this falls back instead, and says so."""
    if not str(GPU).startswith("cuda:"):
        return GPU
    import torch
    n = torch.cuda.device_count()
    if n and int(GPU.split(":")[1]) >= n:
        print("[gpu] {} not visible ({} GPU(s)) -> using cuda:0".format(GPU, n), flush=True)
        return "cuda:0"
    return GPU


_CAPPED = set()


def cap_vram(dev, gib=None):
    """Limit this process to `gib` (default VRAM_GIB) on `dev`. It caps PyTorch's
    allocator; the CUDA context (~0.5 GiB) sits in the remaining buffer. Allocating past
    the cap raises OOM, which env.run_batch answers by halving the batch."""
    gib = VRAM_GIB if gib is None else gib
    d = str(dev)
    if not gib or not d.startswith("cuda") or d in _CAPPED:
        return
    import torch
    if not torch.cuda.is_available():
        return
    idx = torch.device(d).index or 0
    total = torch.cuda.get_device_properties(idx).total_memory
    frac = min(1.0, gib * 1024 ** 3 / total)
    torch.cuda.set_per_process_memory_fraction(frac, idx)
    _CAPPED.add(d)
    print("[gpu] VRAM cap on {}: {:.1f} GiB of {:.1f} GiB".format(
        d, frac * total / 1024 ** 3, total / 1024 ** 3), flush=True)
