"""Make OpenVLA-7B's vendored model code run on transformers 5.x.

OpenVLA-7B pins ``transformers==4.40.1`` and that pin is *uninstallable* on this
machine: 4.40.1 requires ``tokenizers<0.20``, every 0.19.x predates CPython 3.13,
so there is no wheel and the source build is refused by PyO3 0.21 (max 3.12).
The version check inside OpenVLA's own code is a ``logger.warning``, not a raise,
so 5.17 is the only version that can be installed -- which means the vendored
code runs off its supported pin and has to be made to work rather than merely
attempted.

Exactly two things are broken, and both were found by reading the vendored source
rather than by trial and error:

  1. ``transformers.PreTrainedModel`` no longer inherits ``GenerationMixin``.
     OpenVLA's ``predict_action`` is a thin wrapper over ``self.generate(...)``,
     so with 5.17 the class has no ``.generate`` at all -- ``predict_action``
     dies on an AttributeError. Fixed by rebuilding the class with
     ``GenerationMixin`` as an extra base.

  2. ``processing_prismatic.py`` imports ``PaddingStrategy`` and friends from
     ``transformers.tokenization_utils``; 5.x moved them to
     ``transformers.tokenization_utils_base``. Fixed by re-exporting the four
     names into the old module before the remote code is imported.

Both patches are *asserted* to have taken effect. A shim that silently no-ops is
worse than no shim: a missing ``GenerationMixin`` could be papered over, and a
tokenizer that loads but pads differently would produce a model that trains,
moves plausibly and plateaus at an arbitrary success rate. Every patch here
either works or raises.

A third thing is deliberately *not* patched here: ``config.json``'s
``hf_llm_id`` is ``meta-llama/Llama-2-7b-hf``, a manually gated repo that
returns 401 anonymously. OpenVLA's ``__init__`` builds the LLM with
``AutoModelForCausalLM.from_pretrained(hf_llm_id)`` and then overwrites every
single one of those tensors from OpenVLA's own checkpoint, so fetching the
backbone is 13.5 GB of pure waste behind a login. :func:`build_architecture`
builds the LLM from its config instead. That interception is a genuine
behavioural deviation from the vendored code and is asserted too, for the same
reason.

This module is a *compatibility layer*, not a rewrite. The action head is
untouched here -- replacing it with a chunked 2-D planar head is
``adapters/openvla_policy.py``'s job, and keeping the two concerns apart is what
makes both testable.
"""
from __future__ import annotations

import os
import sys
from typing import Optional

import torch
import transformers

__all__ = [
    "REMOTE_MODEL_ENTRY",
    "OPENVLA_REPO",
    "DEFAULT_INSTRUCTION",
    "snapshot_dir",
    "resolve_model_class",
    "build_architecture",
    "setup_hf_offline",
    "greedy_decode",
    "tokens_to_bins",
    "bins_to_action",
    "make_bin_centers",
    "prepare_prompt_ids",
    "action_token_slice",
    "check_decode_agreement",
    "load_processor",
    "compat_report",
]

OPENVLA_REPO = "openvla/openvla-7b"

# The class inside the checkpoint's own remote-code module. Resolved directly
# rather than through ``AutoModelForVision2Seq``, which transformers 5.x deleted
# (its successor is ``AutoModelForImageTextToText``). Going straight to the class
# is also the honest thing to do: we are about to modify the action head, so we
# need the class, not an auto-class that hides it.
REMOTE_MODEL_ENTRY = "modeling_prismatic.OpenVLAForActionPrediction"

# Names processing_prismatic.py imports from transformers.tokenization_utils in
# 4.40.1. All four live in tokenization_utils_base in 5.x.
_MOVED_TOKENIZER_NAMES = (
    "PaddingStrategy",
    "PreTokenizedInput",
    "TextInput",
    "TruncationStrategy",
)

DEFAULT_INSTRUCTION = {
    "push": "push the block into the green seat without hitting the wall",
    "cloth": "fold the cloth corner to the opposite corner",
}


def snapshot_dir(repo: str = OPENVLA_REPO, cache: Optional[str] = None) -> str:
    """Resolve a downloaded snapshot to a path, without touching the network."""
    cache = cache or os.environ.get(
        "HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))
    root = os.path.join(cache, "models--" + repo.replace("/", "--"), "snapshots")
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"no local snapshot for {repo} under {root}. Run "
            f"`hf download {repo}` first; the adapter never downloads implicitly.")
    subs = [os.path.join(root, s) for s in sorted(os.listdir(root))]
    if len(subs) != 1:
        raise RuntimeError(
            f"expected exactly one snapshot of {repo} in {root}, found {len(subs)}: "
            f"{[os.path.basename(s) for s in subs]}. Ambiguous, so refusing to pick.")
    return subs[0]


def _patch_tokenization_utils() -> list[str]:
    """Re-export the names 5.x moved out of transformers.tokenization_utils.

    Returns the names actually restored. The module may be a lazy namespace in
    5.x, hence the explicit sys.modules entry; if it is genuinely absent we
    create a placeholder, because the remote code imports *from* it by name.
    """
    import transformers  # noqa: F401  (ensure the lazy modules materialise)
    from transformers import tokenization_utils_base as base

    mod = sys.modules.get("transformers.tokenization_utils")
    if mod is None:
        mod = type(sys)("transformers.tokenization_utils")
        sys.modules["transformers.tokenization_utils"] = mod
    restored = []
    for name in _MOVED_TOKENIZER_NAMES:
        if not hasattr(base, name):
            raise RuntimeError(
                f"transformers {transformers.__version__} has no "
                f"{name} in tokenization_utils_base either; the tokenizer shim "
                "needs updating for this transformers version.")
        if not hasattr(mod, name):
            setattr(mod, name, getattr(base, name))
        restored.append(name)
    return restored


def resolve_model_class(snapshot: str):
    """Return OpenVLA's model class, made runnable on transformers 5.x.

    The returned class is a *new* type with the fixes applied in its own
    ``__dict__``; the remote module's class object is left untouched so a second
    resolution in the same process cannot stack bases.

    Each fix is a *named* entry in :data:`PATCHES` and each is asserted to have
    had its intended effect. That is not defensive boilerplate: a shim that
    silently no-ops is worse than no shim, because the failure surfaces much later
    as a number that is plausible and wrong. An ``nn.Module`` whose
    ``_supports_sdpa`` property raises reports the *property's* name through
    ``__getattr__``, so the traceback names a symptom and not a cause -- which is
    exactly how a silent shim would present.

    The list is deliberately short and explicit. If a future transformers removes
    one of these breakages, the corresponding patch raises rather than quietly
    doing nothing, so the shim cannot rot into a lie.
    """
    from transformers import LlamaForCausalLM, PreTrainedModel
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from transformers.generation import GenerationMixin

    _patch_tokenization_utils()
    base_cls = get_class_from_dynamic_module(REMOTE_MODEL_ENTRY, snapshot)
    if not issubclass(base_cls, PreTrainedModel):
        raise RuntimeError(
            f"{base_cls.__name__} is not a PreTrainedModel subclass; the compat "
            "shim is pointed at the wrong class.")
    if issubclass(base_cls, GenerationMixin):
        # A future transformers restored generation onto PreTrainedModel; the
        # other patches may also be stale, so still verify rather than assume.
        return base_cls, ["(generation already on base class -- shim unchanged)"]

    applied = {}

    # 1. GenerationMixin. 5.x's PreTrainedModel no longer inherits it, and
    #    OpenVLA's predict_action is a thin wrapper over self.generate(...).
    applied["GenerationMixin"] = None

    # 2. _supports_sdpa. 5.x calls _check_and_adjust_attn_implementation at the
    #    top of PreTrainedModel.__init__, before the subclass has built
    #    self.language_model. OpenVLA overrides _supports_sdpa as a property
    #    returning self.language_model._supports_sdpa, so it raises there.
    #    The property's intent is "does the LLM support SDPA"; in 5.x that is a
    #    class attribute on the LLM class, so the answer is available at patch
    #    time. Read it from LlamaForCausalLM rather than hardcoding True: if a
    #    future transformers drops it we get False, and _sdpa_can_dispatch then
    #    raises with its own message asking for eager -- a real signal rather than
    #    a silently different attention kernel.
    llm_sdpa = getattr(LlamaForCausalLM, "_supports_sdpa", False)
    if not llm_sdpa:
        raise RuntimeError(
            f"LlamaForCausalLM reports no SDPA support on transformers "
            f"{transformers.__version__}, so OpenVLA's fused SDPA path cannot be "
            "preserved. Re-measure the policy's actions under eager attention "
            "before comparing any number to a run on the pinned version.")
    applied["_supports_sdpa"] = llm_sdpa

    # 3. tie_weights signature. 5.x's post_init -> init_weights calls
    #    self.tie_weights(recompute_mapping=False); OpenVLA's override takes no
    #    arguments. Forward whatever we are given, so if a future transformers
    #    grows the call further this keeps working.
    base_tie = base_cls.tie_weights

    def tie_weights(self, *a, **kw):
        return self.language_model.tie_weights(*a, **kw)

    applied["tie_weights"] = f"(*a, **kw) instead of {base_tie.__code__.co_varnames[:base_tie.__code__.co_argcount]}"

    ns = {
        "__module__": base_cls.__module__,
        "__doc__": base_cls.__doc__,
        "_supports_sdpa": llm_sdpa,
        "tie_weights": tie_weights,
        "_uoe_transformers5_patches": tuple(applied),
    }
    patched = type(base_cls.__name__, (base_cls, GenerationMixin), ns)

    # Verify each fix did what it was for, rather than trusting it.
    if not hasattr(patched, "generate"):
        raise RuntimeError("GenerationMixin was mixed in but .generate is still missing")
    if getattr(PreTrainedModel, "generate", None) is not None:
        raise RuntimeError(
            "this transformers already provides PreTrainedModel.generate; the shim "
            "is stale and would shadow a supported path")
    if patched.tie_weights is base_tie:
        raise RuntimeError("tie_weights was not overridden; the shim is stale")
    patched.__name__ = "OpenVLAForActionPrediction"
    return patched, list(applied)


def build_architecture(cfg, snapshot: str, dtype=torch.bfloat16):
    """Build OpenVLA's architecture without allocating or downloading anything.

    The build happens on ``meta`` under ``accelerate.init_empty_weights``.
    Materialising 6.74 B random bf16 LLM weights before the checkpoint is read
    would cost 13.5 GB on top of the 15 GB checkpoint -- 29 GB against a 24 GB
    A5000. With meta tensors and ``load_state_dict(assign=True)`` the peak is the
    checkpoint itself.

    On the gated backbone: ``config.json`` carries ``hf_llm_id =
    meta-llama/Llama-2-7b-hf``, a manually gated repo that 401s anonymously, and
    it is tempting to assume the vendored code fetches it. It does not -- line
    253 of ``modeling_prismatic.py`` is
    ``AutoModelForCausalLM.from_config(config.text_config, ...)``, so the LLM is
    built from the config and every tensor is then filled from OpenVLA's own
    checkpoint. The field is vestigial for this revision. That is not taken on
    trust: with ``HF_HUB_OFFLINE=1`` a real fetch would raise rather than
    silently succeed, and the post-build assertions require every parameter to be
    on ``meta``, which a download could not satisfy. So ``setup_hf_offline()``
    is part of using this, not an optimisation.
    """
    from accelerate import init_empty_weights

    # Resolve the class outside the meta context: only its *parameters* are
    # placeholders, the class object itself is real.
    cls, patches = resolve_model_class(snapshot)

    with init_empty_weights(include_buffers=False):
        model = cls(cfg)

    llm = getattr(model, "language_model", None)
    if llm is None:
        raise RuntimeError(
            "OpenVLA built no self.language_model, so its forward pass has nothing "
            "to run. The vendored code's structure has changed and this shim is "
            "pointed at the wrong class.")
    n_layers = len(getattr(llm, "model", llm).layers)
    if n_layers != getattr(cfg.text_config, "num_hidden_layers", n_layers):
        raise RuntimeError(
            f"LLM has {n_layers} layers but config says "
            f"{cfg.text_config.num_hidden_layers}; the config and the built model "
            "disagree, so the loaded weights would land in the wrong places")
    n_params = sum(p.numel() for p in model.parameters())
    if n_params < 6e9:
        raise RuntimeError(
            f"architecture has only {n_params / 1e9:.2f}B parameters; a 7B Llama-2 "
            "plus a dual 400M vision backbone should be ~7.5B, so the build lost "
            "a submodule rather than saving memory")
    n_meta = sum(1 for p in model.parameters() if p.is_meta)
    if n_meta == 0:
        raise RuntimeError(
            "init_empty_weights did not take effect -- no parameters are on meta, "
            "so the random LLM weights are real and load_state_dict will need "
            "~29 GB rather than ~15 GB")
    return model, {
        "params": n_params,
        "params_on_meta": n_meta,
        "llm_layers": n_layers,
        "hf_llm_id_used": False,      # verified by the offline + all-meta assertions
        "patches": patches,
    }


def setup_hf_offline() -> None:
    """Refuse to let anything in this path reach the network.

    Not paranoia: the point of the meta build is that the gated
    ``hf_llm_id`` is never fetched, and that claim is only evidence if a fetch
    would have failed. This turns it from an assumption into a test.
    """
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ---------------------------------------------------------------------------
# 4. GenerationMixin.generate is unusable on this checkpoint, and fails silently
# ---------------------------------------------------------------------------
# The measured symptom, on 5.17, from real weights and a real frame:
#
#   generate(..., max_new_tokens=7)  ->  [31872] * 7
#   per-step argmax of its own scores ->  31872 at every step
#   per-step max logit                ->  4.719 at every step, to 3 decimals
#   the model's own last-position logit at step 0 -> 11.062
#   a hand-written greedy loop over the same forward -> [31744, 31999, 31955,
#       31842, 31856, 31871, 31872]
#
# Seven autoregressive steps over seven different prefixes cannot produce an
# identical distribution to three decimals, so generate is scoring a constant.
# The constant's logit scale does not match the model's either, so it is not
# reading the right tensor: the shape of it is OpenVLA's cached-generation
# branch, which is entered on ``input_ids.shape[1] == 1``, runs the LLM on one
# token with no cache, and **ignores pixel_values entirely**.
#
# Why this one matters more than the other three: it does not raise. It returns a
# plausible, finite, in-range 7-D action. Fed to the noise control in
# tests/, that action is bit-identical across every input, and the natural
# reading -- "a 256-bin head collapses to the middle bin on an
# out-of-distribution task, so zero-shot is worthless and fine-tuning is
# mandatory" -- is a fabrication. The bins from a real decode are
# [255, 0, 44, 157, 143, 128, 127], which is not a collapse. Anything concluded
# from this checkpoint's generate() output before this was found is void.
#
# The supported path is :func:`greedy_decode` below, which is the decode a chunk
# head replaces anyway. generate() is left in place and unpatched on purpose:
# patching it to look right would mean reimplementing 5.x's cache handling
# inside someone else's forward signature, and the failure mode of getting that
# subtly wrong is the same silent one.

ACTION_TOKEN_BLOCK = (31536, 32064)   # the 256 bins occupy the top of the vocab

# Token that terminates the instruction in OpenVLA's training format.
# predict_action appends it when the prompt does not already end with it, so a
# decode that skips it is conditioning on a format the model never saw. The
# consequence of getting it wrong is different action bins and no error.
PROMPT_END_TOKEN = 29871


def prepare_prompt_ids(input_ids):
    """Append the training-format terminator, exactly as predict_action does.

    Kept separate from the decode rather than folded into it, because it is a
    property of the *prompt* and getting it wrong is silent: the model answers a
    differently-formatted question and returns a plausible action.
    """
    if not torch.all(input_ids[:, -1] == PROMPT_END_TOKEN):
        pad = torch.full((1, 1), PROMPT_END_TOKEN, dtype=torch.long,
                         device=input_ids.device)
        input_ids = torch.cat((input_ids, pad), dim=1)
    return input_ids


def make_bin_centers(n_action_bins: int = 256):
    """OpenVLA's bin centres, from ``np.linspace(-1, 1, n_bins)`` midpoints.

    This returns **n_bins - 1** values, not n_bins: the 256 numbers are *edges* of
    255 intervals, and the clip in predict_action is therefore to
    ``bin_centers.shape[0] - 1`` = 254. Clipping to 255 instead is an off-by-one
    that reaches past the end of the array.
    """
    import numpy as np
    bins = np.linspace(-1, 1, n_action_bins)
    return (bins[:-1] + bins[1:]) / 2.0


def tokens_to_bins(tokens, vocab_size: int, n_action_bins: int = 256):
    """OpenVLA's own token -> bin map, verbatim from ``predict_action``.

    ``discretized = vocab_size - token``, then ``- 1``, then clipped to
    ``[0, bin_centers.shape[0] - 1]``. Reproduced here so the reference decode is
    comparable to predict_action without a 15 GB model in the loop, and so a
    change to the arithmetic shows up in a cheap test rather than a 7B forward.

    The clip is load-bearing: without it a low token id maps to a bin of ~31744
    and, after unnormalisation, to an action a hundred times outside the
    pretraining dataset's range.
    """
    import numpy as np
    n_centres = len(make_bin_centers(n_action_bins))
    return np.clip(vocab_size - np.asarray(tokens) - 1, a_min=0, a_max=n_centres - 1)


def bins_to_action(bins, q01, q99, mask=None, n_action_bins: int = 256):
    """Bin index -> the pretraining dataset's real units, as predict_action does.

    ``0.5 * (centre + 1) * (q99 - q01) + q01``, with masked dimensions left in
    normalised units. ``q01``/``q99`` come from the checkpoint's ``norm_stats`` and
    describe the *dataset's* action space -- for bridge_orig, end-effector deltas
    in metres, not this sim's m/s. Converting between the two is
    ``adapters.openvla_policy``'s job and is deliberately not guessed here.
    """
    import numpy as np
    norm = make_bin_centers(n_action_bins)[np.asarray(bins)]
    lo = np.asarray(q01, dtype=np.float64)
    hi = np.asarray(q99, dtype=np.float64)
    out = 0.5 * (norm + 1) * (hi - lo) + lo
    return out if mask is None else np.where(np.asarray(mask, dtype=bool), out, norm)


def greedy_decode(model, input_ids, pixel_values, attention_mask=None,
                  n_tokens: int = 7, use_cache: bool = False,
                  add_prompt_end: bool = True, return_logits: bool = False):
    """Greedy-decode ``n_tokens`` action tokens, one forward pass per step.

    This is the ground truth for what this checkpoint does, and the reference
    :func:`check_decode_agreement` holds any faster path against.

    Deliberately implemented as plain repeated forwards over the *full* sequence
    rather than through a cache. The uncached path goes through OpenVLA's
    multimodal branch, which is the branch that actually uses ``pixel_values``;
    the cached branch is the one that ignores them, and reimplementing cache
    plumbing for a 7-token decode buys nothing when a chunk head removes the loop
    altogether. Costs ~7 forwards, which for a 7-token decode is the same order as
    the cached version once the KV transfer is counted.

    ``return_logits`` also returns the full logit row at each step, not just the
    argmax. This matters for measurement, not for decoding: a greedy decode is one
    deterministic sample from a 255-way categorical per dimension, so a per-input
    difference in the *action* conflates a change in what the head wants with a
    change in which bin wins the argmax. Two inputs can have near-identical
    actions and very different logit rows -- notably when a small perturbation
    fails to flip an argmax with a large margin. Measuring "what does the head
    respond to" on the action alone cannot tell those apart; the logits can.
    """
    if add_prompt_end:
        input_ids = prepare_prompt_ids(input_ids)
    ids = input_ids
    if attention_mask is None:
        attention_mask = torch.ones_like(ids)
    out_tokens = []
    logit_rows = []
    past = None
    for _ in range(n_tokens):
        with torch.no_grad():
            o = model(input_ids=ids, attention_mask=attention_mask,
                      pixel_values=pixel_values if past is None else None,
                      past_key_values=past, use_cache=use_cache)
        row = o.logits[0, -1].float()
        logit_rows.append(row.cpu())
        nxt = int(row.argmax())
        out_tokens.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(attention_mask[:, :1])], dim=1)
        if use_cache:
            past = o.past_key_values
    tokens = torch.tensor(out_tokens, dtype=torch.long, device=input_ids.device)
    if return_logits:
        return tokens, torch.stack(logit_rows)      # (n_tokens, vocab)
    return tokens


def action_token_slice(vocab_size: int, n_action_bins: int = 256):
    """Index range of the discretised action tokens inside the vocabulary.

    The bins occupy the top of the vocab, and ``predict_action`` inverts the order
    (``vocab_size - token``), so the block is contiguous and descending. Restricted
    to this block when measuring what the head responds to, because the remaining
    ~31.8k logits are text tokens whose movement says nothing about the action.
    """
    import numpy as np
    n_centres = len(make_bin_centers(n_action_bins))
    lo = vocab_size - 1 - (n_centres - 1)          # smallest token -> top bin
    hi = vocab_size - 1 - 0                        # largest token -> bin 0
    return np.arange(lo, hi + 1)


def check_decode_agreement(model, input_ids, pixel_values, vocab_size: int,
                           n_tokens: int = 7, tol: int = 0) -> dict:
    """Compare a decode path against the uncached ground truth, token by token.

    ``tol`` is the number of positions allowed to differ. Zero is the only
    defensible setting for a deterministic greedy decode: any disagreement means
    one of the two paths is reading something the other is not, and "mostly the
    same" is not a property a decode can have.
    """
    ref = greedy_decode(model, input_ids, pixel_values, n_tokens=n_tokens)
    try:
        gen = model.generate(input_ids, max_new_tokens=n_tokens, do_sample=False)
        got = gen[0, -n_tokens:]
        agreed = int((gen[0, -n_tokens:] == ref).sum())
    except Exception as e:                       # a crash is also a disagreement
        got, agreed = None, 0
        gen_err = f"{type(e).__name__}: {str(e)[:80]}"
    else:
        gen_err = None
    return {
        "reference_tokens": ref.tolist(),
        "reference_bins": tokens_to_bins(ref.tolist(), vocab_size).tolist(),
        "generate_tokens": None if got is None else got.tolist(),
        "generate_bins": (None if got is None
                          else tokens_to_bins(got.tolist(), vocab_size).tolist()),
        "agreeing_positions": agreed,
        "of": n_tokens,
        "generate_error": gen_err,
        "generate_is_usable": agreed == n_tokens - tol,
    }


def load_processor(snapshot: str):
    """OpenVLA's processor, with the moved tokenizer names put back."""
    _patch_tokenization_utils()
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(snapshot, trust_remote_code=True)


def compat_report(snapshot: str) -> dict:
    """What had to change to run a 2024 checkpoint on 2026 transformers.

    Cheap enough to call at import and print into a results file, which is the
    point: the patch list belongs in the paper's reproducibility section, and a
    list that has to be maintained by hand is a list that goes stale.
    """
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(snapshot, trust_remote_code=True)
    cls, patches = resolve_model_class(snapshot)
    return {
        "transformers": transformers.__version__,
        "timm": _timm_version(),
        "pinned_by_checkpoint": "4.40.1",
        "installed_matches_pin": transformers.__version__ == "4.40.1",
        "patches_applied": patches,
        "class_name": cls.__name__,
        "hf_llm_id": getattr(cfg, "hf_llm_id", None),
        "hf_llm_id_used": False,   # from_config, verified by the offline+meta build
        "action_bins": getattr(cfg, "n_action_bins", None),
        "image_sizes": list(getattr(cfg, "image_sizes", []) or []),
        "unnorm_keys": sorted(getattr(cfg, "norm_stats", {}) or {}),
    }


def _timm_version() -> str:
    import timm
    # OpenVLA raises NotImplementedError unless this is an exact match, so it is
    # a real gate, not a warning, and is worth reporting as a hard constraint.
    allowed = {"0.9.10", "0.9.11", "0.9.12", "0.9.16"}
    return f"{timm.__version__}{'' if timm.__version__ in allowed else '  (NOT in OpenVLA gate set)'}"
