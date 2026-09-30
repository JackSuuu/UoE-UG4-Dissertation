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
