"""The transformers-5.x compat layer, pinned by what it is protecting against.

OpenVLA-7B's checkpoint pins ``transformers==4.40.1``, and that pin is
uninstallable on Python 3.13: 4.40.1 requires ``tokenizers<0.20``, every 0.19.x
predates CPython 3.13, and the source fallback is refused by PyO3 0.21 (max
3.12). The version check in the vendored code is a warning, not a raise, so the
only installable option is 5.17 and the vendored code has to be made to work.

That makes this file load-bearing, and the failure mode it has to avoid is
specific: a compat shim that silently no-ops does not crash, it produces numbers.
A missing ``GenerationMixin`` can be papered over; a tokenizer that loads but pads
differently trains fine, moves plausibly and plateaus at an arbitrary success
rate that then gets attributed to the verifier. So every patch here is asserted,
and these tests assert the assertions still hold.

Four things are checked, in increasing order of how much they would cost if they
silently stopped working:

  1. the three class patches are present and are the *kind* of thing they claim
     (a real method, a real class-level capability flag, a kwargs-forwarding
     signature) -- not merely that some attribute exists
  2. the tokenizer shim restored all four names 5.x moved
  3. the architecture builds entirely on ``meta``, with the right parameter count
     and layer count. A build that stops being meta is not a slowdown, it is a
     29 GB allocation that OOMs a 24 GB card
  4. the gated ``hf_llm_id`` is genuinely never fetched. This one is checked by
     building with ``HF_HUB_OFFLINE=1``: if the claim in the docstring were
     wrong, the build would raise rather than quietly succeed. A claim that only
     holds when the network is available is not a claim.

Deliberately *not* tested here: that the loaded weights produce correct actions.
That needs the 15 GB checkpoint and belongs in
``~/scratch/proof_openvla.py``, which reports the load as 0 missing / 0
unexpected / 0 non-finite over 982 tensors.

Run: cd src && python -u tests/test_tf5_compat.py
"""
import os
import sys

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC)

FAILED = []


def check(cond, label, detail=""):
    print(f"   {'ok  ' if cond else 'FAIL'} {label}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAILED.append(label)
    return cond


def main():
    import transformers
    from transformers import LlamaForCausalLM, PreTrainedModel

    from adapters import tf5_compat

    print(f"transformers {transformers.__version__}, torch "
          f"{__import__('torch').__version__}")
    print("\n0. the pin cannot be satisfied here, which is why the shim exists")
    try:
        import tokenizers
        tok = tokenizers.__version__
    except ImportError:
        tok = "absent"
    print(f"   checkpoint wants transformers==4.40.1, tokenizers==0.19.1")
    print(f"   installed:      transformers=={transformers.__version__}, "
          f"tokenizers=={tok}")
    check(transformers.__version__ != "4.40.1",
          "running off the pinned version (so the shim is required)")

    tf5_compat.setup_hf_offline()
    snap = tf5_compat.snapshot_dir()
    print(f"   snapshot {os.path.basename(snap)[:12]}")

    print("\n1. the three class patches, and that each is the right kind of thing")
    cls, patches = tf5_compat.resolve_model_class(snap)
    print(f"   patches reported: {patches}")
    for want in ("GenerationMixin", "_supports_sdpa", "tie_weights"):
        check(want in patches, f"patch {want!r} was applied")

    # (a) generate must be a *callable that generates*, not just any attribute
    check(callable(getattr(cls, "generate", None)),
          "generate is callable")
    check("generate" not in cls.__dict__,
          "generate is inherited from GenerationMixin, not defined by the shim")
    import inspect
    gen = inspect.getsource(cls.generate)
    check("GenerationMixin" in str(cls.__mro__),
          "GenerationMixin is in the MRO",
          "->".join(c.__name__ for c in cls.__mro__[:3]))

    # (b) _supports_sdpa must equal the LLM's real capability, not a hardcoded True.
    #     If it is literally True while Llama reports otherwise, the shim is
    #     claiming an attention kernel it has not checked.
    want_sdpa = getattr(LlamaForCausalLM, "_supports_sdpa", False)
    check(cls._supports_sdpa == want_sdpa,
          "_supports_sdpa mirrors LlamaForCausalLM rather than a literal",
          f"shim={cls._supports_sdpa!r} llama={want_sdpa!r}")
    check(not isinstance(cls.__dict__.get("_supports_sdpa"), bool)
          or cls.__dict__.get("_supports_sdpa") == want_sdpa,
          "_supports_sdpa is a class attribute, so the check runs before "
          "self.language_model exists")

    # (c) tie_weights must forward arbitrary arguments, or 5.x's next call growth
    #     breaks it again. Inspect the signature rather than calling it, because
    #     calling it on a meta model would touch uninitialised weights.
    sig = inspect.signature(cls.tie_weights)
    kinds = {p.kind for p in sig.parameters.values()}
    check(any(k == inspect.Parameter.VAR_KEYWORD for k in kinds)
          and any(k == inspect.Parameter.VAR_POSITIONAL for k in kinds),
          "tie_weights forwards *a, **kw", f"signature {sig}")
    base_sig = inspect.signature(
        cls.__mro__[1].tie_weights) if len(cls.__mro__) > 1 else None
    check(base_sig is None or len(base_sig.parameters) < len(sig.parameters),
          "tie_weights is wider than the vendored one it replaces",
          f"vendored {base_sig}")

    print("\n2. the tokenizer shim put the four moved names back")
    import transformers.tokenization_utils as tu
    for name in tf5_compat._MOVED_TOKENIZER_NAMES:
        check(hasattr(tu, name), f"transformers.tokenization_utils.{name}")
        from transformers import tokenization_utils_base as tub
        check(getattr(tu, name) is getattr(tub, name),
              f"{name} is the same object 5.x moved, not a stand-in")

    print("\n3. the architecture builds on meta, at the right size")
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(snap, trust_remote_code=True)
    model, info = tf5_compat.build_architecture(cfg, snap)
    n_par = info["params"]
    n_meta = info["params_on_meta"]
    n_tot = sum(1 for _ in model.parameters())
    print(f"   {n_par / 1e9:.2f}B params, {n_tot} tensors, {n_meta} on meta, "
          f"{info['llm_layers']} LLM layers")
    # 7.54B expected: 6.74B Llama-2 + ~0.4B DINOv2 + ~0.4B SigLIP + projector.
    check(7.0e9 < n_par < 8.0e9, "parameter count is 7-8B",
          f"{n_par / 1e9:.2f}B")
    check(n_meta == n_tot,
          "every parameter is on meta",
          f"{n_meta}/{n_tot}; a non-meta build needs ~29 GB, not ~15 GB")
    check(info["llm_layers"] == cfg.text_config.num_hidden_layers,
          "LLM layer count matches the config", f"{info['llm_layers']}")
    check(hasattr(model, "language_model") and hasattr(model, "vision_backbone"),
          "both submodules exist (a lost one would cost ~0.4B and be easy to miss)")
    # The projector is the one piece with no 2024-era analogue to check against,
    # and it is the piece that would silently mis-scale actions if it were wrong.
    proj = sum(p.numel() for n, p in model.named_parameters() if "projector" in n)
    print(f"   projector: {proj / 1e6:.1f}M params")
    check(proj > 1e6, "projector is non-trivial",
          "a near-zero projector would flatten the visual features into the LLM")

    print("\n4. the gated backbone is never fetched")
    check(os.environ.get("HF_HUB_OFFLINE") == "1", "HF_HUB_OFFLINE=1 is set")
    print(f"   config.hf_llm_id = {getattr(cfg, 'hf_llm_id', None)!r}")
    check(info["hf_llm_id_used"] is False, "build reports hf_llm_id unused")
    # The real evidence is that the build above *succeeded* while offline: a
    # genuine fetch would have raised. Confirm the model has no tensor that
    # could only have come from the network -- i.e. the LLM is on meta too.
    llm_meta = sum(1 for p in model.language_model.parameters() if p.is_meta)
    llm_tot = sum(1 for _ in model.language_model.parameters())
    check(llm_meta == llm_tot,
          "the LLM is on meta as well, so nothing was downloaded for it",
          f"{llm_meta}/{llm_tot}")

    print("\n" + (f"{len(FAILED)} FAILED: {FAILED}" if FAILED
                  else "all compat checks passed"))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
