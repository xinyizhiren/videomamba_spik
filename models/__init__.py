"""Model entry points retained for clean ANN and optional SNN experiments."""

from .videomamba_clean import CleanVideoMamba, create_videomamba_small_clean

__all__ = [
    "CleanVideoMamba",
    "SpikMambaFixed",
    "create_videomamba_small_clean",
    "spikmamba_fixed",
]


def __getattr__(name):
    if name in {"SpikMambaFixed", "spikmamba_fixed"}:
        from .videomamba_spik_baseline_1_fixed import SpikMambaFixed, spikmamba_fixed

        return {"SpikMambaFixed": SpikMambaFixed, "spikmamba_fixed": spikmamba_fixed}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
