"""Model entry points retained for clean ANN and optional SNN experiments."""

from .videomamba_clean import CleanVideoMamba, create_videomamba_small_clean
from .videomamba_trainable_snn import TrainableVideoMambaSNN, create_videomamba_small_trainable_snn

__all__ = [
    "CleanVideoMamba",
    "SpikMambaFixed",
    "TrainableVideoMambaSNN",
    "create_videomamba_small_clean",
    "create_videomamba_small_trainable_snn",
    "spikmamba_fixed",
]


def __getattr__(name):
    if name in {"SpikMambaFixed", "spikmamba_fixed"}:
        from .videomamba_spik_baseline_1_fixed import SpikMambaFixed, spikmamba_fixed

        return {"SpikMambaFixed": SpikMambaFixed, "spikmamba_fixed": spikmamba_fixed}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
