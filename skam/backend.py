"""Array backend for the dense hot loops.

``numpy`` (default) needs nothing beyond requirements.txt and is the reference.
``torch`` is optional: it runs parallel retrieval (memory.run_parallel) and the
Dictionary overlaps (vsa.Dictionary) on a GPU in float32. torch is imported
only when this backend is selected.

Everything else stays in numpy on every backend: item generation (so a seed
gives the same items on every backend), HRR encoding, the P x P solve of the
pseudo-inverse rule (float64; C becomes ill-conditioned as alpha -> 1), and
serial dynamics (a per-site loop that a GPU does not speed up).

float32 can flip the sign of a local field within ~1e-7 of zero, so the two
backends agree statistically, not bitwise; trial files carry the backend name
and results from different backends are never mixed.
"""
from __future__ import annotations

from typing import Any

BACKENDS = ("numpy", "torch")
_state: dict[str, Any] = {"name": "numpy", "device": None}


def use(name: str, device: str | None = None) -> None:
    """Select the backend for this process. ``device`` (torch only) defaults
    to cuda, else mps, else cpu."""
    if name not in BACKENDS:
        raise ValueError(f"unknown backend {name!r}; choose from {BACKENDS}")
    if name == "numpy":
        if device is not None:
            raise ValueError("the numpy backend takes no device")
        _state.update(name="numpy", device=None)
        return
    torch = torch_module()
    if device is None:
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available() else "cpu")
    _state.update(name="torch", device=device)


def name() -> str:
    return str(_state["name"])


def device() -> Any:
    return _state["device"]


def torch_module() -> Any:
    try:
        import torch
    except ImportError as e:
        raise ImportError("the torch backend needs PyTorch: pip install torch") from e
    return torch
