"""Factory for phase-specific metric collectors.

Centralises the importlib + inspect pattern that was duplicated in eval.py
and post_train_assess.py.  Accepts arbitrary kwargs and passes only those
accepted by the collector's ``__init__`` signature.

Public API:
    make_collector(phase_cfg, **kwargs) -> PhaseCollector
"""

from __future__ import annotations

import importlib
import inspect


def make_collector(phase_cfg: object, **kwargs: object) -> object:
    """Resolve a collector class from a PHASE_REGISTRY entry and instantiate it.

    Handles constructor-specific parameters like ``airborne_height_margin``
    (Phase 2) and ``kill_agent_step`` (Phase 3/4) by inspecting the
    ``__init__`` signature and forwarding only accepted kwargs.

    Args:
        phase_cfg: A ``PhaseConfig`` namedtuple from ``PHASE_REGISTRY`` with a
                   ``collector_cls`` string in ``"module.path.ClassName"`` form.
        **kwargs:  Arbitrary keyword arguments.  Only those whose names appear
                   in the collector's ``__init__`` signature are forwarded.

    Returns:
        An instance of the resolved collector class.

    Raises:
        ImportError: If the collector module cannot be imported.
        AttributeError: If the class name is not found in the module.
    """
    module_path, cls_name = phase_cfg.collector_cls.rsplit(".", 1)
    collector_cls = getattr(importlib.import_module(module_path), cls_name)
    sig = inspect.signature(collector_cls.__init__)
    valid = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return collector_cls(**valid)
