"""DAG-VI training interface."""

from __future__ import annotations

from ..fit import FitConfig, fit

DAGVIFitConfig = FitConfig

__all__ = ["DAGVIFitConfig", "fit"]
