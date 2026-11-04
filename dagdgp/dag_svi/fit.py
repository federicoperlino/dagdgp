"""DAG-SVI training interface."""

from __future__ import annotations

from ..fit import FitConfig, fit

DAGSVIFitConfig = FitConfig

__all__ = ["DAGSVIFitConfig", "fit"]
