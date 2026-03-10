"""
Engine: canonical models, normalize, state reducer, compute, and diagnostics.

Pure, testable pipeline:
  raw feed -> normalize -> Frame
  Frame + State + Config -> reducer -> State'
  State' + Config -> compute -> Derived
  Derived + State' -> HistoryPoint
"""

from engine.models import Config, Frame, State, Derived, HistoryPoint

__all__ = ["Config", "Frame", "State", "Derived", "HistoryPoint"]
