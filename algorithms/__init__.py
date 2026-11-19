"""Các thuật toán tối ưu TDVRP của dự án."""

from pathlib import Path

# Các module legacy đã được gom vào algorithms/algorithm, trong khi nhiều
# experiment/baseline vẫn import theo tên algorithms.<module> cũ.
_legacy_modules = Path(__file__).resolve().parent / "algorithm"
if _legacy_modules.is_dir():
    __path__.append(str(_legacy_modules))

from .rapo_2709 import RAPO2709, RAPO2709Config, RAPO2709Result, optimize


MAIN_ALGORITHM = RAPO2709
MAIN_ALGORITHM_NAME = RAPO2709.name

__all__ = [
    "MAIN_ALGORITHM",
    "MAIN_ALGORITHM_NAME",
    "RAPO2709",
    "RAPO2709Config",
    "RAPO2709Result",
    "optimize",
]
