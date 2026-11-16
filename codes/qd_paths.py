"""Repository-relative paths used by SBIEM-QD research scripts.

This module preserves the directory names expected by the original source while
allowing the repository to run outside the author's original HPC workspace.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RUNS = DATA / "sbiem_runs"
POLES = DATA / "memory_poles"
ANALYSIS = DATA / "analysis"
SUPPLEMENT = DATA / "supplement"

for directory in (RUNS, POLES, ANALYSIS, SUPPLEMENT):
    directory.mkdir(parents=True, exist_ok=True)
