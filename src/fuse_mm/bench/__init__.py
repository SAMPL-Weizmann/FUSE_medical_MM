"""Benchmark suite: reproduce the FUSE-paper Table 1 on our 3 sets (S_L/S_U/Test).

Per-verifier stats + a methods x sets table (accuracy + balanced accuracy) for
unsupervised (Majority Vote, Naive Ensemble, FUSE), supervised (OBV, Weaver,
Logistic, Gaussian NB) methods, and an Oracle ceiling. Run once with lambda=0
(no TCI) and later with lambda != 0, then compare.
"""

from .run import run_benchmark

__all__ = ["run_benchmark"]
