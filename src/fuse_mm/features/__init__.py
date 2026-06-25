"""Frozen pretrained-backbone feature extraction (Steps 5-6).

These modules import torch / timm lazily, so importing the top-level `fuse_mm`
package on the data-layer env (Python 3.14, no torch) does not fail.
"""
