"""FUSE stage-1: joint training of all verifiers with L_CE + alpha*L_NCL + lambda*L_TCI.

See docs/FUSE_TCI_spec.md. Stage-2 (estimation) and stage-3 (inference) build on
the verifier score matrices this stage saves.
"""

from .config import load_fuse_config
from .losses import tci_loss, ncl_loss, ce_loss
from .model import FuseVerifiers

__all__ = ["load_fuse_config", "tci_loss", "ncl_loss", "ce_loss", "FuseVerifiers"]
