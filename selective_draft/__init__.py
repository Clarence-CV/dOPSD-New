"""Selective Draft OPSD: one early readout adapter -> shared frozen LM head -> single
highest-confidence soft draft -> one residual message at the source position -> native
tail layers -> final commit. See doc/Selective_Draft_OPSD.md and the engineering plan.

Run with `python -m selective_draft.cli --help`.
"""

__version__ = "0.1.0"
