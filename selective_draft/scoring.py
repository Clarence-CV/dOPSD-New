"""One grading entry point for evaluation, training rollouts and the root-success verifier.

MATH is scored with DMax's own scorer (dInfer/evaluations/val_math.py, vendored unmodified): multi-stage matching with
math_verify and sympy, 8 s per-example timeout, gold = the dataset `answer` field. The toy task compares the last
number in the text.
"""

from __future__ import annotations

import re
from typing import Any, Dict


def _dmax(gold: Any, text: str) -> Dict[str, Any]:
    from .vendor import dmax_val_math as dm
    if dm.MATH_VERIFY_IMPORT_ERROR is not None or dm.SYMPY_IMPORT_ERROR is not None:
        # DMax's scorer silently degrades without these; refuse instead of reporting a weaker number as DMax's
        raise RuntimeError("DMax scoring needs math_verify and sympy: pip install 'math-verify[antlr4_11_0]'")
    try:
        with dm.time_limit(8.0):  # DMax's --per-example-timeout default
            res, _, pred = dm.evaluate_example({"answer": str(gold)}, {"answer": text})
        return {"parsed_answer": pred[:3], "is_correct": bool(res.correct)}
    except Exception:  # noqa: BLE001 - DMax counts timeouts and errors as wrong
        return {"parsed_answer": None, "is_correct": False}


def _toy(gold: Any, text: str) -> Dict[str, Any]:
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not nums:
        return {"parsed_answer": None, "is_correct": False}
    return {"parsed_answer": float(nums[-1]), "is_correct": abs(float(nums[-1]) - float(gold)) < 1e-6}


def grade(gold: Any, text: str, task: str) -> Dict[str, Any]:
    """-> {parsed_answer, is_correct}"""
    return _dmax(gold, text) if task == "math" else _toy(gold, text)
