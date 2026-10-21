"""One grading entry point for evaluation, training rollouts and the root_success_v2 verifier.

Every MATH answer is scored by both available matchers; `data.scorer` only picks which verdict is primary
(`is_correct`). The other one is kept as a cross-check, so a scorer switch can be audited per example instead of
trusted as a new number (plan §6.3). The KLASS gold is extracted from `\\boxed{answer}` of the dataset's answer
field rather than from the full reference solution.
"""

from __future__ import annotations

from typing import Any, Dict

from .data import is_correct, parse_prediction


def _klass(question: str, gold: Any, text: str) -> Dict[str, Any]:
    from .klass_math import compare_answers, extract_math_answer
    try:
        gold_list = extract_math_answer(question, "\\boxed{" + str(gold) + "}")
        pred_list = extract_math_answer(question, text)
        ok = bool(gold_list) and bool(pred_list) and compare_answers(question, gold_list, pred_list)
        return {"parsed": pred_list, "gold": gold_list, "correct": bool(ok), "error": None}
    except Exception as exc:  # noqa: BLE001 - malformed output must not stop an evaluation
        return {"parsed": None, "gold": None, "correct": False, "error": f"{type(exc).__name__}: {exc}"[:200]}


def _dmax(gold: Any, text: str) -> Dict[str, Any]:
    from .vendor import dmax_val_math as dm
    if dm.MATH_VERIFY_IMPORT_ERROR is not None or dm.SYMPY_IMPORT_ERROR is not None:
        # DMax's scorer silently degrades without these; refuse instead of reporting a weaker number as DMax's
        return {"parsed": None, "correct": False, "error": "math_verify/sympy missing"}
    try:
        with dm.time_limit(8.0):  # DMax's --per-example-timeout default
            res, _, pred = dm.evaluate_example({"answer": str(gold)}, {"answer": text})
        return {"parsed": pred[:3], "correct": bool(res.correct), "method": res.method, "error": None}
    except Exception as exc:  # noqa: BLE001 - DMax counts timeouts/errors as wrong
        return {"parsed": None, "correct": False, "error": f"{type(exc).__name__}"}


def grade(question: str, gold: Any, text: str, task: str, scorer: str = "legacy") -> Dict[str, Any]:
    """-> {parsed_answer, is_correct, scorer, verdicts: {legacy: bool, klass: bool?}, klass_parsed?, klass_error?}"""
    legacy_pred = parse_prediction(text, task)
    legacy_ok = is_correct(legacy_pred, gold, task)
    out: Dict[str, Any] = {"scorer": scorer, "verdicts": {"legacy": legacy_ok}}
    if task == "math":
        k = _klass(question, gold, text)
        out["verdicts"]["klass"] = k["correct"]
        out["klass_parsed"] = k["parsed"]
        if k["error"]:
            out["klass_error"] = k["error"]
        dmx = _dmax(gold, text)
        if dmx["error"] != "math_verify/sympy missing" or scorer == "dmax":
            out["verdicts"]["dmax"] = dmx["correct"]
            out["dmax_parsed"] = dmx["parsed"]
            if dmx["error"]:
                out["dmax_error"] = dmx["error"]
    if scorer == "klass" and task == "math":
        out["parsed_answer"] = out["klass_parsed"]
        out["is_correct"] = out["verdicts"]["klass"]
    elif scorer == "dmax" and task == "math":
        if out.get("dmax_error") == "math_verify/sympy missing":
            raise RuntimeError("data.scorer=dmax needs math_verify and sympy (pip install 'math-verify[antlr4_11_0]')")
        out["parsed_answer"] = out["dmax_parsed"]
        out["is_correct"] = out["verdicts"]["dmax"]
    else:
        out["parsed_answer"] = legacy_pred
        out["is_correct"] = legacy_ok
    return out
