"""Prompt, gold extraction, sub-splits, scoring."""
import json
import os

from _helpers import ROOT, run_all
from selective_draft.data import DMAX_BOXED, SUBSPLITS, build_messages, last_boxed, subsplit_ids
from selective_draft.scoring import grade


def test_dmax_prompt_is_one_user_turn_with_the_eval_harness_wording():
    msgs = build_messages("What is 1+1?", "dmax_boxed")
    assert msgs == [{"role": "user", "content": "What is 1+1?\nLet's think step by step, and put your final answer within \\boxed{}."}]
    assert DMAX_BOXED.startswith("\n")


def test_last_boxed_handles_nesting_and_spaced_form():
    assert last_boxed(r"so \boxed{\frac{3}{4}}.") == r"\frac{3}{4}"
    assert last_boxed(r"first \boxed{1} then \boxed{x^{2}}") == "x^{2}"
    assert last_boxed(r"\boxed 5$ end") == "5"
    assert last_boxed("no answer") is None and last_boxed(r"\boxed{unclosed") is None


def test_subsplits_follow_the_documented_slices_and_are_disjoint():
    with open(os.path.join(ROOT, "configs", "math_split_manifest_v1.json")) as f:
        man = json.load(f)
    seen = set()
    for parent, (_, slices) in SUBSPLITS.items():
        for name, size in slices:
            ids = subsplit_ids(man["splits"][parent], parent, name)
            assert len(ids) == size and not (seen & set(ids)), (parent, name)
            seen |= set(ids)
    inside = seen & (set(man["splits"]["train"]) | set(man["splits"]["analysis"]))
    assert len(inside) == 1012  # 256 + 256 readout-train, 128 + 64*2 + 32*2 + 128 + 52 analysis


def test_toy_scorer_compares_the_last_number():
    assert grade(7.0, "so 3 + 4 = 7", "fake") == {"parsed_answer": 7.0, "is_correct": True}
    assert grade(7.0, "no number here", "fake") == {"parsed_answer": None, "is_correct": False}


def test_dmax_math_scorer_when_its_dependencies_are_installed():
    try:
        import math_verify  # noqa: F401
        import sympy  # noqa: F401
    except ImportError:
        print("    (skipped: math_verify not installed locally; runs on the cluster)")
        return
    assert grade("\\frac{3}{4}", "so the answer is \\boxed{\\dfrac{3}{4}}.", "math")["is_correct"]
    assert not grade("5", "the answer is \\boxed{6}", "math")["is_correct"]


if __name__ == "__main__":
    run_all(globals())
