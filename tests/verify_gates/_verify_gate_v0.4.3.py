"""Diagnostic sweep: test all method × family combinations and report results.

Run before and after compatibility fixes to confirm the matrix changes.
Usage:
    python verify_gates/_verify_compat_matrix.py
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

# Suppress unrelated warnings during the sweep
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)

METHODS = [
    "ter_braak",
    "kennedy",
    "kennedy_joint",
    "freedman_lane",
    "freedman_lane_joint",
    "manly",
    "manly_joint",
    "score",
    "score_joint",
    "score_exact",
]

FAMILIES = [
    "linear",
    "logistic",
    "poisson",
    "negative_binomial",
    "ordinal",
    "multinomial",
    "linear_mixed",
    "logistic_mixed",
    "poisson_mixed",
]

_rng = np.random.default_rng(42)


def _make_data(family: str):
    """Minimal dataset for the given family."""
    n, p = 30, 3
    X_np = _rng.standard_normal((n, p))
    # Include confounder as a column in X (confounders= takes column names)
    X = pd.DataFrame(X_np, columns=[f"x{i}" for i in range(p)])
    X["c0"] = _rng.standard_normal(n)

    if family == "linear":
        y_arr = X_np @ _rng.standard_normal(p) + _rng.standard_normal(n)
    elif family == "logistic":
        eta = X_np @ _rng.standard_normal(p)
        y_arr = (1 / (1 + np.exp(-eta)) > 0.5).astype(int)
    elif family in ("poisson", "poisson_mixed"):
        eta = X_np[:, 0] * 0.3
        y_arr = _rng.poisson(np.exp(eta))
    elif family in ("negative_binomial",):
        eta = X_np[:, 0] * 0.3
        y_arr = _rng.negative_binomial(5, 0.5, size=n)
    elif family in ("ordinal", "multinomial"):
        # 4 ordered/unordered categories
        y_arr = _rng.integers(0, 4, size=n)
    elif family in ("linear_mixed", "logistic_mixed"):
        groups = np.repeat(np.arange(6), 5)
        if family == "linear_mixed":
            y_arr = X_np @ _rng.standard_normal(p) + _rng.standard_normal(n)
        else:
            eta = X_np @ _rng.standard_normal(p)
            y_arr = (1 / (1 + np.exp(-eta)) > 0.5).astype(int)
        y = pd.DataFrame({"y": y_arr})
        return X, y, None, groups
    else:
        y_arr = X_np @ _rng.standard_normal(p) + _rng.standard_normal(n)

    y = pd.DataFrame({"y": y_arr})
    return X, y, None, None


def _run_one(method: str, family: str) -> tuple[str, str]:
    """Run one combination, return (status, detail)."""
    from randomization_tests import randomization_test_regression

    try:
        X, y, _, groups = _make_data(family)
    except Exception as e:
        return "DATA_ERROR", str(e)[:60]

    kwargs: dict = dict(
        n_randomizations=20,
        random_state=0,
        method=method,
        family=family,
        confounders=["c0"],
    )
    if groups is not None:
        kwargs["groups"] = groups

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            randomization_test_regression(X, y, **kwargs)
        return "OK", ""
    except ValueError as e:
        return "ValueError", str(e)[:120]
    except NotImplementedError as e:
        return "NotImplementedError", str(e)[:120]
    except Exception as e:
        return f"{type(e).__name__}", str(e)[:120]


def main():
    col_w = max(len(f) for f in FAMILIES) + 2
    method_w = max(len(m) for m in METHODS) + 2
    header = f"{'':>{method_w}}" + "".join(f"{f:>{col_w}}" for f in FAMILIES)
    print(header)
    print("-" * len(header))

    for method in METHODS:
        row = f"{method:>{method_w}}"
        for family in FAMILIES:
            status, detail = _run_one(method, family)
            symbol = (
                " ✓ "
                if status == "OK"
                else " V "
                if status == "ValueError"
                else " N "
                if status == "NotImplementedError"
                else " ? "
            )
            row += f"{symbol:>{col_w}}"
        print(row)

    print()
    print(
        "Legend: ✓=OK  V=ValueError (guarded)  N=NotImplementedError (unguarded crash)  ?=other"
    )
    print()

    # Detailed errors for non-OK, non-ValueError cases (unguarded crashes):
    crashes = []
    for method in METHODS:
        for family in FAMILIES:
            status, detail = _run_one(method, family)
            if status not in ("OK", "ValueError"):
                crashes.append((method, family, status, detail))

    if crashes:
        print("=== UNGUARDED CRASHES (need early guards) ===")
        for m, f, s, d in crashes:
            print(f"  {m:25s} + {f:20s}: {s}: {d}")
    else:
        print("=== No unguarded crashes. ===")

    # Print all ValueError details to understand what's being guarded:
    print()
    print("=== ALL ValueError details ===")
    for method in METHODS:
        for family in FAMILIES:
            status, detail = _run_one(method, family)
            if status == "ValueError":
                print(f"  {method:25s} + {family:20s}: {detail}")


if __name__ == "__main__":
    main()
