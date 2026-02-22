"""Input compatibility layer for optional Polars support.

All public API functions accept pandas DataFrames.  This module adds
transparent support for Polars DataFrames: when a user passes a
``polars.DataFrame`` (or ``polars.LazyFrame``) it is converted to
``pandas.DataFrame`` at the boundary so that internal code — which
operates on NumPy arrays extracted from pandas — remains unchanged.

Polars is **not** a required dependency.  If it is not installed, the
converter simply passes pandas objects through untouched.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

import pandas as pd

if TYPE_CHECKING:
    import polars as pl

    DataFrameLike: TypeAlias = pd.DataFrame | pl.DataFrame | pl.LazyFrame
else:
    DataFrameLike: TypeAlias = pd.DataFrame

# Runtime detection — avoids a hard dependency on Polars.
try:
    import polars as pl

    _HAS_POLARS = True
except ImportError:
    _HAS_POLARS = False


def _ensure_pandas_df(obj: DataFrameLike, *, name: str = "input") -> pd.DataFrame:
    """Convert *obj* to a :class:`pandas.DataFrame` if necessary.

    Accepted types:
        * ``pandas.DataFrame`` — returned as-is.
        * ``polars.DataFrame`` — converted via ``.to_pandas()``.
        * ``polars.LazyFrame`` — collected then converted.

    Args:
        obj: A pandas or Polars DataFrame (or LazyFrame).
        name: Label used in error messages (e.g. ``"X"`` or ``"y"``).

    Returns:
        A pandas ``DataFrame``.

    Raises:
        TypeError: If *obj* is not a recognised DataFrame type.
    """
    if isinstance(obj, pd.DataFrame):
        return obj

    if _HAS_POLARS:
        if isinstance(obj, pl.LazyFrame):
            return obj.collect().to_pandas()
        if isinstance(obj, pl.DataFrame):
            return obj.to_pandas()

    raise TypeError(
        f"'{name}' must be a pandas DataFrame"
        + (" or Polars DataFrame/LazyFrame" if _HAS_POLARS else "")
        + f", got {type(obj).__name__}."
    )
