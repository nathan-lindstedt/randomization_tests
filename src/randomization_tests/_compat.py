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

from typing import TYPE_CHECKING, Any, TypeAlias

import pandas as pd

if TYPE_CHECKING:
    import polars as pl

    DataFrameLike: TypeAlias = pd.DataFrame | pl.DataFrame | pl.LazyFrame
else:
    DataFrameLike: TypeAlias = Any


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

    # Lazy detection for Polars without eager import at package load
    type_mod = getattr(type(obj), "__module__", "")
    if type_mod.startswith("polars"):
        try:
            import polars as pl

            if isinstance(obj, pl.LazyFrame):
                return obj.collect().to_pandas()
            if isinstance(obj, pl.DataFrame):
                return obj.to_pandas()
        except ImportError:
            pass

    raise TypeError(
        f"'{name}' must be a pandas DataFrame or Polars DataFrame/LazyFrame, got {type(obj).__name__}."
    )
