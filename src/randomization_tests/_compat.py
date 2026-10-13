r"""Input compatibility layer for transparent Polars and DataFrame conversions.

All public API functions accept pandas DataFrames. This module adds transparent
support for Polars DataFrames: when a user passes a ``polars.DataFrame`` or
``polars.LazyFrame``, it is converted to a :class:`pandas.DataFrame` at the boundary
so that internal linear algebra and estimation routines operate on unified data structures.

Primary Components
------------------
* :func:`_ensure_pandas_df`: Boundary converter supporting pandas, Polars DataFrames,
  and Polars LazyFrames.
* :data:`DataFrameLike`: Unified type alias accommodating pandas and Polars tabular inputs.

Conversion Semantics & Optional Dependencies
--------------------------------------------
Polars is an **optional** dependency. If Polars is not installed, the converter
passes pandas objects through untouched with zero runtime overhead and rejects
incompatible input types with informative TypeError messages.

Cross-Module Integration
------------------------
* Inbound: Boundary entrypoints in :func:`~.core.randomization_test_regression`,
  :func:`~.confounders.identify_confounders`, :func:`~._invariance.invariance_test`,
  and :func:`~._knockoffs.knockoff_test`.
* Outbound: Interacts with optional ``polars`` library when present.
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
