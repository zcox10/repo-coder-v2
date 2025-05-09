from __future__ import annotations

from ..checks import _check_type
from ..types import PandasDataFrame


def sample(
    df: PandasDataFrame, rows: int | float, seed: int | None = None
) -> PandasDataFrame:
    _check_type(rows, {int, float})
    if rows >= 1:
        if isinstance(rows, float):
            raise ValueError("must be int if > 1")
        df = df.sample(rows, random_state=seed)
    elif 0 < rows < 1:
        df = df.sample(frac=rows, random_state=seed)
    else:
        raise ValueError("must be > 0")
    df = df.reset_index(drop=True)
    return df
