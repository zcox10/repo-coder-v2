from __future__ import annotations

from ..checks import _check_type
from ..types import Column, Func, PandasDataFrame, PandasGroupedFrame


def rollup(
    df: PandasDataFrame | PandasGroupedFrame,
    over: dict[Column, tuple[Column, Func]],
) -> PandasDataFrame:
    _check_type(over, dict)
    if isinstance(df, PandasGroupedFrame):
        groups = set(df.grouper.names)  # type: ignore
        keys = set(over.keys())
        if groups.intersection(keys):
            raise KeyError("unable to overwrite group keys")
        df = df.agg(**over)
        df = df.reset_index(drop=True)
    else:
        df = df.agg(**over)  # type: ignore
        df = df.T  # type: ignore
        df = df.reset_index(drop=True)  # type: ignore
        df = df.fillna(method="ffill")  # type: ignore
        df = df.fillna(method="bfill")  # type: ignore
        df = df.head(1)  # type: ignore
    return df
