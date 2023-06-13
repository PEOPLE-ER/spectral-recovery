import functools
from typing import List, Union, Callable
""" Methods for computing baselines for reference """


def historic_average(stack, reference_range: Union[int, List[int]]):
    """
    Compute the average within a stack over a time range.

    Parameters
    ----------

    Returns
    -------
    """
    if isinstance(reference_range, list):
        if len(reference_range) > 1:
            time_range = f"{reference_range[0]}-{reference_range[1]}"
        else: 
            time_range = str(reference_range[0])
    else:
        time_range = str(reference_range)
    # import dask.array as da
    # print(da.nansum(stack.data).compute())
    ranged_stack = stack.sel(time=time_range)
    baseline_avg = stack.mean(dim=["time", "y", "x"], skipna=True)
    return baseline_avg

