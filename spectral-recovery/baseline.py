""" Methods for computing baselines for reference """
from typing import List, Union

def historic_average(stack, reference_range: Union[int, List[int]]):
    """
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
    ranged_stack = stack.sel(time=time_range)
    baseline_avg = ranged_stack.mean(skipna=True)
    return baseline_avg

# @historic_average
# def with_variation(variation_method: float):
#     """
#     Parameters
#     ----------

#     Returns
#     -------
#     """
#     pass
