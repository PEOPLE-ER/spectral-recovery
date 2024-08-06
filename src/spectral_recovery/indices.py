"""Methods for computing spectral indices.

Most functions are decorated with `compatible_with` and `requires_bands` decorators,
which check that the input stack is compatible with the function and that the stack
contains the required bands.

Most notably, exports the `compute_indices` function, which computes a stack of
spectral indices from a stack of images and str of index names.

"""

import copy
import json

import xarray as xr
import importlib.resources as pkg_resources
import spyndex as spx

from typing import List, Dict

from spectral_recovery._utils import maintain_rio_attrs

# Set up global index configurations:
#    1. Only support vegetation and burn indices
#    2. Init index-specific constant defaults
SUPPORTED_DOMAINS = ["vegetation", "burn"]
with pkg_resources.open_text(
    "spectral_recovery.resources", "constant_defaults.json"
) as f:
    INDEX_CONSTANT_DEFAULTS = json.load(f)


@maintain_rio_attrs
def compute_indices(
    image_stack: xr.DataArray, indices: list[str], constants: dict = {}, **kwargs
):
    """Compute spectral indices using the spyndex package.


    Parameters
    ----------
    image_stack : xr.DataArray
        stack of images.
    indices : list of str
        list of spectral indices to compute
    constants : dict of flt
        constant and value pairs e.g {"L": 0.5}
    kwargs : dict, optional
        Additional kwargs for wrapped spyndex.computeIndex function.

    Returns
    -------jmk,
        xr.DataArray: stack of images with spectral indices stacked along
        the band dimension.

    """
    if _supported_domain(indices):
        params_dict = _build_params_dict(image_stack)
        constants_dict = _build_constants_dict(indices, constants)

        params_dict = params_dict | constants_dict | kwargs
        index_stack = spx.computeIndex(index=indices, params=params_dict)
        try:
            # rename 'index' dim to 'bands' to match tool's expected dims
            index_stack = index_stack.rename({"index": "band"})
        except ValueError:
            # computeIndex will not return an index dim if only 1 index passed
            index_stack = index_stack.expand_dims(dim={"band": indices})
    return index_stack

def GCI(params_dict: Dict[str, xr.DataArray]) -> xr.DataArray:
    try:
        gci = (params_dict["N"] / params_dict["G"]) - 1
    except KeyError as e:
        raise KeyError(f"Missing '{e.args[0]}' in the parameters for GCI")
    return gci

def TCW(params_dict: Dict[str, xr.DataArray]) -> xr.DataArray:
    try:
        tcw = 0.1511*params_dict["B"] + 0.1973*params_dict["G"] + 0.3283*params_dict["R"] + 0.3407*params_dict["N"] - 0.7117*params_dict["S1"] - 0.4559*params_dict["S2"]
    except KeyError as e:
        raise KeyError(f"Missing '{e.args[0]}' in the parameters for TCW")
    return tcw

def TCG(params_dict: Dict[str, xr.DataArray]) -> xr.DataArray:
    try:
        tcw = -0.2941*params_dict["B"] - 0.243*params_dict["G"] - 0.5424*params_dict["R"] + 0.7276*params_dict["N"] + 0.0713*params_dict["S1"] - 0.1608*params_dict["S2"]
    except KeyError as e:
        raise KeyError(f"Missing '{e.args[0]}' in the parameters for TCG")
    return tcw

def _supported_domain(indices: list[str]):
    """Determine whether indices application domains are supported by tool.

    Parameters
    ----------
    indices : list of str
        list of indices

    Raises
    ------
    ValueError
        If any index has an unsupported application domain.
    """
    for i in indices:
        if spx.indices[i].application_domain not in SUPPORTED_DOMAINS:
            raise ValueError(
                "only application domain 'vegetation' and 'burn' are supported (index"
                f" {i} has application domain '{spx.indices[i].application_domain}')"
            ) from None
    return True


def _build_params_dict(image_stack: xr.DataArray):
    """Build dict of standard names and slices required by computeIndex.

    Slices will be taken along the band dimension of image_stack,
    selecting for each of the standard band/constant names that computeIndex
    accepts. Any name that is not in image_stack will not be included
    in the dictionary.

    Parameters
    ----------
    image_stack : xr.DataArray
        DataArray from which to take slices. Must have a band
        dimension and band coordinates value should be standard names
        for the respective band. For more info, see here:
        https://github.com/awesome-spectral-indices/awesome-spectral-indices

    Returns
    -------
    band_dict : dict
        Dictionary mapping standard names to slice of image_stack.

    """
    standard_names = list(spx.bands)
    params_dict = {}
    for standard in standard_names:
        try:
            band_slice = image_stack.sel(band=standard)
            params_dict[standard] = band_slice
        except KeyError:
            continue

    return params_dict


def _build_constants_dict(indices: List, constants: Dict) -> Dict:
    """Build dict of constants and values for the requested indices.


    Parameters
    ----------
    indices: list of str
        The requested indices.
    constants : dict
        Given (i.e non-default) constants and constant values.

    Returns
    -------
    constants_dict : dict
        Dict of required constants for requested indices. Default
        values are used if a required constant is not in `constants`.

    Raises
    ------
    ValueError
        - If a required constant/value is not provided in `constants`
        and the default value is null/None.
        - If a required constant/value is not provided in `constants`
        and more than one index uses a different default value.

    """
    constants_dict = copy.deepcopy(constants)
    given_constants = list(constants_dict.keys())
    for i in indices:
        try:
            default_constants = INDEX_CONSTANT_DEFAULTS[i]
        except KeyError:
            continue
        for c, v in default_constants["defaults"].items():
            if v is None:
                raise ValueError(
                    f"No default value for {c} available (required by {i}). Please provide a value for {c} with the `constants` param."
                )
            # If more than one index needs the same constant and the constant value wasn't given in `constants` dict
            # (i.e must use defaults), check that the default values match o.w fail
            if c not in given_constants:
                if c in constants_dict:
                    if constants_dict[c] != v:
                        raise ValueError(
                            f"Cannot use default values for constants because {c} has more than one default value between the set of indices. Please provide a value for {c} with the `constants` param."
                        )
                else:
                    # Set constant value to the default value
                    constants_dict[c] = v
    return constants_dict
