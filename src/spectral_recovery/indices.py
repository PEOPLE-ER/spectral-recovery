"""Methods for computing spectral indices."""

import copy
import json
import importlib.resources as pkg_resources
from typing import List, Dict
import xarray as xr
import spyndex as spx
from spectral_recovery.utils import maintain_rio_attrs
from spectral_recovery.config import SUPPORTED_DOMAINS

# Set up global index configurations:
#    1. Only support vegetation and burn indices
#    2. Init index-specific constant defaults
with pkg_resources.open_text(
    "spectral_recovery.resources", "constant_defaults.json"
) as f:
    INDEX_CONSTANT_DEFAULTS = json.load(f)


def gci(params_dict: Dict[str, xr.DataArray]) -> xr.DataArray:
    """Compute the Green Chlorophyll Index (GCI) index"""
    try:
        gci_v = (params_dict["N"] / params_dict["G"]) - 1
    except KeyError as e:
        raise KeyError(f"Missing '{e.args[0]}' in the parameters for GCI") from None
    gci_v = gci_v.expand_dims(dim={"band": ["GCI"]})
    return gci_v


def tcw(params_dict: Dict[str, xr.DataArray]) -> xr.DataArray:
    """Compute the Tasselled Cap Wetness (TCW) index with Landsat 8/9 coeff"""
    try:
        tcw_v = (
            0.1511 * params_dict["B"]
            + 0.1973 * params_dict["G"]
            + 0.3283 * params_dict["R"]
            + 0.3407 * params_dict["N"]
            - 0.7117 * params_dict["S1"]
            - 0.4559 * params_dict["S2"]
        )
    except KeyError as e:
        raise KeyError(f"Missing '{e.args[0]}' in the parameters for TCW") from None
    tcw_v = tcw_v.expand_dims(dim={"band": ["TCW"]})
    return tcw_v


def tcg(params_dict: Dict[str, xr.DataArray]) -> xr.DataArray:
    """Compute the Tasselled Cap Greenness (TCW) index with Landsat 8/9 coeff"""
    try:
        tcg_v = (
            -0.2941 * params_dict["B"]
            - 0.243 * params_dict["G"]
            - 0.5424 * params_dict["R"]
            + 0.7276 * params_dict["N"]
            + 0.0713 * params_dict["S1"]
            - 0.1608 * params_dict["S2"]
        )
    except KeyError as e:
        raise KeyError(f"Missing '{e.args[0]}' in the parameters for TCG") from None
    tcg_v = tcg_v.expand_dims(dim={"band": ["TCG"]})
    return tcg_v


SR_REC_IDXS = {"GCI": gci, "TCW": tcw, "TCG": tcg}

@maintain_rio_attrs
def compute_indices(
    timeseries_data: xr.DataArray, indices: list[str], constants: dict = None, **kwargs
):
    """Compute spectral indices.

    Compute spectral indices using the spyndex package or 
    manually implemented indices.

    Parameters
    ----------
    timeseries_data : xr.DataArray
        The timeseries of spectral bands to compute indices with.
        Must contain band, time, y, and x dimensions.
    indices : list of str
        list of spectral indices to compute
    constants : dict of flt, optional
        constant and value pairs e.g {"L": 0.5}
    kwargs : dict, optional
        Additional kwargs for wrapped spyndex.computeIndex function.

    Returns
    -------
    index_stack : xarray.DataArray
        DataArray of spectral indices, with spectral
        indices stacked along the band coordinate dimension.

    """
    spx_indices, sr_indices = _split_indices_by_source(indices)
    params_dict = _build_params_dict(timeseries_data)
    spx_and_sr_outputs = []
    if spx_indices:
        # Compute indexes implemented in spx (spyndex)
        if _supported_domain(spx_indices):
            constants_dict = _build_constants_dict(spx_indices, constants)
            params_dict = params_dict | constants_dict | kwargs
            spx_index_stack = spx.computeIndex(index=spx_indices, params=params_dict)
            # Rename index to band or expand if only one index was computed
            spx_index_stack = (spx_index_stack.rename({"index": "band"})
                   if "index" in spx_index_stack.dims
                   else spx_index_stack.expand_dims(dim={"band": spx_indices}))
            spx_and_sr_outputs.append(spx_index_stack)
    if sr_indices:
        # Compute indexes implemented in sr (spectral-recovery)
        sr_idxs_outputs = []
        for i in sr_indices:
            sr_idxs_outputs.append(SR_REC_IDXS[i](params_dict=params_dict))
        sr_index_stack = xr.concat(sr_idxs_outputs, dim="band")
        spx_and_sr_outputs.append(sr_index_stack)
    print(spx_and_sr_outputs)
    # concatenate spx and sr indexes into one DataArray
    index_stack = xr.concat(spx_and_sr_outputs, dim="band")
    return index_stack


def _split_indices_by_source(indices: List[str]) -> tuple[List[str], List[str]]:
    """Split a list of indices by their source of computation: spyndex or spectral-recovery"""
    spx_list = []
    sr_list = []
    spx_indices = list(spx.indices)
    sr_indices = list(SR_REC_IDXS.keys())
    for i in indices:
        if i in sr_indices:
            sr_list.append(i)
        elif i in spx_indices:
            spx_list.append(i)
        else:
            raise ValueError(f"'{i}' is not a supported index")
    return (spx_list, sr_list)


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
    if constants:
        constants_dict = copy.deepcopy(constants)
    else:
        constants_dict = {}
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
            # If more than one index needs the same constant and the constant value
            # wasn't given in `constants` dict then check that the default values
            # match. If they don't, bail out.
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
