import functools
from enums import Indices

# def maintain_spatial_attrs(func):
#     """ A wrapper/decorator for maintaining spatial attributes between
#     simple stack-based computaions (e.g band math for indices).
#     """
#     @functools.wraps(func)
#     def wrapper_maintain_spatial_attrs(stack):
#         print(stack.attrs)
#         stored_attrs = stack.attrs
#         indice = func(stack)
#         indice.attrs.update(stored_attrs)
#         return indice
#     return wrapper_maintain_spatial_attrs

# If the indice functions are not decorated then the resulting indice
# struct will lose spatial rioxarray derived attributes like 
# AREA_OR_POINT, scale_offset, etc. but spatial_ref information should
# be maintained regardless of whether funcitons are decorated or not.
# @maintain_spatial_attrs
def ndvi(stack):
    nir = stack.sel(band="nir") 
    red = stack.sel(band="red")
    ndvi = (nir - red) / (nir + red)
    return ndvi

# @maintain_spatial_attrs
def nbr(stack):
    nir = stack.sel(band="nir") 
    swir = stack.sel(band="swir")
    nbr = (nir - swir) / (nir + swir)
    return nbr
     
indices_map = {
    Indices.NDVI: ndvi,
    Indices.NBR: nbr,
}
