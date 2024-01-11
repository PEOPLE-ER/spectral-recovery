import xarray as xr


# NOTE: SAME_XR is a hacky solution to get around "ValueErrors" that
# are thrown if you try to assert a mocked function was called with
# xarray's DataArray. Not sure if this indicates some bad design
# with the package ... but for now it stays to ensure correctness.
# Solution from: https://stackoverflow.com/questions/44640717
class SAME_XR:
    def __init__(self, xr: xr.DataArray):
        self.xr = xr

    def __eq__(self, other):
        return isinstance(other, xr.DataArray) and other.equals(self.xr)

    def __repr__(self):
        return repr(self.xr)
