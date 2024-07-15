import pytest

import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd

from shapely import Polygon

from unittest.mock import patch, MagicMock
from xarray.testing import assert_equal
from numpy.testing import assert_array_equal


from spectral_recovery.targets.historic import median, window, _check_reference_years

def test_invalid_scale_throws_value_error():
    with pytest.raises(ValueError):
        valid_poly = Polygon([(3.5, 3.5), (3.5, 5.5), (5.5, 5.5), (5.5, 3.5)])
        test_data = [
            [
                [[1.0]],  # Time 1, band 1
                [[1.0]],  # Time 1, band 2
            ],
            [
                [[3.0]],  # Time 2, band 1
                [[5.0]],  # Time 2, band 2
            ],
        ]
        test_stack = xr.DataArray(
            test_data,
            dims=["time", "band", "y", "x"],
            coords={
                "time": pd.date_range("2010", "2011", freq="YS"),
                "y": [0, 1],
                "x": [0, 1]
            },
        )
        median(polygon=valid_poly, timeseries_data=test_stack, reference_years={0: [2010, 2011]}, scale="not_a_scale")
    
class TestMedianPolygonScale:

    @pytest.fixture()
    def valid_gpd(self):
        polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
        valid_gpd = gpd.GeoDataFrame(geometry=[polygon]).set_crs("EPSG:4326")
        return valid_gpd

    @patch("geopandas.read_file")
    def test_str_polygon_reads_polygon(self, read_mock, valid_gpd):
        read_mock.return_value = valid_gpd
        test_data = [
            [
                [[1.0]],  # Time 1, band 1
                [[1.0]],  # Time 1, band 2
            ],
            [
                [[3.0]],  # Time 2, band 1
                [[5.0]],  # Time 2, band 2
            ],
        ]
        valid_stack = xr.DataArray(
            test_data,
            dims=["time", "band", "y", "x"],
            coords={
                "time": pd.date_range("2010", "2011", freq="YS"),
                "y": [1],
                "x": [1]
            },
        ).rio.write_crs("EPSG:4326", inplace=True)
        _ = median(restoration_sites="pathy", timeseries_data=valid_stack, reference_years={0: [2010, 2011]}, scale="polygon")

        read_mock.assert_called_once()
        assert read_mock.call_args.args[0] == "pathy"


    def test_no_nan_returns_avg_over_time(self, valid_gpd):
        test_data = [
            [
                [[1.0]],  # Time 1, band 1
                [[1.0]],  # Time 1, band 2
            ],
            [
                [[3.0]],  # Time 2, band 1
                [[5.0]],  # Time 2, band 2
            ],
        ]
        test_stack = xr.DataArray(
            test_data,
            dims=["time", "band", "y", "x"],
            coords={
                "time": pd.date_range("2010", "2011", freq="YS"),
                "y": [1],
                "x": [1]
            },
        ).rio.write_crs("EPSG:4326", inplace=True)
        expected_data = [2.0, 3.0]
        expected_dict = {0: xr.DataArray(
            expected_data,
            dims=["band"],
            coords={
                "band": [0, 1]
            },
        ).rio.write_crs("EPSG:4326", inplace=True)
        }

        out_targets = median(restoration_sites=valid_gpd, timeseries_data=test_stack, reference_years={0: [2010, 2011]}, scale="polygon")

        assert len(out_targets.keys()) == 1
        assert list(out_targets.keys())[0] == 0
        assert_equal(expected_dict[0], out_targets[0])

    def test_odd_time_dim_returns_median(self, valid_gpd):
        test_data = [
            [
                [[1.0]],  # Time 1, band 1
                [[1.0]],  # Time 1, band 2
            ],
            [
                [[3.0]],  # Time 2, band 1
                [[5.0]],  # Time 2, band 2
            ],
            [
                [[9.0]],  # Time 3, band 1
                [[7.0]],  # Time 3, band 2
            ],
        ]
        test_stack = xr.DataArray(
            test_data,
            dims=["time", "band", "y", "x"],
            coords={
                "time": pd.date_range("2010", "2012", freq="YS"),
                "y": [1],
                "x": [1]
            },
        ).rio.write_crs("EPSG:4326", inplace=True)
        expected_data = [3.0, 5.0]
        expected_dict = {0: xr.DataArray(
            expected_data,
            dims=["band"],
            coords={
                "band": [0, 1],
            },
        ).rio.write_crs("EPSG:4326", inplace=True)
        }

        out_dict = median(restoration_sites=valid_gpd, timeseries_data=test_stack, reference_years={0: [2010,2012]}, scale="polygon")

        assert len(out_dict.keys()) == 1
        assert list(out_dict.keys())[0] == 0
        assert_equal(expected_dict[0], out_dict[0])

    def test_nan_timeseries_is_nan(self, valid_gpd):
        test_data = [
            [
                [[np.nan]],
                [[3.0]],
            ],
            [
                [[np.nan]],
                [[5.0]],
            ],
        ]
        test_stack = xr.DataArray(
            test_data,
            dims=["time", "band", "y", "x"],
            coords={
                "time": pd.date_range("2010", "2011", freq="YS"),
                "y": [1],
                "x": [1]
            },
        ).rio.write_crs("EPSG:4326", inplace=True)
        expected_data = [np.nan, 4.0]
        expected_dict = {0: xr.DataArray(
            expected_data,
            dims=["band"],
            coords={
                "band": [0, 1]
            },
        ).rio.write_crs("EPSG:4326", inplace=True)
        }
        out_dict = median(restoration_sites=valid_gpd, timeseries_data=test_stack, reference_years={0: [2010, 2011]}, scale="polygon")

        assert len(out_dict.keys()) == 1
        assert list(out_dict.keys())[0] == 0
        assert_equal(expected_dict[0], out_dict[0])

    def test_nan_in_timeseries_ignored(self, valid_gpd):
        test_data = [
            [
                [[np.nan]],
                [[3.0]],
            ],
            [
                [[9.0]],
                [[5.0]],
            ],
        ]
        test_stack = xr.DataArray(
            test_data,
            dims=["time", "band", "y", "x"],
            coords={
                "time": pd.date_range("2010", "2011", freq="YS"),
                "y": [1],
                "x": [1]
            },
        ).rio.write_crs("EPSG:4326", inplace=True)
        expected_data = [9.0, 4.0]
        expected_dict = {0: xr.DataArray(
            expected_data,
            dims=["band"],
            coords={
                "band": [0, 1],
            },
        ).rio.write_crs("EPSG:4326", inplace=True)
        }
        out_dict = median(restoration_sites=valid_gpd, timeseries_data=test_stack, reference_years={0: [2010, 2011]}, scale="polygon")
        assert len(out_dict.keys()) == 1
        assert list(out_dict.keys())[0] == 0
        assert_equal(expected_dict[0], out_dict[0])


class TestMedianPixelScale:
    @pytest.fixture()
    def valid_gpd(self):
        polygon = Polygon([(-0.5, -0.5), (-0.5, 1.5), (1.5, 1.5), (1.5, -0.5)])
        valid_gpd = gpd.GeoDataFrame(geometry=[polygon]).set_crs("EPSG:4326")
        return valid_gpd
    
    def test_scale_pixel_returns_correct_dimensions(self, valid_gpd):
        test_data = np.arange(1, 9).reshape(1, 2, 2, 2).astype(float)
        test_stack = xr.DataArray(
            test_data,
            dims=["band", "time", "y", "x"],
            coords={
                "time": pd.date_range("2010", "2011", freq="YS"), 
                "y":[0, 1],
                "x": [0, 1],
            },
        ).rio.write_crs("EPSG:4326", inplace=True)
        out_dict = median(restoration_sites=valid_gpd, timeseries_data=test_stack, reference_years={0: [2010, 2011]}, scale="pixel")

        assert len(out_dict.keys()) == 1
        assert list(out_dict.keys())[0] == 0
        assert out_dict[0].dims == ("band", "y", "x")
        assert out_dict[0].shape == (1, 2, 2)

    def test_scale_pixel_returns_per_pixel_median(self, valid_gpd):
        test_data = [
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [8.0, 9.0]],
            ],
        ]
        test_stack = xr.DataArray(
            test_data,
            dims=["band", "time", "y", "x"],
            coords={
                "time": pd.date_range("2010", "2011", freq="YS"), 
                "y":[0, 1],
                "x": [0, 1],    
            },
        ).rio.write_crs("EPSG:4326", inplace=True)

        expected_data = [[[3.0, 4.0], [5.5, 6.5]]]
        expected_dict = {0: xr.DataArray(
            expected_data,
            dims=["band", "y", "x"],
            coords={
                "band": [0],
                "y":[0, 1],
                "x": [0, 1],  
            },
        ).rio.write_crs("EPSG:4326", inplace=True)
        }
        out_dict = median(restoration_sites=valid_gpd, timeseries_data=test_stack, reference_years={0: [2010, 2011]}, scale="pixel")
        assert len(out_dict.keys()) == 1
        assert list(out_dict.keys())[0] == 0
        assert_equal(expected_dict[0], out_dict[0])


class TestMedianMultipleSites:

    def test_multiple_restoration_sites_returns_pixel_target(self):
        test_data = [
            [
                [
                    [1., 2., 3.],
                    [4., 5., 6.],
                    [7., 8., 9.]
                ]
            ],

            [
                [
                    [10., 11., 12.],
                    [13., 14., 15.],
                    [16., 17., 18.]
                ]
            ]
        ]
        test_stack = xr.DataArray(
                test_data,
                dims=["time", "band", "y", "x"],
                coords={
                    "time": pd.date_range("2010", "2011", freq="YS"),
                    "y": [1, 0, -1],
                    "x": [-1, 0, 1]
                },
        ).rio.write_crs("EPSG:3348", inplace=True)
        polygon1 = Polygon([(-2, 2), (-0.25, 1), (-0.25, -2), (-2, -2), (-2, 2)])
        polygon2 = Polygon([(-1, 1), (0.75, 1), (0.75, -1), (-1, -1), (-1, 1)])
        polygon3 = Polygon([(0.75, 2), (0, 2), (2, 2), (0.75, -1.75), (0.75, 2)])
        valid_gpd = gpd.GeoDataFrame(geometry=[polygon1, polygon2, polygon3]).set_crs("EPSG:3348")

        expected_dict = {
            0: xr.DataArray(
                    [[[5.5], [8.5], [11.5]]],
                    dims=["band", "y", "x"],
                    coords={
                        "band" : [0],   
                        "y": [1, 0, -1],
                        "x": [-1]
                    },
                ).rio.write_crs("EPSG:3348", inplace=True),
            1: xr.DataArray(
                    [[[6.5], [9.5], [12.5]]],
                    dims=["band", "y", "x"],
                    coords={
                        "band" : [0],
                        "y": [1, 0, -1],
                        "x": [0]
                    },
                ).rio.write_crs("EPSG:3348", inplace=True),
            2: xr.DataArray(
                    [[[7.5], [10.5], [13.5]]],
                    dims=["band", "y", "x"],
                    coords={
                        "band" : [0],
                        "y": [1, 0, -1],
                        "x": [1]
                    },  
                ).rio.write_crs("EPSG:3348", inplace=True)
        }

        out_dict = median(
            restoration_sites=valid_gpd,
            timeseries_data=test_stack,
            reference_years={
                    0: [2010, 2011],
                    1: [2010, 2011],
                    2: [2010, 2011]
                },
            scale="pixel"
        )
        assert len(out_dict) == 3
        assert list(out_dict.keys()) == [0, 1, 2]
        assert_equal(out_dict[0], expected_dict[0])
        assert_equal(out_dict[1], expected_dict[1])
        assert_equal(out_dict[2], expected_dict[2])
    
    def test_multiple_restoration_sites_returns_polygon_target(self):
        test_data = [
            [
                [
                    [1., 2., 3.],
                    [4., 5., 6.],
                    [7., 8., 9.]
                ]
            ],

            [
                [
                    [10., 11., 12.],
                    [13., 14., 15.],
                    [16., 17., 18.]
                ]
            ]
        ]
        test_stack = xr.DataArray(
                test_data,
                dims=["time", "band", "y", "x"],
                coords={
                    "time": pd.date_range("2010", "2011", freq="YS"),
                    "y": [1, 0, -1],
                    "x": [-1, 0, 1]
                },
        ).rio.write_crs("EPSG:3348", inplace=True)
        polygon1 = Polygon([(-2, 2), (-0.25, 1), (-0.25, -2), (-2, -2), (-2, 2)])
        polygon2 = Polygon([(-1, 1), (0.75, 1), (0.75, -1), (-1, -1), (-1, 1)])
        polygon3 = Polygon([(0.75, 2), (0, 2), (2, 2), (0.75, -1.75), (0.75, 2)])
        valid_gpd = gpd.GeoDataFrame(geometry=[polygon1, polygon2, polygon3]).set_crs("EPSG:3348")

        expected_dict = {
            0: xr.DataArray(
                    [8.5],
                    dims=["band"],
                    coords={
                        "band" : [0],   
                    },
                ).rio.write_crs("EPSG:3348", inplace=True),
            1: xr.DataArray(
                    [9.5],
                    dims=["band"],
                    coords={
                        "band" : [0],
                    },
                ).rio.write_crs("EPSG:3348", inplace=True),
            2: xr.DataArray(
                    [10.5],
                    dims=["band"],
                    coords={
                        "band" : [0],
                    },  
                ).rio.write_crs("EPSG:3348", inplace=True)
        }

        out_dict = median(
            restoration_sites=valid_gpd,
            timeseries_data=test_stack,
            reference_years={
                    0: [2010, 2011],
                    1: [2010, 2011],
                    2: [2010, 2011]
                },
            scale="polygon"
        )
        assert len(out_dict) == 3
        assert list(out_dict.keys()) == [0, 1, 2]
        assert_equal(out_dict[0], expected_dict[0])
        assert_equal(out_dict[1], expected_dict[1])
        assert_equal(out_dict[2], expected_dict[2])

    def test_unique_reference_years_per_poly_returns_correct_targets(self):
        test_data = [
            [
                [
                    [1., 2., 3.],
                    [4., 5., 6.],
                    [7., 8., 9.]
                ]
            ],

            [
                [
                    [10., 11., 12.],
                    [13., 14., 15.],
                    [16., 17., 18.]
                ]
            ],

            [
                [
                    [19., 20., 21.],
                    [22., 23., 24.],
                    [25., 26., 27.]
                ]
            ]

        ]
        test_stack = xr.DataArray(
                test_data,
                dims=["time", "band", "y", "x"],
                coords={
                    "time": pd.date_range("2010", "2012", freq="YS"),
                    "y": [1, 0, -1],
                    "x": [-1, 0, 1]
                },
        ).rio.write_crs("EPSG:3348", inplace=True)
        polygon1 = Polygon([(-2, 2), (-0.25, 1), (-0.25, -2), (-2, -2), (-2, 2)])
        polygon2 = Polygon([(-1, 1), (0.75, 1), (0.75, -1), (-1, -1), (-1, 1)])
        polygon3 = Polygon([(0.75, 2), (0, 2), (2, 2), (0.75, -1.75), (0.75, 2)])
        valid_gpd = gpd.GeoDataFrame(geometry=[polygon1, polygon2, polygon3]).set_crs("EPSG:3348")

        expected_dict = {
            0: xr.DataArray(
                    [8.5],
                    dims=["band"],
                    coords={
                        "band" : [0],   
                    },
                ).rio.write_crs("EPSG:3348", inplace=True),
            1: xr.DataArray(
                    [18.5],
                    dims=["band"],
                    coords={
                        "band" : [0],
                    },
                ).rio.write_crs("EPSG:3348", inplace=True),
            2: xr.DataArray(
                    [15],
                    dims=["band"],
                    coords={
                        "band" : [0],
                    },  
                ).rio.write_crs("EPSG:3348", inplace=True)
        }

        out_dict = median(
            restoration_sites=valid_gpd,
            timeseries_data=test_stack,
            reference_years={
                    0: [2010, 2011],
                    1: [2011, 2012],
                    2: [2010, 2012]
                },
            scale="polygon"
        )
        assert len(out_dict) == 3
        assert list(out_dict.keys()) == [0, 1, 2]
        assert_equal(out_dict[0], expected_dict[0])
        assert_equal(out_dict[1], expected_dict[1])
        assert_equal(out_dict[2], expected_dict[2])
        
    def test_polygon_ids_maintained(self):
        test_data = [
            [
                [
                    [1., 2., 3.],
                    [4., 5., 6.],
                    [7., 8., 9.]
                ]
            ],
        ]
        test_stack = xr.DataArray(
                test_data,
                dims=["time", "band", "y", "x"],
                coords={
                    "time": pd.date_range("2010", "2010", freq="YS"),
                    "y": [1, 0, -1],
                    "x": [-1, 0, 1]
                },
        ).rio.write_crs("EPSG:3348", inplace=True)
        polygon1 = Polygon([(-2, 2), (-0.25, 1), (-0.25, -2), (-2, -2), (-2, 2)])
        polygon2 = Polygon([(-1, 1), (0.75, 1), (0.75, -1), (-1, -1), (-1, 1)])
        polygon3 = Polygon([(0.75, 2), (0, 2), (2, 2), (0.75, -1.75), (0.75, 2)])
        valid_gpd = gpd.GeoDataFrame(geometry=[polygon1, polygon2, polygon3]).set_crs("EPSG:3348")
        new_indices = [12, 8, 33]
        valid_gpd.index = new_indices

        expected_keys = [12, 8, 33]
        expected_dict = {
            12: xr.DataArray(
                    [4.0],
                    dims=["band"],
                    coords={
                        "band" : [0],   
                    },
                ).rio.write_crs("EPSG:3348", inplace=True),
            8: xr.DataArray(
                    [5.0],
                    dims=["band"],
                    coords={
                        "band" : [0],
                    },
                ).rio.write_crs("EPSG:3348", inplace=True),
            33: xr.DataArray(
                    [6.0],
                    dims=["band"],
                    coords={
                        "band" : [0],
                    },  
                ).rio.write_crs("EPSG:3348", inplace=True)
        }

        out_dict = median(
            restoration_sites=valid_gpd,
            timeseries_data=test_stack,
            reference_years={
                    12: [2010, 2010],
                    8: [2010, 2010],
                    33: [2010, 2010]
                },
            scale="polygon"
        )
        print(out_dict)
        assert len(out_dict) == 3
        assert list(out_dict.keys()) == expected_keys
        assert_equal(out_dict[12], expected_dict[12])
        assert_equal(out_dict[8], expected_dict[8])
        assert_equal(out_dict[33], expected_dict[33])

class TestCheckReferenceYears:

    test_stack = xr.DataArray(
                [[[[1.],[4.],[7.]]]],
                dims=["time", "band", "y", "x"],
                coords={
                    "time": pd.date_range("2010", "2010", freq="YS"),
                    "y": [-1, 0, 1],
                    "x": [0]
                },
        ).rio.write_crs("EPSG:3348", inplace=True)
    valid_gpd = gpd.GeoDataFrame(geometry=[Polygon([(-1, 1), (0.75, 1), (0.75, -1), (-1, -1), (-1, 1)]), Polygon([(-1, 1), (0.75, 1), (0.75, -1), (-1, -1), (-1, 1)])]).set_crs("EPSG:3348")

    def test_valid_reference_years_successful(self):
        _check_reference_years(
            restoration_sites=self.valid_gpd,
            timeseries_data=self.test_stack,
            reference_years={
                0: [2010, 2010],
                1: [2010, 2010]
            },
        )

    def test_invalid_ref_years_throws_value_err(self):
        with pytest.raises(ValueError):
            _check_reference_years(
                restoration_sites=self.valid_gpd,
                timeseries_data=self.test_stack,
                reference_years={
                    0: [0, 2010],
                    1: [2010, 0]
                },
            )
    
    def test_oob_ref_years_throws_value_err(self):
        with pytest.raises(ValueError):
            _check_reference_years(
                restoration_sites=self.valid_gpd,
                timeseries_data=self.test_stack,
                reference_years={
                    0: [2009, 2010],
                    1: [2010, 2011]
                },
            )

    def test_missing_ref_years_throws_value_err(self):
        with pytest.raises(ValueError):
            _check_reference_years(
                restoration_sites=self.valid_gpd,
                timeseries_data=self.test_stack,
                reference_years={
                    0: [2010, 2010],
                },
            )


class TestWindow:
    valid_poly = Polygon([(-1.5, -1.5), (-1.5, 2.5), (2.5, 2.5), (1.5, -1.5)])

    @pytest.fixture()
    def valid_gpd(self):
        valid_gpd = gpd.GeoDataFrame(geometry=[self.valid_poly]).set_crs("EPSG:4326")
        return valid_gpd
    

    @pytest.fixture()
    def valid_array(self):
        data = np.ones((2, 5, 10, 10))
        latitudes = np.arange(-5, 5)
        longitudes = np.arange(-5, 5)
        time = pd.date_range("2010", "2014", freq="YS")
        xarr = xr.DataArray(
            data,
            dims=["band", "time", "y", "x"],
            coords={"band": ["NBR", "SAVI"], "time": time, "y": latitudes[::-1], "x": longitudes},
        )
        xarr = xarr.rio.write_crs("EPSG:3348", inplace=True)
        return xarr

    @patch("geopandas.read_file")
    def test_str_polygon_reads_polygon(self, read_mock, valid_array, valid_gpd):
        read_mock.return_value = valid_gpd
        _ = window(restoration_sites="pathy", timeseries_data=valid_array, reference_years={0: [2010, 2011]})

        read_mock.assert_called_once()
        assert read_mock.call_args.args[0] == "pathy"

    def test_neg_or_0_N_throws_value_err(self, valid_array, valid_gpd):

        with pytest.raises(
            ValueError, 
        ):
            window(restoration_sites=valid_gpd, timeseries_data=valid_array, reference_years={0: [2010, 2011]}, N=-1)

    def test_even_N_throws_value_err(self, valid_array, valid_gpd):
        with pytest.raises(
            ValueError, 
        ):
            window(restoration_sites=valid_gpd, timeseries_data=valid_array, reference_years={0: [2010, 2011]}, N=2)

    @patch("xarray.DataArray.rolling")
    def test_rolling_called_with_N_center_True_and_minp_None(self, roll_mock, valid_array, valid_gpd):

        window(restoration_sites=valid_gpd, timeseries_data=valid_array, reference_years={0: [2010, 2011]})
        assert roll_mock.called_once
        roll_mock.call_args.kwargs["dim"] == {"x": 3, "y": 3}
        roll_mock.call_args.kwargs["center"] == True
        roll_mock.call_args.kwargs["min_periods"] == None

    def test_window_returns_correct_dict(self, valid_array, valid_gpd):
        expected_dims_and_sizes = {"band": 2, "y": 4, "x": 4}

        result = window(restoration_sites=valid_gpd, timeseries_data=valid_array, reference_years={0: [2010, 2012]})
        print(result[0].sizes)
        assert len(result.keys()) == 1
        assert list(result.keys())[0] == 0
        assert len(result[0].dims) == len(expected_dims_and_sizes.keys())
        print(result[0])
        for dim in result[0].dims:
            assert result[0].sizes[dim] == expected_dims_and_sizes[dim]

    def test_default_window_returns_correct_target_values(self, valid_gpd):
        data = np.arange(-8, 19).reshape((1, 3, 3, 3))
        latitudes = np.arange(0, 3)
        longitudes = np.arange(0, 3)
        time = pd.date_range("2010", "2012", freq="YS")
        input_data = xr.DataArray(
            data,
            dims=["band", "time", "y", "x"],
            coords={"band": ["NBR"], "time": time, "y": latitudes[::-1], "x": longitudes}
        ).rio.write_crs("EPSG:3348", inplace=True)
        
        # Median of `data` along time dim will be 3x3 array with values 1-9. Mean of 5.
        # Since only the centre pixel can have a full 3x3 window, centre should be set 
        # to 5 and all others should be NaN.
        expected_data = np.array([[[np.nan, np.nan, np.nan],[np.nan, 5.0, np.nan], [np.nan, np.nan, np.nan]]])
        result = window(restoration_sites=valid_gpd, timeseries_data=input_data, reference_years={0: [2010, 2012]})

        assert_array_equal(expected_data, result[0])
    
    def test_nan_rm_true_computes_without_NaN(self,):
        """
        Given:

            o, o, o, o
            o, x, x, o
            o, x, o, o
            o, o, o, o
        
        where o is NaN space and x is non-Nan, ensure
        WindowedTarget returns mean values for all cells where a 
        3x3 window contains at least one value.
        
        """
        latitudes = np.arange(-2, 2)
        longitudes = np.arange(-2, 2)
        time = pd.date_range("2010", "2010", freq="YS")
        test_data = xr.DataArray([[[
                [np.nan, np.nan, np.nan, np.nan],
                [np.nan,      1,      2, np.nan],
                [np.nan,      6, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan],
            ]]],
            dims=["band", "time", "y", "x"],
            coords={"time": time, "y": latitudes[::-1], "x": longitudes}
        ).rio.write_crs("EPSG:3348", inplace=True)
        test_poly = Polygon([(-2.5, -2.5), (-2.5, 2.5), (2.5, 2.5), (2.5, -2.5)])
        test_gpd = gpd.GeoDataFrame(geometry=[test_poly]).set_crs("EPSG:4326")

        valid_build = {"restoration_sites": test_gpd, "timeseries_data": test_data, "reference_years": {0: [2010, 2010]}}

        
        expected_data = xr.DataArray([[
                [1.0, 1.5, 1.5, 2.0],
                [3.5, 3.0, 3.0, 2.0],
                [3.5, 3.0, 3.0, 2.0],
                [6.0, 6.0, 6.0, np.nan],
            ]],
            dims=["band", "y", "x"],
            coords={"y": latitudes[::-1], "x": longitudes}
        ).rio.write_crs("EPSG:3348", inplace=True)
        result_dict = window(**valid_build, na_rm=True)
        result_stack = result_dict[0]
        xr.testing.assert_equal(expected_data, result_stack)

    def test_default_computes_correct_means(self,):

        data = np.ones((1,1,6,6))
        for i in range(1, 7):
            data[:,:,i-1,:] = data[:,:,i-1,:] * i

        latitudes = np.arange(-3, 3)
        longitudes = np.arange(-3, 3)
        time = pd.date_range("2010", "2010", freq="YS")
        test_array = xr.DataArray(
            data,
            dims=["band", "time", "y", "x"],
            coords={"time": time, "y": latitudes[::-1], "x": longitudes}
        ).rio.write_crs("EPSG:3348", inplace=True)
        test_poly = Polygon([(-3.5, -3.5), (-3.5, 3.5), (3.5, 3.5), (3.5, -3.5)])
        test_gpd = gpd.GeoDataFrame(geometry=[test_poly]).set_crs("EPSG:4326")
        valid_build = {"restoration_sites": test_gpd, "timeseries_data": test_array, "reference_years": {0: [2010, 2010]}}
        
        expected_stack = xr.DataArray([[
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan,    2.0,    2.0,    2.0,    2.0, np.nan],
                [np.nan,    3.0,    3.0,    3.0,    3.0, np.nan],
                [np.nan,    4.0,    4.0,    4.0,    4.0, np.nan],
                [np.nan,    5.0,    5.0,    5.0,    5.0, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            ]],
            dims=["band", "y", "x"]
        )

        result_dict = window(**valid_build)
        result_stack = result_dict[0]
        assert_array_equal(expected_stack, result_stack)
        
    def test_nan_rm_false_computes_with_NaN(self):
        """
        Given an array like:

            o, o, o, o
            x, x, x, o
            x, !, x, o
            x, x, x, o
        
        where o is NaN space and x/! is non-Nan space,
        ensure WindowedTarget return only values for values
        with a full 3x3 window (i.e !).
        
        """

        latitudes = np.arange(-2, 2)
        longitudes = np.arange(-2, 2)
        time = pd.date_range("2010", "2010", freq="YS")
        data = [[[
            [np.nan, np.nan, np.nan, np.nan],
            [     1,      1,   5, np.nan],
            [     1,      6,   1, np.nan],
            [     1,      1,   1, np.nan],
        ]]]
        test_array = xr.DataArray(
            data,
            dims=["band", "time", "y", "x"],
            coords={"time": time, "y": latitudes[::-1], "x": longitudes}
        ).rio.write_crs("EPSG:3348", inplace=True)
        test_poly = Polygon([(-2.5, -2.5), (-2.5, 2.5), (2.5, 2.5), (2.5, -2.5)])
        test_gpd = gpd.GeoDataFrame(geometry=[test_poly]).set_crs("EPSG:4326")
        valid_build = {"restoration_sites": test_gpd, "timeseries_data": test_array, "reference_years": {0: [2010, 2010]}}

        expected_data = xr.DataArray([[
                [np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan],
                [np.nan, 2.0,    np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan],
            ]],
            dims=["band", "y", "x"],
        )

        result_dict = window(**valid_build, na_rm=False)
        result_stack = result_dict[0]
        assert_array_equal(expected_data, result_stack)
    



