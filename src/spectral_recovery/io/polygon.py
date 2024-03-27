import geopandas as gpd
import pandas as pd


# TODO: allow users to pass attribute col names for date cols
def read_restoration_polygons(
        path: str,
        disturbance_start: str | int= None,
        restoration_start: str | int = None,
        reference_start: str | int = None,
        reference_end: str | int = None, 
    ):
    """Read restoration polygons

    A loose wrapper of the geopandas.read_file function. If
    check_dates=True then this function will check that the
    vector file at path contains 2 or 4 columns containing
    str or int values (the restoration site dates).

    Parameters
    ----------
    path : str
        path to restoration polygon vector file

    """
    # Read the vector file and check there is only one polygon
    restoration_polygons = gpd.read_file(path)
    if len(restoration_polygons.index) > 1:
        raise ValueError(
            "Only one restoration polygon is currently supported"
            f" ({len(restoration_polygons.index)} provided)"
        )

    if (disturbance_start and not restoration_start) or (not disturbance_start and restoration_start):
        raise ValueError("Both disturbance_start and restoration_start must be provided. Not one or the other.")
    
    if disturbance_start and restoration_start:
        if not reference_end and not reference_start:
            dates_no_ref = {
                "dist_start": disturbance_start,
                "rest_start": restoration_start,
            }
            restoration_polygons = restoration_polygons.assign(**dates_no_ref)
            # Dates must be in order: dist, rest, ref start, ref end. 
            restoration_polygons = restoration_polygons[["dist_start", "rest_start", "geometry"]]
        else:
            dates_w_ref = {
                "dist_start": disturbance_start,
                "rest_start": restoration_start,
                "ref_start": reference_start,
                "ref_end": reference_end,
            }
            restoration_polygons = restoration_polygons.assign(**dates_w_ref)
            # Dates must be in order: dist, rest, ref start, ref end. 
            restoration_polygons = restoration_polygons[["dist_start", "rest_start", "ref_start", "ref_end", "geometry"]]

    # Look at the dates within the geodataframe (data from attribute table)
    dates_frame = pd.DataFrame(restoration_polygons.drop(columns="geometry"))
    if len(dates_frame.columns) < 2:
        raise ValueError(
            "Attribute table must have at least 2 columns containing disturbance start"
            f" year and restoration start year ({len(dates_frame.columns)} column"
            " given)"
        )
    # Check that years are either str or int types
    types = dates_frame.dtypes
    for column_name, data_type in types.items():
        if data_type != "int64" and data_type != "object" :
            raise ValueError(
                f"Date fields must be type int or str (given {data_type}) in field"
                f" {column_name})"
            )
    # Check that the dates make sense
    if dates_frame.iloc[:, 0][0] >= dates_frame.iloc[:, 1][0]:
        raise ValueError(
            "Disturbance year cannot be greater than or equal to the restoration year"
            f" ({dates_frame.iloc[:,0][0]} >= {dates_frame.iloc[:,0][0]})"
        )
    if len(dates_frame.columns) > 2:
        try:
            if dates_frame.iloc[:, 2][0] > dates_frame.iloc[:, 3][0]:
                raise ValueError(
                    "Reference start year cannot be greater than the"
                    f" reference end year ({dates_frame.iloc[:,0][0]} >="
                    f" {dates_frame.iloc[:,0][0]})"
                )
        except IndexError:
            raise ValueError(
                "Attribute table must have exactly 2 or 4 columns for dates"
                f" ({len(dates_frame.columns)} given)"
            )

    return restoration_polygons


def read_reference_polygons(
        path: str,
        reference_start: str | int = None,
        reference_end: str | int = None
    ):
    """Read reference polygons

    A loose wrapper of the geopandas.read_file function. If
    check_dates=True then this function will check that the
    vector file at path contains 2 or 4 columns containing
    str or int values (the restoration site dates)

    Parameters
    ----------
    path : str
        path to restoration polygon vector file
    ignore_dates : bool, optional
        flag indicating whether function will
        check for dates in vector file's attribute table

    """
    reference_polygons = gpd.read_file(path)
    
    if reference_start and reference_end:
        dates = {
            "ref_start": reference_start,
            "ref_end": reference_end,
        }
        reference_polygons = reference_polygons.assign(**dates)
        # Dates must be in order: ref start, ref end. 
        reference_polygons = reference_polygons[["ref_start", "ref_end", "geometry"]]

    dates_frame = pd.DataFrame(reference_polygons.drop(columns="geometry"))
    if (reference_start and not reference_end) or (not reference_start and reference_end):
        raise ValueError("Both reference_start and reference_end must be provided. Not one or the other.")

    types = dates_frame.dtypes
    for column_name, data_type in types.items():
        if data_type != "int64" and data_type != "object":
            raise ValueError(
                f"Date fields must be type str or int (given {data_type} in field"
                f" {column_name})"
            )
    if len(dates_frame.columns) != 2:
        raise ValueError(
            "Attribute table must contain 2 columns with reference start year and"
            f" reference end year ({len(dates_frame.columns)} column(s) given)"
        )
    for column_name in dates_frame.columns:
        unique_values = dates_frame[column_name].nunique()
        if unique_values != 1:
            raise ValueError(
                "All date fields (start and end years) must be the same for each"
                " polygon"
            )

    if (dates_frame.iloc[:, 0] > dates_frame.iloc[:, 1]).all():
        raise ValueError(
            "Reference start year cannot be greater than reference end year"
            f" ({dates_frame.iloc[:,0][0]} >= {dates_frame.iloc[:,0][0]})"
        )

    return reference_polygons
