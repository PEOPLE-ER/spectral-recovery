import geopandas as  gpd
import pandas as pd

# TODO: allow users to pass attribute col names for date cols
def read_restoration_polygons(path: str):
    """ Read restoration polygons
    
    A loose wrapper of the geopandas.read_file function. If
    check_dates=True then this function will check that the 
    vector file at path contains 2 or 4 columns containing 
    str or int values (the restoration site dates).

    Parameters
    ----------
    path : str
        path to restoration polygon vector file
    ignore_dates : bool, optional
        flag indicating whether function will
        check for dates in vector file's attribute table

    """
    # Read the vector file and check there is only one polygon
    restoration_polygons = gpd.read_file(path)
    if len(restoration_polygons.index) > 1:
        raise ValueError(f"Only one restoration polygon is currently supported ({len(restoration_polygons.index)} provided)")
    
    # Look at the dates within the geodataframe (data from attribute table)
    dates_frame = pd.DataFrame(restoration_polygons.drop(columns='geometry'))
    if len(dates_frame.columns) < 2:
        raise ValueError(f"Attribute table must have at least 2 columns containing disturbance start year and restoration start year ({len(dates_frame.columns)} column given)")
    # Check that years are either str or int types
    types = dates_frame.dtypes
    for column_name, data_type in types.items():
        if data_type != 'int64':
            raise ValueError(f"Date fields must be type int (given {data_type} in field {column_name})")
    # Check that the dates make sense
    if dates_frame.iloc[:,0][0] >= dates_frame.iloc[:,1][0]:
        raise ValueError(f"Disturbance year cannot be greater than or equal to the restoration year ({dates_frame.iloc[:,0][0]} >= {dates_frame.iloc[:,0][0]})")
    if len(dates_frame.columns) > 2:
        try:
            if dates_frame.iloc[:,2][0] >= dates_frame.iloc[:,3][0]:
                raise ValueError(f"Reference start year cannot be greater than or equal to the reference end year ({dates_frame.iloc[:,0][0]} >= {dates_frame.iloc[:,0][0]})")
        except IndexError:
            raise ValueError(f"Attribute table must have exactly 2 or 4 columns for dates ({len(dates_frame.columns)} given)")

    return restoration_polygons
    
def read_reference_polygons(path: str):
    """ Read reference polygons
    
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
    
    dates_frame = pd.DataFrame(reference_polygons.drop(columns='geometry'))
    types = dates_frame.dtypes
    for column_name, data_type in types.items():
        if data_type != 'int64' and data_type != 'object':
            raise ValueError(f"Date fields must be type str or int (given {data_type} in field {column_name})")
    if len(dates_frame.columns) != 2:
        raise ValueError(f"Attribute table must contain 2 columns with reference start year and reference end year ({len(dates_frame.columns)} column(s) given)")
    types = dates_frame.dtypes
    for column_name, data_type in types.items():
        if data_type != 'int64':
            raise ValueError(f"Date fields must be type int (given {data_type} in field {column_name})")
    
    for column_name in dates_frame.columns:
        unique_values = dates_frame[column_name].nunique()
        if unique_values != 1:
            raise ValueError("All date fields (start and end years) must be the same for each polygon")

    if (dates_frame.iloc[:,0] >= dates_frame.iloc[:,1]).all():
        raise ValueError(f"Reference start year cannot be greater than reference end year ({dates_frame.iloc[:,0][0]} >= {dates_frame.iloc[:,0][0]})")

    return reference_polygons