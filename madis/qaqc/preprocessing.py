"""Preprocessing functions for MADIS observational data and HRRR"""

#add_abs_diff

import xarray as xr
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.spatial import cKDTree

def calc_windspeed(
    wind_v: np.ndarray | xr.DataArray,
    wind_u: np.ndarray | xr.DataArray,
) -> np.ndarray | xr.DataArray:
    """
    Compute wind speed from eastward (u) and northward (v) wind velocities.

    Parameters
    ----------
    wind_u: np.ndarray | xr.DataArray
        east (+) west (-) wind velocity (m/s)
    wind_v: np.ndarray | xr.DataArray
        north (+) south (-) wind velocity (m/s)

    Return
    ---------
    : np.ndarray | xr.DataArray
        Wind speed (m/s)
    """
    assert wind_v.shape == wind_u.shape, "arguments have mismatched shape"
    return np.hypot(wind_u, wind_v)

def calc_winddirection(
    wind_v: np.ndarray | xr.DataArray,
    wind_u: np.ndarray | xr.DataArray,
) -> np.ndarray | xr.DataArray:
    """
    Compute wind direction in clockwise degrees from north.

    Parameters
    ----------
    v: np.ndarray | xr.DataArray
        north (+) south (-) wind velocity (m/s)
    u: np.ndarray | xr.DataArray
        east (+) west (-) wind velocity (m/s)

    Return
    ---------
    deg_north: np.ndarray | xr.DataArray
        Wind direction in clockwise degrees from north (0 through 360)
    """
    assert wind_v.shape == wind_u.shape, "arguments have mismatched shape"
    deg_north = np.degrees(np.arctan2(-wind_u, -wind_v)) % 360
    return deg_north

def mask_stations(
    ds: xr.Dataset,
    lat_bounds: tuple[float, float],
    lon_bounds: tuple[float, float],
) -> xr.Dataset:
    """
    Mask a dataset to stations within given lat/lon bounds.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with 'latitude' and 'longitude' coordinates along 'space' dim.
    lat_bounds : tuple of (min_lat, max_lat)
        Latitude bounds.
    lon_bounds : tuple of (min_lon, max_lon)
        Longitude bounds.

    Returns
    -------
    xr.Dataset
        Dataset filtered to stations within the bounds.
    """
    lat_obs = ds["latitude"].values
    lon_obs = ds["longitude"].values
    mask = (
        (lat_obs >= lat_bounds[0])
        & (lat_obs <= lat_bounds[1])
        & (lon_obs >= lon_bounds[0])
        & (lon_obs <= lon_bounds[1])
    )
    return ds.isel(space=mask)

def read_madis(
    file_paths: list[str],
    lat_bounds: tuple[float, float] | None = None,
    lon_bounds: tuple[float, float] | None = None,
) -> xr.Dataset:
    """
    Load and merge MADIS NetCDF files where each file may have different
    space values. Optionally filter to stations within lat/lon bounds.

    Sorts filepaths so METAR files are loaded first, then other files
    (such as Mesonet).
    This ensures that if there are duplicate stations across sources,
    the METAR station is kept. Files are grouped by their first data
    variable name, so files sharing a variable (e.g. from both METAR
    and Mesonet) are concatenated along the space dimension, then data
    is merged across variable groups.

    Parameters
    ----------
    file_paths : list of str
        List of MADIS NetCDF file paths, with each file
        containing a single variable.
    lat_bounds : tuple of (min_lat, max_lat)
        Latitude bounds.
    lon_bounds : tuple of (min_lon, max_lon)
        Longitude bounds.

    Returns
    -------
    xr.Dataset
        Dataset with unique stations, all variables.

    """
    file_paths = sorted(file_paths, key=lambda p: (0 if "metar" in p else 1))
    seen_vars = defaultdict(list)

    for path in file_paths:
        ds = xr.open_dataset(path, chunks={})
        ds["time"] = pd.to_datetime(ds["time"])
        if lat_bounds is not None and lon_bounds is not None:
            ds = mask_stations(ds, lat_bounds, lon_bounds)
        ds = ds.sortby("time")
        ds = ds.set_index(space=("latitude", "longitude"))
        var = list(ds.data_vars)[0]
        seen_vars[var].append(ds)

    var_datasets = []
    for var, dss in seen_vars.items():
        concatted = xr.concat(dss, dim="space")
        index = concatted.get_index("space")
        mask = ~index.duplicated(keep="first")
        var_datasets.append(concatted.isel(space=mask))

    return xr.merge(var_datasets, compat="override", join="outer")

def subset_grid_to_point_xy(
    ds: xr.Dataset,
    point_ds: xr.Dataset,
) -> xr.Dataset:
    """
    Subset a gridded dataset with (time, x, y) dimensions to the nearest grid
    point for each station in a point dataset.

    The returned dataset's 'space' coordinate is a new integer index
    (0..N-1) built in the same order as `point_ds`'s stations. The returned
    dataset's latitude and longitude coordinates are replaced with the station
    coordinates from `point_ds`.

    Parameters
    ----------
    ds : xr.Dataset
        Gridded data with dims ('time', 'y', 'x') and 2D 'latitude',
        'longitude'.
    point_ds : xr.Dataset
        Point datat with dims ('time', 'space') and 1D 'latitude', 'longitude'
        for each site.

    Returns
    -------
    xr.Dataset
        Subset of gridded data with dims ('time', 'space'). Space is an index
        1-N, and latitude/longitude coordinates are replaced
        with the station coordinates.
    """
    point_ds = point_ds.where(point_ds.time.isin(ds.time), drop=True)

    grid_lat = ds["latitude"].values
    grid_lon = ds["longitude"].values

    grid_points = np.column_stack([grid_lat.ravel(), grid_lon.ravel()])
    tree = cKDTree(grid_points)

    obs_points = np.column_stack(
        [
            point_ds["latitude"].values,
            point_ds["longitude"].values,
        ]
    )
    _, flat_indices = tree.query(obs_points)

    nearest_y, nearest_x = np.unravel_index(flat_indices, grid_lat.shape)

    nearest_y = xr.DataArray(nearest_y, dims="space")
    nearest_x = xr.DataArray(nearest_x, dims="space")

    grid_subset = ds.isel(y=nearest_y, x=nearest_x)

    #lat lon become the true station coordinates
    grid_subset = grid_subset.assign_coords(
        space=np.arange(point_ds.sizes["space"]),
        latitude=("space", point_ds["latitude"].values),
        longitude=("space", point_ds["longitude"].values),
    )

    return grid_subset

def forecast_obs_merge(
    ds1: xr.Dataset,
    ds2: xr.Dataset,
    ds1_timevar: str,
    ds2_timevar: str,
    ds1_prefix: str = None,
    ds2_prefix: str = None,
):
    """
    Subsets both inputs to their common time values, sorts by time, unifies
    the time coordinate name to "time", and combines variables into a single
    dataset. Optionally add prefixes to variable names from either input.

    Both inputs must have a dimension named
    "time", the values used to specify the time coordinate are given by
    `ds1_timevar` and `ds2_timevar`. This function is designed to be used after
    subset_grid_to_point_xy, and use the same point dataset for both inputs so
    that the space coordinate is aligned.

    The "space" coordinate is taken from `ds1`, If used with
    subset_grid_to_point_xy, it is recommended that ds_1 is the gridded dataset
    and ds_2 is the point dataset so the "space" coordinate carried forward is
    type integer rather than a multi-index.

    Parameters
    ----------
    ds1 : xarray.Dataset
        First dataset (e.g., forecast data such as HRRR), with a time
        coordinate specified by `ds1_timevar` and a "space" coordinate.
    ds2 : xarray.Dataset
        Second dataset (e.g., observation data), with a time coordinate
        specified by `ds2_timevar` and a "space" coordinate.
    ds1_timevar : str
        Name of the time coordinate in `ds1` (e.g., "valid_time").
    ds2_timevar : str
        Name of the time coordinate in `ds2` (e.g., "time").
    ds1_prefix : str or None, optional
        Prefix to add to all variables in `ds1`. If None, variable names are
        unchanged. Default is None.
    ds2_prefix : str or None, optional
        Prefix to add to all variables in `ds2`. If None, variable names are
        unchanged. Default is None.

    Returns
    -------
    xarray.Dataset
        Merged dataset containing variables from both inputs, aligned on the
        intersection of time values.

    Raises
    ------
    ValueError
        If `ds1` and `ds2` share no common time values.

    """
    # Find common times and subset
    common_times = np.intersect1d(
        ds1[ds1_timevar].values, ds2[ds2_timevar].values
    )

    if len(common_times) == 0:
        raise ValueError(
            f"No common times found between {ds1_timevar} and {ds2_timevar}"
        )

    ds1_subset = ds1.isel(time=ds1[ds1_timevar].isin(common_times).values)
    ds2_subset = ds2.isel(time=ds2[ds2_timevar].isin(common_times).values)

    # Sort both by their time variable to ensure positional alignment
    ds1_subset = ds1_subset.sortby(ds1_timevar)
    ds2_subset = ds2_subset.sortby(ds2_timevar)

    # Unify time coordinate
    ds1_subset = ds1_subset.assign_coords(time=ds1_subset[ds1_timevar].values)
    ds2_subset = ds2_subset.assign_coords(time=ds2_subset[ds2_timevar].values)

    # Build merged dataset, applying prefixes to both ds1 and ds2 vars
    merged = xr.Dataset(
        coords={
            "time": ds1_subset["time"],
            "space": ds1_subset["space"],
            **{
                k: v
                for k, v in ds1_subset.coords.items()
                if k not in ["time", "space"]
            },
        }
    )

    for var in ds1_subset.data_vars:
        name = f"{ds1_prefix}_{var}" if ds1_prefix else var
        merged[name] = ds1_subset[var]

    for var in ds2_subset.data_vars:
        name = f"{ds2_prefix}_{var}" if ds2_prefix else var
        merged[name] = ds2_subset[var]

    merged = merged.transpose("time", "space")
    merged = merged.copy()

    return merged

def add_abs_diff(
    ds: xr.Dataset,
    var1: str,
    var2: str,
    var_name: str = None,
) -> xr.Dataset:
    """

    Compute absolute difference between two variables.
    The new variable is named `var_name` and is added to the dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing both variables.
    var1 : str
        Name of first variable.
    var2 : str
        Name of second variable.
    var_name : str, optional
        Name of the new variable to store the absolute difference. If None,
        the new variable will be named `abs_diff_{var1}`, which assumes var1
        and var2 are the same variable from different datasets(e.g., forecast
        and observation).

    Returns
    -------
    xarray.Dataset
        Dataset with new variable `var_name`.
    """
    if ds[var1].dims != ds[var2].dims:
        raise ValueError(
            f"'{var1}' and '{var2}' have mismatched dimensions: "
            f"{ds[var1].dims} vs {ds[var2].dims}"
        )

    var_name = var_name if var_name else f"abs_diff_{var1}"
    ds[var_name] = np.abs(ds[var2] - ds[var1])
    return ds
