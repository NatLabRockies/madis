"""Process and apply QA/QC filters to MADIS data by region and variable."""

import geopandas as gpd
import numpy as np
import pandas as pd
from copy import deepcopy
from madis.qaqc.flags import flag_key, flag_var_name, build_flag
from madis.qaqc.filters import filter_run_on_other_da
from madis.qaqc.configs import CONFIG_MAP, CLEAN_ORDER, MASK_DEPENDENCIES


def resolve_run_on_other_da(config, ds):
    """
    Replace "run_on_da" string variable names in 'filter_run_on_other_da'
    filters with the actual DataArrays from ds.

    Parameters
    ----------
    config: list of dict
            Sequence of stage configuration dictionaries. Some stages may
            specify 'filter_run_on_other_da' key with a 'run_on_da' kwarg that
            is a string name of a variable, this function replaces the string
            with the corresponding data array.
    ds: xr.Dataset
            The dataset containing the variables referenced in the config.

    Returns
    -------
    resolved_config: list of dict
            A copy of the input config where any 'run_on_da' string variable
            names in 'filter_run_on_other_da' filters have been replaced with
            the actual DataArrays from ds.
    """
    cfg = deepcopy(config)

    for i, stage_cfg in enumerate(cfg):
        if filter_run_on_other_da not in stage_cfg:
            continue
        stage_cfg = dict(stage_cfg)
        value = stage_cfg[filter_run_on_other_da]
        if isinstance(value, list):
            resolved = []
            for kwargs in value:
                kwargs = dict(kwargs)
                kwargs['run_on_da'] = ds[kwargs['run_on_da']].transpose(
                    'time', 'space', transpose_coords=True
                )
                resolved.append(kwargs)
            stage_cfg[filter_run_on_other_da] = resolved
        else:
            kwargs = dict(value)
            kwargs['run_on_da'] = ds[kwargs['run_on_da']].transpose(
                'time', 'space', transpose_coords=True
            )
            stage_cfg[filter_run_on_other_da] = kwargs
        cfg[i] = stage_cfg
    return cfg


def process_region(
    ds,
    clean_order=CLEAN_ORDER,
    config_map=CONFIG_MAP,
    mask_dependencies=MASK_DEPENDENCIES,
    region_polygon=None,
):
    """Clean observational data for a region using run_filter_bank.

    Optionally subsets ds to stations within the given region_polygon, then
    cleans each variable in the order given by clean_order using its config
    from config_map. Variables in mask_dependencies also inherit the mask of
    their parent variable (e.g. wind direction inherits wind speed's mask).

    Parameters
    ----------
    ds : xr.Dataset
        Observation dataset with dims ("time", "space").
    clean_order : list of str, default CLEAN_ORDER
        Variables to clean, in order. Skipped if not in ds.
    config_map : dict of {str: dict}, default CONFIG_MAP
        Maps variable name to its filter config dictionary.
    mask_dependencies : dict of {str: str}, default MASK_DEPENDENCIES
        Maps variable name to a parent variable whose mask should be OR'd in.
        Example: {"winddirection_10m": "windspeed_10m"} means that the wind
        direction variable will inherit the mask of the wind speed variable.
    region_polygon : shapely.geometry.Polygon, optional
        If provided, subset ds to stations within this polygon before cleaning.

    Returns
    -------
    xr.Dataset
        Regional subset of ds with cleaned variables and a uint16=
        ``<var>_qc_flag`` bitmask for each cleaned variable .
        Space is reindexed to a 0-based integer range.
    """

    if region_polygon is not None:
        for coord in ('latitude', 'longitude'):
            if coord not in ds and coord not in ds.coords:
                raise KeyError(
                    f"ds must contain '{coord}' "
                    'as a data variable or coordinate.'
                )
            if 'space' not in ds[coord].dims:
                raise ValueError(
                    f"'{coord}' must have a 'space' dimension, "
                    f'got dims {tuple(ds[coord].dims)}.'
                )
        stations_gdf = gpd.GeoDataFrame(
            pd.DataFrame(
                {'lon': ds['longitude'].values, 'lat': ds['latitude'].values}
            ),
            geometry=gpd.points_from_xy(
                ds['longitude'].values, ds['latitude'].values
            ),
            crs='EPSG:4326',
        )
        mask = stations_gdf.geometry.within(region_polygon)
        ds_region = ds.isel(space=mask.values)
        ds_region = ds_region.assign_coords(
            space=np.arange(ds_region.sizes['space'])
        )
    else:
        ds_region = ds

    mask_cache = {}

    ds_region_cleaned = ds_region.copy()

    for name in clean_order:
        if name not in ds_region.data_vars:
            continue
        da = ds_region[name].transpose('time', 'space', transpose_coords=True)

        var_config = config_map.get(name)
        if var_config is None:
            continue

        var_config = resolve_run_on_other_da(var_config, ds_region)

        cleaned, var_final_mask, filter_masks = run_filter_bank(da, var_config)

        # Combine with dependency mask if it exists
        parent_mask = None
        if name in mask_dependencies:
            dependent_var = mask_dependencies[name]
            if dependent_var not in mask_cache:
                raise ValueError(
                    f'Dependency {dependent_var} not found in '
                    f'mask_cache for {name}',
                )
            parent_mask = mask_cache[dependent_var]
            final_mask = var_final_mask | parent_mask
            # apply the combined mask to the cleaned data
            cleaned = cleaned.where(~final_mask, np.nan)
        else:
            final_mask = var_final_mask

        mask_cache[name] = final_mask

        ds_region_cleaned[name] = cleaned.transpose(
            'time', 'space', transpose_coords=True
        )
        ds_region_cleaned[flag_var_name(name)] = build_flag(
            da, filter_masks, parent_mask
        )

    return ds_region_cleaned


def _apply_filters(da, stage_config, filter_masks, stage=0):
    """Run a stage of filters, returning the combined mask for this stage.

    Parameters
    ----------
    da : xr.DataArray
        2D array with coordinates ("time", "space") for multiple observation
        stations in a given region for a single variable
    stage_config : dict
        Dictionary of format {filter_function: {kwarg: value}} such that all of
        the requested "filter_function(s)" will be run on ``da`` and
        kwarg/value will be passed in as arguments to the function.
    filter_masks : dict
        Dictionary to be updated with the masks output by each filter
        function, keyed by QC flag name (see ``flags.flag_key``).
    stage : int, default 0
        0-based index of this stage in the variable's config. Used to
        separate initial and final completeness flags.

    Returns
    -------
    stage_mask : xr.DataArray
        Mask showing which values are bad (True where bad) in ANY filters from
        this stage.
    """
    stage_mask = False
    for fun, kwargs in stage_config.items():
        if not isinstance(kwargs, list):
            kwargs = [kwargs]
        for kwarg in kwargs:
            _, mask = fun(da.copy(), **kwarg)
            key = flag_key(fun, kwarg, stage)
            filter_masks[key] = filter_masks.get(key, [])
            filter_masks[key].append(mask)
            stage_mask |= mask
    return stage_mask


def run_filter_bank(da, var_config):
    """Run a set of filters on a collection of observation stations in a
    single region for a single variable. Each stage is run sequentially on
    progressively cleaned data — stage_2 filters see data already cleaned
    by stage_1, and stage_3 sees data cleaned by both stage_1 and stage_2.
    The final_mask accumulates bad values across all stages, so the returned
    mask reflects everything flagged in any stage.

    Parameters
    ----------
    da : xr.DataArray
        2D array with coordinates ("time", "space") for multiple observation
        stations in a given region for a single variable
    var_config : dict
        Nested dictionary of format {stage: {filter_function: kwargs}} where
        stage is one of 'stage_1', 'stage_2', 'stage_3'. kwargs can be a single
        dict or a list of dicts if the same filter is run multiple times with
        different parameters.

    Returns
    -------
    da : xr.DataArray
        Same as input but cleaned!
    final_mask : xr.DataArray
        Mask showing which values are bad (True where bad) in ANY of the
        filters that were run. Same dimensionality as da.
    filter_masks : dict
        Dictionary mapping each QC flag name (see ``flags.flag_key``) to a
        list of masks produced under that name across all stages. If a
        filter is run multiple times, all resulting masks are appended to
        the same list.
    """

    filter_masks = {}
    final_mask = False

    if isinstance(var_config, dict):
        var_config = [var_config]

    for stage, stage_config in enumerate(var_config):
        stage_mask = _apply_filters(da, stage_config, filter_masks, stage)
        final_mask |= stage_mask
        da = da.where(~stage_mask, np.nan)

    return da, final_mask, filter_masks


def get_region_ds(ds, region_polygon):
    """Subset ds to stations within the given region_polygon.

    Parameters
    ----------
    ds : xr.Dataset
        Observation dataset with dims ("time", "space").
    region_polygon : shapely.geometry.Polygon
        Polygon defining the region to subset to.

    Returns
    -------
    xr.Dataset
        Subset of ds containing only stations within the region_polygon.
        Space is reindexed to a 0-based integer range.

    """
    stations_gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(
            ds['longitude'].values, ds['latitude'].values
        ),
        crs='EPSG:4326',
    )
    mask = stations_gdf.geometry.within(region_polygon)
    ds_region = ds.isel(space=mask.values)
    ds_region = ds_region.assign_coords(
        space=np.arange(ds_region.sizes['space'])
    )
    return ds_region
