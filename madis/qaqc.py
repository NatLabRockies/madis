"""Custom QA/QC data filters for MADIS observational data"""

import numpy as np
import pandas as pd
import xarray as xr


def _preflight(da):
    """Check xr.dataarray object with observations for correct format for qa
    functions"""
    assert isinstance(da, xr.DataArray)
    assert da.ndim == 2
    assert da.dims == ('time', 'space')


def filter_linear_gaps(da, diff_threshold=1e-4, n_threshold=3):
    """Some MADIS timeseries data appears to have large gaps that were
    gap-filled using linear interpolation. This function identifies long blocks
    of consecutive linear data and sets that data to NaN

    This filter was developed on Mesonet station data for temperature and
    humidity and has not yet been tested on other variables. It has also
    successfully been applied to windspeed, wind direction, wind gust, and
    surface pressure. Leave comments here when other variables have been
    successfully filtered with this.

    Parameters
    ----------
    da : xr.DataArray
        2D array with coordinates ("time", "space") for multiple observation
        stations in a given region for a single variable
    diff_threshold : float
        If the second-derivative of da is under this threshold
        it is considered linear
    n_threshold : int
        Number of consecutive linear steps to be considered for
        screening out

    Returns
    -------
    da : xr.DataArray
        Same as input but cleaned!
    bad_mask : xr.DataArray
        Mask showing which values are bad (True where bad).
        Same dimensionality as da.
    """

    _preflight(da)
    bad_mask = da.copy().astype(bool)
    bad_mask.name = 'bad_linear_gaps'

    for idx in da['space']:
        ida = da[:, idx]

        diff2 = ida.diff('time').diff('time').values
        linear_mask = np.abs(diff2) < diff_threshold
        linear_mask = np.concatenate(([False], linear_mask, [False]))

        # fix mask to include starting/ending points
        extended_mask = linear_mask.copy()
        extended_mask[1:] |= linear_mask[:-1]
        extended_mask[:-1] |= linear_mask[1:]

        groups = (~extended_mask).cumsum()
        groups[~extended_mask] = 0  # Valid diff values
        unique_groups, group_counts = np.unique(
            groups[groups > 0], return_counts=True
        )
        size_map = dict(zip(unique_groups, group_counts))
        long_lines = np.vectorize(lambda g: size_map.get(g, 0) >= n_threshold)
        long_lines = long_lines(groups)

        ibad_mask = ida.copy().astype(bool)
        ibad_mask[...] = long_lines

        ida[ibad_mask] = np.nan
        da[:, idx] = ida
        bad_mask[:, idx] = ibad_mask

    return da, bad_mask


def filter_spikes(da, diff_threshold=15, ex_percentile=2):
    """Some MADIS timeseries data appears to have momentary extreme spikes.
    This function identifies spikes that are both dramatic (large 1st
    derivative) and also extreme (outside a percentile threshold)

    This filter was developed on Mesonet station data for temperature,
    humidity, windspeed, and wind gust and has not yet been tested on other
    variables. Leave comments here when other variables have been successfully
    filtered with this.

    Parameters
    ----------
    da : xr.DataArray
        2D array with coordinates ("time", "space") for multiple observation
        stations in a given region for a single variable
    diff_threshold : float
        If the first-derivative of da is greater than this value
        (including spanning across NaN values in da) then a value
        is considered to be a spike if it also meets the
        ex_percentile threhsold
    ex_percentile : float
        If the value of the spike is less than this percentile or
        greater than the percentile (100-ex_percentile), then the
        value is considered to be a spike if it also meets the
        diff_threshold

    Returns
    -------
    da : xr.DataArray
        Same as input but cleaned!
    bad_mask : xr.DataArray
        Mask showing which values are bad (True where bad).
        Same dimensionality as da.
    """

    _preflight(da)
    bad_mask = da.copy().astype(bool)
    bad_mask.name = 'bad_spikes'

    for idx in da['space']:
        ida = da[:, idx]

        diff_nonan = np.abs(ida[~np.isnan(ida)].diff('time').values)
        diff_nonan = np.concatenate((diff_nonan, [0]))

        diff1 = np.zeros(len(ida))
        diff1[~np.isnan(ida)] = diff_nonan
        diff1_roll = np.roll(diff1, 1)
        spike_mask = (diff1 > diff_threshold) & (diff1_roll > diff_threshold)

        top_p = 100 - ex_percentile
        extreme = (ida > np.nanpercentile(ida, top_p)) | (
            ida < np.nanpercentile(ida, ex_percentile)
        )

        ibad_mask = spike_mask & extreme
        ida[ibad_mask] = np.nan
        da[:, idx] = ida
        bad_mask[:, idx] = ibad_mask

    return da, bad_mask


def filter_seasonal_anomalies(da, nstd=4):
    """Use seasonal statistics across multiple stations in a single region to
    filter out values that are unrealistic based on time of year in the area

    This filter was developed on Mesonet station data for temperature and
    humidity and has not yet been tested on other variables. Leave comments
    here when other variables have been successfully filtered with this.

    Parameters
    ----------
    da : xr.DataArray
        2D array with coordinates ("time", "space") for multiple observation
        stations in a given region for a single variable
    nstd : float
        Number of standard deviations beyond which to screen out values

    Returns
    -------
    da : xr.DataArray
        Same as input but cleaned!
    bad_mask : xr.DataArray
        Mask showing which values are bad (True where bad).
        Same dimensionality as da.
    """

    _preflight(da)

    ti = pd.to_datetime(da['time'])
    df = pd.DataFrame(da.values, index=ti)
    seasonal = pd.DataFrame(index=df.index)
    rolling = df.rolling(120, center=True)
    bad_mask = da.copy().astype(bool)
    bad_mask.name = 'bad_seasonal_values'

    for idx, window in enumerate(rolling):
        ind = df.index[idx]
        seasonal.at[ind, 'mean'] = np.nanmean(window.values)
        seasonal.at[ind, 'std'] = np.nanstd(window.values)

    for idx in df.columns:
        ub = seasonal['mean'] + nstd * seasonal['std']
        lb = seasonal['mean'] - nstd * seasonal['std']
        mask = (df[idx] < lb) | (df[idx] > ub)

        bad_mask[:, idx] = mask
        da[:, idx] = df[idx].values

    return da, bad_mask

def filter_value_range(da, lower=None, upper=None):
    """Some MADIS observations appear to have values outside of the expected
    valid variable range. This function identifies values outside of specified
    bounds and sets those to NaN.

    Parameters
    ----------
    da : xr.DataArray
        2D array with coordinates ("time", "space") for multiple observation
        stations in a given region for a single variable
    lower : float
        Lower bound for acceptable values. Values below this will be flagged.
    upper : float
        Upper bound for acceptable values. Values above this will be flagged.

    Returns
    -------
    da : xr.DataArray
        Same as input but cleaned!
    bad_mask : xr.DataArray
        Mask showing which values are bad (True where bad).
        Same dimensionality as da.
    """
    if lower is None and upper is None:
        bad_mask = xr.zeros_like(da, dtype=bool)
        bad_mask.name = 'bad_value_range'
        return da, bad_mask

    _preflight(da)
    bad_mask = da.copy().astype(bool)
    bad_mask[...] = False
    bad_mask.name = 'bad_value_range'

    if lower is not None:
        too_low = da < lower
        da = xr.where(too_low, np.nan, da)
        bad_mask = xr.where(too_low, True, bad_mask)
    if upper is not None:
        too_high = da > upper
        da = xr.where(too_high, np.nan, da)
        bad_mask = xr.where(too_high, True, bad_mask)

    return da, bad_mask

def run_filter_bank(da, config):
    """Run a set of filters on a collection of observation stations in a
    single region for a single variable. The filter banks are combined such
    that each filter is run on the raw input data separately and then
    observations that fail ANY of the filters is set to NaN

    Parameters
    ----------
    da : xr.DataArray
        2D array with coordinates ("time", "space") for multiple observation
        stations in a given region for a single variable
    config : dict
        Dictionary of format {filter_function: {kwarg: value}} such that all of
        the requested "filter_function(s)" will be run on ``da`` and
        kwarg/value will be passed in as arguments to the function

    Returns
    -------
    da : xr.DataArray
        Same as input but cleaned!
    final_mask : xr.DataArray
        Mask showing which values are bad (True where bad) in ANY of the
        filters that were run. Same dimensionality as da.
    filter_masks : dict
        Dictionary with the same keys as config and values as the masks output
        by each respective filter
    """

    final_mask = False
    filter_masks = {}
    for fun, kwargs in config.items():
        _, mask = fun(da.copy(), **kwargs)
        filter_masks[fun] = mask
        final_mask |= mask

    da = da.where(~final_mask, np.nan)

    return da, final_mask, filter_masks


CONFIG_WS10M ={
    filter_linear_gaps: dict(diff_threshold=1e-4, n_threshold=3),
    filter_spikes: dict(diff_threshold=15, ex_percentile=1),
    filter_value_range: dict(lower = 0, upper = 66),
}
"""Config for run_filter_bank to clean 10-meter windspeed.
Upper limit of based on regional maximum official wind gust of 66 m/s.
Ex-percentile of 1 to retain extreme windspeed events."""

CONFIG_WDIR10M = {
    filter_linear_gaps: dict(diff_threshold=1e-4, n_threshold=3),
    filter_value_range: dict(lower = 0, upper = 360),
}
"""Config for run_filter_bank to clean 10-meter wind direction."""

CONFIG_GUST10M ={
    filter_linear_gaps: dict(diff_threshold=1e-4, n_threshold=3),
    filter_spikes: dict(diff_threshold=20, ex_percentile=1),
    filter_value_range: dict(lower = 0, upper = 66),
}
"""Config for run_filter_bank to clean 10-meter wind gusts.
Upper limit of based on regional maximum official wind gust of 66 m/s.
Ex-percentile of 1 to retain extreme windspeed events."""

CONFIG_P0M ={
    filter_seasonal_anomalies: dict(nstd=4),
    filter_linear_gaps: dict(diff_threshold=1e-4, n_threshold=3),
}
"""Config for ``run_filter_bank`` to clean near-surface pressure"""

CONFIG_T2M = {
    filter_seasonal_anomalies: dict(nstd=4),
    filter_linear_gaps: dict(diff_threshold=1e-4, n_threshold=3),
    filter_spikes: dict(diff_threshold=15, ex_percentile=2),
}
"""Config for ``run_filter_bank`` to clean near-surface 2-meter temperature"""

CONFIG_RH2M = {
    filter_seasonal_anomalies: dict(nstd=4),
    filter_linear_gaps: dict(diff_threshold=1e-4, n_threshold=3),
    filter_spikes: dict(diff_threshold=15, ex_percentile=2),
    filter_value_range: dict(lower = 0, upper = 100),
}
"""Config for ``run_filter_bank`` to clean near-surface 2-meter relative
humidity"""
