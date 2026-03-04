"""Custom QA/QC data filters for MADIS observational data"""

import logging

import numpy as np
import pandas as pd
import xarray as xr
import ruptures as rpt
from warnings import warn

logger = logging.getLogger(__name__)


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

    CAUTION!! If MADIS data is interpolated from hourly to 30min then
    n_threshold=3 will find linear blocks EVERYWHERE!

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

    ti = pd.to_datetime(da['time'])
    if ti.inferred_freq == '30min' and n_threshold == 3:
        msg = ('CAUTION!! If MADIS data is interpolated from hourly to 30min '
               'then n_threshold=3 will find linear blocks EVERYWHERE!')
        warn(msg)
        logger.warning(msg)

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


def filter_spikes(da, diff_threshold=15, ex_percentile=2, max_spike_length=3):
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
        valid_mask = ~np.isnan(ida)
        valid_idx = np.where(valid_mask)[0]
        valid_values = ida[valid_mask].values

        diff_nonan = np.abs(ida[~np.isnan(ida)].diff('time').values)
        
        spike_mask_compressed = np.zeros(len(valid_values), dtype=bool)
        spike_mask_compressed[1:-1] = (diff_nonan[:-1] > diff_threshold) & (diff_nonan[1:] > diff_threshold)

        top_p = 100 - ex_percentile
        extreme_compressed = (valid_values > np.nanpercentile(valid_values, top_p)) | \
                             (valid_values < np.nanpercentile(valid_values, ex_percentile))

        bad_compressed = spike_mask_compressed & extreme_compressed

        ibad_mask = np.zeros(len(ida), dtype=bool)
        ibad_mask[valid_idx] = bad_compressed

        ida[ibad_mask] = np.nan
        da[:, idx] = ida
        bad_mask[:, idx] = ibad_mask

    return da, bad_mask


def filter_spike_groups(
    da,
    N=3,
    value_thresh=20,
    up_thresh=20,
    down_thresh=20,
    nan_spikes=True,
    ):
    """Some MADIS timeseries data appears to have extreme spikes with multiple
    consecutive extreme values. This function identifies spikes that are
    dramatic (large 1st derivative) and have N consecutive values over a
    threshold.

    This filter was developed on Mesonet station data for windspeed,

    Parameters
    ----------
    da : xr.DataArray
        2D array with coordinates ("time", "space") for multiple observation
        stations in a given region for a single variable
    N : int
        Number of extreme consecutive values in a row that can be filtered out
    value_thresh : float
        All the values N must be above this threshold to be filtered out.
    up_thresh : float
        If the first-derivative of da is greater than this value then a value
        is considered to be the start of a spike
    down_thresh : float
        If the first-derivative of da is greater than this value on the way
        down after an up spike, then a value is considered to be the start of a
        spike
    nan_spikes : bool
        If this is true, consider a NaN immediately before or after the group N
        to have a value zero so it can contribute to the derivative
        calculation.

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
    bad_mask.name = 'bad_spike_grpup'
    bad_mask[...] = False
    T = len(da['time'])

    for idx in da['space']:
        xv = da[:, idx].values
        if nan_spikes:
            xv = np.nan_to_num(xv, nan=0)

        # mask of "high/bad" values (NaNs are treated as False)
        high = np.isfinite(xv) & (xv >= value_thresh)

        # find contiguous True runs in `high`
        # pad with False on both ends so diff catches edges
        edges = np.diff(np.r_[False, high, False].astype(np.int8))
        starts = np.where(edges == 1)[0]  # index in xv where run starts
        ends = (
            np.where(edges == -1)[0] - 1
        )  # index in xv where run ends (inclusive)

        # length constraint: 1..N
        lengths = ends - starts + 1
        keep = (lengths >= 1) & (lengths <= N)

        starts = starts[keep]
        ends = ends[keep]

        # require we can evaluate jump up (start-1 exists)
        # and jump down (end+1 exists)
        in_bounds = (starts > 0) & (ends < T - 1)
        starts = starts[in_bounds]
        ends = ends[in_bounds]

        # jump conditions
        jump_up = xv[starts] - xv[starts - 1] >= up_thresh
        jump_down = xv[ends + 1] - xv[ends] <= -down_thresh

        ok = jump_up & jump_down
        starts = starts[ok]
        ends = ends[ok]

        for idstart, idstop in zip(starts, ends):
            da[slice(idstart, idstop + 1), idx] = np.nan
            bad_mask[slice(idstart, idstop + 1), idx] = np.nan

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
    n_removed = bad_mask.sum().values
    n_total = bad_mask.size
    print(f"Seasonal anomaly filter (nstd={nstd}): removed {n_removed} values out of {n_total} ({100 * n_removed / n_total:.2f}%)")


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
    bad_mask.name = 'bad_value_range'
    bad_mask[...] = False

    if lower is not None:
        bad_mask = bad_mask | (da < lower)
    if upper is not None:
        bad_mask = bad_mask | (da > upper)

    da = da.where(~bad_mask)

    return da, bad_mask


def filter_constant(da, value=None):
    """Some MADIS observation locations have constant 0 values. This sets them
    to NaN so they can be dropped later.

    Parameters
    ----------
    da : xr.DataArray
        2D array with coordinates ("time", "space") for multiple observation
        stations in a given region for a single variable
    value : float | None
        Option to set a specific constant value to screen out. If None, this
        function will check for any station with arbitrary constant value

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
    bad_mask.name = 'bad_constant'
    bad_mask[...] = False

    if value is None:
        vmin = da.min(dim='time', skipna=True)
        vmax = da.max(dim='time', skipna=True)
        constant = vmin == vmax
    else:
        constant = (da == value).all('time')

    da[:, constant] = np.nan
    bad_mask[:, constant] = np.nan

    return da, bad_mask

def filter_percentile(da, percentile=None, nstd=None, threshold=None, direction = None):
    """
    Filter out an ENTIRE stations where a chosen percentile of observed values
    is abnormally high or low, indicating systematic sensor issues or
    miscalibration.

    This filter operates in two ways:
    1. Calculates a threshold based on the distribution of the chosen
    percentile across all stations in the network.
    2. Uses a pre-defined threshold.

    This filter was developed on Mesonet and METAR station data for wind speed
    and has not yet been tested on other variables. Leave comments here when
    other variables have been successfully filtered with this.

    Example: For Mesonet network in 2024 in the greater COWY region with
    percentile=10, nstd=3, direction = 'greater' generates a 10th percentile
    threshold of 4.2 m/s, which flags 17 out of 3415 stations.

    Parameters
    ----------
    da : xr.DataArray
        2D array with dims ("time", "space") for multiple observation
        stations in a given region for a single variable
    percentile : float
        Percentile to calculate for each station (0-100). 
        Use 10 for 10th percentile, 50 for median, etc.
    nstd : float
        Number of standard deviations above mean for threshold calculation.
        Only used if threshold=None (network-based mode)
    threshold : float, optional
        Pre-defined percentile threshold. If provided, uses fixed threshold
        instead of calculating from network statistics.
    direction : str
        Whether to flag stations with percentile "greater" or "less" than the
        threshold.

    Returns
    -------
    da : xr.DataArray
        Same as input but cleaned!
    bad_mask : xr.DataArray
        Mask showing which values are bad (True where bad).
        Same dimensionality as da.
    """


    _preflight(da)
    
    if not 0 <= percentile <= 100:
        raise ValueError(f"percentile must be between 0 and 100, got {percentile}")
    if direction not in ('greater', 'less'):
        raise ValueError(f"direction must be 'greater' or 'less', got {direction}")
    
    percentile_per_station = da.quantile(percentile / 100.0, dim='time')
    
    if threshold is None:
        mean_p = percentile_per_station.mean().values
        std_p = percentile_per_station.std().values
        threshold = mean_p + nstd * std_p if direction == 'greater' else mean_p - nstd * std_p
        mode_msg = f"network-based (mean={mean_p:.2f}, std={std_p:.2f}, nstd={nstd})"
    else:
        mode_msg = "fixed"
    
    if direction == 'greater':
        flagged_stations = percentile_per_station > threshold
    else:
        flagged_stations = percentile_per_station < threshold
    
    bad_mask = da.copy().astype(bool)
    bad_mask.name = f'bad_high_p{int(percentile)}_stations'
    bad_mask[...] = False
    bad_mask[:, flagged_stations] = True
    
    da[:, flagged_stations] = np.nan
    
    logger.info(
        f"P{int(percentile)} filter ({mode_msg}): threshold={threshold:.2f}, "
        f"flagged {flagged_stations.sum().values} stations"
    )
    msg = (
        f"P{int(percentile)} filter ({mode_msg}): threshold={threshold:.2f}, "
        f"flagged {flagged_stations.sum().values} stations"
    )
    logger.info(msg)
    print(msg)

    return da, bad_mask

def filter_mean(da, threshold, direction='greater'):
    """Filter out an ENTIRE stations where the mean of a variable across time
     is greater than a chosen threshold. This is designed to be applied to the
     error between MADIS observations and a reference dataset (e.g. HRRR f02)
     to catach stations with persisten sensor bias. 

    da : xr.DataArray
        2D array with coordinates ("time", "space") for multiple observation
        stations in a given region for a single variable
    threshold : float
        The mean value threshold used to flag stations.
    direction : str
        Whether to flag segments with percentile "greater" or "less" than
        the threshold.

    Returns
    -------
    da : xr.DataArray
        Same as input but cleaned!
    bad_mask : xr.DataArray
        Mask showing which values are bad (True where bad).
        Same dimensionality as da.
    """

    _preflight(da)
    if direction not in ('greater', 'less'):
        raise ValueError(f"direction must be 'greater' or 'less', got {direction}")

    mean_per_station = da.mean(dim='time', skipna=True)
    flagged = mean_per_station > threshold if direction == 'greater' else mean_per_station < threshold

    bad_mask = xr.zeros_like(da, dtype=bool)
    bad_mask.name = 'bad_mean_stations'
    bad_mask[:, flagged] = True
    da[:, flagged] = np.nan

    return da, bad_mask

def filter_completeness (da, completeness = 80, window = 24, station_frac=None):
    """
    Filter out time periods within individual stations where the percentage of
    NA values within a rolling window exceeds a specified completeness threshold.
    This is designed to catch sensors that are temporarily offline or have
    communication issues, resulting in many missing values for a period of time.
    Parameters
    ----------
    da : xr.DataArray
        The input data array containing observation variables with dimensions ("time", "space").
    completeness : float
        The minimum percentage of non-NA values required within the rolling window
        for the data to be considered valid. For example, completeness=50 means
        at least 50% of the values in the window must be non-NA.
    window : int
        The size of the rolling window in number of observations. For hourly data,
        a window of 24 would correspond to a 24-hour period.
    station_frac : float, optional
        The minimum fraction of flagged values required to flag an entire station.
        If None, this feature is disabled.

    Returns
    -------
    da : xr.DataArray
        Same as input but cleaned!
    bad_mask : xr.DataArray
        Mask showing which values are bad (True where bad).
        Same dimensionality as da.
    """

    _preflight(da)

    if not (0 <= completeness <= 100):
        raise ValueError("completeness must be between 0 and 100")
    
    bad_mask = da.copy().astype(bool)
    bad_mask.name = 'bad_completeness'
    bad_mask[...] = False

    min_non_na = completeness / 100 * window
    
    for idx in da['space']:
        ida = da[:, idx]
        rolling_non_na = ida.notnull().rolling(time=window, center=True, min_periods = 1).sum()
        ibad_mask = (rolling_non_na < min_non_na).values
        ida[ibad_mask] = np.nan
        da[:, idx] = ida
        bad_mask[:, idx] = ibad_mask

    if station_frac is not None:
        frac_flagged = bad_mask.mean(dim='time')
        bad_stations = (frac_flagged > station_frac).values  # .values here
        
        bad_mask[:, bad_stations] = True
        n_flagged = int(bad_stations.sum())
        print(f"Station completeness filter (station_frac={station_frac}): flagged {n_flagged} stations entirely")

    return da, bad_mask

def filter_changepoint(da, percentile=10, threshold=None, nstd=None, 
                       direction='greater', pen=250, model = 'l1', jump=20, min_size=48):
    """
    Filter out time periods within individual stations where a chosen percentile
    of observed values within a change point segment is abnormally high or low,
    indicating a temporarily blown or miscalibrated sensor.
    Uses change point detection (ruptures library) to identify segments
    within each station's time series, then flags segments where the chosen
    percentile exceeds a threshold. This is designed to catch sensors that are
    only bad for a portion of the year, complementing filter_percentile which
    flags stations that are bad for the entire record.
    This filter should be run AFTER other filters (e.g. filter_linear_gaps,
    filter_spike_groups) so that artifacts do not create change points.
    
    Parameters
    ----------
    da : xr.DataArray
        2D array with dims ("time", "space") for multiple observation
        stations in a given region for a single variable
    percentile : float
        Percentile to calculate for each segment (0-100).
        Use 10 for 10th percentile, 50 for median, etc.
    threshold : float, optional
        Pre-defined percentile threshold. If provided, uses fixed threshold
        instead of calculating from network statistics.
    nstd : float, optional
        Number of standard deviations from mean for network-based threshold.
        Only used if threshold=None.
    direction : str
        Whether to flag segments with percentile "greater" or "less" than
        the threshold.
    pen : float
        Penalty parameter. Higher values detect only larger shifts.
        This changes and requires tuning depending on the model cost function.
    model: str
        Cost function model to use in ruptures. 'l1' is Least Absolute
        Deviation the median.
    jump : int
        Jump parameter. Only every nth index is considered as a candidate
        change point. Higher values are faster but less precise in location.
    min_size : int
        Minimum number of observations per segment. Prevents change points
        from being placed too close together. With hourly data, min_size=48
        means no segment shorter than 48 hours.

    Returns
    -------
    da : xr.DataArray
        Same as input but cleaned!
    bad_mask : xr.DataArray
        Mask showing which values are bad (True where bad).
        Same dimensionality as da.
    """

    da = da.compute()
    _preflight(da)
    if not 0 <= percentile <= 100:
        raise ValueError(f"percentile must be between 0 and 100, got {percentile}")
    if direction not in ('greater', 'less'):
        raise ValueError(f"direction must be 'greater' or 'less', got {direction}")
    if threshold is None and nstd is None:
        raise ValueError("Must provide either threshold or nstd")
    bad_mask = da.copy().astype(bool)
    bad_mask.name = f'bad_changepoint_p{int(percentile)}'
    bad_mask[...] = False

    if threshold is None:
        percentile_per_station = da.quantile(percentile / 100.0, dim='time')
        mean_p = float(percentile_per_station.mean().values)
        std_p = float(percentile_per_station.std().values)
        threshold = (mean_p + nstd * std_p if direction == 'greater' 
                     else mean_p - nstd * std_p)
        mode_msg = f"network-based (mean={mean_p:.2f}, std={std_p:.2f}, nstd={nstd})"
    else:
        mode_msg = "fixed"
    n_stations_flagged = 0
    n_obs_flagged = 0
    n_stations_skipped = 0
    for idx in da['space']:
        print(idx.values)
        ida = da.sel(space=idx)
        # Skip stations that are already fully NaN 
        if ida.isnull().all():
            n_stations_skipped += 1
            continue
        # Extract valid values and their timestamps
        valid_mask = ~np.isnan(ida.values)
        valid_values = ida.values[valid_mask]
        valid_times = da['time'].values[valid_mask]
        # Skip stations too short to segment
        if len(valid_values) < min_size * 2:
            n_stations_skipped += 1
            continue
        # Run change point detection 
        signal = valid_values.reshape(-1, 1)
        try:
            algo = rpt.Pelt(model=model, jump=jump, min_size=min_size).fit(signal)
            result = algo.predict(pen=pen)
        except Exception as e:
            logger.warning(f"Ruptures failed for station {idx.values}: {e}")
            n_stations_skipped += 1
            continue
        # Evaluate each segment
        segment_boundaries = [0] + result
        station_flagged = False
        for start, end in zip(segment_boundaries[:-1], segment_boundaries[1:]):
            segment_values = valid_values[start:end]
            segment_times = valid_times[start:end]
            seg_percentile = np.percentile(segment_values, percentile)
            if direction == 'greater' and seg_percentile > threshold:
                bad_mask.loc[dict(time=segment_times, space=idx)] = True
                n_obs_flagged += len(segment_times)
                station_flagged = True
            elif direction == 'less' and seg_percentile < threshold:
                bad_mask.loc[dict(time=segment_times, space=idx)] = True
                n_obs_flagged += len(segment_times)
                station_flagged = True
        if station_flagged:
            n_stations_flagged += 1
    # Apply mask to data 
    da = da.where(~bad_mask, np.nan)
    msg = (
        f"Change point filter (P{int(percentile)}, {mode_msg}): "
        f"threshold={threshold:.2f}, pen={pen}, jump={jump}, min_size={min_size}, "
        f"flagged {n_stations_flagged} stations ({n_obs_flagged} obs), "
        f"skipped {n_stations_skipped} stations"
    )
    logger.info(msg)
    print(msg)
    return da, bad_mask

def filter_run_on_other_da(da, *, run_on_da, base_filter, base_filter_kwargs=None):
    """
    Generic adapter:
      - compute mask by running base_filter on run_on_da
      - return mask to be applied to da by run_filter_bank
    
    Parameters
    ----------
    da: xr.DataArray
        The data array to be filtered (but not used for mask calculation)
    run_on_da: xr.DataArray
        The data array to run the base_filter on to compute the mask.
    base_filter: function
        A filter function that takes an xr.DataArray and returns (filtered_da, mask)
    base_filter_kwargs: dict, optional
        Keyword arguments to pass to the base_filter function
    
    Returns
    -------
    da: xr.DataArray
        The input data array (unmodified)
    mask: xr.DataArray
        The mask computed by running base_filter on run_on_da, to be applied
        to da.

    """
    _preflight(da)
    _preflight(run_on_da)

    if base_filter_kwargs is None:
        base_filter_kwargs = {}

    _, mask = base_filter(run_on_da.copy(), **base_filter_kwargs)

    return da, mask

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
        kwarg/value will be passed in as arguments to the function. If a 
        ``'post'`` key is included in the config, those filters will be run on
        cleaned data after the initial filters have been applied. 

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
    
    post_config = config.get('post', {})

    final_mask = False
    filter_masks = {}

    for fun, kwargs in config.items():
        if fun == 'post':
            continue
        if not isinstance(kwargs, list):
            kwargs = [kwargs]
        for kwarg in kwargs:
            _, mask = fun(da.copy(), **kwarg)
            print(f"{fun.__name__} mask sum: {int(mask.sum())}")
            print(f"da NaN count before: {int(np.isnan(da).sum())}")
            filter_masks[fun] = filter_masks.get(fun, [])
            filter_masks[fun].append(mask)
            final_mask |= mask

    da = da.where(~final_mask, np.nan)

    for fun, kwargs in post_config.items():
        if not isinstance(kwargs, list):
            kwargs = [kwargs]
        for kwarg in kwargs:
            _, mask = fun(da, **kwarg)
            if fun not in filter_masks:
                filter_masks[fun] = []
            filter_masks[fun].append(mask)
            final_mask |= mask

    da = da.where(~final_mask, np.nan)

    return da, final_mask, filter_masks

CONFIG_WS10M = {
    filter_spike_groups: dict(N=3, value_thresh=20, up_thresh=20,
                              down_thresh=20, nan_spikes=True),
    filter_value_range: dict(lower=0, upper=66),
    filter_constant: dict(value=None),
    filter_percentile: [dict(percentile=10, nstd=3, threshold=None, direction='greater'),
                        dict(percentile=95, threshold=1, direction='less')],
    filter_run_on_other_da: dict(run_on_da='abs_diff_windspeed_10m', base_filter=filter_changepoint, base_filter_kwargs=dict(percentile=50, threshold=4.2, direction='greater', pen=250, jump=20, min_size=168)),
    'post': {filter_run_on_other_da: dict(run_on_da='abs_diff_windspeed_10m', base_filter=filter_mean, base_filter_kwargs=dict(threshold=4.2, direction = 'greater')),
             filter_completeness: dict(completeness=80, window=24, station_frac=None),}
}

"""Config for run_filter_bank to clean 10-meter windspeed.
Upper limit of based on regional maximum official wind gust of 66 m/s.
Ex-percentile of 1 to retain extreme windspeed events. Median of station mean
absolute windspeed difference is 1.799 with a scaled MAD of 0.64. Threshold of
4.2 for the percentile filter is based on median + 5*scaled MAD of
the distribution of station mean absolute windspeed differences."""

CONFIG_WDIR10M = {
    filter_value_range: dict(lower=0, upper=360),
    filter_constant: dict(value=None),
    'post': {filter_completeness: dict(completeness=80, window=24, station_frac=None),}
}
"""Config for run_filter_bank to clean 10-meter wind direction."""

CONFIG_GUST10M = {
    filter_spikes: dict(diff_threshold=20, ex_percentile=1),
    filter_value_range: dict(lower=0, upper=66),
    filter_constant: dict(value=None),
    'post': {filter_completeness: dict(completeness=80, window=24, station_frac=0.5),}
}
"""Config for run_filter_bank to clean 10-meter wind gusts.
Upper limit of based on regional maximum official wind gust of 66 m/s.
Ex-percentile of 1 to retain extreme windspeed events."""

CONFIG_P0M = {
    filter_constant: dict(value=None),
    'post': {filter_completeness: dict(completeness=80, window=24, station_frac=0.5),}
}
"""Config for ``run_filter_bank`` to clean near-surface pressure"""

CONFIG_T2M = {
    filter_spikes: dict(diff_threshold=15, ex_percentile=2),
    filter_constant: dict(value=None),
    filter_value_range: dict(lower=-60, upper=60),
    filter_run_on_other_da: dict(run_on_da='abs_diff_temperature_2m', base_filter=filter_changepoint, base_filter_kwargs=dict(percentile=50, threshold=4.4, direction='greater', pen=250, jump=20, min_size=168)),
    'post': {filter_run_on_other_da: dict(run_on_da='abs_diff_temperature_2m', base_filter=filter_mean, base_filter_kwargs=dict(threshold=4.4, direction = 'greater')),
            filter_completeness: dict(completeness=80, window=24, station_frac=None),}
}
"""Config for ``run_filter_bank`` to clean near-surface 2-meter temperature.
Threshold of 4.4 for the percentile filter is based on the median
+ 4*scaled MAD of the distribution of station mean absolute temperature differences."""

CONFIG_RH2M = {
    filter_spikes: dict(diff_threshold=15, ex_percentile=2),
    filter_value_range: dict(lower=0, upper=100),
    filter_constant: dict(value=None),
    'post': {filter_completeness: dict(completeness=80, window=24, station_frac=None),}
}
"""Config for ``run_filter_bank`` to clean near-surface 2-meter relative
humidity"""

CONFIG_PRECIP1H = {
        filter_value_range: dict(lower=0, upper=.5) }
"""Config for ``run_filter_bank`` to clean near-surface 1-hour precipitation"""