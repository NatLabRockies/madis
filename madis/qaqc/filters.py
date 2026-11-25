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
    functions

    Parameters
    ----------
    da : xr.DataArray
        2D array with coordinates ("time", "space") for multiple observation
        stations in a given region for a single variable

    Raises
    ------
    AssertionError
        If da is not an xr.DataArray, is not 2-dimensional, or does not have
        dims ("time", "space")
    """
    assert isinstance(da, xr.DataArray)
    assert da.ndim == 2
    assert da.dims == ('time', 'space')


def filter_linear_gaps(da, diff_threshold=1e-4, n_threshold=3):
    """Some MADIS timeseries data appears to have large gaps of linear values.
    This function identifies long blocks of consecutive linear data and sets
    that data to NaN

    This filter has been applied to MADIS station data for temperature,
    humidity, windspeed, wind direction, and wind gust.

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
        msg = (
            'CAUTION!! If MADIS data is interpolated from hourly to 30min '
            'then n_threshold=3 will find linear blocks EVERYWHERE!'
        )
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


def filter_spikes(da, diff_threshold=15, ex_percentile=2):
    """Some MADIS timeseries data appears to have momentary extreme spikes.
    This function identifies spikes that are both dramatic (large 1st
    derivative) and also extreme (outside a percentile threshold).

    This filter has been applied to MADIS station data for temperature,
    humidity, windspeed, wind direction, and wind gust.

    Parameters
    ----------
    da : xr.DataArray
        2D array with coordinates ("time", "space") for multiple observation
        stations in a given region for a single variable
    diff_threshold : float
        If the first-derivative of da is greater than this value
        (including spanning across NaN values in da) then a value
        is considered to be a spike if it also meets the
        ex_percentile threshold.
    ex_percentile : float
        If the value of the spike is less than this percentile or
        greater than the percentile (100-ex_percentile), then the
        value is considered to be a spike if it also meets the
        diff_threshold.

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
        spike_mask_compressed[1:-1] = (diff_nonan[:-1] > diff_threshold) | (
            diff_nonan[1:] > diff_threshold
        )

        top_p = 100 - ex_percentile
        extreme_compressed = (
            valid_values > np.nanpercentile(valid_values, top_p)
        ) | (valid_values < np.nanpercentile(valid_values, ex_percentile))

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

    This filter was developed on Mesonet station data for windspeed, and has
    been applied to wind speed and wind gust observations.

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
        If this is true, replace NaN values with 0 before calculating the
        derivative.This allows a NaN immediately before or after the group N to
        contribute to the derivative calculation.

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
    bad_mask.name = 'bad_spike_group'
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
        ends = np.where(edges == -1)[0] - 1  # index in where ends (inclusive)

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
    filter out values that are unrealistic based on time of year in the area.

    This filter was developed on Mesonet station data for temperature and
    humidity and has not yet been tested on other variables. Leave comments
    here when other variables have been successfully filtered with this.

    CAUTION!! This filter is most appropriate for networks within a small,
    climatically similar region.

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
    print(
        f'Seasonal anomaly filter (nstd={nstd}): removed {n_removed} values'
        f' out of {n_total} ({100 * n_removed / n_total:.2f}%)'
    )

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
    """Some MADIS observation locations have constant values for all time
    steps. This sets those values to NaN so they can be dropped later.

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
    bad_mask[:, constant] = True

    return da, bad_mask


def filter_percentile(
    da, percentile=None, method=None, nstd=3, threshold=None, direction=None
):
    """
     Filter out ENTIRE stations where a chosen percentile of observed values
     is abnormally high or low, indicating systematic sensor issues or
     miscalibration.

     This filter operates in two ways:
     1. Calculates a threshold based on the distribution of the chosen
     percentile across all stations in the network (requires method,
    nstd, and direction).
     2. Uses a pre-defined threshold (requires threshold and direction).

     When using network-based thresholding (method is not None), two approaches
     are available:
     1. method='mean': threshold is mean ± nstd * std of the per-station
     percentiles across the network.
     2. method='median': threshold is median ± nstd * scaled_MAD of the
     per-station percentiles across the network, where scaled_MAD =
     1.4826 * MAD.

     This filter was developed on MADIS station data for wind speed
     and wind gusts.

     Example: For Mesonet network in 2024 in the greater Colorado-Wyoming
     region with percentile=10, method = mean, nstd=3, direction = 'greater'
     generates a 10th percentile threshold of 4.2 m/s, which flags 17 out of
     3415 stations.

    Parameters
    ----------
     da : xr.DataArray
         2D array with dims ("time", "space") for multiple observation
         stations in a given region for a single variable
     percentile : float
         Percentile to calculate for each station (0-100).
         Use 10 for 10th percentile, 50 for median, etc.
     method : str, optional
         Statistic used to compute the network-wide center for threshold
         calculation. Must be 'mean' or 'median'. Required if threshold=None.
     nstd : float
         Number of standard deviations (or MAD equivalents) from the
         network-wide mean (or median) used to compute a threshold.
         Only used if threshold=None (network-based mode). Default is 3.
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
        raise ValueError(
            f'percentile must be between 0 and 100, got {percentile}'
        )
    if direction not in ('greater', 'less'):
        raise ValueError(
            f"direction must be 'greater' or 'less', got {direction}"
        )

    percentile_per_station = da.quantile(
        percentile / 100.0, dim='time', skipna=True
    )

    if threshold is not None:
        mode_msg = 'fixed'
    elif method == 'mean':
        mean_p = percentile_per_station.mean().values
        std_p = float(percentile_per_station.std(skipna=True).values)
        threshold = (
            mean_p + nstd * std_p
            if direction == 'greater'
            else mean_p - nstd * std_p
        )
        mode_msg = (
            f'network-based (mean={mean_p:.2f}, std={std_p:.2f}, nstd={nstd})'
        )
    elif method == 'median':
        median_p = percentile_per_station.median().values
        mad_p = np.nanmedian(np.abs(percentile_per_station.values - median_p))
        scaled_mad_p = 1.4826 * mad_p
        threshold = (
            median_p + nstd * scaled_mad_p
            if direction == 'greater'
            else median_p - nstd * scaled_mad_p
        )
        mode_msg = (
            f'network-based (median={median_p:.2f},'
            f'scaled_mad={scaled_mad_p:.2f}, nstd={nstd})'
        )
    else:
        raise ValueError(
            f"method must be 'mean' or 'median' when threshold "
            f"is not provided, got '{method}'"
        )

    if direction == 'greater':
        flagged_stations = percentile_per_station > threshold
    else:
        flagged_stations = percentile_per_station < threshold

    bad_mask = da.copy().astype(bool)
    bad_mask.name = f'bad_high_p{int(percentile)}_stations'
    bad_mask[...] = False
    bad_mask[:, flagged_stations] = True

    da[:, flagged_stations] = np.nan

    msg = (
        f'P{int(percentile)} filter ({mode_msg}): threshold={threshold:.2f}, '
        f'flagged {flagged_stations.sum().values} stations'
    )
    logger.info(msg)
    print(msg)

    return da, bad_mask


def filter_mean(da, threshold, direction='greater'):
    """Filter out an ENTIRE stations where the mean of a variable across time
     is greater than a chosen threshold. This is designed to be applied to the
     error between MADIS observations and a reference dataset (e.g. HRRR f02)
     to catch stations with persistent sensor bias.

    Parameters
    ----------
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
        raise ValueError(
            f"direction must be 'greater' or 'less', got {direction}"
        )

    mean_per_station = da.mean(dim='time', skipna=True)
    flagged = (
        mean_per_station > threshold
        if direction == 'greater'
        else mean_per_station < threshold
    )

    bad_mask = xr.zeros_like(da, dtype=bool)
    bad_mask.name = 'bad_mean_stations'
    bad_mask[:, flagged] = True
    da[:, flagged] = np.nan

    return da, bad_mask


def filter_error(da, threshold):
    """Filter individual time step where the absolute error between the
    observations and the HRRR f02 forecast exceeds a threshold.

    Parameters
    ----------
    da : xr.DataArray
        2D array with coordinates ("time", "space") for multiple observation
        stations in a given region for a single variable
    threshold : float
        Absolute value threshold above which individual time steps are
        flagged as erroneous.

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
    bad_mask[...] = False
    bad_mask.name = 'bad_error'
    bad_mask = abs(da) > threshold
    da = da.where(~bad_mask)
    return da, bad_mask


def filter_completeness(da, completeness=80, window=24, station_frac=None):
    """
    Filter out time periods within individual stations where the percentage of
    NA values within a rolling window exceeds a specified completeness
    threshold.
    This is designed to catch sensors that are temporarily offline or have
    communication issues, resulting in many missing values for a period of
    time.

    Parameters
    ----------
    da : xr.DataArray
        The input data array containing observation variables with dimensions
        ("time", "space").
    completeness : float
        The minimum percentage of non-NA values required within the rolling
        window for the data to be considered valid. For example,
        completeness=50 means at least 50% of the values in the window must
        be non-NA.
    window : int
        The size of the rolling window in number of observations.
        For hourly data, a window of 24 would correspond to a 24-hour period.
    station_frac : float, optional
        The minimum fraction of flagged values required to flag an entire
        station. If None, this feature is disabled.

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
        raise ValueError('completeness must be between 0 and 100')

    bad_mask = da.copy().astype(bool)
    bad_mask.name = 'bad_completeness'
    bad_mask[...] = False

    min_non_na = completeness / 100 * window

    for idx in da['space']:
        ida = da[:, idx]
        rolling_non_na = (
            ida.notnull()
            .rolling(time=window, center=True, min_periods=1)
            .sum()
        )
        ibad_mask = (rolling_non_na < min_non_na).values
        ida[ibad_mask] = np.nan
        da[:, idx] = ida
        bad_mask[:, idx] = ibad_mask

    if station_frac is not None:
        frac_flagged = bad_mask.mean(dim='time')
        bad_stations = (frac_flagged > station_frac).values  # .values here

        bad_mask[:, bad_stations] = True
        n_flagged = int(bad_stations.sum())
        print(
            f'Station completeness filter (station_frac={station_frac}): '
            f'flagged {n_flagged} stations entirely'
        )

    return da, bad_mask


def _process_station_changepoint(
    idx,
    ida_values,
    time_values,
    percentile,
    threshold,
    direction,
    model,
    jump,
    min_size,
    pen,
):
    """
    Helper function to process a single station for the filter_changepoint
    function.

    Parameters
    ----------
    idx : str
        Value from the space coordinate.
    ida_values : np.ndarray
        1D array of observation values for the station.
    time_values : np.ndarray
        1D array of timestamps corresponding to the observation values.
    percentile : float
        Percentile to calculate for each segment (0-100).
        Use 10 for 10th percentile, 50 for median, etc.
    threshold : float
        Threshold to compare the segment percentile against.
    direction : str
        Whether to flag segments with percentile "greater" or "less" than
        the threshold.
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
    pen : float
        Penalty parameter. Higher values detect only larger shifts.
        This changes and requires tuning depending on the model cost function.

    Returns
    -------
    idx : str
        Value from the space coordinate.
    bad_times : list or None
        Timestamps to flag, or None if skipped.
    n_obs_flagged : int
        Count of flagged observations.
    skipped : bool
        True if the station was skipped, False otherwise.
    """

    skipped = (idx, None, 0, True)

    # Skip stations that are already fully NaN
    if np.all(np.isnan(ida_values)):
        return skipped

    # Extract valid values and their timestamps
    valid_mask = ~np.isnan(ida_values)
    valid_values = ida_values[valid_mask]
    valid_times = time_values[valid_mask]

    # Skip stations too short to segment
    if len(valid_values) < min_size * 2:
        return skipped

    # Run change point detection
    signal = valid_values.reshape(-1, 1)
    try:
        algo = rpt.Pelt(model=model, jump=jump, min_size=min_size).fit(signal)
        result = algo.predict(pen=pen)
    except Exception as e:
        logger.warning(f'Ruptures failed for station {idx}: {e}')
        return skipped

    # Evaluate each segment
    segment_boundaries = [0] + result
    station_flagged = False
    bad_times = []
    n_obs_flagged = 0
    for start, end in zip(segment_boundaries[:-1], segment_boundaries[1:]):
        segment_values = valid_values[start:end]
        segment_times = valid_times[start:end]
        seg_percentile = np.percentile(segment_values, percentile)
        if (direction == 'greater' and seg_percentile > threshold) or (
            direction == 'less' and seg_percentile < threshold
        ):
            bad_times.extend(segment_times)
            n_obs_flagged += len(segment_times)
            station_flagged = True

    return idx, bad_times, n_obs_flagged, station_flagged


def filter_changepoint(
    da,
    percentile=10,
    threshold=None,
    nstd=None,
    direction='greater',
    pen=250,
    model='l1',
    jump=20,
    min_size=48,
):
    """
    Filter out time periods within individual stations where a chosen
    percentileof observed values within a change point segment is abnormally
    high or low, indicating a malfunctioning sensor for that period of time.
    Uses change point detection (ruptures library) to identify segments
    within each station's time series, then flags segments where the chosen
    percentile exceeds a threshold. This is designed to catch sensors that are
    only bad for a portion of the year, complementing filter_percentile which
    flags stations that are bad for the entire record.

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
        raise ValueError(
            f'percentile must be between 0 and 100, got {percentile}'
        )
    if direction not in ('greater', 'less'):
        raise ValueError(
            f"direction must be 'greater' or 'less', got {direction}"
        )
    if threshold is None and nstd is None:
        raise ValueError('Must provide either threshold or nstd')
    bad_mask = da.copy().astype(bool)
    bad_mask.name = f'bad_changepoint_p{int(percentile)}'
    bad_mask[...] = False

    if threshold is None:
        percentile_per_station = da.quantile(
            percentile / 100.0, dim='time', skipna=True
        )
        mean_p = float(percentile_per_station.mean().values)
        std_p = float(percentile_per_station.std().values)
        threshold = (
            mean_p + nstd * std_p
            if direction == 'greater'
            else mean_p - nstd * std_p
        )
        mode_msg = (
            f'network-based (mean={mean_p:.2f}, std={std_p:.2f}, nstd={nstd})'
        )
    else:
        mode_msg = 'fixed'
    n_stations_flagged = 0
    n_obs_flagged = 0
    n_stations_skipped = 0
    da_values = da.values
    time_values = da['time'].values
    space_values = da['space'].values

    for i, idx in enumerate(space_values):
        idx_val, bad_times, n_obs, skip_reason = _process_station_changepoint(
            idx,
            da_values[:, i],
            time_values,
            percentile,
            threshold,
            direction,
            model,
            jump,
            min_size,
            pen,
        )
        if skip_reason:
            n_stations_skipped += 1
            continue
        if bad_times:
            bad_mask.loc[dict(time=bad_times, space=idx_val)] = True
            n_obs_flagged += n_obs
            n_stations_flagged += 1

    # Apply mask to data
    da = da.where(~bad_mask, np.nan)
    msg = (
        f'Change point filter (P{int(percentile)}, {mode_msg}): '
        f'threshold={threshold:.2f}, pen={pen}, jump={jump},'
        f'min_size={min_size},'
        f'flagged {n_stations_flagged} stations ({n_obs_flagged} obs), '
        f'skipped {n_stations_skipped} stations'
    )
    logger.info(msg)
    print(msg)
    return da, bad_mask


def filter_run_on_other_da(
    da, *, run_on_da, base_filter, base_filter_kwargs=None, apply_to=None
):
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
        A filter function that takes an xr.DataArray and returns (filtered_da,
        mask)
    base_filter_kwargs: dict, optional
        Keyword arguments to pass to the base_filter function
    apply_to: str, optional
        Specifies which data array(s) to apply the mask to. Options are:
        - 'da': apply mask to da only
        - 'run_on_da': apply mask to run_on_da only
        - 'both': apply mask to both da and run_on_da

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

    if apply_to in ('da', 'both'):
        da = da.where(~mask, np.nan)
    if apply_to in ('run_on_da', 'both'):
        run_on_da[...] = run_on_da.where(~mask, np.nan)

    return da, mask
