"""Custom QA/QC configurations for MADIS observational data across CONUS"""

from madis.qaqc.filters import filter_completeness, filter_spikes, \
    filter_spike_groups, filter_value_range, \
    filter_linear_gaps, filter_constant, filter_percentile, \
    filter_changepoint, filter_error, filter_mean, filter_run_on_other_da

MASK_DEPENDENCIES = {
    "winddirection_10m": "windspeed_10m",
    "windgust_10m": "windspeed_10m",
}
"""Config for run_filter_bank to apply windspeed mask to all wind variables"""

CLEAN_ORDER = [
    "windspeed_10m",
    "winddirection_10m",
    "windgust_10m",
    "temperature_2m",
    "relativehumidity_2m",
]
"""Order in which to clean variables, windspeed first to apply mask to be
applied to all wind variables"""

CONFIG_WS10M = [
    { # stage 1
        filter_completeness: dict(
            completeness=8, window=8760, station_frac=None
        ),
        filter_spike_groups: dict(
            N=3, value_thresh=20, up_thresh=20, down_thresh=20, nan_spikes=True
        ),
        filter_spikes: dict(diff_threshold=20, ex_percentile=1),
        filter_value_range: dict(lower=0, upper=50),
        filter_linear_gaps: dict(diff_threshold=1e-4, n_threshold=4),
        filter_constant: dict(value=None),
        filter_percentile: [
            dict(
                percentile=10,
                method="median",
                nstd=6,
                threshold=None,
                direction="greater",
            ),
            dict(percentile=95, nstd=None, threshold=1, direction="less"),
        ],
        filter_run_on_other_da: [
            dict(
                run_on_da="abs_diff_windspeed_10m",
                base_filter=filter_changepoint,
                base_filter_kwargs=dict(
                    percentile=50,
                    threshold=8,
                    direction="greater",
                    pen=300,
                    jump=20,
                    min_size=168,
                ),
                apply_to="both",
            ),
            dict(
                run_on_da="abs_diff_windspeed_10m",
                base_filter=filter_error,
                base_filter_kwargs=dict(threshold=45),
                apply_to="both",
            ),
        ],
    },
    { # stage 2
        filter_run_on_other_da: dict(
            run_on_da="abs_diff_windspeed_10m",
            base_filter=filter_mean,
            base_filter_kwargs=dict(threshold=8, direction="greater"),
            apply_to="run_on_da",
        ),
        filter_spikes: dict(diff_threshold=20, ex_percentile=1),
        filter_spike_groups: dict(
            N=3, value_thresh=20, up_thresh=20, down_thresh=20, nan_spikes=True
        ),
    },
    { # stage 3
        filter_completeness: dict(
            completeness=50, window=24, station_frac=None
        ),
    },
]
"""Config for run_filter_bank to clean near-surface wind speed.
"abs_diff_windspeed_10m" is the difference between the observation and HRRR
10-m windspeed using nearest-neighbor interpolation."""

CONFIG_WDIR10M = [
    { # stage 1
        filter_completeness: dict(
            completeness=8, window=8760, station_frac=None
        ),
        filter_value_range: dict(lower=0, upper=360),
        filter_linear_gaps: dict(diff_threshold=1e-4, n_threshold=4),
        filter_constant: dict(value=None),
    },
    { # stage 2
        filter_completeness: dict(
            completeness=50, window=24, station_frac=None
        ),
    },
]
"""Config for run_filter_bank to clean near-surface wind direction."""

CONFIG_GUST10M = [
    { # stage 1
        filter_completeness: dict(
            completeness=8, window=8760, station_frac=None
        ),
        filter_spikes: dict(diff_threshold=20, ex_percentile=1),
        filter_spike_groups: dict(
            N=3, value_thresh=20, up_thresh=20, down_thresh=20, nan_spikes=True
        ),
        filter_linear_gaps: dict(diff_threshold=1e-4, n_threshold=4),
        filter_percentile: [
            dict(
                percentile=10,
                method="median",
                nstd=8,
                threshold=None,
                direction="greater",
            ),
            dict(percentile=95, nstd=None, threshold=2, direction="less"),
        ],
        filter_value_range: dict(lower=0, upper=103),
        filter_constant: dict(value=None),
        filter_run_on_other_da: [
            dict(
                run_on_da="abs_diff_windgust_10m",
                base_filter=filter_changepoint,
                base_filter_kwargs=dict(
                    percentile=50,
                    threshold=8,
                    direction="greater",
                    pen=300,
                    jump=20,
                    min_size=168,
                ),
                apply_to="both",
            ),
            dict(
                run_on_da="abs_diff_windgust_10m",
                base_filter=filter_error,
                base_filter_kwargs=dict(threshold=43),
                apply_to="both",
            ),
        ],
    },
    { # stage 2
        filter_run_on_other_da: dict(
            run_on_da="abs_diff_windgust_10m",
            base_filter=filter_mean,
            base_filter_kwargs=dict(threshold=13, direction="greater"),
            apply_to="run_on_da",
        ),
        filter_spikes: dict(diff_threshold=20, ex_percentile=1),
        filter_spike_groups: dict(
            N=3, value_thresh=20, up_thresh=20, down_thresh=20, nan_spikes=True
        ),
    },
    { # stage 3
        filter_completeness: dict(
            completeness=50, window=24, station_frac=None
        ),
    },
]
"""Config for run_filter_bank to clean near-surface wind gusts.
Upper limit of based on regional maximum official wind gust of 103 m/s.
Ex-percentile of 1 to retain extreme wind events.
"abs_diff_windgust_10m" is the difference between the observation and HRRR
10-m windgust using nearest-neighbor interpolation."""

CONFIG_T2M = [
    { # stage 1
        filter_completeness: dict(
            completeness=8, window=8760, station_frac=None
        ),
        filter_spikes: dict(diff_threshold=15, ex_percentile=2),
        filter_constant: dict(value=None),
        filter_linear_gaps: dict(diff_threshold=0.05, n_threshold=4),
        filter_value_range: dict(lower=-60, upper=60),
        filter_run_on_other_da: [
            dict(
                run_on_da="abs_diff_temperature_2m",
                base_filter=filter_changepoint,
                base_filter_kwargs=dict(
                    percentile=50,
                    threshold=10,
                    direction="greater",
                    pen=300,
                    jump=20,
                    min_size=24,
                ),
                apply_to="both",
            ),
            dict(
                run_on_da="abs_diff_temperature_2m",
                base_filter=filter_error,
                base_filter_kwargs=dict(threshold=8.5),
                apply_to="both",
            ),
        ],
    },
    { # stage 2
        filter_run_on_other_da: dict(
            run_on_da="abs_diff_temperature_2m",
            base_filter=filter_mean,
            base_filter_kwargs=dict(threshold=8.5, direction="greater"),
            apply_to="run_on_da",
        ),
        filter_spikes: dict(diff_threshold=15, ex_percentile=2),
    },
    { # stage 3
        filter_completeness: dict(
            completeness=50, window=24, station_frac=None
        ),
    },
]
"""Config for ``run_filter_bank`` to clean near-surface temperature.
"abs_diff_temperature_2m" is the difference between the observation and HRRR
2-m temperature using nearest-neighbor interpolation."""

CONFIG_RH2M = [
    { # stage 1
        filter_completeness: dict(
            completeness=8, window=8760, station_frac=None
        ),
        filter_value_range: dict(lower=0, upper=100),
        filter_linear_gaps: dict(diff_threshold=1e-4, n_threshold=4),
        filter_constant: dict(value=None),
        filter_run_on_other_da: [
            dict(
                run_on_da="abs_diff_relativehumidity_2m",
                base_filter=filter_changepoint,
                base_filter_kwargs=dict(
                    percentile=50,
                    threshold=30,
                    direction="greater",
                    pen=300,
                    jump=20,
                    min_size=168,
                ),
                apply_to="both",
            ),
            dict(
                run_on_da="abs_diff_relativehumidity_2m",
                base_filter=filter_error,
                base_filter_kwargs=dict(threshold=56),
                apply_to="both",
            ),
        ],
    },
    { #stage 2
        filter_run_on_other_da: dict(
            run_on_da="abs_diff_relativehumidity_2m",
            base_filter=filter_mean,
            base_filter_kwargs=dict(threshold=25, direction="greater"),
            apply_to="run_on_da",
        ),
    },
    { # stage 3
        filter_completeness: dict(
            completeness=50, window=24, station_frac=None
        ),
    },
]

"""Config for ``run_filter_bank`` to clean near-surface relative
humidity. "abs_diff_relativehumidity_2m" is the difference between the
observation and HRRR 2-m relative humidity using nearest-neighbor
interpolation."""


CONFIG_MAP = {
    "windspeed_10m": CONFIG_WS10M,
    "winddirection_10m": CONFIG_WDIR10M,
    "windgust_10m": CONFIG_GUST10M,
    "temperature_2m": CONFIG_T2M,
    "relativehumidity_2m": CONFIG_RH2M,
}
