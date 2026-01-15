"""MADIS data processing functionality - QC, conversion, and netCDF writing."""

import logging
import multiprocessing
import os
from glob import glob

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from rex import init_logger
from tqdm import tqdm

from madis.common import get_dates
from madis.constants import BASEURL, FREQ, OUT_VARIABLES, RAW_VARIABLES

init_logger(__name__, log_level='INFO')
logger = logging.getLogger(__name__)


class MadisProcessor:
    """Class to process MADIS unpacked data files."""

    def __init__(
        self,
        network,
        download_dir,
        unpack_dir=None,
        process_dir=None,
        start_date=None,
        end_date=None,
        min_qc_level=0,
    ):
        """Initialize the MadisProcessor.

        Parameters
        ----------
        network : str
            The MADIS network.
        download_dir : str
            The base directory.
        unpack_dir : str, optional
            The directory containing unpacked CSV files. If None, uses
            download_dir.
        process_dir : str, optional
            The directory to save processed files. If None, uses
            download_dir.
        start_date : str, optional
            Start date in 'YYYY-MM-DD' format.
        end_date : str, optional
            End date in 'YYYY-MM-DD' format.
        min_qc_level : int, optional
            Minimum QC level (0, 1, 2, or 3). Default is 0. 0 means no
            QC applied.
        """

        unpack_dir = unpack_dir or download_dir
        process_dir = process_dir or download_dir
        self.network = network
        self.unpack_dir = f'{unpack_dir}/unpacked/{network}'
        self.process_dir = f'{process_dir}/processed/{network}'
        self.start_date = start_date
        self.end_date = end_date
        self.dates = get_dates(start_date, end_date)
        self.min_level = min_qc_level
        os.makedirs(self.process_dir, exist_ok=True)

    def unpacked_files(self, year):
        """Get a list of unpacked files for the network."""
        return sorted(glob(f'{self.unpack_dir}/{year}/*.csv', recursive=True))

    def processed_files(self, year):
        """Get a list of processed files for the network."""
        return sorted(glob(f'{self.process_dir}/{year}/*.csv', recursive=True))

    def run(self, bbox, max_workers=32, overwrite=False):
        """Process the unpacked data into cleaned timeseries csv files.

        Parameters
        ----------
        bbox : list
            Bounding box [max_lat, min_lon, min_lat, max_lon].
        max_workers : int, optional
            Maximum number of parallel workers. Default is 32.
        overwrite : bool, optional
            Whether to overwrite existing files. Default is False.
        """
        for year in {date.year for date in self.dates}:
            files = self.unpacked_files(year)
            if not files:
                logger.warning(
                    f'No unpacked files found for {self.network} network for '
                    f'year {year}.'
                )
                continue
            os.makedirs(f'{self.process_dir}/{year}', exist_ok=True)
            files = self.filter_files_for_region(files, bbox)
            out_files = [
                f'{self.process_dir}/{year}/{os.path.basename(file)}'
                for file in files
            ]
            overwrites = [overwrite] * len(out_files)
            min_levels = [self.min_level] * len(out_files)
            years = [year] * len(out_files)
            if (
                all(os.path.exists(file) for file in out_files)
                and not overwrite
            ):
                logger.info(
                    f'Processed files for {self.network} network for year '
                    f'{year} and bbox {bbox} already exist. Skipping '
                    'processing.'
                )
                continue
            logger.info(
                f'Processing {len(files)} unpacked files for {self.network} '
                f'network for year {year}...'
            )
            with multiprocessing.Pool(max_workers) as pool:
                pool.starmap(
                    self.write_processed_csv,
                    zip(files, out_files, years, min_levels, overwrites),
                )

    @classmethod
    def set_regular_timestep(cls, data, year):
        """Sample the data to a regular timestep and interpolate missing
        values, as long as they are close to existing values.

        Parameters
        ----------
        data : pd.DataFrame
            The dataframe to resample.
        year : int
            The year.

        Returns
        -------
        pd.DataFrame
            Resampled dataframe with regular timestep.
        """
        ti = pd.date_range(
            f'01-01-{year} 00:00:00',
            f'01-01-{year + 1} 00:00:00',
            freq=FREQ,
        )[:-1]
        tmp = (
            data.copy()
            if 'time' not in data.columns
            else data.set_index('time')
        )
        tmp.index = pd.to_datetime(tmp.index)
        tmp = tmp.groupby(tmp.index).first()
        new_ti = tmp.index.append(ti).sort_values().drop_duplicates()
        tmp = tmp.reindex(new_ti)
        tmp = tmp.interpolate(method='time', limit=2)
        tmp = tmp.resample(FREQ).first()
        if len(tmp) != len(ti):
            msg = (
                f'Error: {len(tmp)} time steps found, expected '
                f'{len(ti)} for {year}.'
            )
            logger.error(msg)
            raise ValueError(msg)
        return tmp

    @classmethod
    def select_process_variables(cls, data):
        """Select only the variables to process from the unpacked data.

        Parameters
        ----------
        data : pd.DataFrame
            The dataframe to filter.

        Returns
        -------
        pd.DataFrame
            Filtered dataframe with only process variables.
        """
        var_names = list(RAW_VARIABLES) + [f'{v}DD' for v in RAW_VARIABLES]
        return data[list(set(var_names) & set(data.columns))]

    @classmethod
    def merge_dup_cols(cls, df, col):
        """Merge duplicate columns in a dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe.
        col : str
            The column name to merge.

        Returns
        -------
        pd.DataFrame
            Dataframe with merged columns.
        """
        if col in df.columns and len(df[col].shape) > 1:
            tmp = np.nanmean(df[col].values, axis=1)
            df = df.drop(col, axis=1)
            df[col] = tmp
        return df

    @classmethod
    def write_processed_csv(
        cls, infile, outfile, year, min_level=0, overwrite=False
    ):
        """Write the processed data to a csv file.

        Parameters
        ----------
        infile : str
            Input unpacked CSV file path.
        outfile : str
            Output processed CSV file path.
        year : int
            The year.
        min_level : int, optional
            Minimum QC level (0, 1, 2, or 3). Default is 0. 0 means no QC
            applied.
        overwrite : bool, optional
            Whether to overwrite existing file. Default is False.
        """
        if os.path.exists(outfile) and not overwrite:
            logger.debug(f'File {outfile} already exists. Skipping write.')
            return

        df = cls._read_csv(infile)

        df = cls.select_process_variables(df)

        df = cls.set_regular_timestep(df, year=year)

        logger.debug(f'Doing QC on {infile}.')
        df = cls.do_quality_control(df, min_level=min_level)

        qc_cols = [c for c in df.columns if 'DD' in c]
        df = df.drop(qc_cols, axis=1)
        df = df.rename(RAW_VARIABLES, axis=1)
        df = cls.merge_dup_cols(df, 'pressure_0m')

        # Ensure all required output variables are present
        for var_name in set(RAW_VARIABLES.values()):
            if var_name not in df:
                df[var_name] = np.nan
        df = df[list(set(RAW_VARIABLES.values()))]

        if len(df) == 0:
            logger.warning(
                f'No data found in {infile} after processing. Skipping file.'
            )
            return

        for col in set(RAW_VARIABLES.values()):
            mask = np.abs(df[col].values) > 1e30
            df.loc[mask, col] = np.nan

        try:
            df = cls.convert_vars(df, do_qc=(min_level > 0))
            df = cls.enforce_ranges(df, do_qc=(min_level > 0))
        except Exception as e:
            logger.error(f'Error processing {infile}: {e}')
            raise

        df['time'] = df.index.astype(str)
        df.to_csv(outfile + '.tmp')
        os.replace(outfile + '.tmp', outfile)
        logger.debug(f'Saved {outfile}')

    @classmethod
    def filter_files_for_region(cls, files, bbox):
        """Filter files based on the bounding box.

        Parameters
        ----------
        files : list
            List of file paths.
        bbox : list
            Bounding box [max_lat, min_lon, min_lat, max_lon].

        Returns
        -------
        np.ndarray
            Array of filtered file paths.
        """
        basefiles = [
            os.path
            .basename(file)
            .replace('.csv', '')
            .replace('.gz', '')
            .split('_')
            for file in files
        ]
        lats = np.array([bf[1] for bf in basefiles], dtype=float)
        lons = np.array([bf[2] for bf in basefiles], dtype=float)
        coord_mask = cls.get_coord_mask(lats=lats, lons=lons, bbox=bbox)
        lats = lats[coord_mask]
        lons = lons[coord_mask]
        return np.array(files)[coord_mask]

    @classmethod
    def get_coord_mask(cls, lats, lons, bbox):
        """Get the valid mask for the given region.

        Parameters
        ----------
        lats : np.ndarray
            Array of latitudes.
        lons : np.ndarray
            Array of longitudes.
        bbox : list
            Bounding box [max_lat, min_lon, min_lat, max_lon].

        Returns
        -------
        np.ndarray
            Boolean mask for valid coordinates.
        """
        region_mask = (
            (lats >= bbox[2])
            & (lats <= bbox[0])
            & (lons >= bbox[1])
            & (lons <= bbox[3])
        )
        return region_mask

    @classmethod
    def _read_csv(cls, file):
        """Read a csv file and return a dataframe. Picklable method with
        logging.

        Parameters
        ----------
        file : str
            Path to CSV file.

        Returns
        -------
        pd.DataFrame
            The loaded dataframe.
        """
        logger.debug(f'Reading file: {file}')
        try:
            out = pd.read_csv(file, index_col=0)
        except Exception as e:
            logger.error(f'Error reading file: {file}.')
            raise e
        return out

    @classmethod
    def _read_and_filter(cls, file):
        """Read a file and filter out rows with NaN values.

        Parameters
        ----------
        file : str
            Path to CSV file.

        Returns
        -------
        pd.DataFrame or None
            Filtered dataframe or None if all values are NaN.
        """
        df = cls._read_csv(file)
        features = [f for f in df.columns if f in OUT_VARIABLES]
        try:
            df = None if df[features].isnull().all().all() else df
        except KeyError as e:
            logger.error(f'Error filtering file {file}: {e}')
            return None
        return df

    @staticmethod
    def constant_check(x, N=4):
        """Check for constant values in the array and return a mask.

        Parameters
        ----------
        x : np.ndarray
            Array to check.
        N : int, optional
            Minimum number of consecutive constant values. Default is 4.

        Returns
        -------
        np.ndarray
            Boolean mask marking constant regions.
        """
        mask = np.zeros_like(x, dtype=bool)
        i = 0
        while i < len(x):
            count = 1
            while i + 1 < len(x) and x[i] == x[i + 1]:
                count += 1
                i += 1
            if count >= N:
                mask[i - count + 1 : i + 1] = True
            i += 1

        return mask

    @classmethod
    def do_quality_control(cls, df, min_level=0):
        """Make sure all variables pass all QC checks, are greater than 0, and
        non-repeating.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to apply QC to.
        min_level : int, optional
            Minimum QC level (0, 1, 2, or 3). Default is 0.

        Returns
        -------
        pd.DataFrame
            QC-filtered dataframe.
        """
        if min_level == 0:
            logger.warning('Skipping QC checks as min_level is 0.')
            return df

        var_names = [var_name for var_name in RAW_VARIABLES if var_name in df]
        for var_name in var_names:
            # pass level 1, 2, 3 checks == V, pass level 1, 2 checks == S,
            # pass level 1 checks == C, subjective good checks == G, positive
            # values only, non-repeating values

            cor_check = df[f'{var_name}DD'].values.astype(str) == 'C'
            scr_check = df[f'{var_name}DD'].values.astype(str) == 'S'
            ver_check = df[f'{var_name}DD'].values.astype(str) == 'V'

            assert min_level in [1, 2, 3], 'min_level must be 1, 2, or 3'
            if min_level == 1:
                # Keep all levels
                qc_check = scr_check | ver_check | cor_check
            elif min_level == 2:
                # Keep only level 2 and 3
                qc_check = scr_check | ver_check
            else:
                # Keep only level 3
                qc_check = ver_check

            good_check = df[f'{var_name}DD'].values.astype(str) == 'G'
            qc_check |= good_check

            positive_check = df[var_name] > 0
            varying_check = ~cls.constant_check(df[var_name].values)
            check = qc_check & positive_check & varying_check
            df.loc[~check, var_name] = np.nan
            logger.debug(
                f'Filtered {(~check).sum()} / {len(check)} time steps '
                f'from {var_name} based on QC checks.'
            )

        return df

    @classmethod
    def enforce_ranges(cls, data, do_qc=False):
        """Make sure all variables are within acceptable ranges.

        Parameters
        ----------
        data : pd.DataFrame
            Dataframe to enforce ranges on.

        Returns
        -------
        pd.DataFrame
            Range-enforced dataframe.
        """
        if not do_qc:
            logger.warning('Skipping range enforcement as do_qc is False.')
            return data

        features = [feat for feat in data.columns if feat in OUT_VARIABLES]
        for feature in features:
            valid_range = OUT_VARIABLES.get(feature)
            out_of_range_mask = data[feature].values > valid_range[1]
            out_of_range_mask |= data[feature].values < valid_range[0]
            not_nan = ~np.isnan(data[feature].values)
            out_of_range_mask &= not_nan
            arr = data[feature].values
            arr[out_of_range_mask] = np.nan
            data[feature] = arr
            logger.debug(
                f'Filtered {(out_of_range_mask).sum()} / {not_nan.sum()} '
                f'out of range time steps for {feature}'
            )
        return data

    @classmethod
    def derive_relativehumidity(cls, data):
        """Calculate relative humidity from dewpoint and temperature if not
        availble directly.

        Parameters
        ----------
        data : pd.DataFrame
            Dataframe with d2m and temperature_2m columns.

        Returns
        -------
        pd.Series
            Relative humidity values.
        """
        water_vapor_pressure = 6.1078 * np.exp(
            17.1 * data['d2m'] / (235 + data['d2m'])
        )
        saturation_water_vapor_pressure = 6.1078 * np.exp(
            17.1 * data['temperature_2m'] / (235 + data['temperature_2m'])
        )
        return 100 * water_vapor_pressure / saturation_water_vapor_pressure

    @classmethod
    def get_relativehumidity(cls, data):
        """Calculate relative humidity from dewpoint and temperature if not
        availble directly.

        Parameters
        ----------
        data : pd.DataFrame
            Dataframe with relevant columns.

        Returns
        -------
        pd.Series
            Relative humidity values.
        """
        if 'relativehumidity_2m' in data:
            out = data['relativehumidity_2m']
            mask = np.isnan(out.values)
            if mask.any():
                out[mask] = cls.derive_relativehumidity(data[mask])
            return out
        water_vapor_pressure = 6.1078 * np.exp(
            17.1 * data['d2m'] / (235 + data['d2m'])
        )
        saturation_water_vapor_pressure = 6.1078 * np.exp(
            17.1 * data['temperature_2m'] / (235 + data['temperature_2m'])
        )
        return 100 * water_vapor_pressure / saturation_water_vapor_pressure

    @classmethod
    def convert_vars(cls, data, do_qc=False):
        """Convert variables to correct units and calculate u/v components.

        Parameters
        ----------
        data : pd.DataFrame
            Dataframe with raw variables.

        Returns
        -------
        pd.DataFrame
            Dataframe with converted variables.
        """
        data['d2m'] -= 273.15
        data['temperature_2m'] -= 273.15
        data['u_10m'] = -data['windspeed_10m'] * np.sin(
            np.deg2rad(data['winddirection_10m'])
        )
        data['v_10m'] = -data['windspeed_10m'] * np.cos(
            np.deg2rad(data['winddirection_10m'])
        )
        data['relativehumidity_2m'] = cls.get_relativehumidity(data)

        if do_qc:
            ws = np.hypot(data['u_10m'].values, data['v_10m'].values)
            mask = ~np.isnan(data['windspeed_10m'].values)
            mask &= np.abs(data['windspeed_10m'].values - ws) > 0.01
            msg = (
                'Wind speed and direction do not match u/v components: '
                f'{ws[mask]}, {data["windspeed_10m"].values[mask]}'
            )
            assert mask.sum() == 0, msg
        return data


def _write_nc(
    year,
    data_vars,
    feature,
    valid_files,
    all_files,
    out_file,
    bbox,
    network,
    overwrite=False,
):
    """Write a single feature to a netcdf file.

    Parameters
    ----------
    year : int
        The year.
    data_vars : dict
        Dictionary of data variables.
    feature : str
        The feature name.
    valid_files : dict
        Dictionary of valid files per feature.
    all_files : list
        List of all files.
    out_file : str
        Output netCDF file path.
    bbox : list
        Bounding box [max_lat, min_lon, min_lat, max_lon].
    network : str
        The MADIS network.
    overwrite : bool, optional
        Whether to overwrite existing file. Default is False.
    """
    if os.path.exists(out_file) and not overwrite:
        logger.info(
            f'File {out_file} for year {year} already exists. Skipping write.'
        )
        return
    splits = [
        os.path.basename(file).replace('.csv', '').split('_')
        for file in valid_files[feature]
    ]
    lats = np.array([bf[1] for bf in splits]).astype(float)
    lons = np.array([bf[2] for bf in splits]).astype(float)

    logger.info(
        f'Keeping {len(valid_files[feature])} / {len(all_files)} coordinates '
        f'for {feature} based on region and NaN filtering. min / max '
        f'latitude: {min(lats):.2f} / {max(lats):.2f}. min / max longitude: '
        f'{min(lons):.2f} / {max(lons):.2f}.',
    )

    meta = pd.DataFrame()
    meta['latitude'] = lats
    meta['longitude'] = lons
    time_index = pd.read_csv(valid_files[feature][0], index_col=0).index
    logger.info(
        f'Creating {feature} dataset with {len(time_index)} timesteps and '
        f'{len(meta.index)} coordinates',
    )
    coords = {
        'time': (
            'time',
            da.asarray(
                pd.DatetimeIndex(time_index).astype(int), dtype=np.int64
            ),
        ),
        'latitude': (
            'space',
            da.asarray(
                meta['latitude'].values.astype(np.float64),
                dtype=np.float64,
            ),
        ),
        'longitude': (
            'space',
            da.asarray(
                meta['longitude'].values.astype(np.float64),
                dtype=np.float64,
            ),
        ),
    }
    ds = xr.Dataset(
        data_vars={feature: data_vars[feature]},
        coords=coords,
        attrs={
            'description': f'MADIS {feature} data for {year}',
            'network': network,
            'bbox': bbox,
            'source': BASEURL,
        },
    )
    encoding = {feature: {'chunksizes': (2000, 500)}}
    logger.info(f'Writing {feature} to netcdf file {out_file}.')
    ds.to_netcdf(out_file + '.tmp', encoding=encoding)
    os.replace(out_file + '.tmp', out_file)
    logger.info(f'Saved {out_file}')


def write_nc(
    year,
    files,
    out_dir,
    bbox,
    network,
    features='all',
    overwrite=False,
    max_workers=32,
    nan_cap=1.0,
):
    """Write all csv data to an nc file.

    Parameters
    ----------
    year : int
        The year.
    files : list
        List of processed CSV file paths.
    out_dir : str
        Output directory for netCDF files.
    bbox : list
        Bounding box [max_lat, min_lon, min_lat, max_lon].
    network : str
        The MADIS network.
    features : str or list, optional
        List of features to write or 'all'. Default is 'all'.
    overwrite : bool, optional
        Whether to overwrite existing files. Default is False.
    max_workers : int, optional
        Maximum number of parallel workers. Default is 32.
    nan_cap : float, optional
        Maximum fraction of NaN values allowed. Default is 1.0.
    """
    os.makedirs(out_dir, exist_ok=True)

    if features == 'all':
        features = pd.read_csv(files[0], index_col=0).columns
    features = [f for f in OUT_VARIABLES if f in features]

    out_files = {
        feat: os.path.join(out_dir, f'madis_{network}_{feat}_{year}.nc')
        for feat in features
    }
    if (
        all(os.path.exists(out_file) for out_file in out_files.values())
        and not overwrite
    ):
        logger.info(
            f'Files {list(out_files.values())} for year {year} already exist '
            f'in {out_dir}. Skipping write.'
        )
        return
    logger.info(f'Writing netCDF files for {network} {year}...')
    filt_files = MadisProcessor.filter_files_for_region(files, bbox)
    logger.info(
        f'Filtered {len(files) - len(filt_files)} / {len(files)} files for '
        f'region {bbox}'
    )
    data_vars = {feat: [] for feat in features}
    valid_files = {feat: [] for feat in features}
    logger.info(
        f'Reading {len(filt_files)} files for year {year} with features: '
        f'{features}',
    )

    with multiprocessing.Pool(max_workers) as pool:
        dfs = pool.map(MadisProcessor._read_and_filter, filt_files)

    filt_files = [file for file, df in zip(filt_files, dfs) if df is not None]
    dfs = [df for df in dfs if df is not None]
    dfs = dict(zip(filt_files, dfs))

    logger.info('Filtering files based on NaN fraction...')
    for file, df in tqdm(dfs.items(), total=len(dfs)):
        for feature in features:
            try:
                num_nans = np.isnan(df[feature].values).sum()
                if num_nans < nan_cap * len(df[feature].values):
                    valid_files[feature].append(file)
                    data_vars[feature].append(df[feature].values[:, None])
            except OSError as e:
                logger.error(
                    f'Error parsing {feature} data from file {file}: {e}'
                )

    good_feats = []
    for feature in features:
        if len(data_vars[feature]) == 0:
            logger.warning(
                f'No valid data found for feature {feature} in year {year}. '
                'Skipping this feature.'
            )
        else:
            good_feats.append(feature)

    for feat in good_feats:
        try:
            data_vars[feat] = da.concatenate(data_vars[feat], axis=-1)
        except Exception as e:
            msg = f'Error concatenating {feat}: {e}'
            logger.error(msg)
            raise OSError(msg) from e
        data_vars[feat] = (('time', 'space'), data_vars[feat])

    for feat in good_feats:
        _write_nc(
            year=year,
            data_vars=data_vars,
            feature=feat,
            valid_files=valid_files,
            all_files=filt_files,
            out_file=out_files[feat],
            bbox=bbox,
            network=network,
            overwrite=overwrite,
        )
