"""MADIS Downloading and Processing. (1) Download all MADIS data as netcdf
files for a given time period. (2) Combine netcdf files into timeseries csvs
for each coordinate within the given bounding box. (3) Process the csvs to
filter according to quality control checks and convert variables to correct
units. (4) Write processed csvs to a single netcdf file."""

import logging
import multiprocessing
import os
import pickle
import subprocess
from datetime import datetime
from glob import glob

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from dateutil import rrule
from rex import init_logger
from tqdm import tqdm

init_logger(__name__, log_level='DEBUG')
logger = logging.getLogger(__name__)


class MadisDownloader:
    """Class to download and process MADIS data."""

    # variables to extract from the netcdf files
    VARIABLES = (
        'altimeter',
        'stationPressure',
        'dewpoint',
        'relHumidity',
        'windSpeed',
        'windDir',
        'temperature',
    )

    # Time step for the time series data
    FREQ = '30min'

    # mapping of variable names from MADIS to the desired output names
    NAME_MAP = {
        'windSpeed': 'windspeed_10m',
        'dewpoint': 'd2m',
        'altimeter': 'pressure_0m',
        'stationPressure': 'pressure_0m',
        'windDir': 'winddirection_10m',
        'temperature': 'temperature_2m',
        'relHumidity': 'relativehumidity_2m',
    }

    # valid ranges for each variable
    RANGES = {
        'u_10m': [-50, 50],
        'v_10m': [-50, 50],
        'relativehumidity_2m': [0, 100],
        'temperature_2m': [-50, 50],
        'd2m': [-50, 50],
        'pressure_0m': [80000, 110000],
    }

    def __init__(self, network, out_dir):
        self.network = network
        self.out_dir = f'{out_dir}/raw/{network}'
        self.unpack_dir = f'{out_dir}/unpacked/{network}'
        self.process_dir = f'{out_dir}/processed/{network}'
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(self.unpack_dir, exist_ok=True)
        os.makedirs(self.process_dir, exist_ok=True)
        assert network in [
            'metar',
            'mesonet',
            'crn',
            'maritime',
        ], f'Network {network} not supported.'

    def unpacked_files(self, year):
        """Get a list of unpacked files for the network."""
        return sorted(glob(f'{self.unpack_dir}/{year}/*.csv', recursive=True))

    def downloaded_files(self, year):
        """Get a list of downloaded files for the network."""
        return sorted(glob(f'{self.out_dir}/{year}/**/*.gz', recursive=True))

    def processed_files(self, year):
        """Get a list of processed files for the network."""
        return sorted(glob(f'{self.process_dir}/{year}/*.csv', recursive=True))

    @classmethod
    def get_dates(cls, start_date, end_date):
        """Get a list of dates between start_date and end_date."""
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        dates = list(rrule.rrule(rrule.HOURLY, dtstart=start, until=end))[:-1]
        return dates

    @classmethod
    def get_url(cls, network, year, month, day, hour):
        """Get the URL for the given network and date."""
        baseurl = 'https://madis-data.ncep.noaa.gov/madisPublic1/data/archive'
        baseurl += f'/{year}/{month:02d}/{day:02d}'
        if network == 'metar':
            url = baseurl + '/point/metar/netcdf/'
        elif network == 'mesonet':
            url = baseurl + '/LDAD/mesonet/netCDF/'
        elif network == 'crn':
            url = baseurl + '/LDAD/crn/netCDF/'
        elif network == 'maritime':
            url = baseurl + '/point/maritime/netcdf/'
        return f'{url}{year}{month:02d}{day:02d}_{hour:02d}00.gz'

    @classmethod
    def get_fp(cls, date, out_dir):
        """Get the file path for the given network and date."""
        return (
            f'{out_dir}/{date.year}/{date.month}/'
            f'{date.year}{date.month:02d}{date.day:02d}_{date.hour:02d}00.gz'
        )

    @classmethod
    def download_date(cls, network, date, out_dir):
        """Download the data for the given date."""
        url = cls.get_url(network, date.year, date.month, date.day, date.hour)
        out_file = cls.get_fp(date, out_dir)
        if os.path.exists(out_file):
            logger.info(f'File {out_file} exists. Skipping download.')
            return

        os.makedirs(os.path.dirname(out_file), exist_ok=True)

        try:
            logger.info(f'Downloading {url} to {out_file}')
            _ = subprocess.run(
                ['wget', url, '-O', out_file, '--no-check-certificate'],
                capture_output=True,
                text=True,
                check=True,
            )
            logger.info(f'Successfully downloaded {out_file}')
        except subprocess.CalledProcessError as e:
            logger.error(f'Error downloading {url}: {e.stderr}')
            if os.path.exists(out_file):
                logger.info(f'Removing {out_file} due to download failure')
                os.remove(out_file)

    @classmethod
    def download(cls, network, start_date, end_date, out_dir, max_workers=32):
        """Run the downloader for the given network and date range."""
        downloader = cls(network, out_dir)
        dates = cls.get_dates(start_date, end_date)
        logger.info(
            f'Starting download for {network} from {start_date} to {end_date}'
        )
        with multiprocessing.Pool(max_workers) as pool:
            out_dirs = [downloader.out_dir] * len(dates)
            networks = [network] * len(dates)
            pool.starmap(
                downloader.download_date, zip(networks, dates, out_dirs)
            )

    @classmethod
    def write_unpacked_csv(cls, data, outfile, year):
        """Write the unpacked data to a csv file."""
        if os.path.exists(outfile):
            logger.info(f'File {outfile} already exists. Skipping write.')
        else:
            tmp = data.set_index('time')
            tmp = tmp.groupby(tmp.index).first()
            tmp = cls.reindex(tmp, year)
            tmp.to_csv(outfile)
            logger.info(f'Saved {outfile}')

    @classmethod
    def unpack(
        cls,
        network,
        start_date,
        end_date,
        out_dir,
        bbox,
        max_workers=32,
        cache_pattern=None,
    ):
        """Unpack the downloaded data into timeseries csv files."""
        downloader = cls(network, out_dir)
        dates = cls.get_dates(start_date, end_date)

        for year in sorted({date.year for date in dates}):
            files = downloader.downloaded_files(year)
            if not files:
                logger.warning(
                    f'No files found for {network} network for year {year}.'
                )
                continue
            os.makedirs(f'{downloader.unpack_dir}/{year}', exist_ok=True)
            logger.info(
                f'Unpacking {len(files)} files for {network} network for '
                f'year {year}...'
            )
            cache_file = (
                None if not cache_pattern else cache_pattern.format(year=year)
            )
            dfs = downloader.group_coordinate_data(
                files, bbox, cache_file=cache_file, max_workers=max_workers
            )

            logger.info('Writing files to disk...')

            out_files = [
                f'{downloader.unpack_dir}/{year}/{year}_{key}.csv'
                for key in dfs
            ]
            years = [year] * len(out_files)
            with multiprocessing.Pool(max_workers) as pool:
                pool.starmap(
                    cls.write_unpacked_csv, zip(dfs.values(), out_files, years)
                )

    @classmethod
    def get_coord_mask(cls, lats, lons, bbox):
        """Get the valid mask for the given region."""
        region_mask = (
            (lats >= bbox[2])
            & (lats <= bbox[0])
            & (lons >= bbox[1])
            & (lons <= bbox[3])
        )
        return region_mask

    @classmethod
    def filter_files_for_region(cls, files, bbox):
        """Filter files based on the bounding box."""
        basefiles = [
            os.path.basename(file)
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
    def _read_single_file(cls, file):
        """Read a single file and return a dataframe."""
        logger.info(f'Reading file: {file}')
        try:
            ds = xr.open_dataset(file)
        except Exception as e:
            logger.info(f'Error reading file: {file}, {e}')
            return None

        var_names = list(cls.VARIABLES) + [f'{v}DD' for v in cls.VARIABLES]
        var_names += ['timeObs', 'observationTime', 'latitude', 'longitude']
        ds = ds[[v for v in ds if v in var_names]]
        for varname in ds:
            if 'DD' in varname:
                try:
                    ds[varname] = ds[varname].astype(str)
                except ValueError:
                    logger.warning(
                        f'Could not convert {varname} to string, skipping.'
                    )
                    ds = ds.drop(varname)
        return ds.to_dataframe()

    @classmethod
    def read_raw_files(cls, files, cache_file=None, max_workers=32):
        """Read raw files and return a concatenated dataframe."""
        if cache_file and os.path.exists(cache_file):
            logger.info(f'Loading cached data from {cache_file}')
            with open(cache_file, 'rb') as f:
                df = pickle.load(f)
            return df
        with multiprocessing.Pool(max_workers) as pool:
            df_list = pool.map(cls._read_single_file, files)
        df_list = [df for df in df_list if df is not None]
        if not df_list:
            logger.error('No valid dataframes found in the files.')
            return pd.DataFrame()
        logger.info(f'Concatenating {len(df_list)} dataframes')
        df = pd.concat(df_list, axis=0)
        if cache_file:
            logger.info(f'Saving data to cache file {cache_file}')
            with open(cache_file + '.tmp', 'wb') as f:
                pickle.dump(df, f)
            os.replace(cache_file + '.tmp', cache_file)
        return df

    @classmethod
    def group_coordinate_data(
        cls, files, bbox, cache_file=None, max_workers=32
    ):
        """Group coordinate data for all files for the given region."""
        df = cls.read_raw_files(
            files, cache_file=cache_file, max_workers=max_workers
        )
        df['lat_lon_str'] = df[['latitude', 'longitude']].apply(
            lambda x: f'{x.latitude:.8f}_{x.longitude:.8f}', axis=1
        )
        coord_mask = cls.get_coord_mask(
            lats=df['latitude'].values,
            lons=df['longitude'].values,
            bbox=bbox,
        )
        total_sites = df['lat_lon_str'].nunique()
        df = df[coord_mask]
        keep_sites = df['lat_lon_str'].nunique()
        logger.info(
            f'Keeping {keep_sites} / {total_sites} sites based on region.'
        )
        df = df.drop(['latitude', 'longitude'], axis=1, errors='ignore')
        df = df.rename(
            {'observationTime': 'time', 'timeObs': 'time'},
            axis=1,
            errors='ignore',
        )
        df = df.drop_duplicates()
        logger.info('Grouping time steps by coordinate')
        dfs = {k: v for k, v in df.groupby('lat_lon_str')}
        return dfs

    @classmethod
    def reindex(cls, df, year):
        """Reindex dataframes to have a complete time index for the year."""
        time_index = pd.date_range(
            f'01-01-{year} 00:00:00', f'12-31-{year} 23:30:00', freq=cls.FREQ
        )
        new_ti = df.index.append(time_index).sort_values()
        new_ti = new_ti.drop_duplicates()
        tmp = df.reindex(new_ti)
        tmp = tmp.interpolate(method='time')
        tmp = tmp.ffill()
        tmp = tmp.resample(cls.FREQ).first()
        if tmp.index[0].year == year - 1:
            tmp = tmp.iloc[1:]
        if len(tmp) != len(time_index):
            logger.error('Dataframe was not reindexed correctly')
        return tmp

    @classmethod
    def write_processed_csv(cls, infile, outfile):
        """Write the processed data to a csv file."""
        if os.path.exists(outfile):
            logger.info(f'File {outfile} already exists. Skipping write.')
            return
        df = pd.read_csv(infile, index_col=0)
        logger.info(f'Doing QC on {infile}.')
        df = cls.do_quality_control(df)
        qc_cols = [c for c in df.columns if 'DD' in c]
        df = df.drop(qc_cols, axis=1)
        tmp = df[[v for v in cls.VARIABLES if v in df.columns]]
        df = df.rename(cls.NAME_MAP, axis=1)
        if not np.isnan(tmp.values).all():
            df = cls.convert_vars(df)
            df.to_csv(outfile)
            logger.info(f'Saved {outfile}')
        else:
            logger.info(f'No valid data in {infile}. Skipping {outfile}.')

    @classmethod
    def process(
        cls,
        network,
        start_date,
        end_date,
        out_dir,
        bbox,
        max_workers=32,
    ):
        """Unpack the downloaded data into timeseries csv files."""
        downloader = cls(network, out_dir)
        dates = cls.get_dates(start_date, end_date)

        for year in {date.year for date in dates}:
            files = downloader.unpacked_files(year)
            if not files:
                logger.warning(
                    f'No unpacked files found for {network} network for year '
                    f'{year}.'
                )
                continue
            os.makedirs(f'{downloader.process_dir}/{year}', exist_ok=True)
            files = cls.filter_files_for_region(files, bbox)
            logger.info('Processing unpacked files for year %s', year)
            out_files = [
                f'{downloader.process_dir}/{year}/{os.path.basename(file)}'
                for file in files
            ]
            with multiprocessing.Pool(max_workers) as pool:
                pool.starmap(cls.write_processed_csv, zip(files, out_files))

    @staticmethod
    def constant_check(x, N=4):
        """Check for constant values in the array and return a mask."""
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
    def do_quality_control(cls, df):
        """Make sure all variables pass all QC checks, are greater than 0, and
        non-repeating"""
        var_names = [var_name for var_name in cls.VARIABLES if var_name in df]
        for var_name in var_names:
            # pass level 1, 2, 3 checks == V
            verified_check = df[f'{var_name}DD'].values.astype(str) == 'V'
            good_check = df[f'{var_name}DD'].values.astype(str) == 'G'
            check = verified_check | good_check
            df.loc[~check, var_name] = np.nan
            logger.info(
                f'Filtered {(~check).sum()} / {len(check)} time steps '
                f'from {var_name} based on QC checks.'
            )

        for var in cls.VARIABLES:
            if var not in df:
                logger.info(f'{var} not found in dataframe.')
                df[var] = np.nan
            zero_mask = df[var] > 0
            constant_mask = ~cls.constant_check(df[var].values)
            mask = zero_mask & constant_mask
            df.loc[~mask, var] = np.nan
            logger.info(
                f'Filtered {(~zero_mask).sum()} / {len(mask)} zero value time '
                f'steps for {var}'
            )
            logger.info(
                f'Filtered {(~constant_mask).sum()} / {len(mask)} repeated '
                f'value time steps for {var}'
            )

        return df

    @classmethod
    def enforce_ranges(cls, arr, feature):
        """Make sure all variables are within acceptable ranges"""
        valid_range = cls.RANGES.get(feature, [-np.inf, np.inf])
        out_of_range_mask = arr > valid_range[1]
        out_of_range_mask |= arr < valid_range[0]
        not_nan = ~np.isnan(arr)
        out_of_range_mask &= not_nan
        arr[out_of_range_mask] = np.nan
        logger.info(
            f'Filtered {(out_of_range_mask).sum().compute()} / '
            f'{not_nan.sum().compute()} out of range time steps for {feature}'
        )
        return arr

    @classmethod
    def get_relativehumidity(cls, data):
        """Calculate relative humidity from dewpoint and temperature if not
        availble directly."""
        if 'relativehumidity_2m' in data:
            return data['relativehumidity_2m']
        water_vapor_pressure = 6.1078 * np.exp(
            17.1 * data['d2m'] / (235 + data['d2m'])
        )
        saturation_water_vapor_pressure = 6.1078 * np.exp(
            17.1 * data['temperature_2m'] / (235 + data['temperature_2m'])
        )
        return 100 * water_vapor_pressure / saturation_water_vapor_pressure

    @classmethod
    def convert_vars(cls, data):
        """Convert variables to correct units and calculate u/v components."""
        data['d2m'] -= 273.15
        data['temperature_2m'] -= 273.15
        data['u_10m'] = -data['windspeed_10m'] * np.sin(
            np.deg2rad(data['winddirection_10m'])
        )
        data['v_10m'] = -data['windspeed_10m'] * np.cos(
            np.deg2rad(data['winddirection_10m'])
        )
        data['relativehumidity_2m'] = cls.get_relativehumidity(data)
        data = data.drop(['windspeed_10m', 'winddirection_10m'], axis=1)
        return data


def write_nc(year, files, out_dir, bbox, overwrite=False):
    """Write all csv data to an nc file"""
    os.makedirs(out_dir, exist_ok=True)
    out_file = out_dir + f'madis_{year}.nc'
    if os.path.exists(out_file) and not overwrite:
        return

    meta = pd.DataFrame()
    filt_files = MadisDownloader.filter_files_for_region(files, bbox)
    basefiles = [
        os.path.basename(file).replace('.csv', '').split('_')
        for file in filt_files
    ]
    lats = [bf[1] for bf in basefiles]
    lons = [bf[2] for bf in basefiles]

    logger.info(
        'Keeping %s / %s coordinates based on region.',
        len(filt_files),
        len(files),
    )

    meta['latitude'] = lats
    meta['longitude'] = lons
    time_index = pd.read_csv(files[0], index_col=0)['time']
    logger.info(
        'Creating dataset with %s timesteps and %s coordinates',
        len(time_index),
        len(meta.index),
    )
    coords = {
        'time': (
            'time',
            da.asarray(
                pd.DatetimeIndex(time_index).astype(int), dtype=np.int32
            ),
        ),
        'latitude': (
            'space',
            da.asarray(
                meta['latitude'].values.astype(np.float32),
                dtype=np.float32,
            ),
        ),
        'longitude': (
            'space',
            da.asarray(
                meta['longitude'].values.astype(np.float32),
                dtype=np.float32,
            ),
        ),
    }
    data_vars = {}
    features = pd.read_csv(files[0], index_col=0).columns
    features = [col for col in features if 'time' not in col]
    for feat in features:
        logger.info('Prepping array for feature = %s', feat)
        arr = []
        for file in tqdm(files):
            df = pd.read_csv(file, index_col=0)
            arr.append(df[feat].values)
        arr = [a[:, None] for a in arr]
        try:
            arr = da.concatenate(arr, axis=-1)
        except Exception as e:
            msg = f'Error concatenating {feat}: {e}'
            logger.error(msg)
            raise OSError(msg) from e
        arr = MadisDownloader.enforce_ranges(arr, feat)
        data_vars[feat] = (('time', 'space'), arr)

    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    ds.to_netcdf(out_file + '.tmp')
    os.replace(out_file + '.tmp', out_file)
