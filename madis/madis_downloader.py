"""MADIS Downloading and Processing. (1) Download all MADIS data as netcdf
files for a given time period. (2) Combine netcdf files into timeseries csvs
for each coordinate within the given bounding box. (3) Process the csvs to
filter according to quality control checks and convert variables to correct
units. (4) Write processed csvs to a single netcdf file."""

import logging
import multiprocessing
import os
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

init_logger(__name__, log_level='INFO')
logger = logging.getLogger(__name__)


VALID_NETWORKS = {
    'metar': 'point/metar/netcdf',
    'sao': 'point/sao/netcdf',
    'maritime': 'point/maritime/netcdf',
    'crn': 'LDAD/crn/netCDF',
    'hcn': 'LDAD/hcn/netCDF',
    'nepp': 'LDAD/nepp/netCDF',
    'rwis': 'LDAD/rwis/netCDF',
    'hfmetar': 'LDAD/hfmetar/netCDF',
    'snow': 'LDAD/snow/netCDF',
    'wisdom': 'LDAD/WISDOM/netCDF',
    'radiometer': 'point/radiometer/netcdf',
    'radiosonde': 'point/raob/netcdf',
    'coop': 'LDAD/coop/netCDF',
    'urbanet': 'LDAD/urbanet/netCDF',
    'hydro': 'LDAD/hydro/netCDF',
    'ldadprofiler': 'LDAD/profiler/netCDF',
    'mesonet': 'LDAD/mesonet/netCDF',
    'satwind3h': 'point/HDW/netcdf',
    'satwind1h': 'point/HDW1h/netcdf',
    'poes': 'point/POES/netcdf',
    'satrad': 'point/satrad/netcdf',
    'acars': 'point/acars/netcdf',
    'acarsprofiles': 'point/acarsProfiles/netcdf',
    'pointprofiler': 'point/profiler/netcdf',
}


DOMAINS = {
    'americas': {'area': [53, -132, -60, -25]},
    'pnw': {'area': [44, -125, 42, -123]},
    'conus': {'area': [53, -132, 20, -60]},
    'columbia': {'area': [15, -82, -7, -64]},
    'south_america': {'area': [16, -84, -60, -25]},
    'central_america': {'area': [36, -120, 0, -58]},
    'lac': {'area': [36, -120, -65, -15]},
    'ukraine': {'area': [52.5, 22, 43, 40.5]},
    'seasia': {'area': [30, 85, -14, 141.5]},
    'africa': {'area': [40, -22, -38, 64]},
}


BASEURL = 'https://madis-data.ncep.noaa.gov/madisPublic1/data/archive'


# variables to extract from the netcdf files - these will be stored in the
# unpacked csv files. The variables in the processed csv files will be
# determined by the value of the corresponding key and the OUT_VARIABLES keys.
RAW_VARIABLES = {
    'windSpeed': 'windspeed_10m',
    'dewpoint': 'd2m',
    'altimeter': 'pressure_0m',
    'stationPressure': 'pressure_0m',
    'windDir': 'winddirection_10m',
    'temperature': 'temperature_2m',
    'relHumidity': 'relativehumidity_2m',
}

# Time step for the final time series data - this the frequency used in time
# series interpolation and resampling.
FREQ = '30min'

# valid ranges for each variable - this needs to include all variables
# that are expected in the final processed data.
OUT_VARIABLES = {
    'u_10m': [-100, 100],
    'v_10m': [-100, 100],
    'relativehumidity_2m': [0, 100],
    'temperature_2m': [-100, 100],
    'pressure_0m': [40000, 120000],
}


class MadisDownloader:
    """Class to download and process MADIS data."""

    def __init__(
        self,
        network,
        download_dir,
        unpack_dir=None,
        process_dir=None,
        start_date=None,
        end_date=None,
        min_qc_level=1,
    ):
        unpack_dir = unpack_dir or download_dir
        process_dir = process_dir or download_dir
        self.network = network
        self.download_dir = f'{download_dir}/raw/{network}'
        self.unpack_dir = f'{unpack_dir}/unpacked/{network}'
        self.process_dir = f'{process_dir}/processed/{network}'
        self.start_date = start_date
        self.end_date = end_date
        self.dates = self.get_dates(start_date, end_date)
        self.min_level = min_qc_level
        os.makedirs(self.download_dir, exist_ok=True)
        os.makedirs(self.unpack_dir, exist_ok=True)
        os.makedirs(self.process_dir, exist_ok=True)
        assert network in VALID_NETWORKS, f'Network {network} not supported.'

    def unpacked_files(self, year):
        """Get a list of unpacked files for the network."""
        return sorted(glob(f'{self.unpack_dir}/{year}/*.csv', recursive=True))

    def downloaded_files(self, year):
        """Get a list of downloaded files for the network."""
        return sorted(
            glob(f'{self.download_dir}/{year}/**/*.gz', recursive=True)
        )

    def processed_files(self, year):
        """Get a list of processed files for the network."""
        return sorted(glob(f'{self.process_dir}/{year}/*.csv', recursive=True))

    def download(self, max_workers=32, overwrite=False):
        """Download raw files for the given network and date range."""
        out_files = [
            self.get_fp(date, self.download_dir) for date in self.dates
        ]
        if all(os.path.exists(file) for file in out_files) and not overwrite:
            logger.info(
                f'All files for {self.network} network from {self.start_date} '
                f'to {self.end_date} already exist. Skipping download.'
            )
            return
        logger.info(
            f'Starting download for {self.network} from {self.start_date} to '
            f'{self.end_date}'
        )
        with multiprocessing.Pool(max_workers) as pool:
            download_dirs = [self.download_dir] * len(self.dates)
            networks = [self.network] * len(self.dates)
            overwrites = [overwrite] * len(self.dates)
            pool.starmap(
                self.download_date,
                zip(networks, self.dates, download_dirs, overwrites),
            )

    def unpack(
        self, bbox, max_workers=32, cache_pattern=None, overwrite=False
    ):
        """Unpack the downloaded data into timeseries csv files."""

        for year in sorted({date.year for date in self.dates}):
            if self.unpacked_files_exist(year, bbox) and not overwrite:
                logger.info(
                    f'Unpacked data for {self.network} network for year '
                    f'{year} and bbox {bbox} already exists. Skipping '
                    'unpacking.'
                )
                continue

            files = self.downloaded_files(year)
            if not files:
                logger.warning(
                    f'No downloaded files found for {self.network} network '
                    f'for year {year}.'
                )
                continue
            os.makedirs(f'{self.unpack_dir}/{year}', exist_ok=True)
            logger.info(
                f'Unpacking {len(files)} files for {self.network} network for '
                f'year {year}...'
            )
            cache_file = (
                None if not cache_pattern else cache_pattern.format(year=year)
            )
            dfs = self.group_coordinate_data(
                files, bbox, cache_file=cache_file, max_workers=max_workers
            )

            out_files = [
                f'{self.unpack_dir}/{year}/{year}_{key}.csv' for key in dfs
            ]
            years = [year] * len(out_files)
            overwrites = [overwrite] * len(out_files)
            logger.info(
                f'Writing unpacked data for {self.network} network for year '
                f'{year} to {len(out_files)} files.'
            )
            with multiprocessing.Pool(max_workers) as pool:
                pool.starmap(
                    self.write_unpacked_csv,
                    zip(dfs.values(), out_files, years, overwrites),
                )
            self.write_unpacked_meta(year, bbox, dfs, overwrite=overwrite)

    def process(self, bbox, max_workers=32, overwrite=False):
        """Unpack the downloaded data into timeseries csv files."""

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
                    zip(files, out_files, min_levels, overwrites),
                )

    def get_unpacked_meta_file(self, year, bbox):
        """Get the list of coordinates for the unpacked data within the
        bounding box. Used to check if data has been unpacked already."""
        bbox_str = '_'.join([f'{b:.2f}' for b in bbox])
        return f'{self.unpack_dir}/meta_{year}_{bbox_str}.csv'

    def unpacked_files_exist(self, year, bbox):
        """Check if the unpacked meta exists for the given year and bbox."""
        meta_file = self.get_unpacked_meta_file(year, bbox)
        if not os.path.exists(meta_file):
            logger.debug(f'Meta file {meta_file} does not exist.')
            return False
        meta_file = pd.read_csv(meta_file)
        out_files = [
            f'{self.unpack_dir}/{year}/{year}_{key}.csv'
            for key in meta_file['lat_lon_str']
        ]
        return all(os.path.exists(file) for file in out_files)

    def write_unpacked_meta(self, year, bbox, dfs, overwrite=False):
        """Write the unpacked meta file for the given year and bbox."""
        meta_file = self.get_unpacked_meta_file(year, bbox)
        if os.path.exists(meta_file) and not overwrite:
            logger.debug(
                f'Meta file {meta_file} already exists. Skipping write.'
            )
            return
        meta_df = pd.DataFrame({'lat_lon_str': list(dfs.keys())})
        meta_df.to_csv(meta_file + '.tmp', index=False)
        os.replace(meta_file + '.tmp', meta_file)
        logger.debug(f'Saved unpacked meta file {meta_file}')

    @classmethod
    def download_date(cls, network, date, download_dir, overwrite=False):
        """Download the data for the given date."""
        url = cls.get_url(network, date.year, date.month, date.day, date.hour)
        out_file = cls.get_fp(date, download_dir)
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        if os.path.exists(out_file) and not overwrite:
            logger.debug(f'File {out_file} exists. Skipping download.')
            return

        try:
            check = subprocess.run(
                [
                    'wget',
                    url,
                    '--spider',
                    '--quiet',
                    '--no-check-certificate',
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if check.returncode == 0:
                logger.info(f'Downloading {url} to {out_file}')
                _ = subprocess.run(
                    [
                        'wget',
                        url,
                        '-O',
                        out_file + '.tmp',
                        '--no-check-certificate',
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                os.replace(out_file + '.tmp', out_file)
                logger.debug(f'Successfully downloaded {out_file}')
        except subprocess.CalledProcessError as e:
            logger.error(f'Error downloading {url}: {e.stderr}')
            os.remove(out_file + '.tmp')

    @classmethod
    def read_raw_files(
        cls, files, cache_file=None, max_workers=32, overwrite=False
    ):
        """Read raw files and return a concatenated dataframe."""
        if cache_file and os.path.exists(cache_file) and not overwrite:
            logger.info(f'Loading cached data from {cache_file}')
            return xr.open_dataset(cache_file).to_dataframe()
        with multiprocessing.Pool(max_workers) as pool:
            df_list = pool.map(cls._read_single_file, files)
        df_list = [df for df in df_list if df is not None]
        if not df_list:
            logger.error('No valid dataframes found in the files.')
            return pd.DataFrame()
        logger.info(f'Concatenating {len(df_list)} dataframes')
        df = pd.concat(df_list, axis=0)
        if cache_file and (not os.path.exists(cache_file) or overwrite):
            logger.info(f'Saving data to cache file {cache_file}')
            xr.Dataset.from_dataframe(df).to_netcdf(cache_file + '.tmp')
            os.replace(cache_file + '.tmp', cache_file)
            logger.info(f'Saved cache file {cache_file}')
        return df

    @classmethod
    def write_unpacked_csv(cls, data, outfile, year, overwrite=False):
        """Write the unpacked data to a csv file."""
        var_names = list(RAW_VARIABLES) + [f'{v}DD' for v in RAW_VARIABLES]
        if os.path.exists(outfile) and not overwrite:
            logger.debug(f'File {outfile} already exists. Skipping write.')
        else:
            ti = pd.date_range(
                f'01-01-{year} 00:00:00',
                f'01-01-{year + 1} 00:00:00',
                freq=FREQ,
            )[:-1]
            tmp = data.set_index('time')
            tmp = tmp.groupby(tmp.index).first()
            new_ti = tmp.index.append(ti).sort_values().drop_duplicates()
            tmp = tmp.reindex(new_ti)
            tmp = tmp[list(set(var_names) & set(tmp.columns))]
            tmp = tmp.interpolate(method='time').ffill()
            tmp = tmp.resample(FREQ).first()
            if len(tmp) != len(ti):
                msg = (
                    f'Error: {len(tmp)} time steps found, expected '
                    f'{len(ti)} for {year}.'
                )
                logger.error(msg)
                raise ValueError(msg)
            tmp.to_csv(outfile + '.tmp')
            os.replace(outfile + '.tmp', outfile)
            logger.debug(f'Saved {outfile}')

    @classmethod
    def write_processed_csv(
        cls, infile, outfile, min_level=1, overwrite=False
    ):
        """Write the processed data to a csv file."""
        if os.path.exists(outfile) and not overwrite:
            logger.debug(f'File {outfile} already exists. Skipping write.')
            return
        df = cls._read_csv(infile)

        logger.debug(f'Doing QC on {infile}.')
        df = cls.do_quality_control(df, min_level=min_level)

        qc_cols = [c for c in df.columns if 'DD' in c]
        df = df.drop(qc_cols, axis=1)
        df = df.rename(RAW_VARIABLES, axis=1)

        # Ensure all required output variables are present
        for var_name in set(RAW_VARIABLES.values()):
            if var_name not in df:
                df[var_name] = np.nan
        df = df[list(RAW_VARIABLES.values())]

        if len(df) == 0:
            logger.warning(
                f'No data found in {infile} after processing. Skipping file.'
            )
            return
        try:
            df = cls.convert_vars(df)
            df = cls.enforce_ranges(df)
        except Exception as e:
            logger.error(f'Error processing {infile}: {e}')
            raise

        df.to_csv(outfile + '.tmp')
        os.replace(outfile + '.tmp', outfile)
        logger.debug(f'Saved {outfile}')

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
        baseurl = BASEURL + f'/{year}/{month:02d}/{day:02d}/'
        url = baseurl + VALID_NETWORKS.get(network)
        return f'{url}/{year}{month:02d}{day:02d}_{hour:02d}00.gz'

    @classmethod
    def get_fp(cls, date, download_dir):
        """Get the file path for the given network and date."""
        return (
            f'{download_dir}/{date.year}/{date.month}/'
            f'{date.year}{date.month:02d}{date.day:02d}_{date.hour:02d}00.gz'
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
    def do_time_check(cls, ds, year):
        """Check if the dataset has a valid time dimension."""
        time_key = (
            'timeObs'
            if 'timeObs' in ds
            else 'observationTime'
            if 'observationTime' in ds
            else None
        )
        if time_key is None:
            raise ValueError('No valid time dimension found in dataset.')
        ti = pd.DatetimeIndex(ds[time_key].values)
        msg = (
            f'Year mismatch between file(s) and dataset: '
            f'file year {year}, dataset years {ti.year.unique()}'
        )
        mask = ti.year == int(year)
        if not mask.any():
            logger.warning(msg)
            raise ValueError(msg)
        if not mask.all():
            ds = ds.load()
            ds = ds.isel(recNum=mask)
        return ds

    @classmethod
    def _read_csv(cls, file):
        """Read a csv file and return a dataframe. Picklable method with
        logging."""
        logger.debug(f'Reading file: {file}')
        try:
            out = pd.read_csv(file, index_col=0)
        except Exception as e:
            logger.error(f'Error reading file: {file}.')
            raise e
        return out

    @classmethod
    def _read_and_filter(cls, file):
        """Read a file and filter out rows with NaN values."""
        df = cls._read_csv(file)
        features = [f for f in df.columns if f in OUT_VARIABLES]
        try:
            df = None if df[features].isnull().all().all() else df
        except KeyError as e:
            logger.error(f'Error filtering file {file}: {e}')
            return None
        return df

    @classmethod
    def _read_single_file(cls, file):
        """Read a single file and return a dataframe."""
        logger.debug(f'Reading file: {file}')
        try:
            ds = xr.open_dataset(file)
        except Exception as e:
            logger.info(f'Error reading file: {file}, {e}. Removing...')
            os.remove(file)
            return None
        return cls._trim_dataset(ds, os.path.basename(file)[:4])

    @classmethod
    def _trim_dataset(cls, ds, year):
        """Select only the relevant variables and convert to dataframe."""
        var_names = list(RAW_VARIABLES) + [f'{v}DD' for v in RAW_VARIABLES]
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
        try:
            ds = cls.do_time_check(ds, year)
            return ds.to_dataframe()
        except Exception as e:
            logger.error(f'Error converting dataset to dataframe: {e}')
            return None

    @classmethod
    def group_coordinate_data(
        cls, files, bbox, cache_file=None, max_workers=32
    ):
        """Group coordinate data for all files for the given region."""
        df = cls.read_raw_files(
            files, cache_file=cache_file, max_workers=max_workers
        )
        if df.empty:
            logger.warning('No data found in the files.')
            return {}
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
        time_key = 'timeObs' if 'timeObs' in df else 'observationTime'
        df = df.rename({time_key: 'time'}, axis=1, errors='ignore')
        df = df.drop_duplicates()
        logger.info('Grouping time steps by coordinate')
        dfs = {k: v for k, v in df.groupby('lat_lon_str')}
        return dfs

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
    def do_quality_control(cls, df, min_level=1):
        """Make sure all variables pass all QC checks, are greater than 0, and
        non-repeating"""
        var_names = [var_name for var_name in RAW_VARIABLES if var_name in df]
        for var_name in var_names:
            # pass level 1, 2, 3 checks == V, pass level 1, 2 checks == S,
            # pass level 1 checks == C, subjective good checks == G, positive
            # values only, non-repeating values

            cor_check = df[f'{var_name}DD'].values.astype(str) == 'C'
            ver_check = df[f'{var_name}DD'].values.astype(str) == 'V'
            scr_check = df[f'{var_name}DD'].values.astype(str) == 'S'

            assert min_level in [1, 2, 3], 'min_level must be 1, 2, or 3'
            if min_level == 1:
                # Keep all levels
                qc_check = scr_check | ver_check | cor_check
            elif min_level == 2:
                # Keep only level 2 and 3
                qc_check = scr_check | ver_check
            else:
                # Keep only level 3
                qc_check = scr_check | ver_check | cor_check

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
    def enforce_ranges(cls, data):
        """Make sure all variables are within acceptable ranges"""
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
    """Write a single feature to a netcdf file."""
    if os.path.exists(out_file) and not overwrite:
        logger.info(
            f'File {out_file} for year {year} already exists. Skipping write.'
        )
        return
    splits = [
        os.path.basename(file).replace('.csv', '').split('_')
        for file in valid_files[feature]
    ]
    lats = [bf[1] for bf in splits]
    lons = [bf[2] for bf in splits]

    logger.info(
        f'Keeping {len(valid_files[feature])} / {len(all_files)} coordinates '
        f'for {feature} based on region and NaN filtering.',
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

    logger.info(f'Writing {feature} to netcdf file {out_file}.')
    ds.to_netcdf(out_file + '.tmp')
    os.replace(out_file + '.tmp', out_file)
    logger.info(f'Saved {out_file}')


def write_nc(
    year,
    files,
    out_dir,
    bbox,
    network,
    overwrite=False,
    max_workers=32,
    nan_cap=1.0,
):
    """Write all csv data to an nc file"""
    os.makedirs(out_dir, exist_ok=True)
    features = pd.read_csv(files[0], index_col=0).columns
    features = [f for f in features if f in OUT_VARIABLES]
    out_files = {
        feat: out_dir + f'madis_{network}_{feat}_{year}.nc'
        for feat in features
    }
    if (
        all(os.path.exists(out_file) for out_file in out_files)
        and not overwrite
    ):
        logger.info(
            f'Files {out_files} for year {year} already exist in {out_dir}. '
            'Skipping write.'
        )
        return

    filt_files = MadisDownloader.filter_files_for_region(files, bbox)
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
        dfs = pool.map(MadisDownloader._read_and_filter, filt_files)
    dfs = [df for df in dfs if df is not None]

    logger.info('Filtering files based on NaN fraction...')
    for df, file in tqdm(zip(dfs, filt_files), total=len(dfs)):
        for feat in features:
            try:
                num_nans = np.isnan(df[feat].values).sum()
                nan_frac = num_nans / df[feat].size
                if nan_frac < nan_cap:
                    valid_files[feat].append(file)
                    data_vars[feat].append(df[feat].values[:, None])
            except IOError as e:
                logger.error(f'Error parsing data from file {file}: {e}')
                continue

    good_feats = []
    for feature in features:
        if len(data_vars[feature]) == 0:
            logger.warning(
                f'No valid data found for feature {feature} in year {year}. '
                'Skipping this feature.'
            )
            continue
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
