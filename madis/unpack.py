"""MADIS data unpacking functionality - convert raw netCDF files to CSV timeseries."""

import logging
import multiprocessing
import os
from glob import glob

import numpy as np
import pandas as pd
import xarray as xr
from rex import init_logger

from madis.common import get_dates
from madis.constants import RAW_VARIABLES

init_logger(__name__, log_level='INFO')
logger = logging.getLogger(__name__)


class MadisUnpacker:
    """Class to unpack MADIS raw data files into CSV timeseries."""

    def __init__(
        self,
        network,
        download_dir,
        unpack_dir=None,
        start_date=None,
        end_date=None,
    ):
        """Initialize the MadisUnpacker.

        Parameters
        ----------
        network : str
            The MADIS network.
        download_dir : str
            The directory containing raw downloaded data.
        unpack_dir : str, optional
            The directory to save unpacked CSV files. If None, uses download_dir.
        start_date : str, optional
            Start date in 'YYYY-MM-DD' format.
        end_date : str, optional
            End date in 'YYYY-MM-DD' format.
        """

        unpack_dir = unpack_dir or download_dir
        self.network = network
        self.download_dir = f'{download_dir}/raw/{network}'
        self.unpack_dir = f'{unpack_dir}/unpacked/{network}'
        self.start_date = start_date
        self.end_date = end_date
        self.dates = get_dates(start_date, end_date)
        os.makedirs(self.unpack_dir, exist_ok=True)

    def unpacked_files(self, year):
        """Get a list of unpacked files for the network."""
        return sorted(glob(f'{self.unpack_dir}/{year}/*.csv', recursive=True))

    def downloaded_files(self, year):
        """Get a list of downloaded files for the network."""
        return sorted(
            glob(f'{self.download_dir}/{year}/**/*.gz', recursive=True)
        )

    def run(
        self, bbox, max_workers=32, cache_pattern=None, overwrite=False
    ):
        """Unpack the downloaded data into timeseries csv files.

        Parameters
        ----------
        bbox : list
            Bounding box [max_lat, min_lon, min_lat, max_lon].
        max_workers : int, optional
            Maximum number of parallel workers. Default is 32.
        cache_pattern : str, optional
            Pattern for cache file with {year} placeholder.
        overwrite : bool, optional
            Whether to overwrite existing files. Default is False.
        """
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
            overwrites = [overwrite] * len(out_files)
            logger.info(
                f'Writing unpacked data for {self.network} network for year '
                f'{year} to {len(out_files)} files.'
            )
            with multiprocessing.Pool(max_workers) as pool:
                pool.starmap(
                    self.write_unpacked_csv,
                    zip(dfs.values(), out_files, overwrites),
                )
            self.write_unpacked_meta(year, bbox, dfs, overwrite=overwrite)

    def get_unpacked_meta_file(self, year, bbox):
        """Get the list of coordinates for the unpacked data within the
        bounding box. Used to check if data has been unpacked already.

        Parameters
        ----------
        year : int
            The year.
        bbox : list
            Bounding box [max_lat, min_lon, min_lat, max_lon].

        Returns
        -------
        str
            Path to the meta file.
        """
        bbox_str = '_'.join([f'{b:.2f}' for b in bbox])
        return f'{self.unpack_dir}/meta_{year}_{bbox_str}.csv'

    def unpacked_files_exist(self, year, bbox):
        """Check if the unpacked meta exists for the given year and bbox.

        Parameters
        ----------
        year : int
            The year.
        bbox : list
            Bounding box [max_lat, min_lon, min_lat, max_lon].

        Returns
        -------
        bool
            True if all unpacked files exist.
        """
        meta_file = self.get_unpacked_meta_file(year, bbox)
        if not os.path.exists(meta_file):
            logger.debug(f'Meta file {meta_file} does not exist.')
            return False
        meta_file = pd.read_csv(meta_file)
        out_files = [
            f'{self.unpack_dir}/{year}/{year}_{key}.csv'
            for key in meta_file['lat_lon_str']
        ]
        out = all(os.path.exists(file) for file in out_files)
        if not out:
            logger.debug(
                f'Not all unpacked files exist for {self.network} network '
                f'for year {year} and bbox {bbox}.'
            )
        return out

    def write_unpacked_meta(self, year, bbox, dfs, overwrite=False):
        """Write the unpacked meta file for the given year and bbox.

        Parameters
        ----------
        year : int
            The year.
        bbox : list
            Bounding box [max_lat, min_lon, min_lat, max_lon].
        dfs : dict
            Dictionary of dataframes keyed by lat_lon_str.
        overwrite : bool, optional
            Whether to overwrite existing file. Default is False.
        """
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
    def read_raw_files(
        cls, files, cache_file=None, max_workers=32, overwrite=False
    ):
        """Read raw files and return a concatenated dataframe.

        Parameters
        ----------
        files : list
            List of raw file paths.
        cache_file : str, optional
            Path to cache file for faster loading.
        max_workers : int, optional
            Maximum number of parallel workers. Default is 32.
        overwrite : bool, optional
            Whether to overwrite cache file. Default is False.

        Returns
        -------
        pd.DataFrame
            Concatenated dataframe of all raw data.
        """
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
    def write_unpacked_csv(cls, data, outfile, overwrite=False):
        """Write the unpacked data to a csv file.

        Parameters
        ----------
        data : pd.DataFrame
            The dataframe to write.
        outfile : str
            Output file path.
        overwrite : bool, optional
            Whether to overwrite existing file. Default is False.
        """
        if os.path.exists(outfile) and not overwrite:
            logger.debug(f'File {outfile} already exists. Skipping write.')
        else:
            tmp = data.set_index('time')
            tmp = tmp.groupby(tmp.index).first()
            tmp.to_csv(outfile + '.tmp')
            os.replace(outfile + '.tmp', outfile)
            logger.debug(f'Saved {outfile}')

    @classmethod
    def _read_single_file(cls, file):
        """Read a single file and return a dataframe.

        Parameters
        ----------
        file : str
            Path to the raw file.

        Returns
        -------
        pd.DataFrame or None
            Dataframe if successful, None otherwise.
        """
        logger.debug(f'Reading file: {file}')
        try:
            ds = xr.open_dataset(file)
        except Exception as e:
            logger.info(f'Error reading file: {file}, {e}. Removing...')
            os.remove(file)
            return None
        year = os.path.basename(file)[:4]
        return cls._trim_dataset(ds, year)

    @classmethod
    def _trim_dataset(cls, ds, year):
        """Select only the relevant variables and convert to dataframe.

        Parameters
        ----------
        ds : xr.Dataset
            The dataset to trim.
        year : str
            The year for time checking.

        Returns
        -------
        pd.DataFrame or None
            Trimmed dataframe if successful, None otherwise.
        """
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
    def do_time_check(cls, ds, year):
        """Check if the dataset has a valid time dimension.

        Parameters
        ----------
        ds : xr.Dataset
            The dataset to check.
        year : str or int
            The expected year.

        Returns
        -------
        xr.Dataset
            The dataset filtered to the correct year.
        """
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
    def group_coordinate_data(
        cls, files, bbox, cache_file=None, max_workers=32
    ):
        """Group coordinate data for all files for the given region.

        Parameters
        ----------
        files : list
            List of raw file paths.
        bbox : list
            Bounding box [max_lat, min_lon, min_lat, max_lon].
        cache_file : str, optional
            Path to cache file.
        max_workers : int, optional
            Maximum number of parallel workers. Default is 32.

        Returns
        -------
        dict
            Dictionary of dataframes keyed by lat_lon_str.
        """
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
            f'Keeping {keep_sites} / {total_sites} sites based on region. min '
            f'/ max lattiude: {df.latitude.min()} / {df.latitude.max()}. min '
            f'/ max longitude: {df.longitude.min()} / {df.longitude.max()}.'
        )
        df = df.drop(['latitude', 'longitude'], axis=1, errors='ignore')
        time_key = 'timeObs' if 'timeObs' in df else 'observationTime'
        df = df.rename({time_key: 'time'}, axis=1, errors='ignore')
        df = df.drop_duplicates()
        logger.info('Grouping time steps by coordinate')
        dfs = {k: v for k, v in df.groupby('lat_lon_str')}
        return dfs

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
