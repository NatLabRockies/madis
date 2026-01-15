"""MADIS data downloading functionality."""

import logging
import multiprocessing
import os
import subprocess
from datetime import datetime
from glob import glob

from dateutil import rrule
from rex import init_logger

from madis.common import get_dates
from madis.constants import BASEURL, VALID_NETWORKS

init_logger(__name__, log_level='INFO')
logger = logging.getLogger(__name__)


class MadisDownloader:
    """Class to download MADIS raw data files."""

    def __init__(
        self,
        network,
        download_dir,
        start_date=None,
        end_date=None,
    ):
        """Initialize the MadisDownloader.

        Parameters
        ----------
        network : str
            The MADIS network to download data from.
        download_dir : str
            The directory to download raw data to.
        start_date : str, optional
            Start date in 'YYYY-MM-DD' format.
        end_date : str, optional
            End date in 'YYYY-MM-DD' format.
        """
        self.network = network
        self.download_dir = f'{download_dir}/raw/{network}'
        self.start_date = start_date
        self.end_date = end_date
        self.dates = get_dates(start_date, end_date)
        os.makedirs(self.download_dir, exist_ok=True)
        assert network in VALID_NETWORKS, f'Network {network} not supported.'

    def downloaded_files(self, year):
        """Get a list of downloaded files for the network."""
        return sorted(
            glob(f'{self.download_dir}/{year}/**/*.gz', recursive=True)
        )

    def run(self, max_workers=32, overwrite=False):
        """Download raw files for the given network and date range.

        Parameters
        ----------
        max_workers : int, optional
            Maximum number of parallel download workers. Default is 32.
        overwrite : bool, optional
            Whether to overwrite existing files. Default is False.
        """
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

    @classmethod
    def download_date(cls, network, date, download_dir, overwrite=False):
        """Download the data for the given date.

        Parameters
        ----------
        network : str
            The MADIS network to download from.
        date : datetime
            The date to download data for.
        download_dir : str
            The directory to save the downloaded file.
        overwrite : bool, optional
            Whether to overwrite existing files. Default is False.
        """
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
    def get_url(cls, network, year, month, day, hour):
        """Get the URL for the given network and date.

        Parameters
        ----------
        network : str
            The MADIS network.
        year : int
            Year.
        month : int
            Month.
        day : int
            Day.
        hour : int
            Hour.

        Returns
        -------
        str
            The download URL.
        """
        baseurl = BASEURL + f'/{year}/{month:02d}/{day:02d}/'
        url = baseurl + VALID_NETWORKS.get(network)
        return f'{url}/{year}{month:02d}{day:02d}_{hour:02d}00.gz'

    @classmethod
    def get_fp(cls, date, download_dir):
        """Get the file path for the given network and date.

        Parameters
        ----------
        date : datetime
            The date for the file.
        download_dir : str
            The base download directory.

        Returns
        -------
        str
            The file path.
        """
        return (
            f'{download_dir}/{date.year}/{date.month}/'
            f'{date.year}{date.month:02d}{date.day:02d}_{date.hour:02d}00.gz'
        )
