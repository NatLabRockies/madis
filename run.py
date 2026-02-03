"""MADIS Downloading and Processing. (1) Download all MADIS data as netcdf
files for a given time period. (2) Combine netcdf files into timeseries csvs
for each coordinate within the given bounding box. (3) Process the csvs to
filter according to quality control checks and convert variables to correct
units. (4) Write processed csvs to a single nc file."""

import argparse
import logging
import warnings
from glob import glob

from rex import init_logger
from xarray import SerializationWarning

from madis.constants import DOMAINS
from madis.download import MadisDownloader
from madis.process import MadisProcessor, write_nc
from madis.unpack import MadisUnpacker

init_logger(__name__, log_level='DEBUG')
logger = logging.getLogger(__name__)

NETWORKS = ['maritime', 'crn', 'metar', 'mesonet']

desc = """Download madis data for the given region and dates and unpack into
timeseries csvs for each valid location."""
parser = argparse.ArgumentParser(prog='madis_downloader.py', description=desc)
parser.add_argument(
    'region',
    type=str,
    help="""Region to perform action (unpack, process) for. This
    selects the bounding box from the `DOMAINS` dictionary. The bounding box
    is only applied following the download step, since the MADIS API does
    not support filtering by region, so global data for the given time period
    will be downloaded.""",
)
parser.add_argument(
    'start_year', type=int, help="""First year to perform action for."""
)
parser.add_argument(
    'end_year',
    type=int,
    help="""Last year to perform action for. All time steps between
    start_year (Jan 1st) and end_year (Dec 31st) will be downloaded,
    inclusive.""",
)
parser.add_argument(
    '-download_dir',
    type=str,
    default=None,
    help="""Parent download directory. Raw downloads will be stored in
    subdirectories of this directory, e.g., `download_dir/<network>`""",
)
parser.add_argument(
    '-unpack_dir',
    type=str,
    default=None,
    help="""Parent unpack directory. Unpacked files will be stored in
    subdirectories of this directory, e.g., `unpack_dir/<network>`.
    If None this defaults to `out_dir/unpacked/<network>`.""",
)
parser.add_argument(
    '-process_dir',
    type=str,
    default=None,
    help="""Parent process directory. Processed files will be stored in
    subdirectories of this directory, e.g., `process_dir/<network>`.
    If None this defaults to `out_dir/processed/<network>`.""",
)
parser.add_argument(
    '-nc_dir',
    type=str,
    default=None,
    help="""Output directory for netcdf file. netcdf files will be stored in
    `nc_dir`. If this directory does not exist, it will be created. This
    does not do anything if `--write_nc` is not set.""",
)
parser.add_argument(
    '-networks',
    type=str,
    nargs='*',
    default=NETWORKS,
    help="""List of networks to download data for. Default is all networks in
    `NETWORKS`. If you want to download data for a single network, you can
    specify it here, e.g. `-networks crn`.""",
)
parser.add_argument(
    '-features',
    type=str,
    nargs='*',
    default='all',
    help="""List of features to write to netcdf files. Default is all features
    in the processed files.""",
)
parser.add_argument(
    '-max_workers',
    type=int,
    default=64,
    help="""Maximum number of workers to use for downloading, unpacking, and
    processing data. Default is 64. This should be set according to the number
    of available CPU cores.""",
)
parser.add_argument(
    '-nan_cap',
    default=1.0,
    type=float,
    help="""Maximum NaN fraction to allow in final netcdf file for each site.
    """,
)
parser.add_argument(
    '-min_qc_level',
    type=int,
    default=0,
    help="""Minimum quality control level to use in processing. Level 3 is the
    highest and level 0 is the lowest. If this is set to 0, no quality
    control checks will be performed.""",
)
parser.add_argument(
    '--download',
    action='store_true',
    default=False,
    help="""Flag to download raw MADIS data for the given time period and all
    networks in `NETWORKS`. The raw data will be stored in
    `out_dir/raw/<network>`.""",
)
parser.add_argument(
    '--unpack',
    action='store_true',
    default=False,
    help="""Flag to unpack the downloaded raw data into timeseries csv files
    for each coordinate within the bounding box defined by the `region`
    argument. The unpacked files will be stored in
    `out_dir/unpacked/<network>`.""",
)
parser.add_argument(
    '--use_cache',
    action='store_true',
    default=False,
    help="""Flag to save download data to a combined netcdf cache file and to
    load raw data prior to unpacking if cache exists. If this is set, the
    script will use existing combined files for unpacking instead of individual
    raw files. These files will only include data for the subset of variables
    defined in the `MadisDownloader`""",
)
parser.add_argument(
    '--process',
    action='store_true',
    default=False,
    help="""Flag to process the unpacked csv files. This will filter data with
    the strictest quality control flag, remove repeated values and zeros,
    convert variables to correct units, and derive some variables
    (e.g. u / v from wind speed and wind direction). The processed files will
    be stored in `out_dir/processed/<network>`. NOTE: This will include files
    with all NaNs if all data was filtered for that location. These are still
    written to help keep track of which files have already been processed.""",
)
parser.add_argument(
    '--write_nc',
    action='store_true',
    default=False,
    help="""Flag to write the processed csv files to a single netCDF file for
    each year. The netCDF files will be stored in `out_dir/<region>/final/`.
    """,
)
parser.add_argument(
    '--overwrite',
    action='store_true',
    default=False,
    help="""Flag to overwrite existing files. If this is not set, the script
    will skip files that already exist in the output directories.""",
)
args = parser.parse_args()


if __name__ == '__main__':
    if not any([args.download, args.unpack, args.process, args.write_nc]):
        raise ValueError(
            'At least one of --download, --unpack, --process, or --write_nc '
            'must be specified.'
        )

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        warnings.simplefilter('ignore', SerializationWarning)

        for year in range(args.start_year, args.end_year + 1):
            start_date = f'{year}-01-01'
            end_date = f'{year}-12-31'
            for network in args.networks:
                if args.download:
                    assert args.download_dir is not None, (
                        'download_dir must be specified when downloading data.'
                    )
                    downloader = MadisDownloader(
                        network=network,
                        out_dir=args.download_dir,
                        start_date=start_date,
                        end_date=end_date,
                    )
                    downloader.run(
                        max_workers=args.max_workers, overwrite=args.overwrite
                    )

                if args.unpack:
                    check = (args.unpack_dir is not None) and (
                        args.download_dir is not None
                    )
                    msg = (
                        'unpack_dir and download_dir must be specified '
                        'when unpacking data.'
                    )
                    assert check, msg
                    unpacker = MadisUnpacker(
                        network=network,
                        in_dir=args.download_dir,
                        out_dir=args.unpack_dir,
                        start_date=start_date,
                        end_date=end_date,
                    )
                    cfp = (
                        f'{args.unpack_dir}/combined_{network}_{{year}}.nc'
                        if args.use_cache
                        else None
                    )
                    unpacker.run(
                        max_workers=args.max_workers,
                        overwrite=args.overwrite,
                        bbox=DOMAINS[args.region]['area'],
                        cache_pattern=cfp,
                    )

                if args.process:
                    check = (args.unpack_dir is not None) and (
                        args.process_dir is not None
                    )
                    msg = (
                        'unpack_dir and process_dir must be specified when '
                        'processing data.'
                    )
                    assert check, msg
                    processor = MadisProcessor(
                        network=network,
                        in_dir=args.unpack_dir,
                        out_dir=args.process_dir,
                        start_date=start_date,
                        end_date=end_date,
                        min_qc_level=args.min_qc_level,
                    )
                    processor.run(
                        max_workers=args.max_workers,
                        overwrite=args.overwrite,
                        bbox=DOMAINS[args.region]['area'],
                    )

                check = (
                    args.process_dir is not None and args.nc_dir is not None
                )
                assert check, (
                    'nc_dir and process_dir must be specified when writing '
                    'netcdf files.'
                )
                proc_dir = f'{args.process_dir}/{network}/{year}/'
                proc_files = sorted(glob(f'{proc_dir}/{year}_*.csv'))
                if args.write_nc and len(proc_files) > 0:
                    write_nc(
                        year,
                        features=args.features,
                        files=proc_files,
                        out_dir=args.nc_dir,
                        bbox=DOMAINS[args.region]['area'],
                        overwrite=args.overwrite,
                        max_workers=args.max_workers,
                        network=network,
                        nan_cap=args.nan_cap,
                    )
