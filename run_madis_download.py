"""MADIS Downloading and Processing. (1) Download all MADIS data as netcdf
files for a given time period. (2) Combine netcdf files into timeseries csvs
for each coordinate within the given bounding box. (3) Process the csvs to
filter according to quality control checks and convert variables to correct
units. (4) Write processed csvs to a single nc file."""

import argparse
import logging
import time
import warnings
from glob import glob

from rex import init_logger
from xarray import SerializationWarning

from madis.madis_downloader import MadisDownloader, write_nc

init_logger(__name__, log_level="DEBUG")
logger = logging.getLogger(__name__)

DOMAINS = {
    "americas": {"area": [53, -132, -60, -25]},
    "pnw": {"area": [44, -125, 42, -123]},
    "conus": {"area": [53, -132, 20, -60]},
    "columbia": {"area": [15, -82, -7, -64]},
    "south_america": {"area": [16, -84, -60, -25]},
    "central_america": {"area": [36, -120, 0, -58]},
    "lac": {"area": [36, -120, -65, -15]},
    "ukraine": {"area": [52.5, 22, 43, 40.5]},
    "seasia": {"area": [30, 85, -14, 141.5]},
    "africa": {"area": [40, -22, -38, 64]},
}

NETWORKS = ["crn", "metar", "mesonet", "maritime"]

desc = """Download madis data for the given region and dates and unpack into
timeseries csvs for each valid location."""
parser = argparse.ArgumentParser(prog="madis_downloader.py", description=desc)
parser.add_argument(
    "region",
    type=str,
    help="""Region to download data for. This selects the bounding box for the
    download from the `DOMAINS` dictionary. The bounding box is only applied
    following the download step, since the MADIS API does not support
    filtering by region, so global data for the given time period will be
    downloaded.""",
)
parser.add_argument("start_year", type=int, help="First year to download.")
parser.add_argument(
    "end_year",
    type=int,
    help="""Last year to download. All time steps between start_year (Jan 1st)
    and end_year (Dec 31st) will be downloaded, inclusive.""",
)
parser.add_argument(
    "out_dir",
    type=str,
    help="""Parent output directory. Raw downloads will be stored in
    subdirectories of this directory, e.g., `out_dir/raw/<network>`. Unpacked
    files will be stored in `unpacked/<network>`, and processed files will be
    stored in `processed/<network>`.""",
)
parser.add_argument(
    "--download",
    action="store_true",
    default=False,
    help="""Flag to download raw MADIS data for the given time period and all
    networks in `NETWORKS`. The raw data will be stored in
    `out_dir/raw/<network>`. If this flag is not set, no data will be
    downloaded.""",
)
parser.add_argument(
    "--unpack",
    action="store_true",
    default=False,
    help="""Flag to unpack the downloaded raw data into timeseries csv files
    for each coordinate within the bounding box defined by the `region`
    argument. The unpacked files will be stored in
    `out_dir/unpacked/<network>`. If this flag is not set, no unpacking will
    be done.""",
)
parser.add_argument(
    "--process",
    action="store_true",
    default=False,
    help="""Flag to process the unpacked csv files. This will filter data with
    the strictest quality control flag, remove repeated values and zeros,
    convert variables to correct units, and derive some variables
    (e.g. u / v from wind speed and wind direction). The processed files will
    be stored in `out_dir/processed/<network>`. If this flag is not set, no
    processing will be done.""",
)
parser.add_argument(
    "--write_nc",
    action="store_true",
    default=False,
    help="""Flag to write the processed csv files to a single netCDF file for
    each year. The netCDF files will be stored in `out_dir/<region>/final/`.
    If this flag is not set, no netCDF files will be written.""",
)
args = parser.parse_args()


if __name__ == "__main__":
    if not any([args.download, args.unpack, args.process, args.write_nc]):
        raise ValueError(
            "At least one of --download, --unpack, --process, or --write_nc "
            "must be specified."
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        warnings.simplefilter("ignore", SerializationWarning)
        for network in NETWORKS:
            downloader = MadisDownloader(network=network, out_dir=args.out_dir)
            if args.download:
                downloader.download(
                    network,
                    f"{args.start_year}-01-01",
                    f"{args.end_year}-12-31",
                    args.out_dir,
                    max_workers=64,
                )

            if args.unpack:
                start_time = time.time()
                downloader.unpack(
                    network,
                    f"{args.start_year}-01-01",
                    f"{args.end_year}-12-31",
                    args.out_dir,
                    max_workers=2,
                    bbox=DOMAINS[args.region]["area"],
                )
                elapsed_time = (time.time() - start_time) / 60
                logger.info(f"Unpacking took {elapsed_time:.2f} minutes")

            if args.process:
                start_time = time.time()
                downloader.process(
                    network,
                    f"{args.start_year}-01-01",
                    f"{args.end_year}-12-31",
                    args.out_dir,
                    max_workers=8,
                    bbox=DOMAINS[args.region]["area"],
                )
                elapsed_time = (time.time() - start_time) / 60
                logger.info(f"Processing took {elapsed_time:.2f} minutes")

        for year in range(args.start_year, args.end_year + 1):
            files = sorted(
                glob(
                    f"{args.out_dir}/processed/**/*{year}_*.csv",
                    recursive=True,
                )
            )
            if args.write_nc:
                start_time = time.time()
                write_nc(
                    year,
                    files,
                    out_dir=f"{args.out_dir}/{args.region}/final/",
                    bbox=DOMAINS[args.region]["area"],
                )
                elapsed_time = (time.time() - start_time) / 60
                logger.info(f"Writing h5 file took {elapsed_time:.2f} minutes")
