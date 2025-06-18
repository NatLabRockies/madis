*************************
MADIS Downloader Package
*************************

Designed to handle the following steps:
1. Download all MADIS data as netcdf files for a given time period.
2. Unpack netcdf files into timeseries csvs for each coordinate within the given bounding box.
3. Process the csvs to filter according to quality control checks and convert variables to correct units.
4. Write processed csvs to a single nc file.

Install from the main directory with:
``pip install -e .``
