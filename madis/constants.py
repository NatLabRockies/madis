"""MADIS constants and configuration variables."""

import numpy as np

# Base URL for MADIS data archive
BASEURL = 'https://madis-data.ncep.noaa.gov/madisPublic1/data/archive'

# Valid MADIS networks and their paths
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

# Predefined geographic domains
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

# Variables to extract from the netcdf files - these will be stored in the
# unpacked csv files. The variables in the processed csv files will be
# determined by the value of the corresponding key and the OUT_VARIABLES keys.
RAW_VARIABLES = {
    'windSpeed': 'windspeed_10m',
    'windGust': 'windgust_10m',
    'precip1Hour': 'precip_1h',
    'dewpoint': 'd2m',
    'altimeter': 'pressure_0m',
    'stationPressure': 'pressure_0m',
    'windDir': 'winddirection_10m',
    'temperature': 'temperature_2m',
    'relHumidity': 'relativehumidity_2m',
}

# Time step for the final time series data - this the frequency used in time
# series interpolation and resampling.
FREQ = '60min'

# Valid ranges for each variable - this needs to include all variables
# that are expected in the final processed data.
OUT_VARIABLES = {
    'u_10m': [-100, 100],
    'v_10m': [-100, 100],
    'relativehumidity_2m': [0, 100],
    'temperature_2m': [-100, 100],
    'pressure_0m': [40000, 120000],
    'windgust_10m': [0, np.inf],
    'precip_1h': [0, np.inf],
}
