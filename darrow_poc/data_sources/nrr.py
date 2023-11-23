import tempfile
import requests
import pandas as pd
import datetime
from pandas import json_normalize
from retrying import retry
import logging
import sys

from roer import config


logger = logging.getLogger(__name__)


def _try_parsing_date(text):
    """
    Helper function to try parsing text that either does or does not have a time
    To make the functions below easier, since often time is optional in the apis
    """
    for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d']:
        try:
            return datetime.datetime.strptime(text, fmt)
        except ValueError:
            pass
    raise ValueError('No valid date format found')


@retry(wait_exponential_multiplier=1000, wait_exponential_max=10000, stop_max_attempt_number=10)
def read_regenradar(
    start_date, 
    end_date, 
    latitude=52.11, 
    longitude=5.18, 
    freq='1h', 
    **kwargs
):
    """
    Export historic precipitation from Nationale Regenradar.

    By default, this function collects the best-known information for a single point, given by
    latitude and longitude in coordinate system EPSG:4326 (WGS84). This can be configured
    using `**kwargs`, but this requires some knowledge of the underlying API.

    The parameters `agg=average`, `rasters=730d6675`, `srs='epsg:28992'` are given to the API, as
    well as `start`, `end`, `window` given by `start_date`, `end_date`, `freq`. Lastly `geom`,
    which is `POINT+(latitude+longitude)`.
    Alternatively, a different geometry can be passed via the `geom` argument in `**kwargs`.
    A different coordinate system can be passed via the `srs` argument in `**kwargs`.
    This is a WKT string. For example: `geom='POINT+(191601+500127)', srs='epsg:28992'`.
    Exact information about the API specification and possible arguments is unfortunately unknown.

    Parameters
    ----------
    start_date: str or datetime-like
        the start time of the period from which to export weather
        if str, must be in the format `%Y-%m-%d` or `%Y-%m-%d %H:%M:%S`
    end_date: str or datetime-like
        the end time of the period from which to export weather
        if str, must be in the format `%Y-%m-%d` or `%Y-%m-%d %H:%M:%S`
    latitude: float, optional (default=52.11)
        latitude of the location from which to export weather. By default, use location of weather
        station De Bilt
    longitude: float, optional (default=5.18)
        longitude of the location from which to export weather. By default, use location of
        weather station De Bilt
    freq: str or DateOffset (default='1h')
        frequency of export. Minimum, and default frequency is every 5 minutes. To learn more
        about the frequency strings, see `this link
        <http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases>`__.
    kwargs: dict
        additional parameters passed in the url. Must be convertable to string. Any entries with a
        value of None will be ignored and not passed in the url.

    Returns
    -------
    result: dataframe
        Dataframe with column `VALUE` and column `TIME`.
        `VALUE` is the precipitation in the last 5 minutes, in mm.
    """

    # convert to milliseconds, which the regenradar needs
    window = int(pd.tseries.frequencies.to_offset(freq).nanos / 1000000)
    assert window >= 300 * 1000, "The minimum window for read_regenradar is 300000"

    # will raise exception if the section does not appear in the config file
    user = config["regenradar"]["user"]
    password = config["regenradar"]["password"]

    logger.debug(("Getting regenradar historic data: start_date={}, end_date={}, latitude={}, "
                  "longitude={}, window={}").
                 format(start_date, end_date, latitude, longitude, window))
    if isinstance(start_date, str):
        start_date = _try_parsing_date(start_date)
    if isinstance(end_date, str):
        end_date = _try_parsing_date(end_date)

    regenradar_url = 'https://rhdhv.lizard.net/api/v3/raster-aggregates/?'
    params = {'agg': 'average',
              'rasters': '730d6675',
              'srs': 'epsg:28992',
              'start': str(start_date),
              'stop': str(end_date),
              'window': str(window),
              'geom': 'POINT+({x}+{y})'.format(x=longitude, y=latitude)
              }
    params.update(kwargs)
    params = '&'.join('%s=%s' % (k, v) for k, v in params.items() if v is not None)

    res = requests.get(regenradar_url + params, auth=(user, password))
    res = res.json()
    data = json_normalize(res, 'data')

    # Time in miliseconds, convert to posixct
    data.columns = ['TIME', 'VALUE']
    data['TIME'] = pd.to_datetime(data['TIME'], unit='ms')

    return data


@retry(wait_exponential_multiplier=1000, wait_exponential_max=10000, stop_max_attempt_number=10)
def iter_regenradar(
    geom,
    start_time,
    end_time,
    batch_size='28D',
    freq='1h',
    srs='epsg:28992'
):
    """Collect NRR data via batches
    A workaround to avoid crashing NRR, which might happen for large requests

    Parameters
    ----------
    geom: str or shapely object
        Point or polygon to get precipitation data for
    start_time: str
        the start time of period to get data from
    end_time: str
        the end time of period to get data from
    batch_size: str
        size of batches retrieved from NRR
    freq: str (default='1h')
        frequency of data from NRR
    srs: str (default='epsg:28992')
        string containing srs information, for example to use RDS, 'epsg:28992'

    """
    df_list = []
    date_range = pd.date_range(start_time, end_time, freq=batch_size)
    n_calls = len(date_range) - 1
    for j in range(n_calls):
        start = date_range[j]
        end = date_range[j + 1]
        df_temp = read_regenradar(start, end, freq=freq, geom=geom, srs=srs)
        df_list.append(df_temp)
    df = pd.concat(df_list).\
        reset_index(drop=True)

    return df


def collect_regenradar_data(
    meta_data,
    geom_name='geometry',
    start_time='2015-12-30',
    end_time='2019-02-02',
    freq='1h',
    srs='epsg:28992'
):
    """Collect precipitation data for multiple locations
    Collecting data for all location from Nationale Regenradar, using SAM

    Parameters
    ----------
    meta_data: pandas dataframe,
        Table containing column ID and a column with shapely objects
    geom_name: str
        Column name for geometry objects
    start_time: str
        the start time of period to get data from
    end_time: str
        the end time of period to get data from
    freq: str (default='1h')
        frequency of data from NRR
    srs: str (default='epsg:28992')
        string containing srs information, for example to use RDS, 'epsg:28992'

    """
    precip_list = []
    for _, row in meta_data.iterrows():
        ID = row['ID']
        logger.info(f'Collect precipitation data for location {ID}')
        
        # collect data in batches
        df = iter_regenradar(row[geom_name], start_time, end_time, freq=freq, srs=srs)
        df['ID'] = ID
        
        # convert all data to UTC. (Regenradar is currently UTC + 3)
        # might be adjusted in future versions
        df['TIME'] = df['TIME'].dt.tz_localize('UTC').dt.tz_convert('Europe/Moscow')
        precip_list.append(df)

    precip = pd.concat(precip_list)\
        .drop_duplicates(subset=['TIME', 'ID'])\
        .reset_index(drop=True)

    return precip
