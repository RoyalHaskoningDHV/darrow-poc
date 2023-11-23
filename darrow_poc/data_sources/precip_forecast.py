import base64
import datetime
from io import BytesIO
import json
import logging
import os
from pathlib import Path
import requests

import pandas as pd
import geopandas as gpd
import numpy as np
import shapely

from roer.data_sources.raw_data import RawData


# Dependencies only necessary when using WIWB API and Azure blob store
# I.e. not necessary during production setting
try:
    import jwt
    from dotenv import load_dotenv
    from azure.storage.blob import BlockBlobService

    load_dotenv()

    account_name = os.environ.get('account_name')
    account_key = os.environ.get('account_key')
    api_url_auth = os.environ.get('wiwb_auth_url')
    api_url_wiwb = os.environ.get('wiwb_query_url')
    client_id = os.environ.get('wiwb_clientid')
    client_secret = os.environ.get('wiwb_clientsecret')
except ModuleNotFoundError:
    logging.warning(
        "Not using WIWB API or azure blob store, dependencies not available."
    )


# Functie voor het ophalen van een WIWB access token
def get_wiwb_token(client_id, client_secret):
    authorization = base64.b64encode(
        bytes(client_id + ":" + client_secret, "ISO-8859-1")
    ).decode("ascii")
    headers = {
        "Authorization": f"Basic {authorization}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    body = {"grant_type": "client_credentials"}
    response = requests.post(api_url_auth, data=body, headers=headers)
    wiwb_token = response.json().get("access_token")

    return(wiwb_token)

# Functie om geldigheid WIWB token te controleren
def token_expired(token):
    token_decoded = jwt.decode(token, options={"verify_signature": False})
    token_exp_datetime = datetime.datetime.utcfromtimestamp(token_decoded['exp'])
    current_datetime = datetime.datetime.utcnow() - datetime.timedelta(minutes=1)
    if current_datetime > token_exp_datetime:
        return True
    return False

# Functie om token verversen, indien nodig
def get_or_refresh_token(token):
    if token_expired(token):
        return(get_wiwb_token(client_id, client_secret))
    return(token)

# Functie op HTTP header te genereren inclusief authenticatie
def create_http_header_with_auth(token):
    access_token = get_or_refresh_token(token)
    api_header = {
        "content-type": "application/json", "Authorization": "Bearer " + access_token
    }

    return(api_header)

def get_latest_precipitation_model_date(token):
    """Get latest model date of Knmi.Harmonie API

    Parameters
    ----------
    token : str
        WIWB-API authentication token

    Returns
    -------
    str
    """
    payload = {
        "DataSourceCodes": ["Knmi.Harmonie"]
    }

    resp = requests.post(
        f"{api_url_wiwb}/entity/datasources/get",
        headers=create_http_header_with_auth(token),
        data=json.dumps(payload)
    )

    if resp.status_code == 200:
        model_info = resp.json()
        return model_info['DataSources']['Knmi.Harmonie']['LastModelDate']
    else:
        return None


def query_grid_data(token: str, model_date: str = None):
    """Query Knmi.Harmonie ModelGrid data from WIWB-API

    Parameters
    ----------
    token : str
        WIWB-API authentication token
    model_date : str (default=None)
        If not given will use the most recent model date

    Returns
    -------
    resp : requests.models.Response
    """
    if model_date is None:
        model_date = get_latest_precipitation_model_date(token)

    payload = {
        "Readers": [{
            "DataSourceCode": "Knmi.Harmonie",
            "Settings": {
                "StructureType": "ModelGrid",
                "ModelDate": model_date,
                "VariableCodes": ["APCP"],
                "Interval": {
                    "Type": "Hours",
                    "Value": 1,
                },
                "Extent": {
                    "Xll": 5.9700200,
                    "Yll": 50.4035700,
                    "Xur": 6.8093747,
                    "Yur": 51.2194353,
                    "SpatialReference": {
                        "Epsg": 4326
                    }
                }
            }
        }],
        "Exporter": {
            "DataFormatCode": "json",
        }
    }

    resp = requests.post(
        f"{api_url_wiwb}/modelgrids/get",
        headers=create_http_header_with_auth(token),
        data=json.dumps(payload),
    )

    return resp

def read_json_from_api(bytes_string: bytes):
    """When querying the WIWB api you can get a json response.
    However, it fails to automatically convert to json. This
    function converts the bytes_string to json, by replacing the
    BOM at the beginning of the message.

    Parameters
    ----------
    bytes_string : bytes

    Returns
    -------
    dict
    """
    raw_json = bytes_string.decode('utf8').replace('\ufeff', '')
    data_json = json.loads(raw_json)

    return dict(data_json)

def get_grid_coordinates(grid):
    """Get latitude and longitude of center points
    of grid coordinates
    
    NOTE: Can be extended to return grid squares instead of
    center points.
    
    Parameters
    ----------
    grid : dict
    
    Returns
    -------
    latitude : list
    longitude : list
    """
    grid_def = grid['GridDefinition']
    columns = grid_def['Columns']
    rows = grid_def['Rows']
    cellWidth = grid_def['CellWidth']
    cellHeight = grid_def['CellHeight']
    xll = grid_def['Xll']
    yll = grid_def['Yll']

    latitude, longitude = [], []

    for i in range(len(grid['Data'])):
        column = int(i % columns)
        row = int(np.floor(i / columns))
        cellLowerLeftX = xll + (column * cellWidth)
        cellLowerLeftY = yll + ((rows - row - 1) * cellHeight)
        cellUpperLeftX = xll + ((column + 1) * cellWidth)
        cellUpperLeftY = yll + ((rows - row) * cellHeight)
        cellCenterX = xll + ((column + 0.5) * cellWidth)
        cellCenterY = yll + ((rows - row - 0.5) * cellHeight)

        longitude.append(cellCenterX)
        latitude.append(cellCenterY)

    return latitude, longitude

def convert_json_grids_to_csv(blob_service, grids: list, start_date: str):
    """Convert each grid from a WIWB-API ModelGrids call
    into a pandas dataframe and save them. In our case
    each grid corresponds to one hour of data

    Parameters
    ----------
    blob_service : BlockBlobService
    grids : list of dicts
    start_date : str
        start date of model run, used as parent folder when saving
    """
    for i, grid in enumerate(grids):

        latitude, longitude = get_grid_coordinates(grid)

        df = pd.DataFrame({
            'precipitation': grid['Data'],
            'latitude': latitude,
            'longitude': longitude,
        })

        content = df.to_csv(index=False, encoding="utf-8")

        blob_service.create_blob_from_text(
            'precipitation-forecast',
            Path(start_date) / f"hour{i}.csv",
            content,
        )

def get_precipitation_forecast_grids(model_date: str = None):
    """Retrieve precipitatino forecasts from knmi.harmonie model from WIWB-API and
    save to blob container 'precipitation-forecast'

    Files are written to f"{start_date}/hourX.csv", with X being a number
    from 0 to 48, denoting how many hours into the future it is forecast.

    Parameters
    ----------
    model_date : str (default=None)
        If not given will use the most recent model date

    Returns
    -------
    start_date : str
    """
    blob_service = BlockBlobService(account_name=account_name, account_key=account_key)

    logging.info("Retrieving WIWB-API access token")
    token = get_wiwb_token(client_id, client_secret)

    logging.info("Querying ModelGrid data from Knmi.Harmonie API")
    resp = query_grid_data(token, model_date)

    logging.info("Converting output to json")
    data = read_json_from_api(resp.content)
    start_date = data["Data"][0]["StartDate"]

    logging.info("Converting grid outputs to geopandas")
    convert_json_grids_to_csv(
        blob_service,
        data["Data"][0]['Grids'],
        start_date=start_date,
    )

    return start_date

def assign_regions_to_points(precip_forecast: gpd.GeoDataFrame, precip_regions: gpd.GeoDataFrame):
    """Add column to `precip_forecast` Geodataframe containing what polygon of roer regions
    it belongs to, based on `precip_regions` Geodataframe

    NOTE: Ensure that both GeoDataFrames are in the same projection

    Parameters
    ----------
    precip_forecast : gpd.GeoDataFrame
    precip_regions : gpd.GeoDataFrame

    Returns
    -------
    precip_forecast : gpd.GeoDataFrame
    """
    region_assignment = []
    for grid_point in precip_forecast['geometry']:
        region_assignment.append(None)

        for i, region in enumerate(precip_regions['geometry']):
            if region.contains(grid_point):
                _ = region_assignment.pop()
                region_assignment.append(precip_regions['naam'].iloc[i])

    precip_forecast["region"] = region_assignment

    return precip_forecast

def get_forecast_with_precipitation_regions(hour: int, start_date: str, save_path: str):
    """Get precipitation forecast data with regions assigned to points

    Parameters
    ----------
    hour : int
        If 0 get forecast for hour 0, if 48 get forecast for hour 48
    start_date : str
        start date of model run, used as parent folder when saving
    save_path : str
        Path where to save the .csv outputs

    Returns
    -------
    precip_forecast : gpd.GeoDataFrame
    """
    # Get forecast from file
    df = pd.read_csv(Path(save_path) / start_date / f"hour{hour}")

    precip_forecast = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.longitude, df.latitude)
    ).set_crs(epsg=4326)

    # Add precipitation regions
    blob_service = BlockBlobService(account_name=account_name, account_key=account_key)
    blob = blob_service.get_blob_to_bytes(
        container_name="stroomgebieden",
        blob_name="Roer_merged.csv"
    )

    precip_regions = gpd.GeoDataFrame(pd.read_csv(BytesIO(blob.content)))
    precip_regions.loc[:, 'geometry'] = precip_regions.loc[:, 'geometry'].apply(
        lambda x: shapely.wkt.loads(x)
    )
    precip_regions.crs = "EPSG:4326"
    precip_forecast = assign_regions_to_points(precip_forecast, precip_regions)

    return precip_forecast

def get_open_meteo_historic_rainfall():
    """Quick and dirty function to load in some prepared open_meteo historic data"""

    rd = RawData()
    df_areas = rd.get_smoothed_shapefile_roer()
    df_areas = df_areas.to_crs(epsg=4326)

    dir_history = Path("./data/raw_immutable/precip_history/open_meteo")

    for i, region in enumerate(df_areas["naam"]):
        df_temp = pd.read_csv(dir_history / f"{region}.csv", skiprows=3)
        df_temp = df_temp.rename(columns={"precipitation (mm)": region})
        if i == 0:
            df_history = df_temp
        else:
            df_history = df_history.merge(df_temp, on="time")

    df_history['time'] = pd.to_datetime(df_history['time'])
    df_history = df_history.set_index('time', drop=True)
    df_history = df_history.rename(columns={c: f"forecast_{c}" for c in df_history.columns})
    df_history.index = pd.to_datetime(df_history.index, utc=True)
    df_history.index.name = 'TIME'

    return df_history.loc['2022-11-24 06:00:00': '2023-03-02 23:00:00', :].dropna()

def get_open_meteo_historic_rainfall_plus_noise():
    """Quick and dirty function to load in some prepared open_meteo historic data
    and add some noise to it to roughly align with noise we see in real forecast
    data - in this case based on open meteo forecast data 24 hours into the future.

    # NOTE: Using gaussian noise here is likely incorrect
    """
    stds = {
        'benedenroer': 0.25391060909921437,
        'bovenroer': 0.3438946835993292,
        'bruinkool': 0.27855385853883174,
        'inde': 0.2957083649538393,
        'kall': 0.29490044648899877,
        'merzbeek': 0.2787967662099619,
        'middenroer': 0.2463929227801372,
        'urft': 0.26395196192124976,
        'worm': 0.2678220946896192,
    }
    df_history = get_open_meteo_historic_rainfall()
    df_history += np.random.normal(0, list(stds.values()), df_history.shape)
    df_history = df_history.clip(0, 9999)

    return df_history

def get_open_meteo_forecast():
    """Quick and dirty function to load in some prepared open_meteo forecast data"""

    rd = RawData()
    df_areas = rd.get_smoothed_shapefile_roer()
    df_areas = df_areas.to_crs(epsg=4326)

    dir_forecast = Path("./data/raw_immutable/precip_forecast/open_meteo")

    for i, region in enumerate(df_areas["naam"]):
        df_temp = pd.read_csv(dir_forecast / f"{region}.csv", skiprows=3)
        df_temp = df_temp.rename(columns={"precipitation (mm)": region})
        if i == 0:
            df_forecast = df_temp
        else:
            df_forecast = df_forecast.merge(df_temp, on="time")

    df_forecast['time'] = pd.to_datetime(df_forecast['time'])
    df_forecast = df_forecast.set_index('time', drop=True)
    df_forecast = df_forecast.rename(columns={c: f"forecast_{c}" for c in df_forecast.columns})
    df_forecast.index = pd.to_datetime(df_forecast.index, utc=True)
    df_forecast.index.name = 'TIME'

    return df_forecast.loc['2022-11-24 06:00:00': '2023-03-02 23:00:00', :].dropna()

def get_knmi_forecast(from_local: bool = True):
    """Load all KNMI forecast history available

    Parameters
    ----------
    from_local : bool (default=True)
        If true, try to get the data from a specific local path
    """
    if from_local:
        df_forecast = pd.read_csv("./data/precip_forecast_from_blob/precip_forecast.csv")
    else:
        df_forecast = get_rainfall_forecasts_from_blob_store()

    df_forecast = prepare_precip_forecast_for_modelling(df_forecast)
    df_forecast.index = pd.to_datetime(df_forecast.index, utc=True)
    df_forecast.index.name = 'TIME'

    return df_forecast

def get_rainfall_forecasts_from_blob_store():
    """Get all rainfall forecasts we accumulated in the blob store.
    """
    logging.info("Getting all precipitation forecast blobs")

    blob_service = BlockBlobService(account_name=account_name, account_key=account_key)
    model_blobs = blob_service.list_blobs(container_name="precipitation-forecast")

    rd = RawData()
    precip_regions = rd.get_smoothed_shapefile_roer().to_crs(epsg=4326)

    df_all = None
    for model_blob in model_blobs:

        model_date, model_hour = model_blob.name.split("/")
        model_hour = int(model_hour.split('.')[0][4:])

        if (model_hour == 0) or (model_hour > 30):
            continue
        if model_hour == 1:
            print(model_date)

        blob = blob_service.get_blob_to_bytes(
            container_name="precipitation-forecast",
            blob_name=model_blob.name
        )

        df = pd.read_csv(BytesIO(blob.content))
        df.loc[:, "model_date"] = model_date
        df.loc[:, "hour_horizon"] = model_hour

        precip_forecast = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df.longitude, df.latitude)
        ).set_crs(epsg=4326)

        precip_forecast = assign_regions_to_points(
            precip_forecast, precip_regions
        ).dropna(subset=["region"])

        precip_forecast = precip_forecast.groupby(
            ["region", "model_date", "hour_horizon"]
        ).mean().reset_index()

        if df_all is None:
            df_all = precip_forecast.copy()
        else:
            df_all = pd.concat([df_all, precip_forecast])

    return df_all

def add_time_column(df_model: pd.DataFrame, model_date):
    """Add TIME column to forecasting model dataframe (i.e. of a single model date).
    This also invovles adding the 6 hours between the current hour and the
    hour where a new forecast becomes available, as well as shifting the prediction
    hours (hour_horizon) accordingly.

    Parameters
    ----------
    df_model : pd.DataFrame

    Returns
    -------
    df_model : pd.DataFrame
        Same as input but with TIME column added
    """
    TIME = pd.concat(
        [df_model["model_date"] + pd.Timedelta(i, unit="H") for i in range(0, 7)]
    )
    df_model = pd.concat([df_model for i in range(0, 7)])
    df_model['TIME'] = TIME

    for i in range(0, 7):
        t = model_date + pd.Timedelta(i, unit="H")
        df_model.loc[df_model["TIME"] == t, 'hour_horizon'] -= i

    df_model = df_model.loc[(df_model['hour_horizon'] >= 1) & (df_model['hour_horizon'] <= 24), :]
    df_model = df_model.loc[df_model['TIME'] != model_date, :]

    return df_model

def prepare_precip_forecast_for_modelling(df_forecast: pd.DataFrame):
    """GEnerate training features from forecast dataframe

    Parameters
    ----------
    df_forecast : pd.DataFrame

    Returns
    -------
    df_forecast : pd.DataFrame
        Now in wide format with region x horizon as columns and TIME as index
    """
    print("Preparing forecast features for modelling")

    df_forecast.loc[:, "model_date"] = pd.to_datetime(
        df_forecast["model_date"], format="%Y%m%d%H%M%S"
    )
    model_dates = df_forecast["model_date"].unique()

    model_list = []
    for i, model_date in enumerate(model_dates):

        if i % 100 == 0:
            print(f"{i}/{len(model_dates)}: {model_date}")

        df_model = df_forecast.loc[
            df_forecast["model_date"] == model_date, 
            ["region", "model_date", "hour_horizon", "precipitation"]
        ]
        df_model = add_time_column(df_model, model_date)
        model_list.append(df_model)

    df_forecast = pd.concat(model_list).reset_index(drop=True).pivot(
        columns=["region", "hour_horizon"], index='TIME', values='precipitation'
    )
    df_forecast.columns = [f"forecast_{c[0]}_horizon{c[1]}" for c in df_forecast.columns]

    return df_forecast
