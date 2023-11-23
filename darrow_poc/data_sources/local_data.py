import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path

from .raw_data import RawData


class LocalData(RawData):

    def read_discharge(self, id):
        """ Read local discharge data
        Read csv file with preprocessed data of provided id
        Parameters
            id: str

        Returns
            df: pandas DataFrame

        """
        
        print('Reading {} data'.format(id))

        # Parse new data
        df = self._read_lanuv_from_fews()
        
        if id in df.columns:
            df_new = df.loc[:, ['TIME', id]]

            # Remove rows with integeres in datetime column
            ints = []
            for i, el in enumerate(df_new['TIME']):
                if type(el) == int:
                    ints.append((i, el))

            df_new = df_new.drop(df_new.iloc[[tup[0] for tup in ints], :].index)
            df_new['TIME'] = pd.to_datetime(df_new['TIME'])
            df_new = df_new.set_index('TIME')

            # Get historical data (2000-2020 from Lanuv csv files)
            df_hist = pd.read_csv(f'data/preprocessed/discharge/{id}.csv', 
                        parse_dates=[1], index_col=0) \
                        .set_index('TIME') \
                        .rename(columns={'VALUE': id})

            # Append new data to old data
            df_new_unique = df_new.loc[df_new.index > max(df_hist.index), :]
            df_all = pd.concat([df_hist, df_new_unique], axis=0)
        
        else:
            df_all = pd.read_csv(f'data/preprocessed/discharge/{id}.csv', parse_dates=[1], index_col=0) \
                       .set_index('TIME') \
                       .rename(columns={'VALUE': id}) \
        
        return df_all.resample('H') \
                     .mean() \
                     .tz_localize('UTC').tz_convert('Europe/Berlin')

    def read_precip(self):
        """ Read local precipitation data
        Read areas data and uses names to gather multiple csv's
        Optionally sum them to a single time series

        Parameters
            return_sum: boolean

        Returns
            precip: pandas DataFrame
        """
        areas = self.get_shapefile_roer()
        precip = [
            pd.read_csv(
                f'data/preprocessed/precip/nrr/{id}.csv', parse_dates=[1], index_col=0
            ).drop(['ID', 'index'], axis=1)
            for id in areas['naam']
        ]
        precip = pd.concat([p.set_index('TIME') for p in precip], axis=1)
        precip.index = pd.to_datetime(precip.index, format='%Y-%m-%d %H:%M:%S%z', utc=True)
        precip['TIME'] = precip.index.values
        precip['TIME'] = precip['TIME'].dt.tz_localize('UTC').dt.tz_convert('Europe/Moscow')
        precip.index = precip['TIME']
        precip.drop('TIME', inplace=True, axis=1)

        precip.columns = areas['naam']
    
        return precip

    def read_evaporation(self):
        """ Read and preprocess evaporation data from KNMI,
        which was collected using knmy

        Returns
            evap: pandas DataFrame
        """
        evap = pd.read_csv(
            'data/raw_immutable/knmi/evap.csv',
            parse_dates=[3], index_col=[0]
        ).set_index(keys='TIME'
        ).drop(['STN'], axis=1)

        evap = evap \
            .divide(10) \
            .shift(1) \
            .resample('H') \
            .ffill() \
            .rename(columns={'EV24': 'evap'})
        evap.index = evap.index.tz_localize('UTC').tz_convert('Europe/Amsterdam')

        return evap.dropna()    
