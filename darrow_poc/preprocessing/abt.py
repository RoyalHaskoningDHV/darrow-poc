import pandas as pd
import numpy as np
import roer
from roer.data_sources.local_data import LocalData


def create_abt():
    ld = LocalData()

    precip = ld.read_precip()
    precip.columns = ['precip_' + col for col in precip.columns]

    discharge_ids = [
        'altenburg1',
        'eschweiler',
        'herzogenrath1',
        'herzogenrath2',
        'juelich',
        'juelich_wl',
        'kirchberg1',
        'kornelimuenster',
        'linnich1',
        'obermaubach',
        'randerath',
        'stah',
        'rimburg',
        'zerkall1',
    ]

    discharge = pd.concat([ld.read_discharge(id) for id in discharge_ids], axis=1)
    discharge.columns = ['discharge_' + col for col in discharge.columns]

    evap = ld.read_evaporation()

    df = discharge.join(precip).join(evap)

    return df


if __name__ == '__main__':

    # This interpolation is outdated, use iterative imputer instead.
    df = create_abt()
    df = df.interpolate(method='linear', limit=24, limit_direction='both')  # linearly interpolate gaps up to 24 hours
    df = df.ffill()  # Future improvement (!)
    df.to_csv('data/abt/data.csv')

    # Maybe not doing this: Because of long window based features
    # first do feature engineering, then split

    train = df.loc['2010-01-01':'2017-12-31 23:59:59']
    test = df.loc['2018-01-01':'2020-03-31 23:59:59']

    train.to_csv('data/abt/train.csv')
    test.to_csv('data/abt/test.csv')
