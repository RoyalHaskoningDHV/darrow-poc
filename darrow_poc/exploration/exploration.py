import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import dill as pickle
from scipy.signal import find_peaks

from sklearn.pipeline import Pipeline

from roer.models.feature_engineering import (
    SelectOriginalFeaturesTransformer,
)
from ..metrics import calculate_metrics


def crosscorr(datay, datax, lag=0):
    """ Lag-N cross correlation.
    Parameters
    ----------
    lag : int, default 0
    datay, datax : pandas.Series objects of equal length

    Returns
    ----------
    crosscorr : float
    """
    return datay.corr(datax.shift(lag))


def plot_corr_matrix(corr_matrix, figsize=(15, 15)):
    mask = np.zeros_like(corr_matrix, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=figsize)

    heatmap = sns.heatmap(corr_matrix,
                          mask=mask,
                          square=True,
                          linewidths=.5,
                          cmap='coolwarm',
                          cbar_kws={'shrink': .4,
                                    'ticks': [-1, -.5, 0, 0.5, 1]},
                          vmin=-1,
                          vmax=1,
                          annot=True,
                          annot_kws={'size': 9})

    # add the column names as labels
    ax.set_yticklabels(corr_matrix.columns, rotation=0)
    ax.set_xticklabels(corr_matrix.columns)

    sns.set_style({'xtick.bottom': True}, {'ytick.left': True})


def select_high_water_periods(
    df,
    threshold=90,
    sensor='discharge_linnich1',
    pre=24 * 7,
    post=24 * 7,
):
    """Given a dataset of water discharge values for various sensors over
    time, find all peaks above a given threshold value at the given sensor.
    Then return the IDs of those peak periods, plus thre previous `pre` and
    following `post` IDs.

    Parameters
    ----------
    df : pd.DataFrame
    threshold : numeric (default=90)
    sensor : str (default='linnich1')
        Sensor where we apply the threshold
    pre : int
        Number of samples to include prior to peak
    post : int
        Number of samples to include post peak

    Returns
    -------
    ids : np.array
        Can be used to index the periods with and around high water
    """

    peaks, _ = find_peaks(df.loc[:, sensor].values, height=threshold)

    ids = list()
    for peak in peaks:
        ids.append(np.arange(peak - pre, peak + post))

    res = np.unique(np.array(ids).flatten())
    high_water_ids = res[(res >= 0) & (res <= df.shape[0])]

    return high_water_ids


def save_r2_for_missing_channel_combinations(
    model,
    channel_combinations,
    X_test,
    y_test,
    save_path,
):
    """Compute r2 scores for many missing channel combinations and save
    to .pkl file.

    Parameters
    ----------
    model : sklearn.Pipeline
        Includes both the imputer and forecaster
    channel_combinations : list
        Each item is a tuple of column names denoting the channels
        to simulate with missing data
    X_test : pd.DataFrame
    y_test : pd.DataFrame
    save_path : path-like
        .pkl filepath to store the resulting dictionary

    Returns
    -------
    score_table : dict
        Keys are tuples of missing channels names. Values are np.arrays of
        r2 scores.
    """
    try:
        with open(save_path, 'rb') as file:
            score_table = pickle.load(file)
    except:
        score_table = dict()

    for i, channel_combination in enumerate(channel_combinations):

        if channel_combination not in score_table.keys():

            X_test_missing = X_test.copy()
            X_test_missing.loc[:, list(channel_combination)] = np.nan

            # transform all input data
            imputer = Pipeline([
                ('ift', model.named_steps['imputer'].named_steps['ift']),
                ('it', model.named_steps['imputer'].named_steps['it']),
                ('soft', SelectOriginalFeaturesTransformer(X_test_missing)),  # important to use input data
            ])
            preproc_pipeline = Pipeline([
                ('imputer', imputer),
                ('forecaster', model.named_steps['estimator'][:-1])
            ])
            X_test_transform = preproc_pipeline.transform(X_test_missing)

            # make predictions
            pred_test = pd.DataFrame(
                model.named_steps['estimator'].named_steps['lm'].predict(X_test_transform),
                index=y_test.index,
                columns=y_test.columns
            )

            # keep track of performance
            scores = pd.concat([
                pd.DataFrame(calculate_metrics(
                    np.array(y_test[col]), 
                    np.array(pred_test[col])), 
                    index=[0]
                )
                for col in y_test.columns
            ]).reset_index(drop=True)

            score_table[channel_combination] = scores['r2']

            print(channel_combination, np.mean(scores['r2']))

            if i % 10 == 0:
                print('saving')
                with open(save_path, 'wb') as file:
                    pickle.dump(score_table, file)

    return score_table
