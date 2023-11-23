import pytest
import numpy as np
import dill as pkl
import logging
from pathlib import Path

from fews.RoerStah.Work.predict import (
    etl,
    main as predict,
)

logging.basicConfig(level=logging.DEBUG)

ns = '{http://www.wldelft.nl/fews/PI}'
FEWS_DIR = Path("./fews/RoerStah")
INPUT_DIR = Path("Input")
MODEL_DIR = Path("Work/Models")

@pytest.fixture
def full_models():
    with open(FEWS_DIR / MODEL_DIR / Path("full_models.pkl"), "rb") as f:
        full_models = pkl.load(f)
    return full_models

@pytest.fixture
def score_tables():
    with open(FEWS_DIR / MODEL_DIR / Path("score_tables.pkl"), "rb") as f:
        score_tables = pkl.load(f)
    return score_tables

@pytest.fixture
def input_data():
    data_src = str(FEWS_DIR / INPUT_DIR / Path("Invoer_voorspelling_Stah.xml"))
    input_data = etl(data_src, ns)
    return input_data


# TODO: make missing_channels variable input for more transparency
def test_predict(input_data, full_models, score_tables):

    horizons = np.arange(1, 25)

    # Test case 1: No missing data
    pred_a, a, X_transformed = predict(horizons=horizons, testing=True, missing_channels=[''])
    assert (a == ('',)) or (a == ''), f"Something is off in model selection step (a: {a})"

    # Test case 2: Stah is missing
    pred_b, b, X_transformed = predict(
        horizons=horizons,
        testing=True,
        missing_channels=['discharge_stah']
    )
    assert (b == ('',)) or (b == ''), f"Something is off in model selection step (b: {b})"

    # Test case 3:
    pred_c, c, X_transformed = predict(
        horizons=horizons,
        testing=True,
        missing_channels=[
            'discharge_juelich',
            'discharge_herzogenrath2',
            'discharge_herzogenrath1',
            'discharge_juelich_wl'
        ]
    )
    assert (c == ('',)) or (c == ''), f"Something is off in model selection step (c: {c})"

    # Test case 4:
    pred_d, d, X_transformed = predict(
        horizons=horizons,
        testing=True,
        missing_channels=[
            'discharge_altenburg1',
            'discharge_herzogenrath2',
            'discharge_juelich',
            'discharge_linnich1',
            'discharge_stah',
        ]
    )
    assert d == (
        'discharge_altenburg1',
        'discharge_herzogenrath2',
        'discharge_juelich',
        'discharge_linnich1',
        'discharge_stah',
    ), f"Something is off in model selection step (d: {d})"

    # Test case 5: Precipitation data is missing (1 channel)
    pred_e, e, X_transformed = predict(
        horizons=horizons,
        testing=True,
        missing_channels=[
            'precip_worm',
        ]
    )
    assert e == (
        'precip_benedenroer',
        'precip_bovenroer',
        'precip_bruinkool',
        'precip_inde',
        'precip_kall',
        'precip_merzbeek',
        'precip_middenroer',
        'precip_urft',
        'precip_worm',
    ), f"Something is off in model selection step (e: {e})"

    # Test case 6: Precipitation data is missing (all channels)
    pred_f, f, X_transformed = predict(
        horizons=horizons,
        testing=True,
        missing_channels=[cl for cl in input_data.columns if 'precip_' in cl]
    )
    assert f == (
        'precip_benedenroer',
        'precip_bovenroer',
        'precip_bruinkool',
        'precip_inde',
        'precip_kall',
        'precip_merzbeek',
        'precip_middenroer',
        'precip_urft',
        'precip_worm',
    ), f"Something is off in model selection step (f: {f})"
