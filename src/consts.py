from pathlib import Path
import random

import numpy as np
import torch

SRC_DIR = Path(__file__).parent

APPCAT_VARS = [
    'appCat.builtin',
    'appCat.communication',
    'appCat.entertainment',
    'appCat.finance',
    'appCat.game',
    'appCat.office',
    'appCat.other',
    'appCat.social',
    'appCat.travel',
    'appCat.unknown',
    'appCat.utilities',
    'appCat.weather'
]

USER_VARS = [
    'circumplex.arousal',
    'circumplex.valence',
    'mood'
]

SENSOR_VARS = APPCAT_VARS + ['screen', 'activity']

EVENT_VARS = ['call', 'sms']

ALL_VARS = SENSOR_VARS + USER_VARS + EVENT_VARS

VAR_NAMES_ORDER = [
    'activity',
    'appCat.builtin',
    'appCat.communication',
    'appCat.entertainment',
    'appCat.finance',
    'appCat.game',
    'appCat.office',
    'appCat.other',
    'appCat.social',
    'appCat.travel',
    'appCat.unknown',
    'appCat.utilities',
    'appCat.weather',
    'screen',
    'circumplex.arousal',
    'circumplex.valence',
    'mood',
    'call',
    'sms',
]

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)