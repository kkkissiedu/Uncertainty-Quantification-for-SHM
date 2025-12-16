# %% Imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# %% Constants
FEATURE_NAMES = [
    'Cement', 'BlastFurnaceSlag', 'FlyAsh',         'Water',
    'Superplasticizer', 'CoarseAggregate', 'FineAggregate', 'Age',
]
TARGET_NAME = 'Strength'


# =============================================================================
# Data loading
# =============================================================================

def load_concrete_data(path: str) -> tuple:
    df         = pd.read_excel(path)
    df.columns = FEATURE_NAMES + [TARGET_NAME]
    X          = df[FEATURE_NAMES].values
    y          = df[TARGET_NAME].values
    return X, y


# =============================================================================
# Splitting utilities
# =============================================================================

def split_raw(
    X:           np.ndarray,
    y:           np.ndarray,
    test_size:   float,
    random_seed: int,
) -> tuple:
    # Returns unscaled splits — callers apply their own saved scalers
    return train_test_split(X, y, test_size=test_size, random_state=random_seed)


def create_splits(
    X:           np.ndarray,
    y:           np.ndarray,
    test_size:   float,
    random_seed: int,
) -> tuple:
    X_train, X_test, y_train, y_test = split_raw(X, y, test_size, random_seed)
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler
