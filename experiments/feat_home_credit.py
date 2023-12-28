import os
import gc
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xfeat

HOME_PATH = Path(__file__).resolve().parents[1]
try:
    sys.path.append(str(HOME_PATH))
    from ayniy.utils import reduce_mem_usage
except Exception as e:
    raise e


def load_application() -> pd.DataFrame:
    filepath = HOME_PATH / "input/features/application_train_test.ftr"
    if not os.path.exists(filepath):
        # Convert dataset into feather format.
        train = pd.read_csv(HOME_PATH / "input/home-credit-default-risk/application_train.csv")
        test = pd.read_csv(HOME_PATH / "input/home-credit-default-risk/application_test.csv")
        xfeat.utils.compress_df(pd.concat([train, test], sort=False)).reset_index(
            drop=True
        ).to_feather(filepath)

    return pd.read_feather(filepath)


def convert_category_type(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype=="O":
            df[col] = df[col].astype("category")
    return df



def main():
    application_train = load_application()
    application_train = reduce_mem_usage(application_train)

    X_train = application_train.drop(columns=["TARGET", "SK_ID_CURR"])
    y_train = application_train["TARGET"]
    id_train = application_train[["SK_ID_CURR"]]

    X_train = convert_category_type(X_train)





if __name__ == "__main__":
