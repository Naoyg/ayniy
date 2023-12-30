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
    from ayniy.utils import Data, reduce_mem_usage
except Exception as e:
    raise e

id_col = "SK_ID_CURR"
target_col = "TARGET"
output_dir = HOME_PATH / "input/pickle"


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
    df_application = load_application()
    df_application = reduce_mem_usage(df_application)
    df_application.set_index(id_col, inplace=True)
    df_application_train, df_application_test = df_application[df_application["TARGET"].notna()], df_application[df_application["TARGET"].isna()]

    X_train, X_test = df_application_train.drop(columns=[target_col]), df_application_test.drop(columns=[target_col])
    y_train = df_application_train[target_col]

    Data.dump(X_train, output_dir / "X_train_fe000.pkl")
    Data.dump(X_test, output_dir / "X_test_fe000.pkl")
    Data.dump(y_train, output_dir / "y_train_fe000.pkl")


if __name__ == "__main__":
    main()
