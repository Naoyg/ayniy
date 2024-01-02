import sys
from pathlib import Path

import numpy as np
import pandas as pd

HOME_PATH = Path(__file__).resolve().parents[1]

try:
    sys.path.append(str(HOME_PATH))
    from ayniy.utils import Data
except Exception as e:
    raise e

id_col = "SK_ID_CURR"
target_col = "TARGET"
output_dir = HOME_PATH / "input/pickle"


def add_application_features(df_application):
    # 特徴量1：総所得を世帯人数で割った値
    df_application["INCOME_div_PERSON"] = df_application["AMT_INCOME_TOTAL"] / df_application["CNT_FAM_MEMBERS"]

    # 特徴量2：総所得を就労期間で割った値
    df_application["INCOME_div_EMPLOYED"] = df_application["AMT_INCOME_TOTAL"] / df_application["DAYS_EMPLOYED"]

    # 特徴量3：外部スコアの平均値など
    cols_ext = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    df_application["EXT_SOURCE_mean"] = df_application[cols_ext].mean(axis="columns")
    df_application["EXT_SOURCE_max"] = df_application[cols_ext].max(axis="columns")
    df_application["EXT_SOURCE_min"] = df_application[cols_ext].min(axis="columns")
    df_application["EXT_SOURCE_std"] = df_application[cols_ext].std(axis="columns")
    df_application["EXT_SOURCE_count"] = df_application[cols_ext].notnull().sum(axis="columns")

    # 特徴量4：就労期間を年齢で割った値
    df_application["DAYS_EMPLOYED_div_BIRTH"] = df_application["DAYS_EMPLOYED"] / df_application["DAYS_BIRTH"]

    # 特徴量5：年金支払額を所得金額で割った値
    df_application["ANNUITY_div_INCOME"] = df_application["AMT_ANNUITY"] / df_application["AMT_INCOME_TOTAL"]

    # 特徴量6：年金支払額を借入金で割った値
    df_application["ANNUITY_div_CREDIT"] = df_application["AMT_ANNUITY"] / df_application["AMT_CREDIT"]

    return df_application


def main():
    X_train = Data.load(output_dir / "X_train_fe000.pkl")
    y_train = Data.load(output_dir / "y_train_fe000.pkl")
    X_test = Data.load(output_dir / "X_test_fe000.pkl")

    X_train["DAYS_EMPLOYED"] = X_train["DAYS_EMPLOYED"].replace(365243, np.nan)
    X_test["DAYS_EMPLOYED"] = X_test["DAYS_EMPLOYED"].replace(365243, np.nan)

    X_train, X_test = add_application_features(X_train), add_application_features(X_test)

    fe_name = "fe001"
    Data.dump(X_train, output_dir / f"X_train_{fe_name}.pkl")
    Data.dump(X_test, output_dir / f"X_test_{fe_name}.pkl")
    Data.dump(y_train, output_dir / f"y_train_{fe_name}.pkl")


if __name__ == "__main__":
    main()
