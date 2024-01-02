import sys
import yaml
from pathlib import Path

import pandas as pd

HOME_PATH = Path(__file__).resolve().parents[1]
try:
    sys.path.append(str(HOME_PATH))
    from ayniy.utils import Data, reduce_mem_usage
    from ayniy.preprocessing.tabular import GroupbyTransformer
except Exception as e:
    raise e


id_col = "SK_ID_CURR"
raw_data_dir = HOME_PATH / "input/home-credit-default-risk"
output_dir = HOME_PATH / "input/pickle"


def main():
    df_pos = pd.read_csv(raw_data_dir / "POS_CASH_balance.csv")
    df_left = df_pos[[id_col]].drop_duplicates(subset=id_col)

    df_pos_ohe = pd.get_dummies(df_pos, columns=["NAME_CONTRACT_STATUS"], dummy_na=True)
    with open(HOME_PATH / "experiments/conf/data/fe002.yaml", "r") as f:
        configs = yaml.safe_load(f)


    groupby = GroupbyTransformer(param_dict=configs["aggregation"]["groupby_dict"], df_left=df_left)
    df_pos_group1 = groupby.transform(df_pos_ohe)
    groupby = GroupbyTransformer(param_dict=configs["aggregation"]["nunique_dict"], df_left=df_left)
    df_pos_group2 = groupby.transform(df_pos_ohe)

    df_pos_processed = df_pos_group1.merge(df_pos_group2, on=id_col, how="left", validate="one_to_one")
    filepath = HOME_PATH / "input/features/pos_cash_balance.ftr"
    df_pos_processed.to_feather(filepath)

    X_train = Data.load(output_dir / "X_train_fe001.pkl")
    y_train = Data.load(output_dir / "y_train_fe001.pkl")
    X_test = Data.load(output_dir / "X_test_fe001.pkl")

    X_train = X_train.merge(df_pos_processed, on=id_col, how="left", validate="one_to_one")
    X_test = X_test.merge(df_pos_processed, on=id_col, how="left", validate="one_to_one")

    fe_name = "fe002"
    Data.dump(X_train, output_dir / f"X_train_{fe_name}.pkl")
    Data.dump(X_test, output_dir / f"X_test_{fe_name}.pkl")
    Data.dump(y_train, output_dir / f"y_train_{fe_name}.pkl")


if __name__ == "__main__":
    main()
