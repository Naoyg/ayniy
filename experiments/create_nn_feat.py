from pathlib import Path

from ayniy.utils import Data
from sklearn.preprocessing import StandardScaler

HOME_PATH = Path(__file__).resolve().parents[1]


if __name__ == "__main__":

    fe_id = "fe000"
    fe_name = f"{fe_id}_nn_small"

    X_train = Data.load(HOME_PATH / f"input/pickle/X_train_{fe_id}.pkl")
    y_train = Data.load(HOME_PATH / f"input/pickle/y_train_{fe_id}.pkl")
    X_test = Data.load(HOME_PATH / f"input/pickle/X_test_{fe_id}.pkl")

    del_col = []
    for c in X_train.columns:
        X_train[c].fillna(-1, inplace=True)
        X_test[c].fillna(-1, inplace=True)
        try:
            prep = StandardScaler()
            X_train[c] = prep.fit_transform(X_train[[c]])
            X_test[c] = prep.transform(X_test[[c]])
        except ValueError:
            del_col.append(c)
    print(del_col)
    print(len(del_col))
    X_train.drop(del_col, axis=1, inplace=True)
    X_test.drop(del_col, axis=1, inplace=True)
    print(X_train.shape)

    X_train = X_train.loc[:100]
    y_train = y_train.loc[:100]

    Data.dump(X_train, HOME_PATH / f"input/pickle/X_train_{fe_name}.pkl")
    Data.dump(y_train, HOME_PATH / f"input/pickle/y_train_{fe_name}.pkl")
    Data.dump(X_test, HOME_PATH / f"input/pickle/X_test_{fe_name}.pkl")
