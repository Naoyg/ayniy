from pathlib import Path

import catboost as cb
import numpy as np
import pandas as pd
from ayniy.model.model import Model
from ayniy.utils import Data

HOME_PATH = Path(__file__).resolve().parents[2]
MODEL_DIR = HOME_PATH / "output/model"
MODEL_DIR.mkdir(exist_ok=True)


class ModelCatClassifier(Model):
    def train(
        self,
        tr_x: pd.DataFrame,
        tr_y: pd.DataFrame,
        va_x: pd.DataFrame = None,
        va_y: pd.DataFrame = None,
        te_x: pd.DataFrame = None,
    ) -> None:

        # ハイパーパラメータの設定
        params = dict(self.params)
        self.model: cb.CatBoostClassifier = cb.CatBoostClassifier(**params)

        self.model.fit(
            tr_x,
            tr_y,
            cat_features=self.categorical_features,
            eval_set=(va_x, va_y),
            verbose=100,
            use_best_model=True,
            plot=False,
        )

    def predict(self, te_x: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(te_x)[:, 1]

    def feature_importance(self, te_x: pd.DataFrame) -> pd.DataFrame:
        fold_importance_df = pd.DataFrame()
        fold_importance_df["Feature"] = te_x.columns.values
        fold_importance_df["importance"] = self.model.feature_importances_
        return fold_importance_df

    def save_model(self) -> None:
        model_path = MODEL_DIR / f"{self.run_fold_name}.model"
        Data.dump(self.model, model_path)

    def load_model(self) -> None:
        model_path = MODEL_DIR /  f"{self.run_fold_name}.model"
        self.model = Data.load(model_path)


class ModelCatRegressor(Model):
    def train(
        self,
        tr_x: pd.DataFrame,
        tr_y: pd.DataFrame,
        va_x: pd.DataFrame = None,
        va_y: pd.DataFrame = None,
        te_x: pd.DataFrame = None,
    ) -> None:

        # ハイパーパラメータの設定
        params = dict(self.params)
        self.model: cb.CatBoostRegressor = cb.CatBoostRegressor(**params)

        self.model.fit(
            tr_x,
            tr_y,
            cat_features=self.categorical_features,
            eval_set=(va_x, va_y),
            verbose=100,
            use_best_model=True,
            plot=False,
        )

    def predict(self, te_x: pd.DataFrame) -> np.ndarray:
        return self.model.predict(te_x)

    def feature_importance(self, te_x: pd.DataFrame) -> pd.DataFrame:
        fold_importance_df = pd.DataFrame()
        fold_importance_df["Feature"] = te_x.columns.values
        fold_importance_df["importance"] = self.model.feature_importances_
        return fold_importance_df

    def save_model(self) -> None:
        model_path = MODEL_DIR / f"{self.run_fold_name}.model"
        Data.dump(self.model, model_path)

    def load_model(self) -> None:
        model_path = MODEL_DIR / f"{self.run_fold_name}.model"
        self.model = Data.load(model_path)
