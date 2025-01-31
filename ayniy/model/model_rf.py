from pathlib import Path

import numpy as np
import pandas as pd
from ayniy.model.model import Model
from ayniy.utils import Data
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

HOME_PATH = Path(__file__).resolve().parents[2]
MODEL_DIR = HOME_PATH / "output/model"
MODEL_DIR.mkdir(exist_ok=True)


class ModelRFRegressor(Model):
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
        self.model: RandomForestRegressor = RandomForestRegressor(**params)
        self.model.fit(tr_x, tr_y)

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


class ModelRFClassifier(Model):
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
        self.model: RandomForestClassifier = RandomForestClassifier(**params)
        self.model.fit(tr_x, tr_y)

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
        model_path = MODEL_DIR / f"{self.run_fold_name}.model"
        self.model = Data.load(model_path)
