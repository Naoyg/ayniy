import sys
from pathlib import Path

import hydra
import mlflow
import numpy as np
import sweetviz as sv
from omegaconf import DictConfig

HOME_PATH = Path(__file__).resolve().parents[1]
try:
    sys.path.append(str(HOME_PATH))
    from ayniy.utils import Data
except Exception as e:
    raise e


def load_dataset(cfg):
    X_train = Data.load(HOME_PATH / f"input/pickle/X_train_{cfg.fe_name}.pkl")
    y_train = Data.load(HOME_PATH / f"input/pickle/y_train_{cfg.fe_name}.pkl")
    X_test = Data.load(HOME_PATH / f"input/pickle/X_test_{cfg.fe_name}.pkl")

    X_train[cfg.cols_definition.target_col] = y_train
    X_test[cfg.cols_definition.target_col] = np.nan
    return X_train, X_test


@hydra.main(config_name="config_eda", config_path="conf")
def main(cfg: DictConfig):
    df_train, df_test = load_dataset(cfg)

    report = sv.analyze(
        [df_train, "Training Data"], target_feat=cfg.cols_definition.target_col, pairwise_analysis="off")
    report.show_html(
      filepath="sweetviz_report.html",
      open_browser=False,
      layout="widescreen",
      scale=None)

    mlflow.set_experiment(cfg.exp_name)
    with mlflow.start_run(run_name=cfg.run_name):
        mlflow.log_params(cfg)
        mlflow.log_artifacts(".")


if __name__ == "__main__":
    main()
