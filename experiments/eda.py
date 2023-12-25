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


def replace_inf(df):
    to_replace = {np.inf: 9999999999, -np.nan: -9999999999}
    return df.replace(to_replace)


@hydra.main(config_name="config_eda", config_path="conf")
def main(cfg: DictConfig):
    save_dir = HOME_PATH / "output/report"
    save_dir.mkdir(exist_ok=True)
    cwd_hydra = Path.cwd()

    df_train, df_test = load_dataset(cfg)
    df_train, df_test = replace_inf(df_train), replace_inf(df_test)

    report = sv.analyze(
        [df_train, "Training Data"], target_feat=cfg.cols_definition.target_col, pairwise_analysis="off")
    file_report = save_dir / "sweetviz_report.html"
    report.show_html(
      filepath=file_report,
      open_browser=False,
      layout="widescreen",
      scale=None)

    feature_config = sv.FeatureConfig(skip=cfg.cols_definition.target_col)
    compare_report = sv.compare(
        [df_train, "Training Data"], [df_test, "Test Data"], feat_cfg=feature_config, pairwise_analysis="off")
    file_compare_report = save_dir / "sweetviz_compare_report.html"
    compare_report.show_html(
      filepath=file_compare_report,
      open_browser=False,
      layout="widescreen",
      scale=None)

    mlflow.set_tracking_uri("file://"+str(HOME_PATH / "experiments/mlruns"))
    mlflow.set_experiment(cfg.exp_name)
    with mlflow.start_run(run_name=cfg.run_name):
        mlflow.log_params(cfg)
        mlflow.log_artifacts(save_dir)
        mlflow.log_artifacts(cwd_hydra)


if __name__ == "__main__":
    main()
