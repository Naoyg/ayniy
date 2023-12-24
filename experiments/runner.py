import hydra
from ayniy.model.runner import Runner
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedKFold


@hydra.main(config_name="config", config_path="conf")
def main(cfg: DictConfig):
    """
    cd experiments
    python runner.py model=lgbm data=fe000
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)

    run_configs = cfg
    runner = Runner(run_configs, cv)
    runner.run_train_cv()
    runner.run_predict_cv()
    runner.submission()


if __name__ == "__main__":
    main()
