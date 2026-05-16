import mlflow
from omegaconf import DictConfig

import hydra


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run(run_name=cfg.experiment.name):
        mlflow.log_params(dict(cfg.experiment.params))

        # --- your training code here ---
        # df = load_and_validate(cfg.data.input_path)
        # model = ...
        # mlflow.log_metric("accuracy", accuracy)
        # mlflow.sklearn.log_model(model, "model")


if __name__ == "__main__":
    main()
