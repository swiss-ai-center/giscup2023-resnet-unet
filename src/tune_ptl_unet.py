from pathlib import Path

import yaml
from ray import tune
from ray.train import CheckpointConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer
from ray.tune.schedulers import ASHAScheduler

from process_ptl_unet import train


def tune_ptl():
    params = yaml.safe_load(open("params.yaml"))
    prepare_params = params["prepare"][params["prepare"]["method"]]
    train_params = params["process"]["ptl_unet"]
    cwd = Path.cwd()

    base_kwargs = dict(
        seed=train_params["seed"],
        dataset_path=Path(train_params["dataset_path"]),
        image_size=prepare_params["tile_output_size"],
        split=train_params["split"],
        batch_size=train_params["batch_size"],
        precision=train_params["precision"],
        metric_monitor=train_params["metric_monitor"],
        use_best_ckpt=train_params["use_best_ckpt"],
        es_patience=3,
        lr_step_size=train_params.get("lr_step_size"),
        tversky_alpha=train_params["tversky_alpha"],
        tversky_beta=train_params["tversky_beta"],
        lake_loss_gamma=train_params["lake_loss_gamma"],
        epochs=None,
        num_workers=train_params["num_workers"],
        save_ckpt=False,
        is_tune=True,
        cwd=cwd,
    )
    search_space = dict(
        accumulate_grad_batches=tune.choice([1, 2]),
        lr=tune.loguniform(1e-5, 1e-3),
        lr_gamma=tune.choice([1, 0.9, 0.8]),
    )
    # The maximum training epochs
    num_epochs = 30
    # Number of samples from parameter space
    num_samples = -1
    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

    scaling_config = ScalingConfig(
        num_workers=2,
        use_gpu=True,
        resources_per_worker={"CPU": 40, "GPU": 2},
    )

    run_config = RunConfig(
        log_to_file=True,  # Write logs to file (stdout & stderr)
        checkpoint_config=CheckpointConfig(
            num_to_keep=1,
            checkpoint_score_attribute="val_loss",
            checkpoint_score_order="min",
        ),
    )
    # Define a TorchTrainer without hyper-parameters for Tuner
    ray_trainer = TorchTrainer(
        lambda c: train(**{**base_kwargs, **c}),
        scaling_config=scaling_config,
        run_config=run_config,
    )
    tuner = tune.Tuner(
        ray_trainer,
        param_space={"train_loop_config": search_space},
        tune_config=tune.TuneConfig(
            metric="val_loss",
            mode="min",
            num_samples=num_samples,
            scheduler=scheduler,
        ),
    )
    results = tuner.fit()
    print(
        results.get_best_result(
            metric="val_loss",
            mode="min",
        )
    )


if __name__ == "__main__":
    tune_ptl()
