import warnings

warnings.filterwarnings("ignore")

import copy
from typing import List, Tuple

import hydra
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyrootutils
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

import torch

from src import utils

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[dict, dict]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    assert cfg.ckpt_path

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    log.info("Starting perturbation error evaluation!")

    # Load model parameters
    model_ckpt = torch.load(cfg.ckpt_path)["state_dict"]
    model.load_state_dict(model_ckpt)

    # if cfg has perturbation_interval
    if cfg.get("perturbation_interval"):
        perturbation_interval = cfg.perturbation_interval
    else:
        perturbation_interval = 1

    perturbation_list = [0]
    for i in range(cfg.perturb_range):
        perturbation_list.append(1e-3 * 2 ** (i * perturbation_interval))

    perturbation_list.sort()

    perturbation_error = []
    previous_error = 0

    for perturbation_scale in perturbation_list:
        repeats = cfg.perturb_repeats if perturbation_scale > 1e-5 else 1

        for seed in range(repeats):
            torch.manual_seed(seed)

            model.load_state_dict(model_ckpt)
            model.net.perturb_weight_initialization()

            # For every weight in the model, add a scaled random noise
            for name, param in model.named_parameters():
                """
                ======  ==============  ==========================
                ord     matrix norm     vector norm
                ======  ==============  ==========================
                'fro'   Frobenius norm  --
                'nuc'   nuclear norm    --
                Number  --              sum(abs(x)**ord)**(1./ord)
                ======  ==============  ==========================
                """

                if "W" in name:
                    noise = -torch.ones_like(param)
                    noise = noise / torch.norm(noise, p=float("inf")) * perturbation_scale
                    param.data = param.data + noise
                else:
                    noise = torch.randn_like(param)
                    noise = noise / torch.norm(noise, p=float("inf")) * perturbation_scale
                    param.data = param.data + noise

            # Evaluate the model
            model.eval()

            metric = trainer.test(model=model, datamodule=datamodule)
            previous_error = max(previous_error, metric[0]["test/loss"])

            # if previous_error > 1e3:
            #     break

        perturbation_error.append(previous_error)

    # save error to csv
    df = pd.DataFrame(
        {
            "perturbation": perturbation_list,
            "perturbation_error": perturbation_error,
        }
    )
    df.to_csv(cfg.paths.output_dir + "/perturbation_error.csv", index=False)

    print("Perturbation error is", perturbation_error)
    plt.plot(perturbation_list, perturbation_error, label="Perturbation error")

    plt.xlabel(r"Perturbation $\beta$")
    plt.ylabel(r"Perturbed error $E_m(\beta)$")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.savefig(cfg.paths.output_dir + "/perturbation_error.pdf")
    plt.close()

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="perturb.yaml")
def main(cfg: DictConfig) -> None:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    evaluate(cfg)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")  # to avoid some noise message

    main()
