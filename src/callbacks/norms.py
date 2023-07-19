# https://github.com/HazyResearch/state-spaces/blob/main/src/callbacks/norms.py

import torch
from lightning import Callback, LightningModule, Trainer


class TrackNorms(Callback):
    # TODO do callbacks happen before or after the method in the main LightningModule?
    # @rank_zero_only # needed?
    def on_train_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx: int
    ):
        # Log extra metrics
        metrics = {}

        if hasattr(pl_module, "_grad_norms"):
            metrics.update(pl_module._grad_norms)

        self.log_dict(
            metrics,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            add_dataloader_idx=False,
            sync_dist=True,
        )

    def on_after_backward(self, trainer: Trainer, pl_module: LightningModule):
        norms = {}
        for name, p in pl_module.named_parameters():
            if p.grad is None:
                continue

            param_norm = torch.mean(p.grad.data**2)
            norms[f"grad_norm.{name}"] = param_norm
        pl_module._grad_norms = norms
