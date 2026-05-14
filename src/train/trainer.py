"""Boucle d'entrainement modulaire pour la segmentation 3D MONAI."""
from __future__ import annotations
import math
import time
from pathlib import Path
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Compose


class Trainer:
    """Boucle train/val avec early stopping, AMP, sliding-window inference."""

    def __init__(self, model, loss_fn, train_loader, val_loader,
                 num_classes: int, cfg: dict, device: str = "cuda"):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.cfg = cfg
        self.device = device

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=cfg["lr"],
            weight_decay=cfg.get("weight_decay", 1e-5),
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=cfg["epochs"], eta_min=cfg["lr"] * 0.01
        )
        self.scaler = GradScaler("cuda", enabled=cfg.get("amp", True))

        self.dice_metric = DiceMetric(include_background=False, reduction="mean",
                                      get_not_nans=False)
        self.post_pred = Compose([AsDiscrete(argmax=True, to_onehot=num_classes)])
        self.post_label = Compose([AsDiscrete(to_onehot=num_classes)])

        self.patch_size = cfg.get("patch_size", [128, 128, 64])
        self.sw_overlap = cfg.get("sliding_window_overlap", 0.5)

    def _train_epoch(self):
        self.model.train()
        total_loss = 0.0
        n = 0
        for batch in self.train_loader:
            x = batch["image"].to(self.device, non_blocking=True)
            y = batch["label"].to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=self.cfg.get("amp", True)):
                logits = self.model(x)
                loss = self.loss_fn(logits, y)
            self.scaler.scale(loss).backward()
            if self.cfg.get("grad_clip"):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg["grad_clip"]
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
        self.scheduler.step()
        return total_loss / max(n, 1)

    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        self.dice_metric.reset()
        for batch in self.val_loader:
            x = batch["image"].to(self.device, non_blocking=True)
            y = batch["label"].to(self.device, non_blocking=True)
            with autocast("cuda", enabled=self.cfg.get("amp", True)):
                logits = sliding_window_inference(
                    inputs=x, roi_size=self.patch_size,
                    sw_batch_size=2, predictor=self.model,
                    overlap=self.sw_overlap, mode="gaussian",
                )
            preds = [self.post_pred(p) for p in logits]
            labels = [self.post_label(l) for l in y]
            self.dice_metric(y_pred=preds, y=labels)
        return self.dice_metric.aggregate().item()

    def fit(self, epochs: int, output_dir: str | Path):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        history = {"train_loss": [], "val_dsc": [], "lr": []}
        best_dsc = -math.inf
        epochs_no_improve = 0
        patience = self.cfg.get("early_stopping_patience", 30)
        val_interval = self.cfg.get("val_interval", 1)

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch()
            current_lr = self.optimizer.param_groups[0]["lr"]
            history["train_loss"].append(train_loss)
            history["lr"].append(current_lr)

            if epoch % val_interval == 0:
                dsc = self._validate()
                history["val_dsc"].append(dsc)
                if dsc > best_dsc:
                    best_dsc = dsc
                    epochs_no_improve = 0
                    torch.save({
                        "epoch": epoch,
                        "model_state": self.model.state_dict(),
                        "val_dsc": dsc,
                        "config": self.cfg,
                    }, output_dir / "best.pt")
                else:
                    epochs_no_improve += 1
                print(f"[Epoch {epoch:3d}] loss={train_loss:.4f} "
                      f"val_dsc={dsc:.4f} (best={best_dsc:.4f}) "
                      f"lr={current_lr:.2e} ({time.time() - t0:.1f}s)")
                if epochs_no_improve >= patience:
                    print(f"Early stopping a l'epoch {epoch} (patience={patience}).")
                    break
            else:
                print(f"[Epoch {epoch:3d}] loss={train_loss:.4f} "
                      f"({time.time() - t0:.1f}s)")
        return history
