"""Boucle d'entrainement modulaire pour la segmentation 3D MONAI.

Supporte la reprise depuis checkpoint (utile sur Kaggle / sessions limitees a 12h)
via deux fichiers ecrits dans output_dir :
  - best.pt : le meilleur modele (pour inference / eval)
  - last.pt : etat complet du training (pour reprise)

last.pt est ecrit a chaque epoch et contient :
  model, optimizer, scheduler, scaler (AMP), history, best_dsc, epoch courant,
  compteur d'early stopping, et la config. Ce qu'il faut pour reprendre a
  l'identique sans perdre une epoch.

L'ecriture est atomique (fichier temporaire + rename) pour eviter un last.pt
corrompu si le job est coupe pile pendant l'ecriture.
"""
from __future__ import annotations
import math
import os
import time
from pathlib import Path
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Compose


def _atomic_save(payload: dict, path: Path) -> None:
    """Ecrit un checkpoint torch de facon atomique.

    Pourquoi : un Ctrl+C ou une coupure Kaggle pile pendant torch.save() laisse
    un fichier tronque qui plante au prochain load. On ecrit dans un .tmp puis
    on rename (operation atomique sur la plupart des systemes).
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


class Trainer:
    """Boucle train/val avec early stopping, AMP, sliding-window inference, et resume."""

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

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------
    def _build_checkpoint(self, epoch: int, history: dict,
                          best_dsc: float, epochs_no_improve: int) -> dict:
        """Snapshot complet de l'etat d'entrainement pour reprise."""
        return {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "history": history,
            "best_dsc": best_dsc,
            "epochs_no_improve": epochs_no_improve,
            "config": self.cfg,
            "num_classes": self.num_classes,
        }

    def load_checkpoint(self, ckpt_path: str | Path) -> dict:
        """Recharge un checkpoint complet et retourne le contexte (epoch, history, etc.).

        Le contexte retourne sert a la boucle fit() pour redemarrer la ou on s'etait
        arrete. On ne fait PAS confiance aveuglement aux hyperparams du checkpoint :
        si la config a change (LR, epochs, etc.), on utilise la config courante,
        mais on garde les states (optim/sched/scaler) pour la continuite numerique.
        """
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint introuvable : {ckpt_path}")
        # weights_only=False car on charge aussi des states optim/sched/config
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        # Le scaler peut etre absent dans un vieux checkpoint (best.pt avant resume)
        if "scaler_state" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state"])
        return {
            "start_epoch": int(ckpt["epoch"]) + 1,
            "history": ckpt.get("history", {"train_loss": [], "val_dsc": [], "lr": []}),
            "best_dsc": float(ckpt.get("best_dsc", -math.inf)),
            "epochs_no_improve": int(ckpt.get("epochs_no_improve", 0)),
        }

    # ------------------------------------------------------------------
    # Boucles
    # ------------------------------------------------------------------
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

    def fit(self, epochs: int, output_dir: str | Path,
            resume_from: str | Path | None = None):
        """Entraine pendant `epochs` epochs (en tout, pas en plus).

        Args:
            epochs : nombre TOTAL d'epochs vise. Si on reprend a l'epoch 50 et
                qu'on demande 300, on fera les 250 restants.
            output_dir : ou ecrire best.pt et last.pt.
            resume_from : chemin vers un .pt a recharger AVANT de demarrer.
                Si None et qu'un last.pt existe dans output_dir, on NE reprend pas
                automatiquement (decision explicite pour eviter les surprises).
                Passer Path(output_dir) / "last.pt" pour reprendre auto.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        history = {"train_loss": [], "val_dsc": [], "lr": []}
        best_dsc = -math.inf
        epochs_no_improve = 0
        start_epoch = 1

        # Reprise eventuelle
        if resume_from is not None:
            ctx = self.load_checkpoint(resume_from)
            start_epoch = ctx["start_epoch"]
            history = ctx["history"]
            best_dsc = ctx["best_dsc"]
            epochs_no_improve = ctx["epochs_no_improve"]
            print(f"Reprise depuis {resume_from} : start_epoch={start_epoch}, "
                  f"best_dsc={best_dsc:.4f}, epochs_no_improve={epochs_no_improve}")
            if start_epoch > epochs:
                print(f"Le checkpoint est deja a l'epoch {start_epoch - 1} >= "
                      f"epochs cible ({epochs}). Rien a faire.")
                return history

        patience = self.cfg.get("early_stopping_patience", 30)
        val_interval = self.cfg.get("val_interval", 1)

        for epoch in range(start_epoch, epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch()
            current_lr = self.optimizer.param_groups[0]["lr"]
            history["train_loss"].append(train_loss)
            history["lr"].append(current_lr)

            if epoch % val_interval == 0:
                dsc = self._validate()
                history["val_dsc"].append(dsc)
                improved = dsc > best_dsc
                if improved:
                    best_dsc = dsc
                    epochs_no_improve = 0
                    # best.pt : minimal, pour inference/eval (lecture seule par train.py)
                    _atomic_save({
                        "epoch": epoch,
                        "model_state": self.model.state_dict(),
                        "val_dsc": dsc,
                        "config": self.cfg,
                    }, output_dir / "best.pt")
                else:
                    epochs_no_improve += 1
                print(f"[Epoch {epoch:3d}] loss={train_loss:.4f} "
                      f"val_dsc={dsc:.4f} (best={best_dsc:.4f}) "
                      f"lr={current_lr:.2e} ({time.time() - t0:.1f}s)"
                      f"{' *' if improved else ''}")
                stop_now = epochs_no_improve >= patience
            else:
                print(f"[Epoch {epoch:3d}] loss={train_loss:.4f} "
                      f"({time.time() - t0:.1f}s)")
                stop_now = False

            # last.pt : etat complet, ecrit a CHAQUE epoch (validation ou pas).
            # C'est ce qui permet de reprendre apres une coupure Kaggle.
            _atomic_save(
                self._build_checkpoint(epoch, history, best_dsc, epochs_no_improve),
                output_dir / "last.pt",
            )

            if stop_now:
                print(f"Early stopping a l'epoch {epoch} (patience={patience}).")
                break
        return history
