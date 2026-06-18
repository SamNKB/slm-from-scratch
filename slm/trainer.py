import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
import wandb

from .config import TrainConfig, ModelConfig
from .model import SLM
from .dataset import make_dataloader


def _cosine_lr(step: int, warmup: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup:
        return max_lr * step / warmup
    if step > max_steps:
        return min_lr
    progress = (step - warmup) / (max_steps - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


class Trainer:
    def __init__(self, model_cfg: ModelConfig, train_cfg: TrainConfig):
        self.model_cfg = model_cfg
        self.cfg = train_cfg
        self.device = torch.device(train_cfg.device if torch.cuda.is_available() else "cpu")

        self.model = SLM(model_cfg).to(self.device)
        if train_cfg.compile:
            self.model = torch.compile(self.model)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
            betas=(0.9, 0.95),
        )

        self.train_loader = make_dataloader(train_cfg.data_path, model_cfg.context_length, train_cfg.batch_size)
        self.val_loader = make_dataloader(train_cfg.val_data_path, model_cfg.context_length, train_cfg.batch_size, shuffle=False)

        self.step = 0
        if train_cfg.resume_from:
            self._load_checkpoint(train_cfg.resume_from)

        run_name = Path(train_cfg.checkpoint_dir).name
        self.writer = SummaryWriter(log_dir=f"{train_cfg.tensorboard_dir}/{run_name}")

        wandb.init(
            project=train_cfg.wandb_project,
            name=run_name,
            mode=train_cfg.wandb_mode,
            config={**vars(model_cfg), **vars(train_cfg)},
            resume="allow",
        )

        print(f"Model parameters: {self.model.num_params():,}")
        print(f"Device: {self.device}")
        print(f"TensorBoard : tensorboard --logdir {train_cfg.tensorboard_dir}")
        print(f"W&B mode    : {train_cfg.wandb_mode}")

    @torch.no_grad()
    def _eval(self) -> float:
        self.model.eval()
        losses = []
        for i, (x, y) in enumerate(self.val_loader):
            if i >= self.cfg.eval_steps:
                break
            x, y = x.to(self.device), y.to(self.device)
            _, loss = self.model(x, y)
            losses.append(loss.item())
        self.model.train()
        return sum(losses) / len(losses)

    def _save_checkpoint(self):
        path = Path(self.cfg.checkpoint_dir) / f"step_{self.step:06d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "step": self.step,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "model_cfg": self.model_cfg,
                "train_cfg": self.cfg,
            },
            path,
        )
        print(f"  checkpoint saved → {path}")

    def _load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.step = ckpt["step"]
        print(f"Resumed from step {self.step}")

    def train(self):
        self.model.train()
        data_iter = iter(self.train_loader)
        t0 = time.perf_counter()

        while self.step < self.cfg.max_steps:
            lr = _cosine_lr(
                self.step,
                self.cfg.warmup_steps,
                self.cfg.max_steps,
                self.cfg.learning_rate,
                self.cfg.learning_rate / 10,
            )
            for g in self.optimizer.param_groups:
                g["lr"] = lr

            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                x, y = next(data_iter)

            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            _, loss = self.model(x, y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optimizer.step()
            self.step += 1

            if self.step % self.cfg.log_interval == 0:
                elapsed = time.perf_counter() - t0
                print(f"step {self.step:6d} | loss {loss.item():.4f} | lr {lr:.2e} | {elapsed:.1f}s")
                self.writer.add_scalar("train/loss", loss.item(), self.step)
                self.writer.add_scalar("train/lr", lr, self.step)
                wandb.log({"train/loss": loss.item(), "train/lr": lr}, step=self.step)
                t0 = time.perf_counter()

            if self.step % self.cfg.eval_interval == 0:
                val_loss = self._eval()
                print(f"  val loss: {val_loss:.4f}")
                self.writer.add_scalar("val/loss", val_loss, self.step)
                wandb.log({"val/loss": val_loss}, step=self.step)

            if self.step % self.cfg.checkpoint_interval == 0:
                self._save_checkpoint()

        self._save_checkpoint()
        self.writer.close()
        wandb.finish()
        print("Training complete.")
