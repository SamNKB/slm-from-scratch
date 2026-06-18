"""
Train an SLM from scratch.

Usage:
    python scripts/train.py
    python scripts/train.py --config configs/small.yaml
    python scripts/train.py --resume checkpoints/step_005000.pt
"""

import argparse
import yaml
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from slm.config import ModelConfig, TrainConfig
from slm.trainer import Trainer


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = ModelConfig(**cfg.get("model", {}))
    train_cfg = TrainConfig(**cfg.get("train", {}))

    if args.resume:
        train_cfg.resume_from = args.resume

    trainer = Trainer(model_cfg, train_cfg)
    trainer.train()


if __name__ == "__main__":
    main()
