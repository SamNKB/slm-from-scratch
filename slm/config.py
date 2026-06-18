from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    context_length: int = 512
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    dropout: float = 0.1
    bias: bool = False


@dataclass
class TrainConfig:
    # data
    data_path: str = "data/processed/train.bin"
    val_data_path: str = "data/processed/val.bin"

    # training
    batch_size: int = 32
    max_steps: int = 10_000
    eval_interval: int = 500
    eval_steps: int = 50
    log_interval: int = 50

    # optimizer
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_steps: int = 200

    # checkpointing
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 1000
    resume_from: str | None = None

    # device
    device: str = "cuda"
    compile: bool = False  # torch.compile — experimental

    # logging
    tensorboard_dir: str = "runs"
    wandb_project: str = "slm-from-scratch"
    wandb_mode: str = "offline"  # "offline" | "online" | "disabled"
