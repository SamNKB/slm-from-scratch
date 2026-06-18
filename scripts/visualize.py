"""
Visualização do modelo com Netron (arquitetura) e W&B (métricas offline).

Uso:
    # Abre o modelo no Netron (exporta para ONNX automaticamente)
    python scripts/visualize.py --checkpoint checkpoints/address/step_002000.pt

    # Sincroniza runs offline do W&B para a nuvem (precisa de conta)
    python scripts/visualize.py --wandb-sync

    # Só exporta ONNX, sem abrir o Netron
    python scripts/visualize.py --checkpoint checkpoints/address/step_002000.pt --no-browser
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from slm.model import SLM


class _InferenceWrapper(nn.Module):
    """Wrapper para exportar só a inferência (sem loss)."""
    def __init__(self, model: SLM):
        super().__init__()
        self.model = model

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        logits, _ = self.model(idx)
        return logits


def export_onnx(checkpoint_path: str) -> str:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = SLM(ckpt["model_cfg"])
    model.load_state_dict(ckpt["model"])
    model.eval()

    cfg = ckpt["model_cfg"]
    wrapper = _InferenceWrapper(model)

    dummy = torch.randint(0, cfg.vocab_size, (1, cfg.context_length))
    onnx_path = str(Path(checkpoint_path).with_suffix(".onnx"))

    torch.onnx.export(
        wrapper,
        dummy,
        onnx_path,
        input_names=["tokens"],
        output_names=["logits"],
        dynamic_axes={"tokens": {1: "seq_len"}, "logits": {1: "seq_len"}},
        opset_version=17,
    )
    print(f"ONNX exportado: {onnx_path}")
    print(f"  vocab_size={cfg.vocab_size}  context={cfg.context_length}")
    print(f"  d_model={cfg.d_model}  n_heads={cfg.n_heads}  n_layers={cfg.n_layers}")
    return onnx_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", help="Caminho para o .pt a visualizar")
    parser.add_argument("--no-browser", action="store_true", help="Só exporta ONNX, não abre o Netron")
    parser.add_argument("--port", type=int, default=8080, help="Porta do Netron (default: 8080)")
    parser.add_argument("--wandb-sync", action="store_true", help="Sincroniza runs offline do W&B para a nuvem")
    args = parser.parse_args()

    if args.wandb_sync:
        import subprocess
        print("Sincronizando runs offline do W&B...")
        subprocess.run(["wandb", "sync", "--sync-all"], check=True)
        return

    if not args.checkpoint:
        parser.error("--checkpoint é obrigatório (a não ser que use --wandb-sync)")

    onnx_path = export_onnx(args.checkpoint)

    if not args.no_browser:
        import netron
        print(f"\nAbrindo Netron em http://localhost:{args.port}")
        print("Ctrl+C para fechar.")
        netron.start(onnx_path, address=("localhost", args.port))


if __name__ == "__main__":
    main()
