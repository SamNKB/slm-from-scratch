"""
Camada de inferência compartilhada.

Centraliza o carregamento de checkpoint/tokenizador e a geração de texto,
para que API, Spark e CLIs não dupliquem a mesma lógica (e os mesmos bugs).

    from slm.inference import AddressNormalizer

    norm = AddressNormalizer.from_checkpoint(
        "checkpoints/address/step_002000.pt", "data/tokenizer"
    )
    print(norm.normalize("av paulista 1578 ap 42 sp"))
"""

from __future__ import annotations

import torch

from .model import SLM
from .tokenizer import BPETokenizer


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve "auto" para cuda/cpu; aceita "cpu"/"cuda" explícitos."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_model(checkpoint: str, device: str | torch.device = "auto") -> tuple[SLM, torch.device]:
    """Carrega um SLM de um checkpoint em modo de avaliação."""
    dev = resolve_device(device) if isinstance(device, str) else device
    ckpt = torch.load(checkpoint, map_location=dev, weights_only=False)
    model = SLM(ckpt["model_cfg"]).to(dev)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, dev


class TextGenerator:
    """Geração de texto livre a partir de um checkpoint treinado."""

    def __init__(self, model: SLM, tokenizer: BPETokenizer, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @classmethod
    def from_checkpoint(
        cls, checkpoint: str, tokenizer_dir: str, device: str | torch.device = "auto"
    ) -> "TextGenerator":
        model, dev = load_model(checkpoint, device)
        tokenizer = BPETokenizer.load(tokenizer_dir)
        return cls(model, tokenizer, dev)

    def generate(
        self,
        prompt: str = "",
        *,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int | None = 50,
    ) -> str:
        """Gera a continuação de `prompt` e retorna o texto decodificado completo."""
        tok = self.tokenizer
        ids = [tok.bos_id] + (tok.encode(prompt) if prompt else [])
        idx = torch.tensor([ids], dtype=torch.long, device=self.device)
        out = self.model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k)
        return tok.decode(out[0].tolist())


class AddressNormalizer(TextGenerator):
    """Normalização de endereços via completamento `ENTRADA: ... | SAIDA: ...`."""

    PROMPT = "ENTRADA: {endereco} | SAIDA:"

    def normalize(
        self,
        endereco: str,
        *,
        temperature: float = 0.3,
        top_k: int = 20,
        max_new_tokens: int = 60,
    ) -> str:
        """Retorna apenas a forma canônica (o que vem depois de `SAIDA:`)."""
        prompt = self.PROMPT.format(endereco=endereco.strip())
        decoded = self.generate(
            prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k
        )
        if "SAIDA:" in decoded:
            return decoded.split("SAIDA:")[-1].strip().split("\n")[0].strip()
        return decoded.strip()
