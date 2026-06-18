"""
Avalia o modelo de endereços com entradas livres.

Uso:
    python scripts/address_eval.py --checkpoint checkpoints/address/step_020000.pt

Digite endereços no prompt interativo. Ctrl+C para sair.
"""

import argparse
import torch
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from slm.model import SLM
from slm.tokenizer import BPETokenizer


def normalizar(model: SLM, tok: BPETokenizer, endereco: str, device: torch.device) -> str:
    prompt = f"ENTRADA: {endereco} | SAIDA:"
    ids = [tok.bos_id] + tok.encode(prompt)
    idx = torch.tensor([ids], device=device)

    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=80, temperature=0.3, top_k=20)

    gerado = tok.decode(out[0].tolist())
    # extrai só o que vem depois de "SAIDA:"
    if "SAIDA:" in gerado:
        return gerado.split("SAIDA:")[-1].strip().split("\n")[0].strip()
    return gerado.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="data/tokenizer")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = SLM(ckpt["model_cfg"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tok = BPETokenizer.load(args.tokenizer)
    print(f"Modelo carregado. Device: {device}\n")

    exemplos = [
        "r das flores 123 sp",
        "Av. Paulista nº1578 ap 42 Sao Paulo",
        "tv. marechal deodoro, n.50 - BH",
        "rua XV de Novembro 900 sala 3 porto alegre",
    ]
    print("=== Exemplos automáticos ===")
    for e in exemplos:
        resultado = normalizar(model, tok, e, device)
        print(f"  IN : {e}")
        print(f"  OUT: {resultado}\n")

    print("=== Modo interativo (Enter vazio para sair) ===")
    while True:
        try:
            entrada = input("Endereço: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not entrada:
            break
        print(f"  → {normalizar(model, tok, entrada, device)}\n")


if __name__ == "__main__":
    main()
