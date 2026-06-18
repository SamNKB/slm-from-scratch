"""
Avalia o modelo de endereços com entradas livres.

Uso:
    python scripts/address_eval.py --checkpoint checkpoints/address/step_020000.pt

Digite endereços no prompt interativo. Enter vazio para sair.
"""

import argparse
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from slm.inference import AddressNormalizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="data/tokenizer")
    args = parser.parse_args()

    norm = AddressNormalizer.from_checkpoint(args.checkpoint, args.tokenizer)
    print(f"Modelo carregado. Device: {norm.device}\n")

    exemplos = [
        "r das flores 123 sp",
        "Av. Paulista nº1578 ap 42 Sao Paulo",
        "tv. marechal deodoro, n.50 - BH",
        "rua XV de Novembro 900 sala 3 porto alegre",
    ]
    print("=== Exemplos automáticos ===")
    for e in exemplos:
        print(f"  IN : {e}")
        print(f"  OUT: {norm.normalize(e)}\n")

    print("=== Modo interativo (Enter vazio para sair) ===")
    while True:
        try:
            entrada = input("Endereço: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not entrada:
            break
        print(f"  → {norm.normalize(entrada)}\n")


if __name__ == "__main__":
    main()
