"""
Generate text from a trained checkpoint.

Usage:
    python scripts/generate.py --checkpoint checkpoints/step_010000.pt --prompt "Era uma vez"
"""

import argparse
import torch
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from slm.model import SLM
from slm.tokenizer import BPETokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="data/tokenizer")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = SLM(ckpt["model_cfg"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tok = BPETokenizer.load(args.tokenizer)

    ids = [tok.bos_id] + (tok.encode(args.prompt) if args.prompt else [])
    idx = torch.tensor([ids], device=device)

    with torch.no_grad():
        out = model.generate(idx, args.max_tokens, temperature=args.temperature, top_k=args.top_k)

    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
