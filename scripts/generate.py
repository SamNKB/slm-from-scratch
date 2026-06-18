"""
Generate text from a trained checkpoint.

Usage:
    python scripts/generate.py --checkpoint checkpoints/step_010000.pt --prompt "Era uma vez"
"""

import argparse
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from slm.inference import TextGenerator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="data/tokenizer")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    gen = TextGenerator.from_checkpoint(args.checkpoint, args.tokenizer)

    print(gen.generate(
        args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    ))


if __name__ == "__main__":
    main()
