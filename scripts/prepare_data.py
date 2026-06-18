"""
Tokenize a raw text file and save to binary format for fast loading during training.

Usage:
    python scripts/prepare_data.py --input data/raw/corpus.txt --vocab-size 8192
"""

import argparse
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from slm.tokenizer import BPETokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw .txt file")
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--tokenizer-dir", default="data/tokenizer")
    args = parser.parse_args()

    text = Path(args.input).read_text(encoding="utf-8")
    print(f"Corpus: {len(text):,} chars")

    print(f"Training BPE tokenizer (vocab_size={args.vocab_size})...")
    tok = BPETokenizer()
    tok.train(text, args.vocab_size)
    tok.save(args.tokenizer_dir)
    print(f"Tokenizer saved to {args.tokenizer_dir}/ ({len(tok):,} tokens)")

    print("Encoding corpus...")
    ids = tok.encode(text)
    ids = np.array(ids, dtype=np.uint16)
    print(f"Total tokens: {len(ids):,}")

    split = int(len(ids) * (1 - args.val_split))
    train_ids, val_ids = ids[:split], ids[split:]

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_ids.tofile(out / "train.bin")
    val_ids.tofile(out / "val.bin")
    print(f"Saved train ({len(train_ids):,}) and val ({len(val_ids):,}) tokens to {out}/")


if __name__ == "__main__":
    main()
