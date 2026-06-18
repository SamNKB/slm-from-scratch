import json
import regex as re
from pathlib import Path
from collections import defaultdict


class BPETokenizer:
    """Minimal Byte-Pair Encoding tokenizer."""

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"

    SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

    # GPT-4 style pre-tokenization pattern
    _PAT = re.compile(
        r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
    )

    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.merges: dict[tuple[str, str], str] = {}
        self._build_base_vocab()

    def _build_base_vocab(self):
        for tok in self.SPECIAL_TOKENS:
            self.vocab[tok] = len(self.vocab)
        for i in range(256):
            self.vocab[chr(i)] = len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self.vocab[self.PAD_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.vocab[self.BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.vocab[self.EOS_TOKEN]

    def __len__(self) -> int:
        return len(self.vocab)

    def _get_stats(self, corpus: list[list[str]]) -> dict[tuple[str, str], int]:
        counts: dict[tuple[str, str], int] = defaultdict(int)
        for word in corpus:
            for a, b in zip(word, word[1:]):
                counts[(a, b)] += 1
        return counts

    def _merge(self, corpus: list[list[str]], pair: tuple[str, str], merged: str) -> list[list[str]]:
        a, b = pair
        out = []
        for word in corpus:
            new_word: list[str] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            out.append(new_word)
        return out

    def train(self, text: str, vocab_size: int, verbose: bool = True):
        assert vocab_size > len(self.vocab), "vocab_size must be larger than base vocab"
        n_merges = vocab_size - len(self.vocab)

        words = [list(w) for w in re.findall(self._PAT, text)]
        self.merges = {}
        self._merge_rank: dict[tuple[str, str], int] = {}

        for step in range(n_merges):
            stats = self._get_stats(words)
            if not stats:
                break
            best = max(stats, key=stats.get)
            merged = "".join(best)
            self.vocab[merged] = len(self.vocab)
            self.merges[best] = merged
            self._merge_rank[best] = step
            words = self._merge(words, best, merged)
            if verbose and (step + 1) % 500 == 0:
                print(f"  merge {step + 1}/{n_merges}: {best!r} → {merged!r}")

    def encode(self, text: str, add_special: bool = False) -> list[int]:
        tokens: list[int] = []
        if add_special:
            tokens.append(self.bos_id)

        for word in re.findall(self._PAT, text):
            chars = list(word)
            while True:
                if len(chars) < 2:
                    break
                stats = {(a, b): 0 for a, b in zip(chars, chars[1:])}
                candidates = {pair: self.merges[pair] for pair in stats if pair in self.merges}
                if not candidates:
                    break
                best = min(candidates, key=lambda p: self._merge_rank.get(p, float("inf")))
                chars = self._merge([chars], best, candidates[best])[0]

            for ch in chars:
                tokens.append(self.vocab.get(ch, self.vocab[self.UNK_TOKEN]))

        if add_special:
            tokens.append(self.eos_id)
        return tokens

    def decode(self, ids: list[int]) -> str:
        inv = {v: k for k, v in self.vocab.items()}
        return "".join(inv.get(i, self.UNK_TOKEN) for i in ids if i not in {self.pad_id, self.bos_id, self.eos_id})

    def save(self, path: str | Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "vocab.json", "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False)
        merges_list = [[list(k), v] for k, v in self.merges.items()]
        with open(path / "merges.json", "w", encoding="utf-8") as f:
            json.dump(merges_list, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        path = Path(path)
        tok = cls.__new__(cls)
        with open(path / "vocab.json", encoding="utf-8") as f:
            tok.vocab = json.load(f)
        with open(path / "merges.json", encoding="utf-8") as f:
            raw = json.load(f)
        tok.merges = {tuple(k): v for k, v in raw}
        tok._merge_rank = {tuple(k): i for i, (k, _) in enumerate(raw)}
        return tok
