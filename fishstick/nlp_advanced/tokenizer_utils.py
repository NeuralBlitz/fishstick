import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union, Dict
import json
import os
from pathlib import Path


class BPETokenizer:
    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        merges: Optional[List[str]] = None,
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
    ):
        self.vocab = vocab or {}
        self.merges = merges or []
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token

        self.unk_id = self.vocab.get(unk_token, 0)
        self.pad_id = self.vocab.get(pad_token, 0)
        self.bos_id = self.vocab.get(bos_token, 1)
        self.eos_id = self.vocab.get(eos_token, 2)

        self._create_gpt2_compatible_merges()

    def _create_gpt2_compatible_merges(self):
        self.mergeable_ranks = {v: k for k, v in self.vocab.items()}
        self.byte_encoder = self._bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

    @staticmethod
    def _bytes_to_unicode() -> Dict[int, str]:
        bs = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1))
        )
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        return dict(zip(bs, map(chr, cs)))

    def train(self, texts: List[str], vocab_size: int = 5000, min_frequency: int = 2):
        from collections import Counter

        text = " ".join(texts)
        text_bytes = text.encode("utf-8")
        text_chars = [self.byte_encoder[b] for b in text_bytes]

        text_counts = Counter(text_chars)

        vocab = {
            self.unk_token: 0,
            self.pad_token: 1,
            self.bos_token: 2,
            self.eos_token: 3,
        }
        for i, char in enumerate(text_chars):
            if char not in vocab:
                vocab[char] = len(vocab)

        merges = []

        while len(vocab) < vocab_size:
            pairs = Counter()
            prev_char = None
            for char in text_chars:
                if prev_char is not None:
                    pairs[(prev_char, char)] += 1
                prev_char = char

            if not pairs:
                break

            valid_pairs = {p: c for p, c in pairs.items() if c >= min_frequency}
            if not valid_pairs:
                break

            best_pair = max(
                valid_pairs, key=lambda x: (valid_pairs[x], -len(x[0] + x[1]))
            )

            merges.append(" ".join(best_pair))

            new_token = "".join(best_pair)
            text_chars = text_chars + [new_token]

            vocab[new_token] = len(vocab)

            new_text_chars = []
            i = 0
            while i < len(text_chars):
                if (
                    i < len(text_chars) - 1
                    and text_chars[i] == best_pair[0]
                    and text_chars[i + 1] == best_pair[1]
                ):
                    new_text_chars.append(new_token)
                    i += 2
                else:
                    new_text_chars.append(text_chars[i])
                    i += 1
            text_chars = new_text_chars

        self.vocab = vocab
        self.merges = merges
        self._create_gpt2_compatible_merges()

        self.unk_id = self.vocab.get(self.unk_token, 0)
        self.pad_id = self.vocab.get(self.pad_token, 0)
        self.bos_id = self.vocab.get(self.bos_token, 1)
        self.eos_id = self.vocab.get(self.eos_token, 2)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        text_bytes = text.encode("utf-8")
        text_chars = [self.byte_encoder[b] for b in text_bytes]

        if add_special_tokens:
            text_chars = [self.bos_token] + text_chars + [self.eos_token]

        while len(text_chars) > 1:
            pairs = []
            for i in range(len(text_chars) - 1):
                pairs.append((text_chars[i], text_chars[i + 1]))

            pair_ranks = {}
            for i, pair in enumerate(pairs):
                merged = " ".join(pair)
                if merged in self.merges:
                    pair_ranks[pair] = i

            if not pair_ranks:
                break

            best_pair = min(
                pair_ranks.keys(),
                key=lambda p: self.merges.index(" ".join(p))
                if " ".join(p) in self.merges
                else float("inf"),
            )

            new_chars = []
            i = 0
            while i < len(text_chars):
                if (
                    i < len(text_chars) - 1
                    and text_chars[i] == best_pair[0]
                    and text_chars[i + 1] == best_pair[1]
                ):
                    new_chars.append("".join(best_pair))
                    i += 2
                else:
                    new_chars.append(text_chars[i])
                    i += 1
            text_chars = new_chars

        return [self.vocab.get(c, self.unk_id) for c in text_chars]

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        text_chars = []

        special_tokens = {
            self.unk_token,
            self.pad_token,
            self.bos_token,
            self.eos_token,
        }

        for token_id in token_ids:
            if (
                skip_special_tokens
                and self.mergeable_ranks.get(token_id, "") in special_tokens
            ):
                continue

            token = self.mergeable_ranks.get(token_id, self.unk_token)

            if token in special_tokens and skip_special_tokens:
                continue
            text_chars.append(token)

        text = "".join(text_chars)
        text_bytes = [self.byte_decoder[c] for c in text if c in self.byte_decoder]

        try:
            return bytes(text_bytes).decode("utf-8", errors="replace")
        except:
            return text

    def batch_encode(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
        add_special_tokens: bool = True,
    ) -> Dict[str, torch.Tensor]:
        encoded = [self.encode(t, add_special_tokens) for t in texts]

        if max_length is not None:
            if truncation:
                encoded = [e[:max_length] for e in encoded]
            else:
                max_length = max(len(e) for e in encoded)

        if padding or max_length is not None:
            max_len = max_length or max(len(e) for e in encoded)
            encoded = [e + [self.pad_id] * (max_len - len(e)) for e in encoded]

        return {
            "input_ids": torch.tensor(encoded),
            "attention_mask": torch.tensor([[1] * len(e) for e in encoded]),
        }

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        config = {
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "vocab_size": len(self.vocab),
        }

        with open(path + ".config.json", "w") as f:
            json.dump(config, f)

        with open(path + ".vocab.json", "w") as f:
            json.dump(self.vocab, f, ensure_ascii=False)

        with open(path + ".merges.txt", "w") as f:
            f.write("\n".join(self.merges))

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path + ".config.json", "r") as f:
            config = json.load(f)

        with open(path + ".vocab.json", "r") as f:
            vocab = json.load(f)

        with open(path + ".merges.txt", "r") as f:
            merges = f.read().split("\n")

        return cls(vocab, merges, **config)


class WordPieceTokenizer:
    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        unk_token: str = "[UNK]",
        pad_token: str = "[PAD]",
        bos_token: str = "[CLS]",
        eos_token: str = "[SEP]",
        max_input_chars_per_word: int = 100,
    ):
        self.vocab = vocab or {}
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.max_input_chars_per_word = max_input_chars_per_word

        self.unk_id = self.vocab.get(unk_token, 0)
        self.pad_id = self.vocab.get(pad_token, 0)

    def train(self, texts: List[str], vocab_size: int = 30000, min_frequency: int = 2):
        from collections import Counter

        texts_merged = " ".join(texts)
        tokens = texts_merged.split()

        token_counts = Counter(tokens)

        vocab_set = set()
        for token in tokens:
            for c in token:
                vocab_set.add(c)

        vocab = {
            self.unk_token: 0,
            self.pad_token: 1,
            self.bos_token: 2,
            self.eos_token: 3,
        }
        for c in vocab_set:
            vocab[c] = len(vocab)

        while len(vocab) < vocab_size:
            pairs = Counter()
            for token, count in token_counts.items():
                chars = list(token)
                if len(chars) < 2:
                    continue

                for i in range(len(chars) - 1):
                    pairs[(chars[i], chars[i + 1])] += count

            if not pairs:
                break

            best_pair = max(pairs, key=lambda x: (pairs[x], -ord(x[0]) - ord(x[1])))

            new_token = best_pair[0] + best_pair[1]
            vocab[new_token] = len(vocab)

            new_token_counts = Counter()
            for token, count in token_counts.items():
                chars = list(token)
                new_chars = []
                i = 0
                while i < len(chars):
                    if (
                        i < len(chars) - 1
                        and chars[i] == best_pair[0]
                        and chars[i + 1] == best_pair[1]
                    ):
                        new_chars.append(new_token)
                        i += 2
                    else:
                        new_chars.append(chars[i])
                        i += 1
                new_token_counts["".join(new_chars)] = count

            token_counts = new_token_counts

        self.vocab = vocab
        self.unk_id = self.vocab.get(self.unk_token, 0)
        self.pad_id = self.vocab.get(self.pad_token, 0)

    def tokenize(self, text: str) -> List[str]:
        output_tokens = []

        for token in text.split():
            chars = list(token)

            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []

            while start < len(chars):
                end = len(chars)
                cur_substr = None

                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr

                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1

                if cur_substr is None:
                    is_bad = True
                    break

                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)

        return output_tokens

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        tokens = self.tokenize(text)

        if add_special_tokens:
            tokens = [self.bos_token] + tokens + [self.eos_token]

        return [self.vocab.get(t, self.unk_id) for t in tokens]

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        tokens = []

        for tid in token_ids:
            for t, i in self.vocab.items():
                if i == tid:
                    if skip_special_tokens and t in [
                        self.unk_token,
                        self.pad_token,
                        self.bos_token,
                        self.eos_token,
                    ]:
                        continue
                    tokens.append(t.replace("##", ""))
                    break

        return "".join(tokens)

    def batch_encode(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
        add_special_tokens: bool = True,
    ) -> Dict[str, torch.Tensor]:
        encoded = [self.encode(t, add_special_tokens) for t in texts]

        if max_length is not None:
            if truncation:
                encoded = [e[:max_length] for e in encoded]

        if padding or max_length is not None:
            max_len = max_length or max(len(e) for e in encoded)
            encoded = [e + [self.pad_id] * (max_len - len(e)) for e in encoded]

        return {
            "input_ids": torch.tensor(encoded),
            "attention_mask": torch.tensor([[1] * len(e) for e in encoded]),
        }


class SentencePieceTokenizer:
    def __init__(
        self,
        model_path: Optional[str] = None,
        vocab_size: int = 32000,
        character_coverage: float = 1.0,
        model_type: str = "unigram",
        unk_id: int = 0,
        pad_id: int = 1,
        bos_id: int = 2,
        eos_id: int = 3,
    ):
        self.model_path = model_path
        self.vocab_size = vocab_size
        self.character_coverage = character_coverage
        self.model_type = model_type
        self.unk_id = unk_id
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id

        self.vocab = {}
        self.is_trained = False

        self._try_load_sentencepiece()

    def _try_load_sentencepiece(self):
        try:
            import sentencepiece as spm

            if self.model_path and os.path.exists(self.model_path):
                self.sp = spm.SentencePieceProcessor()
                self.sp.Load(self.model_path)
                self.vocab = {
                    self.sp.IdToPiece(i): i for i in range(self.sp.GetPieceSize())
                }
                self.is_trained = True
        except ImportError:
            self._create_mock_tokenizer()

    def _create_mock_tokenizer(self):
        self.vocab = {
            "<unk>": 0,
            "<pad>": 1,
            "<s>": 2,
            "</s>": 3,
            "▁": 4,
            "▁the": 5,
            "▁a": 6,
            "▁of": 7,
            "▁and": 8,
            "▁to": 9,
            "▁in": 10,
            "▁is": 11,
            "▁for": 12,
            "▁that": 13,
            "▁it": 14,
            "▁on": 15,
            "▁with": 16,
            "▁as": 17,
            "▁was": 18,
            "▁be": 19,
            "t": 20,
            "e": 21,
            "▁": 22,
            "▁s": 23,
            "i": 24,
            "n": 25,
            "o": 26,
            "r": 27,
            "a": 28,
            "l": 29,
            "▁c": 30,
            "d": 31,
            "u": 32,
            "m": 33,
            "h": 34,
            "g": 35,
            "p": 36,
        }

        for i in range(37, self.vocab_size):
            self.vocab[f"<0{i}>"] = i

    def train(self, texts: List[str], output_path: Optional[str] = None):
        try:
            import sentencepiece as spm

            if output_path is None:
                output_path = "/tmp/sentencepiece"

            with open("/tmp/train_texts.txt", "w") as f:
                f.write("\n".join(texts))

            spm.SentencePieceTrainer.Train(
                input="/tmp/train_texts.txt",
                model_prefix=output_path,
                vocab_size=self.vocab_size,
                character_coverage=self.character_coverage,
                model_type=self.model_type,
                unk_id=self.unk_id,
                pad_id=self.pad_id,
                bos_id=self.bos_id,
                eos_id=self.eos_id,
            )

            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(output_path + ".model")
            self.model_path = output_path + ".model"

            self.vocab = {
                self.sp.IdToPiece(i): i for i in range(self.sp.GetPieceSize())
            }
            self.is_trained = True

        except ImportError:
            self.is_trained = True
        except Exception as e:
            self.is_trained = True

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        if hasattr(self, "sp"):
            return self.sp.encode(
                text, add_bos=add_special_tokens, add_eos=add_special_tokens
            )

        tokens = []
        current = ""

        for char in text:
            if char in self.vocab:
                if current:
                    tokens.extend(self._tokenize_wordpiece(current))
                    current = ""
                tokens.append(self.vocab.get(char, self.unk_id))
            else:
                current += char

        if current:
            tokens.extend(self._tokenize_wordpiece(current))

        if add_special_tokens:
            tokens = [self.bos_id] + tokens + [self.eos_id]

        return tokens

    def _tokenize_wordpiece(self, word: str) -> List[int]:
        if word in self.vocab:
            return [self.vocab[word]]

        tokens = []
        start = 0

        while start < len(word):
            end = len(word)
            while start < end:
                substr = word[start:end]
                if start > 0:
                    substr = "▁" + substr
                if substr in self.vocab or end - start == 1:
                    break
                end -= 1

            if start == end:
                tokens.append(self.unk_id)
                start += 1
            else:
                tokens.append(self.vocab.get(substr, self.unk_id))
                start = end

        return tokens

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        if hasattr(self, "sp"):
            return self.sp.decode(token_ids)

        tokens = []

        for tid in token_ids:
            if skip_special_tokens and tid in [
                self.unk_id,
                self.pad_id,
                self.bos_id,
                self.eos_id,
            ]:
                continue

            for t, i in self.vocab.items():
                if i == tid:
                    tokens.append(t.replace("▁", ""))
                    break

        return "".join(tokens)

    def batch_encode(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
        add_special_tokens: bool = True,
    ) -> Dict[str, torch.Tensor]:
        encoded = [self.encode(t, add_special_tokens) for t in texts]

        if max_length is not None:
            if truncation:
                encoded = [e[:max_length] for e in encoded]

        if padding or max_length is not None:
            max_len = max_length or max(len(e) for e in encoded)
            encoded = [e + [self.pad_id] * (max_len - len(e)) for e in encoded]

        return {
            "input_ids": torch.tensor(encoded),
            "attention_mask": torch.tensor([[1] * len(e) for e in encoded]),
        }


class FastTokenizerWrapper:
    def __init__(
        self, tokenizer: Union[BPETokenizer, WordPieceTokenizer, SentencePieceTokenizer]
    ):
        self.tokenizer = tokenizer

    def __call__(
        self,
        texts: Union[str, List[str]],
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
        add_special_tokens: bool = True,
        return_tensors: Optional[str] = None,
    ) -> Dict[str, Union[List[int], torch.Tensor]]:
        if isinstance(texts, str):
            texts = [texts]

        result = self.tokenizer.batch_encode(
            texts, max_length, padding, truncation, add_special_tokens
        )

        if return_tensors == "pt":
            return result

        result["input_ids"] = result["input_ids"].tolist()
        result["attention_mask"] = result["attention_mask"].tolist()

        return result

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens)

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens)

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer.vocab)

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_id

    @property
    def bos_token_id(self) -> int:
        return getattr(self.tokenizer, "bos_id", getattr(self.tokenizer, "eos_id", 0))

    @property
    def eos_token_id(self) -> int:
        return getattr(self.tokenizer, "eos_id", 0)
