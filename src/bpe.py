from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import TypeAlias

from src.pretokenizers import Pretokenizer, gpt2_pretokenize

# user-defined type alias
TokenId: TypeAlias = int
Pair: TypeAlias = tuple[TokenId, TokenId]
Word: TypeAlias = tuple[TokenId, ...] # pretoken


@dataclass
class TokenizerSpec:
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]
    special_tokens: tuple[str, ...]

# initial vocab for bpe
def create_initial_vocab(special_tokens: tuple[str, ...]) -> dict[int, bytes]:
    vocab = {idx: bytes([idx]) for idx in range(256) }
    existing = set(vocab.values())

    for sp_token in special_tokens:
        sp_token_bytes = sp_token.encode("utf-8")
        if sp_token_bytes not in existing:
            vocab[len(vocab)] = sp_token_bytes
            existing.add(sp_token_bytes)
        
    return vocab


# merge the top_pair with new token_id which is added to vocab
def merge_pair(word: Word, pair: Pair, new_token_id: TokenId) -> Word:
    output: list[TokenId] = []
    i = 0

    while i < len(word):
        if i < len(word) - 1 and (word[i], word[i+1]) == pair:
            output.append(new_token_id)
            i += 2
        else:
            output.append(word[i])
            i += 1

    return tuple(output)


# calculate pairwise frequences across all words in word_freqs
def compute_pairs_freqs(word_freqs: dict[Word, int]) -> Counter[Pair]:
    pair_freqs: Counter[Pair] = Counter()

    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            pair_freqs[(word[i], word[i+1])] += freq

    return pair_freqs


def train_bpe_slow(
    text: str,
    vocab_size: int,
    special_tokens: tuple[str, ...] = (),
    pretokenizer: Pretokenizer = gpt2_pretokenize,
) -> TokenizerSpec:
    if vocab_size < 256:
        raise ValueError("vocab_size must be atleast 256")
    
    vocab = create_initial_vocab(special_tokens)
    special_set = set(special_tokens)

    pretoken_counts = Counter(pretokenizer(text, special_tokens))
    word_freqs: dict[Word, int] = defaultdict(int)

    for pretoken, count in pretoken_counts.items():
        if pretoken in special_set:
            continue

        word = tuple(pretoken.encode("utf-8"))
        if word:
            word_freqs[word] += count

    merges: list[tuple[bytes, bytes]] = []

    while len(vocab) < vocab_size:
        pair_freqs = compute_pairs_freqs(word_freqs)

        if not pair_freqs:
            break
        
        # highest pair freq and in lexi order
        best_pair = max(pair_freqs, key=lambda pair: (pair_freqs[pair], vocab[pair[0]], vocab[pair[1]]))

        new_token_id = len(vocab)
        left_bytes = vocab[best_pair[0]]
        right_bytes = vocab[best_pair[1]]

        vocab[new_token_id] = left_bytes + right_bytes
        merges.append((left_bytes, right_bytes))

        new_word_freqs: dict[Word, int] = defaultdict(int)
        for word, freq in word_freqs.items():
            new_word = merge_pair(word, best_pair, new_token_id)
            new_word_freqs[new_word] += freq
        
        word_freqs = new_word_freqs

    return TokenizerSpec(vocab=vocab, merges=merges, special_tokens=special_tokens)


class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: tuple[str, ...] = (),
        pretokenizer: Pretokenizer = gpt2_pretokenize,
    ):
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = special_tokens
        self.pretokenizer = pretokenizer

        self.token_to_id = {token_bytes: token_id for token_id, token_bytes in self.vocab.items()}
        self.merge_ranks = {merge: rank for rank, merge in enumerate(self.merges)}

        self.special_token_to_id = {
            token: self.token_to_id[token.encode("utf-8")]
            for token in self.special_tokens
        }

    @classmethod
    def from_spec(
        cls,
        spec: TokenizerSpec,
        pretokenizer: Pretokenizer = gpt2_pretokenize,
    ) -> "BPETokenizer":
        return cls(
            vocab=spec.vocab,
            merges=spec.merges,
            special_tokens=spec.special_tokens,
            pretokenizer=pretokenizer,
        )

    def decode(self, token_ids: list[int]) -> str:
        raw_bytes = b"".join(self.vocab[token_id] for token_id in token_ids)
        return raw_bytes.decode("utf-8", errors="replace")

    def encode(self, text: str) -> list[int]:
        output_ids: list[int] = []
        special_set = set(self.special_tokens)

        for pretoken in self.pretokenizer(text, self.special_tokens):
            if pretoken in special_set:
                # special tokens have a token_id predefined in vocab
                output_ids.append(self.special_token_to_id[pretoken])
            else:
                output_ids.extend(self._encode_pretoken(pretoken))

        return output_ids

    def _encode_pretoken(self, pretoken: str) -> list[int]:
        tokens = [bytes([byte]) for byte in pretoken.encode("utf-8")]

        while len(tokens) >= 2:
            # get the pair with lowest rank
            best_pair = min(
                ((tokens[i], tokens[i+1]) for i in range(len(tokens) - 1)),
                key=lambda pair: self.merge_ranks.get(pair, float("inf")),
            )

            if best_pair not in self.merge_ranks:
                break

            tokens = self._merge_bytes_pair(tokens, best_pair)

        return [self.token_to_id[token] for token in tokens]


    def _merge_bytes_pair(self, tokens: list[bytes], pair: tuple[bytes, bytes],) -> list[bytes]:
        output: list[bytes] = []
        i = 0

        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair:
                output.append(tokens[i] + tokens[i+1])
                i += 2
            else:
                output.append(tokens[i])
                i += 1

        return output

    

    


