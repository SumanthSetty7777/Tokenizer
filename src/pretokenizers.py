# Then BPE learns merges inside each pretoken, not across the whole document. This prevents weird merges like "o," + " how" becoming one token.

from __future__ import annotations
from collections.abc import Callable
import regex as re

GPT2_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

Pretokenizer = Callable[[str, tuple[str, ...]], list[str]]


def split_special_tokens(text: str, special_tokens: tuple[str, ...] = ()) -> list[str]:
    if not special_tokens:
        return [text]
        
    special_tokens_set = set(special_tokens)

    escaped = [re.escape(tok) for tok in special_tokens_set]
    escaped.sort(key=len, reverse=True)

    pattern = "(" + "|".join(escaped) + ")"
    return re.split(pattern, text)


# text split across some standard conventions
def gpt2_pretokenize(text: str, special_tokens: tuple[str, ...] = ()) -> list[str]:
    special_token_set = set(special_tokens)
    pretokens: list[str] = []

    for piece in split_special_tokens(text, special_tokens):
        if piece == "":
            continue

        if piece in special_token_set:
            pretokens.append(piece)
            continue

        pretokens.extend(match.group() for match in GPT2_PATTERN.finditer(piece))
    return pretokens


# text split across whitespace " "
def whitespace_pretokenize(text: str, special_tokens: tuple[str, ...] = ()) -> list[str]:
    special_token_set = set(special_tokens)
    pretokens: list[str] = []

    for piece in split_special_tokens(text, special_tokens):
        if piece == "":
            continue

        if piece in special_token_set:
            pretokens.append(piece)
            continue

        pretokens.extend(piece.split())
    return pretokens


# allow merges across everything except special tokens
def byte_pretokenize(text: str, special_tokens: tuple[str, ...] = ()) -> list[str]:
    special_token_set = set(special_tokens)
    pretokens: list[str] = []

    for piece in split_special_tokens(text, special_tokens):
        if piece == "":
            continue

        pretokens.append(piece)
    return pretokens

