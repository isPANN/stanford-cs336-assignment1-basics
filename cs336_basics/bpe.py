from curses import pair_content
import os
import regex as re

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(input_path, "r", encoding="utf-8") as file:
        content = file.read()
    
    vocab = init_vocab(special_tokens)

    # return vocab, merges

def init_vocab(special_tokens: list[str]) -> dict:
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])
    for i in range(len(special_tokens)):
        vocab[256+i] = special_tokens[i].encode("utf-8")
    return vocab

def split_by_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    # `special_tokens` is a list of boundary strings. 
    # We will only pre-tokenize within each individual segment after they have been split apart.
    pattern = "|".join(re.escape(t) for t in special_tokens)
    segments = re.split(pattern, text)
    return segments

def pretokenize_and_count(segments: list[str]) -> dict[tuple[bytes, ...], int]:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    counts = {}
    for segment in segments:
        for m in re.finditer(PAT, segment):
            m_byte_tuple = tuple(bytes([b]) for b in m.group().encode("utf-8"))
            counts[m_byte_tuple] = counts.get(m_byte_tuple, 0) + 1
    return counts

def count_pairs(counts: dict[tuple[bytes, ...], int]):
    pair_counts = {}
    for seq in counts.keys():
        tmp_pair_counts = {}
        for pair in zip(seq, seq[1:]):
            tmp_pair_counts[pair] = tmp_pair_counts.get(pair, 0) + 1
        for pair in tmp_pair_counts.keys():
            pair_counts[pair] = pair_counts.get(pair, 0) + tmp_pair_counts[pair] * counts[seq]
    return pair_counts


if __name__ == "__main__":
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    corpus_path = PROJECT_ROOT / "tests" / "fixtures" / "corpus.en"

    with open(corpus_path, "r", encoding="utf-8") as file:
        content = file.read()
    vocab = init_vocab(["rr", "sw"])
    assert len(vocab) == 258
    assert vocab[0] == bytes([0]) and vocab[255] == bytes([255])

    result = split_by_special_tokens("Doc1<|endoftext|>Doc2", ["<|endoftext|>"])
    assert result == ["Doc1", "Doc2"]
    assert all("<|endoftext|>" not in s for s in result)
    assert count_pairs(pretokenize_and_count(result)) == {(b'D', b'o'): 2, (b'o', b'c'): 2}