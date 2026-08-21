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
    segments = split_by_special_tokens(content, special_tokens)
    pre_token_counts = pretokenize_and_count(segments)
    pair_counts = count_pairs(pre_token_counts)
    vocab = init_vocab(special_tokens)
    merges = []
    while len(vocab) < vocab_size:
        token_a, token_b = select_best_pair(pair_counts)
        pre_token_counts, pair_counts = apply_merge(pre_token_counts, pair_counts, (token_a, token_b))
        merges.append((token_a, token_b))
        vocab[len(vocab)] = token_a + token_b
    return vocab, merges

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

def select_best_pair(pair_counts: dict[tuple[bytes, bytes], int]) -> tuple[bytes, bytes]:
    best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
    return best_pair

def apply_merge(counts: dict[tuple[bytes, ...], int], pair_counts: dict[tuple[bytes, bytes], int], pair: tuple[bytes, bytes]):
    new_counts = {}
    for key in counts.keys():
        byte_id = 0
        new_token = []
        while byte_id < len(key) - 1:
            if key[byte_id] == pair[0] and key[byte_id+1] == pair[1]:
                combined_pair = pair[0] + pair[1]
                pair_counts[pair] -= counts[key]
                if byte_id > 0: 
                    pair_counts[(new_token[-1], pair[0])] -= counts[key]
                    pair_counts[(new_token[-1], combined_pair)] = pair_counts.get((new_token[-1], combined_pair), 0) + counts[key]
                if byte_id + 2 < len(key):
                    pair_counts[(pair[1], key[byte_id+2])] -= counts[key]
                    pair_counts[(combined_pair, key[byte_id+2])] = pair_counts.get((combined_pair, key[byte_id+2]), 0) + counts[key]
                new_token.append(combined_pair)
                byte_id += 2                    
            else:
                new_token.append(key[byte_id])
                byte_id += 1
        if byte_id == len(key) - 1:
            new_token.append(key[byte_id])
        new_counts[tuple(new_token)] = counts[key]
    return new_counts, pair_counts


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

    pairs = {(b'A', b'B'): 5, (b'BA', b'A'): 5, (b'A', b'C'): 5}
    assert select_best_pair(pairs) == (b'BA', b'A')