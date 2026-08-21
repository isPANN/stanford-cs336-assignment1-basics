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
    pair_counts, pair_dict = count_pairs(pre_token_counts)
    vocab = init_vocab(special_tokens)
    merges = []
    while len(vocab) < vocab_size:
        token_a, token_b = select_best_pair(pair_counts)
        pre_token_counts, pair_counts, pair_dict = apply_merge(pre_token_counts, pair_counts, pair_dict, (token_a, token_b))
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
    pair_dict = {}
    for seq in counts.keys():
        tmp_pair_counts = {}
        for pair in zip(seq, seq[1:]):
            tmp_pair_counts[pair] = tmp_pair_counts.get(pair, 0) + 1
        for pair in tmp_pair_counts.keys():
            pair_counts[pair] = pair_counts.get(pair, 0) + tmp_pair_counts[pair] * counts[seq]
            pair_dict.setdefault(pair, set()).add(seq) 
    return pair_counts, pair_dict

def select_best_pair(pair_counts: dict[tuple[bytes, bytes], int]) -> tuple[bytes, bytes]:
    best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
    return best_pair

def apply_merge(counts: dict[tuple[bytes, ...], int], pair_counts: dict[tuple[bytes, bytes], int], pair_dict: dict[tuple[bytes, bytes], set], pair: tuple[bytes, bytes]):
    for seq in list(pair_dict.get(pair, ())):
        token_id = 0
        new_seq = []
        while token_id < len(seq) - 1:
            if seq[token_id] == pair[0] and seq[token_id + 1] == pair[1]:
                combined_pair = pair[0] + pair[1]
                new_seq.append(combined_pair)
                token_id += 2
            else:
                new_seq.append(seq[token_id])
                token_id += 1
        if token_id == len(seq) - 1:
            new_seq.append(seq[token_id])
        new_seq_t = tuple(new_seq)
        freq = counts[seq]
        if new_seq_t != seq:
            counts[new_seq_t] = counts.get(new_seq_t, 0) + freq
            del counts[seq]

        old_pair_freq = {}
        for old_pair in zip(seq, seq[1:]):
            old_pair_freq[old_pair] = old_pair_freq.get(old_pair, 0) + 1
        new_pair_freq = {}
        for new_pair in zip(new_seq_t, new_seq_t[1:]):
            new_pair_freq[new_pair] = new_pair_freq.get(new_pair, 0) + 1

        for old_pair in old_pair_freq:
            if old_pair in pair_dict:
                pair_dict[old_pair].discard(seq)
        for new_pair in new_pair_freq:
            pair_dict.setdefault(new_pair, set()).add(new_seq_t)

        for changed_pair in old_pair_freq.keys() | new_pair_freq.keys():
            delta = new_pair_freq.get(changed_pair, 0) - old_pair_freq.get(changed_pair, 0)
            pair_counts[changed_pair] = pair_counts.get(changed_pair, 0) + delta * freq
            if pair_counts[changed_pair] == 0:
                del pair_counts[changed_pair]
                pair_dict.pop(changed_pair, None)
    return counts, pair_counts, pair_dict


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