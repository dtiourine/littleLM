import re


class Tokenizer:
    def __init__(self, vocab_size: int):
        self.vocab = {}
        self.vocab_size = vocab_size

    def _pretokenize(self, text: str) -> list[str]:
        return re.findall(r" ?\S+|\s+", text)

    def train(self, text: str):
        chunks = self._pretokenize(text)
        chunks = [list(c.encode("utf-8")) for c in chunks]

        while len(self.vocab) < self.vocab_size - 256:
            pair_counts = self._count_pairs(chunks)
            best = max(pair_counts, key=pair_counts.get)
            new_id = 256 + len(self.vocab)
            self.vocab[new_id] = best
            chunks = [self._merge(c, *best, new_id) for c in chunks]

    def _count_pairs(self, chunks: list[list[int]]) -> dict[tuple[int, int], int]:
        count = {}
        for chunk in chunks:
            for pair in zip(chunk[:-1], chunk[1:]):
                count[pair] = count.get(pair, 0) + 1
        return count

    def _merge(self, chunk: list[int], left: int, right: int, new_id: int):
        result = []
        i = 0
        while i < len(chunk):
            if i + 1 < len(chunk) and chunk[i] == left and chunk[i + 1] == right:
                result.append(new_id)
                i += 2
            else:
                result.append(chunk[i])
                i += 1
        return result

    def encode(self, text: str):
        chunks = self._pretokenize(text)
        chunks = [list(c.encode("utf-8")) for c in chunks]

        result = []
        for chunk in chunks:
            for new_id in sorted(self.vocab.keys()):
                pair = self.vocab[new_id]
                chunk = self._merge(chunk, pair[0], pair[1], new_id)
            result.extend(chunk)
        return result

    def decode(self, ids: list[int]) -> str:
        bytes_out = []
        for token_id in ids:
            bytes_out.extend(self._decode_token(token_id))
        return bytes(bytes_out).decode("utf-8")

    def _decode_token(self, token_id: int) -> list[int]:
        if token_id < 256:
            return [token_id]
        left, right = self.vocab[token_id]
        return self._decode_token(left) + self._decode_token(right)
