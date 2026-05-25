class Tokenizer:
    def __init__(self, vocab_size: int):
        self.vocab = {}
        self.available_key = 256
        self.vocab_size = vocab_size

    def train(self, text: str | list[int]):
        if isinstance(text, str):
            text = [ord(c) for c in list(text)]

        while len(self.vocab) < self.vocab_size - 256:
            most_freq_pair = self._get_most_freq_pair(text)
            pair = (most_freq_pair[0], most_freq_pair[1])

            self.vocab[self.available_key] = pair
            text = self._merge(text, pair[0], pair[1], self.available_key)
            self.available_key += 1

    def _get_most_freq_pair(self, text: list[int]):
        k = 2
        state = []
        count = {}

        for r in range(len(text)):
            state.append(text[r])

            if r >= k:
                state = state[1:]

            if r >= k - 1:
                pair = (state[0], state[1])
                count[pair] = count.get(pair, 0) + 1

        return max(count, key=lambda pair: count[pair])

    def _merge(self, text: list[int], left_c: int, right_c: int, new_c: int):
        k = 2
        state = []

        r = 0
        while r < len(text):
            state.append(text[r])

            if r >= k:
                state = state[1:]

            if r >= k - 1:
                if state[0] == left_c and state[1] == right_c:
                    text[k - 1] = new_c
                    del text[r]
                else:
                    r += 1
            else:
                r += 1

        return text

    def encode(self, text: str | list[int]):
        if isinstance(text, str):
            text = [ord(c) for c in list(text)]

        for i in self.vocab:
            pair = self.vocab[i]
            text = self._merge(text, pair[0], pair[1], i)
        return text

    def decode(self, text: list[int]):
        codes = list(self.vocab.keys())
        codes.reverse()
        print(codes)

        for c in codes:
            pair = self.vocab[c]

            i = 0
            while i < len(text):
                if text[i] == c:
                    text[i] = pair[0]
                    text.insert(i + 1, pair[1])
                else:
                    i += 1

        return "".join([chr(c) for c in text])


if __name__ == "__main__":
    # text = """
    # the cat chased the caterpillar across the cathedral courtyard.
    # cats chase caterpillars, and caterpillars crawl carefully.
    # the cathedral cat was curious, calm, and very, very fast.
    # """
    # tokenizer = Tokenizer(vocab_size=10)
    # # text_ints = [ord(c) for c in text]
    # tokenizer.train(text)
    # encoded = tokenizer.encode(text)
    # print(encoded)
    # decoded = tokenizer.decode(encoded)
    # print(decoded)

    t = Tokenizer(vocab_size=260)
    t.train("aaabdaaabac")
    print(t.vocab)
    print(t.encode("aaabdaaabac"))
    print(t.decode(t.encode("aaabdaaabac")))
