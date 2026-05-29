import pytest

from littlelm.tokenizer import Tokenizer


SAMPLE_TEXT = (
    "the quick brown fox jumps over the lazy dog. "
    "the quick brown fox is quick. "
    "the lazy dog is lazy. "
    "foxes are quick and dogs are lazy. "
    "this is a sample text for tokenization tests. "
    "the rain in spain falls mainly on the plain."
)


@pytest.fixture
def tokenizer():
    """A tokenizer trained on SAMPLE_TEXT with a small vocab for fast tests."""
    tok = Tokenizer(vocab_size=300)
    tok.train(SAMPLE_TEXT)
    return tok


class TestRoundTrip:
    def test_recovers_training_text(self, tokenizer):
        encoded = tokenizer.encode(SAMPLE_TEXT)
        assert tokenizer.decode(encoded) == SAMPLE_TEXT

    def test_recovers_unseen_text(self, tokenizer):
        text = "the cat sat on the mat and watched the fox."
        encoded = tokenizer.encode(text)
        assert tokenizer.decode(encoded) == text

    def test_recovers_unicode(self, tokenizer):
        # BPE works on UTF-8 bytes, so multi-byte chars should round-trip
        # even though the tokenizer was trained only on ASCII.
        text = "café héllo wörld 🚀 こんにちは"
        encoded = tokenizer.encode(text)
        assert tokenizer.decode(encoded) == text

    def test_recovers_single_character(self, tokenizer):
        assert tokenizer.decode(tokenizer.encode("a")) == "a"

    def test_empty_string(self, tokenizer):
        assert tokenizer.encode("") == []
        assert tokenizer.decode([]) == ""


class TestVocab:
    def test_vocab_size_after_training(self):
        vocab_size = 300
        tok = Tokenizer(vocab_size=vocab_size)
        tok.train(SAMPLE_TEXT)
        # 256 base byte tokens + learned merges = vocab_size
        assert len(tok.vocab) == vocab_size - 256

    def test_token_ids_in_valid_range(self, tokenizer):
        encoded = tokenizer.encode(SAMPLE_TEXT)
        assert all(0 <= t < tokenizer.vocab_size for t in encoded)

    def test_training_is_deterministic(self):
        a = Tokenizer(vocab_size=300)
        a.train(SAMPLE_TEXT)
        b = Tokenizer(vocab_size=300)
        b.train(SAMPLE_TEXT)
        assert a.vocab == b.vocab


class TestSaveLoad:
    def test_load_recovers_vocab(self, tokenizer, tmp_path):
        path = tmp_path / "tokenizer.json"
        tokenizer.save(path)

        loaded = Tokenizer.from_file(path)

        assert loaded.vocab_size == tokenizer.vocab_size
        assert loaded.vocab == tokenizer.vocab

    def test_load_preserves_encoding(self, tokenizer, tmp_path):
        path = tmp_path / "tokenizer.json"
        tokenizer.save(path)
        loaded = Tokenizer.from_file(path)

        # The loaded tokenizer must encode identically to the original.
        assert loaded.encode(SAMPLE_TEXT) == tokenizer.encode(SAMPLE_TEXT)

    def test_load_preserves_decoding(self, tokenizer, tmp_path):
        path = tmp_path / "tokenizer.json"
        tokenizer.save(path)
        loaded = Tokenizer.from_file(path)

        ids = tokenizer.encode(SAMPLE_TEXT)
        assert loaded.decode(ids) == tokenizer.decode(ids)

    def test_full_roundtrip_through_disk(self, tokenizer, tmp_path):
        # Sanity check: train -> save -> load -> encode -> decode == original.
        path = tmp_path / "tokenizer.json"
        tokenizer.save(path)
        loaded = Tokenizer.from_file(path)

        text = "a brand new sentence the loaded tokenizer has never seen."
        assert loaded.decode(loaded.encode(text)) == text
