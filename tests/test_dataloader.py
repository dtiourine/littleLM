import numpy as np
import pytest

from littlelm.dataloader import DataLoader


@pytest.fixture
def data():
    # A deterministic 1D corpus where each token equals its position.
    # Makes the "chunks are contiguous" property easy to assert.
    return np.arange(1000)


class TestShapes:
    def test_inputs_and_targets_are_batch_by_context(self, data):
        loader = DataLoader(data, batch_size=8, context_len=32, seed=42)
        inputs, targets = next(iter(loader))
        assert inputs.shape == (8, 32)
        assert targets.shape == (8, 32)

    def test_various_batch_and_context_sizes(self, data):
        for B, T in [(1, 1), (4, 16), (16, 64), (32, 128)]:
            loader = DataLoader(data, batch_size=B, context_len=T, seed=42)
            inputs, targets = next(iter(loader))
            assert inputs.shape == (B, T)
            assert targets.shape == (B, T)


class TestShiftInvariant:
    def test_targets_are_inputs_shifted_by_one(self, data):
        loader = DataLoader(data, batch_size=4, context_len=16, seed=42)
        inputs, targets = next(iter(loader))
        # Position j of targets predicts what input has at position j+1.
        np.testing.assert_array_equal(inputs[:, 1:], targets[:, :-1])

    def test_chunks_are_contiguous_in_data(self, data):
        # Since data = arange, consecutive tokens differ by 1.
        # Each row of inputs and targets should be a contiguous slice.
        loader = DataLoader(data, batch_size=4, context_len=16, seed=42)
        inputs, targets = next(iter(loader))
        assert np.all(np.diff(inputs, axis=1) == 1)
        assert np.all(np.diff(targets, axis=1) == 1)
        # Next-token relationship: target = input + 1
        np.testing.assert_array_equal(inputs + 1, targets)


class TestRange:
    def test_no_out_of_bounds(self, data):
        loader = DataLoader(data, batch_size=8, context_len=32, seed=42)
        inputs, targets = next(iter(loader))
        assert inputs.min() >= 0
        assert inputs.max() < len(data)
        assert targets.min() >= 0
        assert targets.max() < len(data)

    def test_can_sample_last_valid_position(self):
        # If data has length N and context_len is T, the largest valid start
        # is N - T - 1. Verify we can reach it (off-by-one canary).
        N, T = 100, 8
        small_data = np.arange(N)
        # Run many batches to ensure we explore the sampling range.
        loader = DataLoader(small_data, batch_size=64, context_len=T, seed=42)
        it = iter(loader)
        max_target_seen = -1
        for _ in range(50):
            _, targets = next(it)
            max_target_seen = max(max_target_seen, targets.max())
        # The last data value is N - 1; we should be able to see it as a target.
        assert max_target_seen == N - 1


class TestDeterminism:
    def test_same_seed_same_batches(self, data):
        a = DataLoader(data, batch_size=8, context_len=32, seed=42)
        b = DataLoader(data, batch_size=8, context_len=32, seed=42)
        a_it, b_it = iter(a), iter(b)
        for _ in range(5):
            ai, at = next(a_it)
            bi, bt = next(b_it)
            np.testing.assert_array_equal(ai, bi)
            np.testing.assert_array_equal(at, bt)

    def test_different_seeds_produce_different_batches(self, data):
        a = DataLoader(data, batch_size=8, context_len=32, seed=42)
        b = DataLoader(data, batch_size=8, context_len=32, seed=99)
        ai, _ = next(iter(a))
        bi, _ = next(iter(b))
        assert not np.array_equal(ai, bi)


class TestInfiniteIteration:
    def test_yields_many_batches_without_error(self, data):
        loader = DataLoader(data, batch_size=4, context_len=8, seed=42)
        it = iter(loader)
        for _ in range(100):
            inputs, targets = next(it)
            assert inputs.shape == (4, 8)
            assert targets.shape == (4, 8)

    def test_consecutive_batches_differ(self, data):
        # Random sampling means consecutive batches should almost never match.
        loader = DataLoader(data, batch_size=4, context_len=8, seed=42)
        it = iter(loader)
        first, _ = next(it)
        second, _ = next(it)
        assert not np.array_equal(first, second)
