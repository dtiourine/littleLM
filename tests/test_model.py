"""End-to-end smoke tests for the model.

These tests are intentionally small (d_model=32, vocab=64, n_layers=2) so they
run in seconds. The point is to catch integration bugs across the full stack —
forward, backward, optimizer, save/load — not to evaluate model quality.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from littlelm.network.transformer import LittleLM
from littlelm.optimizer import SGD


@pytest.fixture
def model():
    np.random.seed(0)
    return LittleLM(d_model=32, vocab_size=64, n_heads=4, n_layers=2)


@pytest.fixture
def batch():
    np.random.seed(1)
    inputs = np.random.randint(0, 64, size=(2, 16))
    targets = np.random.randint(0, 64, size=(2, 16))
    return inputs, targets


class TestForward:
    def test_inference_shape(self, model, batch):
        inputs, _ = batch
        logits, loss = model.forward(inputs)
        assert logits.data.shape == (2, 1, 64)
        assert loss is None

    def test_training_shape(self, model, batch):
        inputs, targets = batch
        logits, loss = model.forward(inputs, targets=targets)
        assert logits.data.shape == (2, 16, 64)
        assert loss is not None
        assert loss.data.shape == ()

    def test_initial_loss_near_log_vocab(self, model, batch):
        # A correctly initialized LM should output near-uniform distributions,
        # making the initial cross-entropy ~ log(vocab_size).
        inputs, targets = batch
        _, loss = model.forward(inputs, targets=targets)
        assert abs(float(loss.data) - np.log(64)) < 0.5


class TestBackward:
    def test_every_param_gets_gradient(self, model, batch):
        # If any parameter has no gradient after backward, the autograd graph
        # is broken somewhere — that param isn't connected to the loss.
        inputs, targets = batch
        for p in model.parameters():
            p.grad = np.zeros_like(p.data)
        _, loss = model.forward(inputs, targets=targets)
        loss.backward()
        for i, p in enumerate(model.parameters()):
            assert not np.all(p.grad == 0), f"Param {i} (shape {p.data.shape}) got zero gradient"

    def test_gradient_shapes_match_parameter_shapes(self, model, batch):
        inputs, targets = batch
        for p in model.parameters():
            p.grad = np.zeros_like(p.data)
        _, loss = model.forward(inputs, targets=targets)
        loss.backward()
        for p in model.parameters():
            assert p.grad.shape == p.data.shape


class TestOverfit:
    def test_loss_decreases_on_memorizable_batch(self, model, batch):
        # The strongest smoke test: a small model should be able to drive loss
        # well below the random baseline on a single fixed batch within a few
        # steps. If this fails, something in forward/backward/optimizer is wrong.
        inputs, targets = batch
        optimizer = SGD(model.parameters(), lr=0.05)

        initial_loss = None
        final_loss = None
        for step in range(30):
            optimizer.zero_grad()
            _, loss = model.forward(inputs, targets=targets)
            loss.backward()
            optimizer.step()
            if step == 0:
                initial_loss = float(loss.data)
            final_loss = float(loss.data)

        # Loss should have dropped meaningfully — well below log(vocab_size).
        assert final_loss < initial_loss * 0.9, (
            f"Loss did not decrease enough: {initial_loss:.4f} → {final_loss:.4f}"
        )


class TestSaveLoadRoundTrip:
    def test_logits_identical_after_round_trip(self, model, batch):
        inputs, _ = batch
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            model.save(tmp)
            loaded = LittleLM.from_pretrained(tmp)

            l1, _ = model.forward(inputs)
            l2, _ = loaded.forward(inputs)
            np.testing.assert_allclose(l1.data, l2.data)

    def test_config_preserved(self, model):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            model.save(tmp)
            loaded = LittleLM.from_pretrained(tmp)
            assert loaded.config() == model.config()

    def test_state_dict_preserved(self, model):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            model.save(tmp)
            loaded = LittleLM.from_pretrained(tmp)

            s1, s2 = model.state_dict(), loaded.state_dict()
            assert s1.keys() == s2.keys()
            for k in s1:
                np.testing.assert_array_equal(s1[k], s2[k])

    def test_loaded_model_still_trains(self, model, batch):
        # After loading, gradients should still flow and the optimizer should
        # work. Guards against load_state_dict accidentally breaking the
        # parameter references the optimizer holds.
        inputs, targets = batch
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            model.save(tmp)
            loaded = LittleLM.from_pretrained(tmp)

            optimizer = SGD(loaded.parameters(), lr=0.05)
            optimizer.zero_grad()
            _, loss_before = loaded.forward(inputs, targets=targets)
            loss_before.backward()
            optimizer.step()

            optimizer.zero_grad()
            _, loss_after = loaded.forward(inputs, targets=targets)
            assert float(loss_after.data) < float(loss_before.data)
