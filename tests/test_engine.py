import copy
import unittest

import mlx.core as mx
import mlx.nn as nn

mx.set_default_device(mx.cpu)

from selfsteer.decode import Runner  # noqa: E402
from selfsteer.edits import Edit, edit_from_preset, down_column, unit  # noqa: E402
from selfsteer.model import Engine, text_layers  # noqa: E402
from tests.fakes import FakeTokenizer, tiny_model  # noqa: E402

SEQ = [5, 17, 99, 3, 250, 41, 7, 8, 9, 10]


def full_logits(engine, tokens):
    engine.stack.set_span(0, len(tokens))
    return engine.model(mx.array(tokens)[None])


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.plain = tiny_model(seed=1)
        self.engine = Engine.wrap(tiny_model(seed=1), FakeTokenizer())

    def test_wrapped_without_edits_is_bit_identical(self):
        a = self.plain(mx.array(SEQ)[None])
        b = full_logits(self.engine, SEQ)
        self.assertTrue(bool(mx.array_equal(a, b).item()))

    def test_neuron_vectors_match_hand_computation(self):
        preset = {'name': 'tiny', 'neurons': [[1, 5, 0.5], [1, 9, -0.25], [2, 7, 0.3]],
                  'resid_median': {str(i): 2.0 + i for i in range(4)}, 'window': 3}
        edit = edit_from_preset(preset, self.engine.projections, num_layers=4, num_neurons=128)
        p = self.engine.projections[1]
        w = mx.dequantize(p.weight, p.scales, p.biases, group_size=p.group_size, bits=p.bits)
        expect = (0.5 * 3.0 * w[:, 5] / mx.linalg.norm(w[:, 5])
                  - 0.25 * 3.0 * w[:, 9] / mx.linalg.norm(w[:, 9]))
        self.assertTrue(bool(mx.allclose(edit.vectors[1], expect, atol=1e-5).item()))
        self.assertEqual(edit.layers, [1, 2])
        self.assertEqual(self.engine.num_neurons, 128)

    def test_window_only_changes_positions_inside_it(self):
        base = full_logits(self.engine, SEQ)
        v = mx.random.normal((64,)) * 5
        edit = Edit('w', 'add', {1: v}, window=3)
        self.engine.stack.activate(edit, 4, 7)
        steered = full_logits(self.engine, SEQ)
        self.engine.stack.clear()
        same = mx.all(steered[0, :4] == base[0, :4]).item()
        self.assertTrue(bool(same))
        self.assertFalse(bool(mx.allclose(steered[0, 4:7], base[0, 4:7]).item()))

    def test_add_matches_reference_formula(self):
        # Reference: Lazy Genius WindowedResponseScope arithmetic on the raw projection.
        v = mx.random.normal((64,))
        layer = text_layers(self.plain)[2]
        proj = layer.mlp.down_proj

        class Ref(nn.Module):
            def __init__(self, p):
                super().__init__()
                self.p = p

            def __call__(self, x):
                out = self.p(x)
                mask = mx.arange(0, out.shape[1]) >= 2
                return (out.astype(mx.float32) + v[None, None, :] * mask[None, :, None]).astype(out.dtype)
        layer.mlp.down_proj = Ref(proj)
        ref = self.plain(mx.array(SEQ)[None])
        layer.mlp.down_proj = proj
        self.engine.stack.activate(Edit('r', 'add', {2: v}), 2, None)
        ours = full_logits(self.engine, SEQ)
        self.engine.stack.clear()
        self.assertTrue(bool(mx.array_equal(ref, ours).item()))

    def test_ablate_equals_rank1_weight_edit(self):
        plain = tiny_model(seed=2, quantize=False)
        edited = tiny_model(seed=2, quantize=False)
        engine = Engine.wrap(tiny_model(seed=2, quantize=False), FakeTokenizer())
        u = unit(mx.random.normal((64,)))
        lin = text_layers(edited)[1].mlp.down_proj
        lin.weight = lin.weight - mx.outer(u, u) @ lin.weight
        expect = edited(mx.array(SEQ)[None])
        engine.stack.activate(Edit('a', 'ablate', {1: u}), 0, None)
        got = full_logits(engine, SEQ)
        self.assertTrue(bool(mx.allclose(expect, got, atol=1e-4).item()))
        self.assertFalse(bool(mx.allclose(plain(mx.array(SEQ)[None]), got, atol=1e-4).item()))

    def test_incremental_decode_matches_full_prefill(self):
        runner = Runner(self.engine)
        runner.prefill(SEQ[:6], step=4)
        steps = [runner.step([t]) for t in SEQ[6:]]
        full = full_logits(self.engine, SEQ)
        for i, logits in enumerate(steps):
            self.assertTrue(bool(mx.allclose(logits[0], full[0, 6 + i], atol=1e-4).item()))
        self.assertEqual(runner.length, len(SEQ))

    def test_snapshot_restore_gives_identical_continuation(self):
        runner = Runner(self.engine)
        runner.prefill(SEQ[:5])
        snap = runner.snapshot()
        a = runner.step([SEQ[5]])
        runner.step([77])
        runner.restore(snap)
        b = runner.step([SEQ[5]])
        self.assertTrue(bool(mx.array_equal(a, b).item()))

    def test_switch_midstream_keeps_absolute_positions(self):
        runner = Runner(self.engine)
        runner.prefill(SEQ[:6])
        v = mx.random.normal((64,))
        active = self.engine.stack.activate(Edit('m', 'add', {0: v}, window=2), 7, 9)
        runner.step([1])  # position 6: outside
        runner.step([2])  # position 7: inside
        runner.step([3])  # position 8: inside
        runner.step([4])  # position 9: outside
        self.assertEqual(active.steered_positions, 2)
        self.assertEqual(self.engine.stack.names_at(8), ['m'])
        self.assertEqual(self.engine.stack.names_at(9), [])
        self.engine.stack.clear()

    def test_gain_scales_add_vectors(self):
        v = mx.ones((64,))
        edit = Edit('g', 'add', {0: v})
        self.assertAlmostEqual(float(mx.sum(edit.scaled(-2.0).vectors[0]).item()), -128.0)
        self.assertIs(Edit('a', 'ablate', {0: v}).scaled(3.0).kind, 'ablate')


if __name__ == '__main__':
    unittest.main()
