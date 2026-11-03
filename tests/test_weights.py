import unittest

import mlx.core as mx

mx.set_default_device(mx.cpu)

from selfsteer.edits import Edit, columns_from_ops, unit  # noqa: E402
from selfsteer.model import Engine, text_layers  # noqa: E402
from selfsteer import weights  # noqa: E402
from tests.fakes import FakeTokenizer, tiny_model  # noqa: E402

SEQ = [5, 17, 99, 3, 250, 41, 7, 8, 9, 10]


def logits(engine, tokens):
    engine.stack.set_span(0, len(tokens))
    return engine.model(mx.array(tokens)[None])


class ColumnEditTests(unittest.TestCase):
    def test_scale_and_project_equal_explicit_weight_edits(self):
        engine = Engine.wrap(tiny_model(seed=4, quantize=False), FakeTokenizer())
        u = unit(mx.random.normal((64,)))
        ops = [{'layer': 1, 'op': 'scale', 'neurons': [3, 7, 9], 'factor': 0.0},
               {'layer': 2, 'op': 'project', 'neurons': [1, 2, 40], 'direction': u.tolist(), 'alpha': 1.5}]
        cols = columns_from_ops(ops, engine.projections, 4, 128)
        engine.stack.activate(Edit('c', 'columns', {}, columns=cols), 0, None)
        got = logits(engine, SEQ)
        engine.stack.clear()
        ref = tiny_model(seed=4, quantize=False)
        l1 = text_layers(ref)[1].mlp.down_proj
        w = l1.weight
        keep = mx.ones((128,)).at[mx.array([3, 7, 9])].add(-1.0)
        l1.weight = w * keep[None, :]
        l2 = text_layers(ref)[2].mlp.down_proj
        w2 = l2.weight
        s = mx.array([1, 2, 40])
        sub = w2[:, s] - 1.5 * mx.outer(u, u @ w2[:, s])
        l2.weight = w2.at[:, s].add(sub - w2[:, s])
        expect = ref(mx.array(SEQ)[None])
        self.assertTrue(bool(mx.allclose(got, expect, atol=1e-4).item()))

    def test_capture_reads_neuron_activations_without_edits(self):
        engine = Engine.wrap(tiny_model(seed=5), FakeTokenizer())
        out = weights.read_activations(engine, [{'prompt': 'hi', 'answer': 'hello there'},
                                                {'prompt': 'yo', 'answer': 'sup'}])
        self.assertEqual(out['neurons'].shape, (2, 4, 128))
        self.assertEqual(out['resid'].shape, (2, 4, 64))
        self.assertEqual(list(out['answer_tokens']), [11, 3])
        self.assertIsNone(engine.stack.capture)

    def test_top_writers_ranks_by_projection(self):
        engine = Engine.wrap(tiny_model(seed=6, quantize=False), FakeTokenizer())
        w = engine.projections[0].weight
        u = unit(w[:, 17])
        top = weights.top_writers(engine, 0, u.tolist(), k=3)
        self.assertEqual(top[0]['neuron'], 17)


if __name__ == '__main__':
    unittest.main()
