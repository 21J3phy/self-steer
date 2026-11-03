"""CPU tests for the remote worker on a tiny random Qwen3.5 with the real tokenizer.

  ../.venv-kaggle/bin/python -m unittest remote/test_worker.py   (needs torch + transformers 5.5.4)
"""
import json
import os
import tempfile
import unittest
import urllib.request
from unittest import mock

import torch

os.environ['SELFSTEER_WORK'] = tempfile.mkdtemp()
import worker as W  # noqa: E402

TOK_DIR = os.path.expanduser('~/models/Qwen3.5-4B-8bit')


def tiny():
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    W.force_torch_deltanet()
    torch.manual_seed(0)
    cfg = Qwen3_5TextConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
                            num_key_value_heads=2, head_dim=16, vocab_size=248320, linear_num_value_heads=4,
                            linear_num_key_heads=2, linear_key_head_dim=16, linear_value_head_dim=16,
                            linear_conv_kernel_dim=4, tie_word_embeddings=True, full_attention_interval=4,
                            layer_types=['linear_attention'] * 3 + ['full_attention'])
    cfg._attn_implementation = 'eager'
    m = Qwen3_5ForCausalLM(cfg).eval()
    m.requires_grad_(False)
    return m, AutoTokenizer.from_pretrained(TOK_DIR)


def scripted(tokens):
    q = list(tokens)

    class S:
        def __init__(self, *a, **k):
            pass

        def __call__(self, logits, history):
            return q.pop(0)
    return S


class WorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.tok = tiny()
        cls.e = W.Engine(cls.model, cls.tok)
        cls.lazy = W.Edit('lazy', 'add', vectors={1: torch.ones(64) * 2}, window=2, description='short answers')

    def logits(self, ids):
        out, _ = self.e.forward(ids)
        return out

    def test_columns_edit_equals_explicit_weight_edit(self):
        ids = self.e.encode('Hello there, how are you doing today?')
        u = W.unit(torch.randn(64))
        ops = [{'layer': 1, 'op': 'scale', 'neurons': [3, 7], 'factor': 0.0},
               {'layer': 2, 'op': 'project', 'neurons': [1, 40], 'direction': u.tolist(), 'alpha': 1.5}]
        edit = W.Edit('c', 'columns', columns=W.columns_from_ops(self.e, ops))
        self.e.stack.activate(edit, 0, None, tag='t')
        got = self.logits(ids)
        self.e.stack.deactivate('t')
        base = self.logits(ids)
        lin1, lin2 = self.e.layers[1].mlp.down_proj, self.e.layers[2].mlp.down_proj
        w1, w2 = lin1.weight.data.clone(), lin2.weight.data.clone()
        try:
            lin1.weight.data[:, [3, 7]] = 0
            s = [1, 40]
            lin2.weight.data[:, s] = w2[:, s] - 1.5 * torch.outer(u, u @ w2[:, s])
            want = self.logits(ids)
        finally:
            lin1.weight.data, lin2.weight.data = w1, w2
        self.assertTrue(torch.allclose(got, want, atol=1e-4))
        self.assertFalse(torch.allclose(got, base, atol=1e-4))

    def test_bundle_equals_both_edits_live(self):
        ids = self.e.encode('Hello there, how are you doing today?')
        wk = W.Worker(self.e, budget=1)
        wk.lib['lazy'] = self.lazy
        wk.t_save_edit('fix', 'scale two neurons', [{'layer': 1, 'op': 'scale', 'neurons': [3, 7], 'factor': 0.5},
                                                    {'layer': 2, 'op': 'scale', 'neurons': [5], 'factor': 2.0}])
        s = wk.t_bundle_mode('lazy-fixed', 'lazy', 'fix')
        self.assertEqual(s['layers'], [1, 2])
        self.assertEqual(wk.lib['lazy-fixed'].window, 2)
        self.e.stack.activate(wk.lib['lazy-fixed'], 3, 5, tag='t')
        got = self.logits(ids)
        self.e.stack.deactivate('t')
        self.e.stack.activate(self.lazy, 3, 5, tag='t')
        self.e.stack.activate(wk.lib['fix'], 3, 5, tag='t')
        want = self.logits(ids)
        self.e.stack.deactivate('t')
        self.assertTrue(torch.allclose(got, want, atol=1e-5))
        self.assertTrue(torch.allclose(got[:3], self.logits(ids)[:3], atol=1e-5))   # before the window: untouched
        wk.t_add_mode('lazy2', 'lazy-fixed')
        self.e.stack.activate(wk.lib['lazy2'], 3, 5, tag='t')
        self.assertTrue(torch.allclose(self.logits(ids), want, atol=1e-5))
        self.e.stack.deactivate('t')

    def test_incremental_decode_matches_full(self):
        ids = self.e.encode('one two three four five six seven')
        r = W.Runner(self.e)
        r.feed(ids[:4])
        steps = [r.feed([t]) for t in ids[4:]]
        full = self.logits(ids)
        for i, s in enumerate(steps):
            self.assertTrue(torch.allclose(s, full[4 + i], atol=1e-4))

    def test_selfsteer_tool_loop_and_placebo(self):
        call = ('<tool_call>\n<function=set_mode>\n<parameter=name>\nlazy\n</parameter>\n'
                '</function>\n</tool_call>')
        script = self.e.encode(call) + self.e.encode('ok fine') + [self.e.im_end]
        lib = {'lazy': self.lazy}
        with mock.patch.object(W, 'Sampler', scripted(script)):
            r = W.answer(self.e, lib, 'hi', seed=1, tools_for=lib)
        self.assertEqual(r['tool_calls'][0]['call']['name'], 'set_mode')
        self.assertEqual(r['segments'][1]['modes'], ['lazy'])
        self.assertEqual(r['segments'][1]['steered'], 2)
        self.assertEqual(r['reply'], 'ok fine')
        with mock.patch.object(W, 'Sampler', scripted(script)):
            p = W.answer(self.e, lib, 'hi', seed=1, tools_for=lib, placebo=True)
        self.assertEqual(p['segments'][1]['modes'], [])
        self.assertEqual(p['tool_calls'][0]['result'], r['tool_calls'][0]['result'])

    def test_read_activations_and_kernel(self):
        wk = W.Worker(self.e, budget=3)
        out = wk.t_read_activations('t', [{'prompt': 'hi', 'answer': 'hello there'}, {'prompt': 'a', 'answer': 'b c'}])
        self.assertEqual(out['shapes']['neurons'], [2, 4, 128])
        self.assertEqual(out['passes_left'], 2)
        k = wk.t_py("d = load('t'); print(d['neurons'].shape, W(1).shape)")
        self.assertIn('(2, 4, 128) torch.Size([64, 128])', k['output'])
        self.assertNotIn('model', wk.ns)
        top = wk.t_top_writers(1, direction=wk.e.W(1)[:, 9].tolist(), k=1)
        self.assertEqual(top[0]['neuron'], 9)

    def test_tool_context_read_with_edit(self):
        wk = W.Worker(self.e, budget=3)
        wk.lib['lazy'] = self.lazy
        a = wk.t_read_activations('bare', [{'prompt': 'what is 2+2', 'answer': 'It is 4.'}], first=3)
        b = wk.t_read_activations('tool', [{'prompt': 'what is 2+2', 'answer': 'It is 4.'}], context='tool',
                                  mode='lazy', first=3)
        import numpy as np
        x = np.load(W.WORK / 'bare.npz')['resid']
        y = np.load(W.WORK / 'tool.npz')['resid']
        self.assertEqual(x.shape, y.shape)
        self.assertFalse(np.allclose(x, y))

    def test_dla_matches_logit_change_when_neuron_silenced(self):
        # Linearized attribution should track the real logit drop from silencing a neuron in the last layer.
        ids = self.e.encode('one two three four')
        prompt, ans = 'count', ' one two three four'
        target = self.e.encode(' four')[0]
        out = W.read_activations(self.e, [{'prompt': prompt, 'answer': ans}], attribute_to=[target])
        self.assertEqual(int(out['dla_positions'][0]), 1)
        l = self.e.n_layers - 1
        n = int(abs(out['dla'][0, l]).argmax())
        msgs = [{'role': 'user', 'content': prompt}]
        toks = self.e.chat_tokens(msgs) + self.e.encode(ans)
        pos = len(toks) - 2
        base, _ = self.e.forward(toks)
        edit = W.Edit('s', 'columns', columns=W.columns_from_ops(self.e, [{'layer': l, 'neurons': [n], 'op': 'scale', 'factor': 0.0}]))
        self.e.stack.activate(edit, 0, None, tag='t')
        cut, _ = self.e.forward(toks)
        self.e.stack.deactivate('t')
        real = float(base[pos, target] - cut[pos, target])
        self.assertAlmostEqual(real, float(out['dla'][0, l, n]), delta=0.05 * abs(real) + 1e-3)

    def test_http_jobs_with_auth(self):
        wk = W.Worker(self.e, budget=2)
        srv = W.serve(wk, 'secret', port=8799)
        try:
            req = urllib.request.Request('http://127.0.0.1:8799/call', method='POST',
                                         data=json.dumps({'tool': 'budget', 'args': {}}).encode(),
                                         headers={'Authorization': 'Bearer secret'})
            jid = json.loads(urllib.request.urlopen(req).read())['job']
            import time
            for _ in range(50):
                job = json.loads(urllib.request.urlopen(urllib.request.Request(
                    f'http://127.0.0.1:8799/job/{jid}', headers={'Authorization': 'Bearer secret'})).read())
                if job['status'] == 'done':
                    break
                time.sleep(0.05)
            self.assertEqual(job['result']['budget'], 2)
            bad = urllib.request.Request('http://127.0.0.1:8799/health')
            with self.assertRaises(urllib.error.HTTPError):
                urllib.request.urlopen(bad)
        finally:
            srv.shutdown()


if __name__ == '__main__':
    unittest.main()
