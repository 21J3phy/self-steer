import threading
import unittest
from unittest import mock

import mlx.core as mx

mx.set_default_device(mx.cpu)

from selfsteer import session as session_mod  # noqa: E402
from selfsteer.controller import RequestBoard  # noqa: E402
from selfsteer.decode import Runner  # noqa: E402
from selfsteer.edits import Edit  # noqa: E402
from selfsteer.library import Library  # noqa: E402
from selfsteer.model import Engine  # noqa: E402
from selfsteer.probe import build_mode  # noqa: E402
from tests.fakes import FakeTokenizer, scripted_sampler, tiny_model  # noqa: E402

TOOL = ('<tool_call>\n<function=set_mode>\n<parameter=name>\nlazy\n</parameter>\n'
        '</function>\n</tool_call>')


def make(policy='keep', board=None, pump=None):
    tok = FakeTokenizer()
    engine = Engine.wrap(tiny_model(seed=3), tok)
    lib = Library(engine)
    lib.resid_median = {i: 2.0 for i in range(4)}
    lib.add(Edit('lazy', 'add', {1: mx.ones((64,)) * 3}, window=2, description='short answers'))
    s = session_mod.Session(engine, lib, cache_policy=policy, board=board, pump=pump,
                            allow_requests=board is not None)
    return s, tok


class SessionTests(unittest.TestCase):
    def run_turn(self, s, tok, script, user='hi'):
        with mock.patch.object(session_mod, 'make_sampler', scripted_sampler(tok.encode(script))):
            return s.send(user, seed=1)

    def test_tool_call_switches_mode_for_the_answer(self):
        s, tok = make()
        r = self.run_turn(s, tok, TOOL + 'ok<|im_end|>')
        self.assertEqual(r.tool_calls[0]['call']['name'], 'set_mode')
        self.assertIn("Mode 'lazy' is on", r.tool_calls[0]['result'])
        self.assertEqual(r.segments[0]['modes'], [])
        self.assertEqual(r.segments[1]['modes'], ['lazy'])
        self.assertEqual(r.segments[1]['steered_positions'], 2)  # window 2, answer is 3 tokens
        self.assertEqual(r.reply, 'ok')
        self.assertEqual(s.runner.length, len(s.ledger))
        self.assertEqual(s.armed, [])  # answer scope expired
        self.assertTrue(s.dirty)
        # The tool response sits in the ledger, in Qwen's format.
        text = tok.decode(s.ledger)
        self.assertIn("<tool_response>\nMode 'lazy' is on", text)

    def test_second_turn_appends_without_rerendering(self):
        s, tok = make()
        self.run_turn(s, tok, 'a<|im_end|>')
        before = list(s.ledger)
        self.run_turn(s, tok, 'b<|im_end|>', user='again')
        self.assertEqual(s.ledger[:len(before)], before)
        self.assertIn('<|im_start|>user\nagain<|im_end|>', tok.decode(s.ledger))

    def test_reprefill_recomputes_cache_without_edits(self):
        s, tok = make('reprefill')
        self.run_turn(s, tok, TOOL + 'ok<|im_end|>')
        s.reprefill()
        fresh = Runner(s.engine)
        fresh.prefill(s.ledger)
        a = s.runner.step([65])
        b = fresh.step([65])
        self.assertTrue(bool(mx.array_equal(a, b).item()))
        self.assertFalse(s.dirty)

    def test_keep_policy_leaves_steered_states(self):
        s, tok = make('keep')
        self.run_turn(s, tok, TOOL + 'ok<|im_end|>')
        fresh = Runner(s.engine)
        fresh.prefill(s.ledger)
        a = s.runner.step([65])
        b = fresh.step([65])
        self.assertFalse(bool(mx.allclose(a, b).item()))

    def test_unknown_mode_is_reported_to_qwen(self):
        s, tok = make()
        bad = TOOL.replace('lazy', 'pirate')
        r = self.run_turn(s, tok, bad + 'ok<|im_end|>')
        self.assertIn('Unknown mode', r.tool_calls[0]['result'])
        self.assertEqual(r.segments[1]['modes'], [])

    def test_conversation_scope_persists(self):
        s, tok = make()
        script = TOOL.replace('</parameter>\n</function>',
                              '</parameter>\n<parameter=scope>\nconversation\n</parameter>\n</function>')
        self.run_turn(s, tok, script + 'ok<|im_end|>')
        self.assertEqual([a.name for a in s.armed], ['lazy'])
        r = self.run_turn(s, tok, 'x<|im_end|>', user='next')
        self.assertEqual(r.segments[0]['modes'], ['lazy'])

    def test_request_mode_waits_for_controller(self):
        board = RequestBoard()
        holder = {}

        def pump(timeout):
            req = board.claim_next(timeout)
            if req is not None:
                s = holder['s']
                build_mode(s.engine, s.library, name='eli5', description='simple words',
                           examples=[{'prompt': 'a', 'positive': 'easy', 'negative': 'hard'},
                                     {'prompt': 'b', 'positive': 'fun', 'negative': 'dry'}],
                           layers=[1, 2], strength=0.5)
                board.fulfill(req.id, 'eli5')
        s, tok = make(board=board, pump=pump)
        holder['s'] = s
        call = ('<tool_call>\n<function=request_mode>\n<parameter=description>\nexplain simply\n'
                '</parameter>\n</function>\n</tool_call>')
        r = self.run_turn(s, tok, call + 'ok<|im_end|>')
        self.assertIn("Claude built mode 'eli5'", r.tool_calls[0]['result'])
        self.assertIn('eli5', s.library.names())
        edit = s.library.get('eli5')
        norm = float(mx.linalg.norm(edit.vectors[1]).item())
        self.assertAlmostEqual(norm, 0.5 * 2.0, places=4)


class BoardTests(unittest.TestCase):
    def test_claim_and_fulfill(self):
        board = RequestBoard()
        req = board.post('formal tone', 's1')
        got = board.claim_next(0.1)
        self.assertEqual(got.id, req.id)
        self.assertIsNone(board.claim_next(0.05))
        threading.Timer(0.05, lambda: board.fulfill(req.id, 'formal')).start()
        board.wait(req, 2.0)
        self.assertEqual(req.status, 'done')


if __name__ == '__main__':
    unittest.main()
