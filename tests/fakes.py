"""Tiny random Qwen3.5 and a byte-level fake tokenizer. Nothing here loads the 4B."""
from __future__ import annotations

import json

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import qwen3_5

SPECIALS = ['<|im_start|>', '<|im_end|>', '<tool_call>', '</tool_call>', '<think>', '</think>',
            '<tool_response>', '</tool_response>']
SPECIAL_IDS = {s: 256 + i for i, s in enumerate(SPECIALS)}
SKIPPED = {'<|im_start|>', '<|im_end|>'}  # like Qwen: tool/think tokens are not "special"


def tiny_model(seed=0, quantize=True, layers=4):
    mx.random.seed(seed)
    tc = dict(model_type='qwen3_5_text', hidden_size=64, intermediate_size=128, num_hidden_layers=layers,
              num_attention_heads=4, num_key_value_heads=2, head_dim=16, vocab_size=320,
              linear_num_value_heads=4, linear_num_key_heads=2, linear_key_head_dim=16,
              linear_value_head_dim=16, linear_conv_kernel_dim=4, tie_word_embeddings=True,
              full_attention_interval=4, rms_norm_eps=1e-6,
              rope_parameters={'rope_type': 'default', 'mrope_section': [1, 1, 0], 'rope_theta': 10000,
                               'partial_rotary_factor': 0.25})
    model = qwen3_5.Model(qwen3_5.ModelArgs.from_dict({'model_type': 'qwen3_5', 'text_config': tc}))
    if quantize:
        nn.quantize(model, group_size=64, bits=8)
    mx.eval(model.parameters())
    model.eval()
    return model


class Detok:
    def __init__(self, tok):
        self.tok = tok
        self.reset()

    def reset(self):
        self.tokens, self.last_segment = [], ''

    def add_token(self, token):
        self.tokens.append(token)
        self.last_segment = self.tok.decode([token], skip_special_tokens=True)


class FakeTokenizer:
    eos_token_ids = [SPECIAL_IDS['<|im_end|>']]

    def __init__(self):
        self.detokenizer = Detok(self)

    def convert_tokens_to_ids(self, token):
        return SPECIAL_IDS[token]

    def encode(self, text, add_special_tokens=False):
        out, i = [], 0
        while i < len(text):
            for s in SPECIALS:
                if text.startswith(s, i):
                    out.append(SPECIAL_IDS[s])
                    i += len(s)
                    break
            else:
                out.extend(text[i].encode('utf-8'))
                i += 1
        return out

    def decode(self, tokens, skip_special_tokens=False):
        inv = {v: k for k, v in SPECIAL_IDS.items()}
        parts, buf = [], bytearray()
        for t in tokens:
            if t in inv:
                if buf:
                    parts.append(buf.decode('utf-8', 'replace'))
                    buf = bytearray()
                if not (skip_special_tokens and inv[t] in SKIPPED):
                    parts.append(inv[t])
            elif t < 256:
                buf.append(t)
        if buf:
            parts.append(buf.decode('utf-8', 'replace'))
        return ''.join(parts)

    def apply_chat_template(self, messages, tools=None, tokenize=True, add_generation_prompt=False,
                            enable_thinking=False):
        text = ''
        system = [m['content'] for m in messages if m['role'] == 'system']
        if tools:
            block = '\n'.join(json.dumps(t) for t in tools)
            text += f'<|im_start|>system\n# Tools\n<tools>\n{block}\n</tools>'
            text += ('\n\n' + system[0] if system else '') + '<|im_end|>\n'
        elif system:
            text += f'<|im_start|>system\n{system[0]}<|im_end|>\n'
        for m in messages:
            if m['role'] == 'system':
                continue
            text += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        if add_generation_prompt:
            text += '<|im_start|>assistant\n<think>\n\n</think>\n\n'
        return self.encode(text) if tokenize else text


def scripted_sampler(tokens):
    """A sampler that ignores logits and plays back a fixed token list."""
    queue = list(tokens)

    def make(_sampling):
        def sample(_logprobs):
            return mx.array([queue.pop(0)])
        return sample
    return make
