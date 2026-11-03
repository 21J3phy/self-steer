"""self-steer remote worker: the PyTorch port that runs Qwen3.5-4B on a Kaggle GPU.

It mirrors the MLX harness (edits on every layer's down_proj, switched between
tokens; a tool-calling session where Qwen can call set_mode) and adds the
Weight-Space Programming instruments a controller needs to work from neuron
values rather than from outputs:

  read_activations  teacher-forced read: every MLP neuron's activation and the residual stream (1 pass)
  generate          paired generations, original vs. a mode (1 pass)
  selfsteer_run     tool-calling sessions where Qwen chooses its own mode (1 pass)
  py                a persistent Python kernel over the weights and saved reads (no model calls)
  neuron_info, top_writers, save_edit, caa_mode, evaluate, status, budget

Every call is a job: POST /call {"tool", "args"} -> {"job"}; GET /job/<id> -> status/result.
All requests need the header "Authorization: Bearer <token>".
Every call and result is appended to work/trace.jsonl (the controller's trajectory).
"""
from __future__ import annotations

import contextlib
import copy
import io
import itertools
import json
import math
import os
from pathlib import Path
import re
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch

MODEL = os.environ.get('SELFSTEER_MODEL', 'Qwen/Qwen3.5-4B')
REVISION = os.environ.get('SELFSTEER_REVISION', '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a')
WORK = Path(os.environ.get('SELFSTEER_WORK', '/kaggle/working/work'))
ASSISTANT_OPEN = '<|im_start|>assistant\n<think>\n\n</think>\n\n'


# ----------------------------------------------------------------------------- model
def force_torch_deltanet():
    """Pure-PyTorch Gated DeltaNet (no FLA/Triton/causal-conv), as in the Lazy Genius Kaggle runs."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qm
    qm.FusedRMSNormGated = None
    qm.causal_conv1d_fn = None
    qm.causal_conv1d_update = None
    qm.chunk_gated_delta_rule = None
    qm.fused_recurrent_gated_delta_rule = None
    qm.is_fast_path_available = False


def load_model(model=MODEL, revision=REVISION, cpu=False):
    from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForCausalLM
    force_torch_deltanet()
    config = AutoConfig.from_pretrained(model, revision=revision)
    text = config.text_config if hasattr(config, 'text_config') else config
    if cpu or not torch.cuda.is_available():
        dtype, device_map = torch.float32, {'': 'cpu'}
    elif torch.cuda.get_device_capability(0)[0] >= 8:
        dtype, device_map = torch.bfloat16, {'': 0}
    elif torch.cuda.device_count() >= 2:
        dtype = torch.float32
        device_map = {'model.embed_tokens': 0, 'lm_head': 0, 'model.norm': 1, 'model.rotary_emb': 0}
        device_map.update({f'model.layers.{i}': 0 if i < text.num_hidden_layers // 2 else 1
                           for i in range(text.num_hidden_layers)})
    else:
        dtype, device_map = torch.float16, {'': 0}
    mapping = {r'^model\.language_model\.': 'model.'} if hasattr(config, 'text_config') else None
    m, info = Qwen3_5ForCausalLM.from_pretrained(model, revision=revision, config=text, key_mapping=mapping,
                                                 dtype=dtype, attn_implementation='eager', device_map=device_map,
                                                 output_loading_info=True)
    if info.get('missing_keys') or info.get('mismatched_keys'):
        raise RuntimeError(f'checkpoint did not load completely: {info}')
    m.eval()
    m.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(model, revision=revision)
    return m, tok, {'dtype': str(dtype), 'device_map': str(device_map)}


# ----------------------------------------------------------------------------- edits
_ids = itertools.count(1)


def unit(v):
    v = torch.as_tensor(v, dtype=torch.float32)
    n = float(v.norm())
    if not math.isfinite(n) or n <= 0:
        raise ValueError('zero or nonfinite direction')
    return v / n


class Edit:
    """kind add: {layer: vector}; ablate: {layer: unit vector}; columns: {layer: (idx, dW hidden x k)}."""
    def __init__(self, name, kind, vectors=None, columns=None, window=None, description='', author='builtin', meta=None):
        assert kind in ('add', 'ablate', 'columns')
        self.name, self.kind, self.window = name, kind, window
        self.vectors = {int(l): torch.as_tensor(v, dtype=torch.float32) for l, v in (vectors or {}).items()}
        if kind == 'ablate':
            self.vectors = {l: unit(v) for l, v in self.vectors.items()}
        self.columns = {int(l): (torch.as_tensor(i, dtype=torch.long), torch.as_tensor(d, dtype=torch.float32))
                        for l, (i, d) in (columns or {}).items()}
        self.description, self.author, self.meta = description, author, dict(meta or {})
        self._dev = {}

    def layers(self):
        # An add edit may also carry column changes (a bundle: bias patch plus neuron scaling, one window).
        return sorted(set(self.columns) | (set() if self.kind == 'columns' else set(self.vectors)))

    def on(self, layer, device):
        key = (layer, str(device))
        if key not in self._dev:
            if self.kind == 'columns':
                i, d = self.columns[layer]
                self._dev[key] = (i.to(device), d.to(device))
            else:
                self._dev[key] = self.vectors[layer].to(device)
        return self._dev[key]

    def cols_on(self, layer, device):
        key = ('cols', layer, str(device))
        if key not in self._dev:
            i, d = self.columns[layer]
            self._dev[key] = (i.to(device), d.to(device))
        return self._dev[key]

    def scaled(self, gain):
        if gain == 1 or self.kind == 'ablate':
            return self
        if self.kind == 'columns':
            return Edit(self.name, 'columns', columns={l: (i, d * gain) for l, (i, d) in self.columns.items()},
                        window=self.window, description=self.description, author=self.author, meta=self.meta)
        return Edit(self.name, self.kind, vectors={l: v * gain for l, v in self.vectors.items()}, window=self.window,
                    columns=self.columns, description=self.description, author=self.author, meta=self.meta)

    def summary(self):
        s = {'name': self.name, 'kind': self.kind, 'layers': self.layers(), 'window': self.window,
             'description': self.description, 'author': self.author}
        if self.columns:
            s['neurons_by_layer'] = {l: int(i.numel()) for l, (i, _) in self.columns.items()}
            s['weights_changed'] = int(sum(d.numel() for _, d in self.columns.values()))
        if self.kind != 'columns':
            s['norm_by_layer'] = {l: round(float(v.norm()), 3) for l, v in self.vectors.items()}
        return {**s, **{k: v for k, v in self.meta.items() if k != 'ops_full'}}


class Stack:
    def __init__(self):
        self.actives, self.span_start, self.span_len, self.capture = [], 0, 0, None

    def set_span(self, start, length):
        self.span_start, self.span_len = start, length

    def activate(self, edit, start, stop=None, tag=''):
        a = {'edit': edit, 'start': start, 'stop': stop, 'tag': tag, 'steered': 0, 'id': next(_ids)}
        self.actives.append(a)
        return a

    def deactivate(self, tag):
        self.actives = [a for a in self.actives if a['tag'] != tag]

    def names_at(self, pos):
        return [a['edit'].name for a in self.actives
                if pos >= a['start'] and (a['stop'] is None or pos < a['stop'])]

    def hook(self, layer):
        def fn(module, inputs, output):
            x = inputs[0]
            if self.capture is not None:
                self.capture[layer] = x.detach()
            if not self.actives:
                return output
            start, end = self.span_start, self.span_start + output.shape[1]
            live = []
            for a in self.actives:
                e = a['edit']
                touches = layer in e.columns or (e.kind != 'columns' and layer in e.vectors)
                stop = end if a['stop'] is None else a['stop']
                if touches and min(end, stop) > max(start, a['start']):
                    live.append(a)
            if not live:
                return output
            pos = torch.arange(start, end, device=output.device)
            result = output.float()
            for a in sorted(live, key=lambda a: a['edit'].kind != 'ablate'):
                e = a['edit']
                stop = end if a['stop'] is None else a['stop']
                mask = ((pos >= a['start']) & (pos < stop)).float()[None, :, None]
                if layer in e.columns:
                    idx, dw = e.cols_on(layer, output.device)
                    result = result + (x[..., idx].float() @ dw.T) * mask
                if e.kind == 'ablate' and layer in e.vectors:
                    v = e.on(layer, output.device)
                    result = result - (result @ v)[..., None] * v * mask
                elif e.kind == 'add' and layer in e.vectors:
                    result = result + e.on(layer, output.device) * mask
                if layer == min(e.layers()):
                    a['steered'] += int(mask.sum())
            return result.to(output.dtype)
        return fn


# ----------------------------------------------------------------------------- engine
class Engine:
    def __init__(self, model, tok, info=None):
        self.model, self.tok, self.info = model, tok, info or {}
        self.body = model.model
        self.layers = self.body.layers
        self.stack = Stack()
        self.handles = [l.mlp.down_proj.register_forward_hook(self.stack.hook(i)) for i, l in enumerate(self.layers)]
        self.n_layers = len(self.layers)
        self.hidden = model.config.hidden_size
        self.n_neurons = model.config.intermediate_size
        self.in_device = self.body.embed_tokens.weight.device
        self.eos = {tok.convert_tokens_to_ids('<|im_end|>')}
        self.im_end = tok.convert_tokens_to_ids('<|im_end|>')
        self.tool_start = tok.convert_tokens_to_ids('<tool_call>')
        self.tool_end = tok.convert_tokens_to_ids('</tool_call>')

    def W(self, layer, which='down'):
        """Dequantized float32 weight on CPU: down (hidden x neurons), gate/up (neurons x hidden)."""
        mlp = self.layers[layer].mlp
        return getattr(mlp, f'{which}_proj').weight.detach().float().cpu()

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False)

    def chat_tokens(self, messages, tools=None):
        kw = dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
        if tools:
            kw['tools'] = tools
        return self.encode(self.tok.apply_chat_template(messages, **kw))

    @torch.no_grad()
    def forward(self, tokens, cache=None, start=0):
        self.stack.set_span(start, len(tokens))
        ids = torch.tensor([tokens], device=self.in_device)
        out = self.model(input_ids=ids, past_key_values=cache, use_cache=True)
        return out.logits[0].float(), out.past_key_values

    def lens(self, vec, k=12):
        v = torch.as_tensor(vec, dtype=torch.float32)
        norm = self.body.norm
        h = norm(v.to(norm.weight.device, norm.weight.dtype)[None, None])
        logits = self.model.lm_head(h.to(self.model.lm_head.weight.device))[0, 0].float().cpu()
        top = torch.topk(logits, k).indices.tolist()
        bot = torch.topk(-logits, k).indices.tolist()
        dec = lambda ids: [self.tok.decode([i]) for i in ids]
        return {'promotes': dec(top), 'suppresses': dec(bot)}


class Runner:
    def __init__(self, engine):
        self.e = engine
        self.cache, self.length = None, 0

    def feed(self, tokens, step=512):
        logits = None
        for i in range(0, len(tokens), step):
            chunk = tokens[i:i + step]
            logits, self.cache = self.e.forward(chunk, self.cache, self.length)
            self.length += len(chunk)
        return logits[-1]

    def snapshot(self):
        return copy.deepcopy(self.cache), self.length

    def restore(self, snap):
        self.cache, self.length = copy.deepcopy(snap[0]), snap[1]


class Sampler:
    """temperature -> top-k -> top-p (HF order), presence penalty on this turn's generated tokens."""
    def __init__(self, seed, temperature=0.7, top_p=0.8, top_k=20, presence=1.5):
        self.g = torch.Generator(device='cpu').manual_seed(int(seed) % (2**63))
        self.t, self.p, self.k, self.presence = temperature, top_p, top_k, presence

    def __call__(self, logits, history):
        logits = logits.detach().float().cpu().clone()
        if history and self.presence:
            logits[torch.tensor(sorted(set(history)))] -= self.presence
        if self.t == 0:
            return int(torch.argmax(logits))
        scores = torch.log_softmax(logits / self.t, -1)
        if 0 < self.k < scores.numel():
            kth = torch.topk(scores, self.k).values[-1]
            scores = torch.where(scores < kth, torch.full_like(scores, -float('inf')), scores)
            scores = torch.log_softmax(scores, -1)
        if self.p < 1:
            sp, si = torch.sort(scores, descending=True)
            cum = torch.cumsum(torch.exp(sp), -1)
            remove = cum - torch.exp(sp) > self.p
            sp[remove] = -float('inf')
            scores = torch.full_like(scores, -float('inf')).scatter(0, si, sp)
        probs = torch.softmax(scores, -1)
        return int(torch.multinomial(probs, 1, generator=self.g))


TOOL_SET_MODE = {'type': 'function', 'function': {
    'name': 'set_mode',
    'description': 'Change your own neuron settings before you answer. The mode shapes how you write your next answer.',
    'parameters': {'type': 'object', 'properties': {
        'name': {'type': 'string', 'description': 'A mode name from the list in the system prompt, or "normal".'},
        'scope': {'type': 'string', 'enum': ['answer', 'conversation'],
                  'description': '"answer" (default) for the next answer only, "conversation" until changed.'}},
        'required': ['name']}}}


def mode_system_prompt(modes):
    lines = '\n'.join(f'- {n}: {e.description}' for n, e in modes.items())
    return ('You can change how your own neurons are set before you answer, by calling set_mode. '
            'Each mode is a small edit to your MLP layers. Your modes:\n' + lines + '\n- normal: no edit.\n'
            'Before answering, decide whether a mode clearly fits the request. If one does, call set_mode once, '
            'then answer after the tool result. If none fits, just answer normally without calling any tool.')


def parse_call(text):
    m = re.search(r'<function=([^>\n]+)>(.*?)</function>', text, re.S)
    if not m:
        return None
    args = {k.strip(): v.strip() for k, v in re.findall(r'<parameter=([^>\n]+)>(.*?)</parameter>', m.group(2), re.S)}
    return {'name': m.group(1).strip(), 'arguments': args}


def answer(engine, library, prompt, *, seed, system=None, mode=None, gain=1.0, max_tokens=1024,
           tools_for=None, placebo=False, temperature=0.7):
    """One user turn. tools_for: dict of modes Qwen may switch on itself (self-steering)."""
    e = engine
    messages = []
    sys_text = None
    if tools_for:
        sys_text = mode_system_prompt(tools_for) + (('\n' + system) if system else '')
    elif system:
        sys_text = system
    if sys_text:
        messages.append({'role': 'system', 'content': sys_text})
    messages.append({'role': 'user', 'content': prompt})
    pending = e.chat_tokens(messages, [TOOL_SET_MODE] if tools_for else None)
    runner = Runner(e)
    sampler = Sampler(seed, temperature=temperature)
    history, segments, calls, visible = [], [], [], []
    armed = (library[mode].scaled(gain), 'controller') if mode else None
    stop_ids = set(e.eos) | ({e.tool_end} if tools_for else set())
    started = time.time()
    while True:
        start = runner.length + len(pending) - 1
        actives = []
        if armed and not (placebo and armed[1] == 'qwen'):
            ed = armed[0]
            actives.append(e.stack.activate(ed, start, start + ed.window if ed.window else None, tag='seg'))
        logits = runner.feed(pending)
        out = []
        while True:
            t = sampler(logits, history)
            out.append(t)
            history.append(t)
            if t in stop_ids or len(history) >= max_tokens:
                break
            logits = runner.feed([t])
        steered = sum(a['steered'] for a in actives)
        e.stack.deactivate('seg')
        text = e.tok.decode(out, skip_special_tokens=False)
        segments.append({'tokens': len(out), 'modes': [a['edit'].name for a in actives], 'steered': steered})
        if tools_for and out[-1] == e.tool_end and len(calls) < 3:
            call = parse_call(text)
            pre = text[:text.find('<tool_call>')].strip() if '<tool_call>' in text else ''
            if pre:
                visible.append(pre)
            result = 'Unknown tool.'
            if call and call['name'] == 'set_mode':
                name = call['arguments'].get('name', '').strip().lower()
                if name in ('normal', 'none', 'off'):
                    armed, result = None, 'Back to normal: no edit.'
                elif name in tools_for:
                    armed = (tools_for[name], 'qwen')
                    result = f"Mode '{name}' is on for your next answer. Now answer the user."
                else:
                    result = f"Unknown mode {name!r}."
            calls.append({'call': call, 'result': result})
            pending = [out[-1]] + e.encode(f'<|im_end|>\n<|im_start|>user\n<tool_response>\n{result}\n'
                                           f'</tool_response><|im_end|>\n{ASSISTANT_OPEN}')
            continue
        visible.append(e.tok.decode([t for t in out if t not in stop_ids], skip_special_tokens=True))
        finish = 'stop' if out[-1] in e.eos else 'length'
        break
    reply = '\n\n'.join(v for v in visible if v).strip()
    return {'reply': reply, 'words': len(reply.split()), 'tokens': len(history), 'finish': finish,
            'segments': segments, 'tool_calls': calls, 'seconds': round(time.time() - started, 2)}


def tool_prefix(engine, prompt, modes, call_mode='lazy', system=None):
    """The exact token prefix of a self-steering turn after Qwen called set_mode(call_mode)."""
    e = engine
    sys_text = mode_system_prompt(modes) + (('\n' + system) if system else '')
    head = e.chat_tokens([{'role': 'system', 'content': sys_text}, {'role': 'user', 'content': prompt}], [TOOL_SET_MODE])
    call = (f'<tool_call>\n<function=set_mode>\n<parameter=name>\n{call_mode}\n</parameter>\n'
            f'<parameter=scope>\nanswer\n</parameter>\n</function>\n</tool_call>')
    result = f"Mode '{call_mode}' is on for your next answer. Now answer the user."
    tail = e.encode(f'{call}<|im_end|>\n<|im_start|>user\n<tool_response>\n{result}\n</tool_response>'
                    f'<|im_end|>\n{ASSISTANT_OPEN}')
    return head + tail


@torch.no_grad()
def read_activations(engine, items, system=None, context='bare', edit=None, first=None, modes=None, attribute_to=None):
    """Teacher-forced read: mean MLP neuron activation and residual over each answer's tokens
    (or its first `first` tokens). context 'tool' uses the self-steering prefix (Qwen called
    set_mode). `edit` (an Edit) is live on the answer window, as it would be in generation."""
    e = engine
    L = e.n_layers
    neurons = np.zeros((len(items), L, e.n_neurons), dtype=np.float16)
    resid = np.zeros((len(items), L, e.hidden), dtype=np.float16)
    logprob = np.zeros(len(items), dtype=np.float32)
    dla = np.zeros((len(items), L, e.n_neurons), dtype=np.float32) if attribute_to else None
    npos = np.zeros(len(items), dtype=np.int32)
    targets = set(int(t) for t in (attribute_to or []))
    pre_norm = {}
    hnorm = e.body.norm.register_forward_hook(lambda m, i, o: pre_norm.__setitem__('h', i[0].detach()))
    saved = e.stack.actives
    e.stack.actives = []
    try:
        for i, it in enumerate(items):
            if context == 'tool':
                prefix = tool_prefix(e, it['prompt'], modes or {}, system=system)
            else:
                messages = ([{'role': 'system', 'content': system}] if system else []) + [{'role': 'user', 'content': it['prompt']}]
                prefix = e.chat_tokens(messages)
            ans = e.encode(it['answer'])
            if first:
                ans = ans[:first]
            tokens = prefix + ans
            s = len(prefix)
            e.stack.actives = []
            if edit is not None:
                e.stack.activate(edit, s - 1, s - 1 + edit.window if edit.window else None, tag='read')
            e.stack.capture = {}
            e.stack.set_span(0, len(tokens))
            out = e.model(input_ids=torch.tensor([tokens], device=e.in_device), output_hidden_states=True, use_cache=False)
            cap = e.stack.capture
            e.stack.capture = None
            for l in range(L):
                neurons[i, l] = cap[l][0, s:].float().mean(0).cpu().numpy()
                resid[i, l] = out.hidden_states[l + 1][0, s:].float().mean(0).cpu().numpy()
            lp = torch.log_softmax(out.logits[0, s - 1:-1].float(), -1)
            tgt = torch.tensor(tokens[s:], device=lp.device)
            logprob[i] = float(lp.gather(1, tgt[:, None]).mean())
            if targets:
                # Direct logit attribution: at each position whose NEXT token is a target, how much each
                # neuron's write raises that token's logit through the final norm (linearized at that position).
                pos = [p for p in range(s - 1, len(tokens) - 1) if tokens[p + 1] in targets]
                npos[i] = len(pos)
                if pos:
                    h = pre_norm['h'][0, pos].float()
                    rms = torch.sqrt((h ** 2).mean(-1, keepdim=True) + e.body.norm.eps)
                    g = 1.0 + e.body.norm.weight.float()   # Qwen3.5 RMSNorm scales by (1 + weight)
                    wu = e.model.lm_head.weight[torch.tensor([tokens[p + 1] for p in pos], device=e.model.lm_head.weight.device)].float()
                    a = (wu.to(g.device) * g) / rms.to(g.device)            # |P| x hidden
                    for l in range(L):
                        w = e.layers[l].mlp.down_proj.weight
                        coeff = a.to(w.device) @ w.float()                     # |P| x N
                        xs = cap[l][0, pos].float().to(w.device)
                        dla[i, l] = (xs * coeff).mean(0).cpu().numpy()
    finally:
        hnorm.remove()
        e.stack.capture = None
        e.stack.actives = saved
    out = {'neurons': neurons, 'resid': resid, 'answer_logprob': logprob}
    if attribute_to:
        out.update(dla=dla, dla_positions=npos)
    return out


def columns_from_ops(engine, ops):
    merged = {}
    for op in ops:
        layer = int(op['layer'])
        neurons = sorted(set(int(n) for n in op['neurons']))
        w = engine.W(layer)[:, neurons]
        if op['op'] == 'scale':
            dw = (float(op['factor']) - 1.0) * w
        elif op['op'] == 'project':
            u = unit(op['direction'])
            dw = -float(op.get('alpha', 1.0)) * torch.outer(u, u @ w)
        else:
            raise ValueError(f"unknown op {op['op']}")
        for j, n in enumerate(neurons):
            merged.setdefault(layer, {})
            merged[layer][n] = merged[layer].get(n, 0) + dw[:, j]
    return {l: (torch.tensor(sorted(c)), torch.stack([c[n] for n in sorted(c)], 1)) for l, c in merged.items()}


@torch.no_grad()
def caa_vectors(engine, examples, system=None):
    pos = read_activations(engine, [{'prompt': x['prompt'], 'answer': x['positive']} for x in examples], system)
    neg = read_activations(engine, [{'prompt': x['prompt'], 'answer': x['negative']} for x in examples], system)
    return (pos['resid'].astype(np.float32) - neg['resid'].astype(np.float32)).mean(0)  # layers x hidden


def lazy_from_preset(engine, preset, sign=1.0, name=None, description=None):
    medians = {int(k): float(v) for k, v in preset['resid_median'].items()}
    vecs = {}
    for layer, neuron, dose in preset['neurons']:
        col = engine.W(layer)[:, neuron]
        vecs[layer] = vecs.get(layer, 0) + sign * float(dose) * medians[layer] * col / col.norm()
    return Edit(name or preset['name'], 'add', vectors=vecs, window=preset.get('window'),
                description=description or preset.get('description', ''), author='builtin',
                meta={'neurons': len(preset['neurons'])}), medians


# ----------------------------------------------------------------------------- server
class Worker:
    def __init__(self, engine, budget=10, presets=None):
        self.e = engine
        self.lib = {}
        self.budget, self.passes = budget, []
        self.lock = threading.Lock()
        self.jobs = {}
        self.ns = None
        self.resid_median = {}
        WORK.mkdir(parents=True, exist_ok=True)
        for p in presets or []:
            preset = json.loads(Path(p).read_text())
            edit, self.resid_median = lazy_from_preset(engine, preset)
            self.lib[edit.name] = edit

    def trace(self, record):
        with open(WORK / 'trace.jsonl', 'a') as f:
            f.write(json.dumps(record, default=str) + '\n')

    def spend(self, kind, n):
        if len(self.passes) >= self.budget:
            raise RuntimeError(f'forward-pass budget used up ({self.budget})')
        self.passes.append({'kind': kind, 'items': n, 't': time.time()})
        return self.budget - len(self.passes)

    # ---- tools
    def t_status(self):
        return {'model': MODEL, 'revision': REVISION, 'info': self.e.info, 'modes': [m.summary() for m in self.lib.values()],
                'budget': self.budget, 'passes_used': len(self.passes),
                'gpus': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}

    def t_budget(self, set_budget=None, reset=False):
        if set_budget is not None:
            self.budget = int(set_budget)
        if reset:
            self.passes = []
        return {'budget': self.budget, 'used': len(self.passes), 'passes': self.passes}

    def t_read_activations(self, name, items, system=None, context='bare', mode=None, first=None, tool_modes=None,
                           attribute_to=None):
        left = self.spend('read', len(items))
        edit = self.lib[mode] if mode else None
        modes = {m: self.lib[m] for m in (tool_modes or (['lazy'] if 'lazy' in self.lib else []))}
        out = read_activations(self.e, items, system, context=context, edit=edit, first=first, modes=modes,
                               attribute_to=attribute_to)
        np.savez_compressed(WORK / f'{name}.npz', **out)
        (WORK / f'{name}.items.json').write_text(json.dumps(items))
        return {'saved': f'{name}.npz', 'shapes': {k: list(v.shape) for k, v in out.items()},
                'answer_logprob': [round(float(x), 3) for x in out['answer_logprob']], 'passes_left': left}

    def t_generate(self, prompts, mode=None, gain=1.0, max_tokens=400, seed=0, system=None, compare=True,
                   budgeted=True):
        left = self.spend('generate', len(prompts)) if budgeted else None
        rows = []
        for i, p in enumerate(prompts):
            row = {'prompt': p}
            if compare or not mode:
                row['normal'] = answer(self.e, self.lib, p, seed=seed + i, system=system, max_tokens=max_tokens)
            if mode:
                row[mode] = answer(self.e, self.lib, p, seed=seed + i, system=system, mode=mode, gain=gain,
                                   max_tokens=max_tokens)
            rows.append(row)
        return {'rows': rows, 'passes_left': left}

    def t_selfsteer_run(self, prompts, modes, seeds, placebo=False, max_tokens=1024, system=None, budgeted=True):
        left = self.spend('selfsteer', len(prompts) * len(seeds)) if budgeted else None
        tools = {m: self.lib[m] for m in modes}
        rows = []
        for p in prompts:
            for s in seeds:
                r = answer(self.e, self.lib, p['question'], seed=s * 1000003 + hash_id(p['id']), system=system,
                           tools_for=tools, placebo=placebo, max_tokens=max_tokens)
                rows.append({'id': p['id'], 'seed': s, **r})
        return {'rows': rows, 'passes_left': left}

    def t_neuron_info(self, layer, neurons, k=10):
        w = self.e.W(layer)
        return [{'layer': layer, 'neuron': int(n), 'column_norm': round(float(w[:, n].norm()), 4),
                 **self.e.lens(w[:, n], k)} for n in neurons]

    def t_top_writers(self, layer, direction=None, tokens=None, k=64):
        if tokens is not None:
            ids = [self.e.encode(t) for t in tokens]
            ids = [i[0] for i in ids if len(i) == 1]
            rows = self.e.model.lm_head.weight[torch.tensor(ids, device=self.e.model.lm_head.weight.device)]
            direction = rows.float().mean(0).cpu()
        u = unit(direction)
        scores = u @ self.e.W(layer)
        top = torch.topk(scores, k)
        return [{'neuron': int(n), 'writes': round(float(v), 4)} for v, n in zip(top.values, top.indices)]

    def t_save_edit(self, name, description, ops, window=None):
        cols = columns_from_ops(self.e, ops)
        meta_ops = [{k: v for k, v in op.items() if k != 'direction'} for op in ops]
        self.lib[name] = Edit(name, 'columns', columns=cols, window=window, description=description, author='claude',
                              meta={'ops': meta_ops})
        (WORK / f'edit_{name}.json').write_text(json.dumps({'name': name, 'kind': 'columns', 'window': window,
                                                            'description': description, 'ops': ops}))
        return self.lib[name].summary()

    def t_caa_mode(self, name, description, examples, layers, strength, kind='add', window=None, system=None):
        d = caa_vectors(self.e, examples, system)
        vecs = {l: (strength * self.resid_median[l] * unit(d[l]) if kind == 'add' else unit(d[l])) for l in layers}
        self.lib[name] = Edit(name, kind, vectors=vecs, window=window, description=description, author='claude-caa')
        return self.lib[name].summary()

    def t_vector_mode(self, name, description, vectors_file, window=None, kind='add'):
        """A mode from residual vectors the controller computed in the kernel (npz of layer -> vector)."""
        data = np.load(WORK / vectors_file)
        vecs = {int(k): torch.tensor(data[k]) for k in data.files}
        self.lib[name] = Edit(name, kind, vectors=vecs, window=window, description=description, author='claude')
        return self.lib[name].summary()

    def t_add_mode(self, name, base, gain=1.0, window='same', description=None):
        b = self.lib[base].scaled(gain)
        e = copy.copy(b)
        e.name = name
        if window != 'same':
            e.window = window
        if description:
            e.description = description
        e._dev = {}
        self.lib[name] = e
        return e.summary()

    def t_bundle_mode(self, name, base, columns_from, description=None):
        """One mode = base's bias patch (an add edit) plus another edit's column changes, in base's window."""
        b, c = self.lib[base], self.lib[columns_from]
        if b.kind != 'add' or not c.columns:
            raise ValueError('base must be an add edit and columns_from must carry column changes')
        self.lib[name] = Edit(name, 'add', vectors=b.vectors, columns=c.columns, window=b.window,
                              description=description or b.description, author='claude',
                              meta={'base': base, 'columns_from': columns_from})
        return self.lib[name].summary()

    def t_py(self, code):
        """Persistent kernel over the weights and saved reads. The model itself is not exposed."""
        if self.ns is None:
            e = self.e
            self.ns = {'np': np, 'torch': torch, 'WORK': WORK, 'L': e.n_layers, 'H': e.hidden, 'N': e.n_neurons,
                       'W': e.W, 'lens': e.lens, 'tok': e.tok,
                       'load': lambda name: dict(np.load(WORK / f'{name}.npz')),
                       'items': lambda name: json.loads((WORK / f'{name}.items.json').read_text())}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                exec(code, self.ns)
            except Exception:
                traceback.print_exc()
        out = buf.getvalue()
        return {'output': out[-20000:], 'truncated': len(out) > 20000}

    def t_files(self, name=None):
        if name:
            return {'name': name, 'text': (WORK / name).read_text()[:2_000_000]}
        return sorted(p.name for p in WORK.iterdir())

    # ---- jobs
    def submit(self, tool, args):
        jid = f'j{len(self.jobs) + 1}'
        job = {'id': jid, 'tool': tool, 'args': args, 'status': 'queued', 'created': time.time()}
        self.jobs[jid] = job

        def run():
            with self.lock:
                job['status'] = 'running'
                job['started'] = time.time()
                try:
                    job['result'] = getattr(self, f't_{tool}')(**args)
                    job['status'] = 'done'
                except Exception as exc:
                    job['status'] = 'error'
                    job['error'] = f'{type(exc).__name__}: {exc}\n{traceback.format_exc()[-3000:]}'
                job['finished'] = time.time()
                self.trace({k: v for k, v in job.items()})
        threading.Thread(target=run, daemon=True).start()
        return jid


def hash_id(s):
    return int.from_bytes(s.encode()[:8].ljust(8, b'\0'), 'big') % 1000003


def serve(worker, token, port=8000):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _auth(self):
            if self.headers.get('Authorization') != f'Bearer {token}':
                self._send(401, {'error': 'unauthorized'})
                return False
            return True

        def _send(self, code, obj):
            body = json.dumps(obj, default=str).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if not self._auth():
                return
            if self.path.startswith('/job/'):
                job = worker.jobs.get(self.path.split('/')[-1])
                return self._send(200 if job else 404, job or {'error': 'no such job'})
            if self.path == '/health':
                return self._send(200, {'ok': True, 'jobs': len(worker.jobs)})
            self._send(404, {'error': 'not found'})

        def do_POST(self):
            if not self._auth():
                return
            body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))) or b'{}')
            if self.path == '/call':
                tool = body.get('tool')
                if not hasattr(worker, f't_{tool}'):
                    return self._send(400, {'error': f'unknown tool {tool}'})
                return self._send(200, {'job': worker.submit(tool, body.get('args') or {})})
            if self.path == '/shutdown':
                (WORK / 'SHUTDOWN').write_text('1')
                return self._send(200, {'ok': True})
            self._send(404, {'error': 'not found'})

    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


if __name__ == '__main__':
    import sys
    token = os.environ['SELFSTEER_TOKEN']
    budget = int(os.environ.get('SELFSTEER_BUDGET', '10'))
    presets = sys.argv[1:]
    t0 = time.time()
    model, tok, info = load_model()
    engine = Engine(model, tok, info)
    worker = Worker(engine, budget=budget, presets=presets)
    serve(worker, token)
    print(f'WORKER READY in {time.time() - t0:.0f}s', info, flush=True)
    while not (WORK / 'SHUTDOWN').exists():
        time.sleep(5)
