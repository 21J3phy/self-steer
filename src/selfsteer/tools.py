"""The tools Qwen sees, and the system prompt that explains its modes."""
from __future__ import annotations

from mlx_lm.tool_parsers.qwen3_coder import parse_tool_call


def _fn(name, description, properties, required=()):
    return {'type': 'function', 'function': {
        'name': name, 'description': description,
        'parameters': {'type': 'object', 'properties': properties, 'required': list(required)}}}


SET_MODE = _fn(
    'set_mode',
    'Change your own neuron settings before you answer. The mode shapes how you write your next answer.',
    {'name': {'type': 'string', 'description': 'A mode name from the list in the system prompt, or "normal".'},
     'scope': {'type': 'string', 'enum': ['answer', 'conversation'],
               'description': '"answer" (default) for the next answer only, "conversation" until changed.'}},
    ['name'])

REQUEST_MODE = _fn(
    'request_mode',
    'Ask Claude, your controller, to build a new mode for you when none of your modes fits. '
    'Describe the behavior you want. You will get back the new mode name, which you can then switch on with set_mode.',
    {'description': {'type': 'string', 'description': 'The behavior the new mode should produce.'}},
    ['description'])


def tool_schemas(allow_requests):
    return [SET_MODE] + ([REQUEST_MODE] if allow_requests else [])


def system_prompt(library, allow_requests, extra=None):
    parts = [
        'You can change how your own neurons are set before you answer, by calling set_mode. '
        'Each mode is a small edit to your MLP layers. Your modes:',
        library.guide(),
        '- normal: no edit.',
        'Before answering, decide whether a mode clearly fits the request. If one does, call set_mode once, '
        'then answer after the tool result. If none fits, just answer normally without calling any tool.',
    ]
    if allow_requests:
        parts.append('If the user wants a style none of your modes gives, you may call request_mode to ask '
                     'Claude to build one, then switch it on with set_mode.')
    if extra:
        parts.append(extra)
    return '\n'.join(parts)


def parse(text, tools):
    """Parses the body of one <tool_call>...</tool_call> block."""
    start = text.rfind('<tool_call>')
    end = text.rfind('</tool_call>')
    if start < 0 or end < start:
        raise ValueError('No complete tool call')
    body = text[start + len('<tool_call>'):end].strip()
    return parse_tool_call(body, tools)
