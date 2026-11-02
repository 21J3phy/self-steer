"""Call one tool on the running self-steer MCP server (how a controller script talks to it).

  uv run python scripts/mcp_call.py status
  uv run python scripts/mcp_call.py create_mode @examples.json
  uv run python scripts/mcp_call.py try_mode '{"name": "eli5", "prompts": ["Why is the sky blue?"]}'
"""
import asyncio
import json
from pathlib import Path
import sys
import time

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

URL = 'http://127.0.0.1:8765/mcp'


def parse_args(raw):
    if not raw:
        return {}
    if raw.startswith('@'):
        return json.loads(Path(raw[1:]).read_text())
    return json.loads(raw)


async def call(tool, arguments):
    async with streamable_http_client(URL) as (read, write):
        async with ClientSession(read, write, read_timeout_seconds=900) as session:
            await session.initialize()
            started = time.perf_counter()
            result = await session.call_tool(tool, arguments)
            elapsed = time.perf_counter() - started
    payload = getattr(result, 'structured_content', None) or getattr(result, 'structuredContent', None)
    if payload is None:
        payload = [getattr(c, 'text', str(c)) for c in result.content]
    return {'tool': tool, 'seconds': round(elapsed, 2), 'is_error': getattr(result, 'is_error', None) or getattr(result, 'isError', None),
            'result': payload}


if __name__ == '__main__':
    out = asyncio.run(call(sys.argv[1], parse_args(sys.argv[2] if len(sys.argv) > 2 else '')))
    print(json.dumps(out, indent=1, ensure_ascii=False, default=str))
