import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from tpasini.recordatio.proxy.mcp_server import create_mcp, mount_mcp


def parse(response):
    response.raise_for_status()
    if response.headers.get('content-type', '').startswith('application/json'):
        return response.json()
    return json.loads(next(line[6:] for line in response.text.splitlines() if line.startswith('data: ')))


def test_http_mcp_starts_in_parent_lifespan_and_preserves_cleanup():
    events = []

    @asynccontextmanager
    async def parent(app):
        events.append('startup')
        try:
            yield {'parent_ready': True}
        finally:
            events.append('shutdown')

    async def retrieve(query, scope):
        events.append((query, scope))
        return SimpleNamespace(topic_hint=None, units=[], entities=[], rendered_relations=[])

    mcp = create_mcp(retrieve=retrieve, extract_and_store=None, schedule_background=None,
                     settings=SimpleNamespace(recordatio_extraction_api_key=None), allowed_hosts=['recordatio:8787'])
    app = FastAPI(lifespan=parent)
    mount_mcp(app, mcp)
    headers = {'Accept': 'application/json, text/event-stream', 'X-Agent-Id': 'alice/helper'}
    with TestClient(app, base_url='http://recordatio:8787', headers=headers) as client:
        response = client.post('/mcp/mcp', json={'jsonrpc': '2.0', 'id': 1, 'method': 'initialize', 'params': {
            'protocolVersion': '2025-03-26', 'capabilities': {}, 'clientInfo': {'name': 'test', 'version': '1'}}})
        assert parse(response)['result']['serverInfo']['name'] == 'recordatio'
        response = client.post('/mcp/mcp', json={
            'jsonrpc': '2.0', 'id': 2, 'method': 'tools/call',
            'params': {'name': 'recollect', 'arguments': {'query': 'listing deadline'}},
        })
        result = parse(response)['result']
        assert not result.get('isError')
        assert events == ['startup', ('listing deadline', 'alice/helper')]
        denied = client.post('/mcp/mcp', headers={'Host': 'untrusted.example'}, json={
            'jsonrpc': '2.0', 'id': 3, 'method': 'tools/list',
        })
        assert denied.status_code == 421
    assert events[-1] == 'shutdown'


def test_mount_preserves_legacy_shutdown_handlers():
    events = []
    app = FastAPI()
    app.router.add_event_handler('startup', lambda: events.append('startup'))
    app.router.add_event_handler('shutdown', lambda: events.append('shutdown'))
    mcp = create_mcp(retrieve=None, extract_and_store=None, schedule_background=None,
                     settings=SimpleNamespace(recordatio_extraction_api_key=None))
    mount_mcp(app, mcp)
    with TestClient(app):
        assert events == ['startup']
    assert events == ['startup', 'shutdown']
