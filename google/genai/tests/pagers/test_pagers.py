# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Tests that pagers keep iterating past empty pages."""

import pytest

from ... import Client
from ... import pagers
from ... import types

httpx = pytest.importorskip('httpx')

# page_token -> (file names on the page, next_page_token)
_PAGES = {
    None: (['files/a'], 'token-1'),
    'token-1': ([], 'token-2'),
    'token-2': (['files/b'], None),
}


def _list_files_response(page_token):
  names, next_page_token = _PAGES[page_token]
  return types.ListFilesResponse(
      files=[types.File(name=name) for name in names],
      next_page_token=next_page_token,
  )


def test_pager_skips_empty_page():
  pager = pagers.Pager(
      'files',
      lambda config: _list_files_response(config['page_token']),
      _list_files_response(None),
      {'page_size': 1},
  )

  assert [f.name for f in pager] == ['files/a', 'files/b']


async def test_async_pager_skips_empty_page():
  async def request(config):
    return _list_files_response(config['page_token'])

  pager = pagers.AsyncPager(
      'files', request, _list_files_response(None), {'page_size': 1}
  )

  assert [f.name async for f in pager] == ['files/a', 'files/b']


def test_pager_with_only_empty_pages_stops():
  pages = {None: ([], 'token-1'), 'token-1': ([], None)}

  def response(page_token):
    names, next_page_token = pages[page_token]
    return types.ListFilesResponse(
        files=[types.File(name=n) for n in names],
        next_page_token=next_page_token,
    )

  pager = pagers.Pager(
      'files', lambda config: response(config['page_token']), response(None), {}
  )

  assert list(pager) == []


def test_files_list_skips_empty_page(monkeypatch):
  monkeypatch.delenv('GOOGLE_GENAI_USE_VERTEXAI', raising=False)
  monkeypatch.delenv('GOOGLE_GEMINI_BASE_URL', raising=False)
  monkeypatch.setenv('GOOGLE_GENAI_HTTP_CLIENT', 'httpx')

  def handler(request):
    names, next_page_token = _PAGES[request.url.params.get('pageToken')]
    body = {'files': [{'name': name} for name in names]}
    if next_page_token:
      body['nextPageToken'] = next_page_token
    return httpx.Response(200, json=body)

  client = Client(
      api_key='test-key',
      http_options=types.HttpOptions(
          httpx_client=httpx.Client(transport=httpx.MockTransport(handler))
      ),
  )

  names = [f.name for f in client.files.list(config={'page_size': 1})]

  assert names == ['files/a', 'files/b']
