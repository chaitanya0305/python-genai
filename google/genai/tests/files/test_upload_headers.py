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

"""Tests that files.upload sends the headers set in the upload config."""

import io

import pytest

from ... import Client
from ... import types

httpx = pytest.importorskip('httpx')

_UPLOAD_URL = (
    'https://generativelanguage.googleapis.com/upload/session?upload_id=1'
)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.delenv('GOOGLE_GENAI_USE_VERTEXAI', raising=False)
  monkeypatch.delenv('GOOGLE_GEMINI_BASE_URL', raising=False)
  monkeypatch.setenv('GOOGLE_GENAI_HTTP_CLIENT', 'httpx')


def _handler(seen: list):
  def handler(request):
    seen.append(request)
    if request.headers.get('x-goog-upload-command') == 'start':
      return httpx.Response(
          200, headers={'x-goog-upload-url': _UPLOAD_URL}, json={}
      )
    return httpx.Response(
        200,
        headers={'x-goog-upload-status': 'final'},
        json={'file': {'name': 'files/abc'}},
    )

  return handler


def _config() -> types.UploadFileConfig:
  return types.UploadFileConfig(
      mime_type='text/plain',
      http_options=types.HttpOptions(headers={'X-Custom-Header': 'value'}),
  )


def test_upload_sends_config_headers():
  seen = []
  transport = httpx.MockTransport(_handler(seen))
  client = Client(
      api_key='test-key',
      http_options=types.HttpOptions(
          httpx_client=httpx.Client(transport=transport)
      ),
  )
  config = _config()

  file = client.files.upload(file=io.BytesIO(b'hello'), config=config)

  assert file.name == 'files/abc'
  assert len(seen) == 2
  for request in seen:
    assert request.headers.get('x-custom-header') == 'value'
  # The caller's config is not changed.
  assert config.http_options.headers == {'X-Custom-Header': 'value'}


@pytest.mark.asyncio
async def test_async_upload_sends_config_headers():
  seen = []
  transport = httpx.MockTransport(_handler(seen))
  client = Client(
      api_key='test-key',
      http_options=types.HttpOptions(
          httpx_async_client=httpx.AsyncClient(transport=transport)
      ),
  )
  config = _config()

  file = await client.aio.files.upload(
      file=io.BytesIO(b'hello'), config=config
  )

  assert file.name == 'files/abc'
  assert len(seen) == 2
  for request in seen:
    assert request.headers.get('x-custom-header') == 'value'
  assert config.http_options.headers == {'X-Custom-Header': 'value'}
