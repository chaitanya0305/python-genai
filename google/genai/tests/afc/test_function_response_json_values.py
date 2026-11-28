# Copyright 2025 Google LLC
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

"""Tests that tool results with dates, UUIDs, enums and bytes can be sent."""

import datetime
import enum
import json
import uuid

import httpx
import pydantic
import pytest

from ... import _common
from ... import client as client_lib
from ... import types


class _Status(enum.Enum):
  SHIPPED = 'shipped'


class _Order(pydantic.BaseModel):
  id: uuid.UUID
  placed_on: datetime.date
  status: _Status


_ORDER_ID = uuid.UUID('12345678-1234-5678-1234-567812345678')


@pytest.mark.parametrize(
    'value, expected',
    [
        (datetime.date(2026, 1, 2), '2026-01-02'),
        (datetime.time(3, 4), '03:04:00'),
        (datetime.datetime(2026, 1, 2, 3, 4), '2026-01-02T03:04:00'),
        (_ORDER_ID, str(_ORDER_ID)),
        (_Status.SHIPPED, 'shipped'),
        ([b'\x00\x01', b'\x02'], ['AAE=', 'Ag==']),
        ([datetime.date(2026, 1, 2), None], ['2026-01-02', None]),
        ([[b'\x00\x01']], [['AAE=']]),
        ((b'\x00\x01', 1), ['AAE=', 1]),
    ],
)
def test_encode_unserializable_types_values(value, expected):
  encoded = _common.encode_unserializable_types({'response': {'v': value}})

  assert encoded == {'response': {'v': expected}}
  json.dumps(encoded)


def _mock_client(handler):
  return client_lib.Client(
      api_key='test-api-key',
      http_options=types.HttpOptions(
          httpx_client=httpx.Client(transport=httpx.MockTransport(handler))
      ),
  )


def test_chat_afc_sends_tool_result_with_date_uuid_and_enum():
  requests = []

  def handler(request):
    requests.append(json.loads(request.content))
    if len(requests) == 1:
      part = {'functionCall': {'name': 'get_order', 'args': {}, 'id': 'c1'}}
    else:
      part = {'text': 'Your order shipped.'}
    return httpx.Response(
        200,
        json={
            'candidates': [{
                'content': {'role': 'model', 'parts': [part]},
                'finishReason': 'STOP',
            }]
        },
    )

  def get_order() -> _Order:
    """Returns the latest order."""
    return _Order(
        id=_ORDER_ID,
        placed_on=datetime.date(2026, 1, 2),
        status=_Status.SHIPPED,
    )

  chat = _mock_client(handler).chats.create(
      model='gemini-2.5-flash',
      config=types.GenerateContentConfig(tools=[get_order]),
  )
  response = chat.send_message('Where is my order?')

  assert response.text == 'Your order shipped.'
  assert len(requests) == 2
  function_response = requests[1]['contents'][-1]['parts'][0][
      'functionResponse'
  ]
  assert function_response['response'] == {
      'result': {
          'id': str(_ORDER_ID),
          'placed_on': '2026-01-02',
          'status': 'shipped',
      }
  }
