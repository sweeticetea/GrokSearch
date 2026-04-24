import pytest

from grok_search.providers.grok import GrokSearchProvider
from grok_search.sources import extract_sources_from_payload


class _FakeResponse:
    def __init__(self, lines: list[str]):
        self._lines = lines

    async def aiter_lines(self):
        for line in self._lines:
            yield line


@pytest.mark.asyncio
async def test_parse_streaming_response_extracts_text_and_structured_citations():
    provider = GrokSearchProvider("https://example.com/v1", "test-key")
    response = _FakeResponse(
        [
            'data: {"choices":[{"delta":{"content":"今天A股震荡走强。","citations":[{"title":"来源A","url":"https://example.com/a"}]}}]}',
            'data: {"choices":[{"delta":{"content":"两市成交额继续放大。"}}]}',
            "data: [DONE]",
        ]
    )

    content, sources = await provider._parse_streaming_response(response, include_sources=True)

    assert content == "今天A股震荡走强。两市成交额继续放大。"
    assert sources == [{"title": "来源A", "url": "https://example.com/a"}]


@pytest.mark.asyncio
async def test_parse_streaming_response_supports_responses_style_events():
    provider = GrokSearchProvider("https://example.com/v1", "test-key")
    response = _FakeResponse(
        [
            "event: response.output_text.delta",
            'data: {"type":"response.output_text.delta","delta":"今日市场上涨"}',
            "event: response.output_item.added",
            'data: {"type":"response.output_item.added","item":{"type":"message","content":[{"type":"output_text","text":"，量能有所修复"}],"annotations":[{"type":"url_citation","title":"来源B","url":"https://example.com/b"}]}}',
            "data: [DONE]",
        ]
    )

    content, sources = await provider._parse_streaming_response(response, include_sources=True)

    assert content == "今日市场上涨，量能有所修复"
    assert sources == [{"title": "来源B", "url": "https://example.com/b"}]


@pytest.mark.asyncio
async def test_parse_streaming_response_falls_back_to_plain_text_sse():
    provider = GrokSearchProvider("https://example.com/v1", "test-key")
    response = _FakeResponse(
        [
            "data: 今天A股上涨",
            "data: 市场情绪回暖",
            "data: [DONE]",
        ]
    )

    content, sources = await provider._parse_streaming_response(response, include_sources=True)

    assert content == "今天A股上涨\n市场情绪回暖"
    assert sources == []


def test_extract_sources_from_payload_handles_nested_annotations():
    payload = {
        "response": {
            "output": [
                {
                    "type": "message",
                    "annotations": [
                        {"type": "url_citation", "title": "来源C", "url": "https://example.com/c"}
                    ],
                }
            ]
        }
    }

    assert extract_sources_from_payload(payload) == [
        {"title": "来源C", "url": "https://example.com/c"}
    ]
