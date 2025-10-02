import asyncio
from types import SimpleNamespace

import pandas as pd
import pytest
import tiktoken

from graphrag.data_model.entity import Entity
from graphrag.query.context_builder.builders import ContextBuilderResult, LocalContextBuilder
from graphrag.query.structured_search.causal_search.search import CausalSearch


class FakeChatModel:
    """Minimal chat model stub recording prompts and returning canned responses."""

    def __init__(self, responses: list[str], stream_chunks: list[str] | None = None):
        self.responses = responses
        self.stream_chunks = stream_chunks or []
        self.achat_calls: list[dict] = []
        self.achat_stream_calls: list[dict] = []
        self.config = SimpleNamespace()

    async def achat(self, prompt: str, history: list | None = None, **kwargs):
        self.achat_calls.append({"prompt": prompt, "history": history or []})
        content = self.responses.pop(0)
        return SimpleNamespace(output=SimpleNamespace(content=content))

    async def achat_stream(self, prompt: str, history: list | None = None, **kwargs):
        self.achat_stream_calls.append({"prompt": prompt, "history": history or []})
        for chunk in self.stream_chunks:
            await asyncio.sleep(0)
            yield chunk


class RecordingContextBuilder(LocalContextBuilder):
    """Context builder stub that captures build parameters and returns static tables."""

    def __init__(self):
        self.build_calls: list[dict] = []
        self._entities_df = pd.DataFrame(
            [{"id": "1", "entity": "Water Flow", "description": "Water entity"}]
        )
        self._sources_df = pd.DataFrame(
            [{"id": "s-1", "text": "Hydrology report excerpt."}]
        )
        self.context_chunks = "chunk"

        entity = Entity(
            id="1",
            title="Water Flow",
            short_id="E1",
            description="Water entity",
            rank=5,
            text_unit_ids=["s-1"],
        )
        self.entities = {entity.id: entity}
        self.relationships = {}
        self.text_units = {}
        self.community_reports = {}

    def build_context(self, query: str, **kwargs) -> ContextBuilderResult:
        self.build_calls.append({"query": query, "kwargs": kwargs})
        context_records = {
            "entities": self._entities_df.copy(),
            "sources": self._sources_df.copy(),
        }
        return ContextBuilderResult(
            context_chunks=self.context_chunks,
            context_records=context_records,
            llm_calls=0,
            prompt_tokens=42,
            output_tokens=0,
        )


@pytest.mark.asyncio
async def test_causal_search_passes_entity_hints_and_includes_sources():
    long_report = "TOKEN " * 2200
    original_token_count = long_report.count("TOKEN")
    model = FakeChatModel(responses=[long_report, "Final response"])
    context_builder = RecordingContextBuilder()
    encoder = tiktoken.get_encoding("cl100k_base")

    searcher = CausalSearch(
        model=model,
        context_builder=context_builder,
        token_encoder=encoder,
        model_params={},
        context_builder_params={"top_k_mapped_entities": 1},
        s_parameter=1,
        max_context_tokens=200,
    )

    result = await searcher.search("What drives seasonal flooding?")

    assert result.response == "Final response"
    assert "context_build" in result.llm_calls_categories
    assert result.llm_calls_categories["causal_discovery"] == 1
    assert result.llm_calls_categories["response_generation"] == 1

    assert context_builder.build_calls[0]["query"] == "What drives seasonal flooding?"
    second_kwargs = context_builder.build_calls[1]["kwargs"]
    assert "include_entity_names" in second_kwargs
    assert "Water Flow" in second_kwargs["include_entity_names"]

    discovery_prompt = model.achat_calls[0]["history"][0]["content"]
    assert "Hydrology report excerpt." in discovery_prompt

    summary_prompt = model.achat_calls[1]["history"][0]["content"]
    summary_token_count = summary_prompt.count("TOKEN")
    assert summary_token_count < original_token_count
    summary_limit = max(1500, int(searcher.max_context_tokens * 0.5))
    assert summary_token_count <= summary_limit


@pytest.mark.asyncio
async def test_stream_search_streams_llm_tokens():
    model = FakeChatModel(responses=["Stage report"], stream_chunks=["alpha", "beta"])
    context_builder = RecordingContextBuilder()
    encoder = tiktoken.get_encoding("cl100k_base")

    searcher = CausalSearch(
        model=model,
        context_builder=context_builder,
        token_encoder=encoder,
        model_params={},
        context_builder_params={"top_k_mapped_entities": 1},
        s_parameter=1,
        max_context_tokens=200,
    )

    chunks = []
    async for piece in searcher.stream_search("Explain causal map"):
        chunks.append(piece)

    assert chunks == ["alpha", "beta"]
    assert model.achat_stream_calls  # streaming interface invoked
