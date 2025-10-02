import asyncio
import json
from types import SimpleNamespace

import pandas as pd
import pytest
import tiktoken

from graphrag.data_model.entity import Entity
from graphrag.query.context_builder.builders import ContextBuilderResult, LocalContextBuilder
from graphrag.query.context_builder.conversation_history import (
    ConversationHistory,
    ConversationRole,
)
from graphrag.query.structured_search.causal_search.search import (
    CausalSearch,
    CausalSearchError,
)


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


class EmptyResponseModel(FakeChatModel):
    """Chat model stub that returns empty outputs to trigger error handling."""

    def __init__(self):
        super().__init__(responses=[])

    async def achat(self, prompt: str, history: list | None = None, **kwargs):
        self.achat_calls.append({"prompt": prompt, "history": history or []})
        return SimpleNamespace(output=SimpleNamespace(content=""))


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

    def build_context(
        self,
        query: str,
        conversation_history=None,
        **kwargs,
    ) -> ContextBuilderResult:
        call_kwargs = dict(kwargs)
        call_kwargs["conversation_history"] = conversation_history
        self.build_calls.append({"query": query, "kwargs": call_kwargs})
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


class RichContextBuilder(RecordingContextBuilder):
    """Context builder with multiple entities for section testing."""

    def __init__(self):
        super().__init__()
        self._entities_df = pd.DataFrame(
            [
                {"id": str(idx), "entity": f"Entity {idx}", "description": f"Desc {idx}"}
                for idx in range(1, 6)
            ]
        )
        self.context_chunks = "rich chunk"

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


@pytest.mark.asyncio
async def test_causal_search_includes_conversation_history():
    model = FakeChatModel(responses=["Report", "Answer"])
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

    history = ConversationHistory()
    history.add_turn(ConversationRole.USER, "Previous question")
    history.add_turn(ConversationRole.ASSISTANT, "Previous answer")

    await searcher.search("New question", conversation_history=history)

    for call in context_builder.build_calls:
        assert call["kwargs"]["conversation_history"] is history


@pytest.mark.asyncio
async def test_causal_search_output_files_include_trim_markers(tmp_path):
    long_report = "TOKEN " * 2200
    model = FakeChatModel(responses=[long_report, "Final response"])
    context_builder = RecordingContextBuilder()
    encoder = tiktoken.get_encoding("cl100k_base")

    searcher = CausalSearch(
        model=model,
        context_builder=context_builder,
        token_encoder=encoder,
        model_params={},
        context_builder_params={
            "top_k_mapped_entities": 1,
            "save_network_data": True,
            "save_causal_report": True,
            "output_folder": "causal_test",
            "output_base_dir": str(tmp_path),
        },
        s_parameter=1,
        max_context_tokens=200,
        final_context_config={
            "include_causal_summary": True,
            "include_network_sections": ["entities"],
            "section_token_ratios": {"summary": 0.4, "entities": 0.6},
            "section_row_limits": {"entities": 1},
            "max_final_context_tokens": 200,
        },
    )

    searcher._generate_query_id = lambda _: "case"  # type: ignore[assignment]

    await searcher.search("Explain causes")

    output_dir = tmp_path / "causal_test"
    network_file = output_dir / "causal_search_network_data_case.json"
    report_file = output_dir / "causal_search_report_case.md"

    assert network_file.exists()
    assert report_file.exists()

    network_payload = json.loads(network_file.read_text())
    assert "_trimmed_meta" in network_payload
    assert "payload" in network_payload
    assert "_final_context_meta" in network_payload
    final_meta = network_payload["_final_context_meta"]
    assert "summary" in final_meta
    assert "sections" in final_meta

    report_text = report_file.read_text()
    assert "Trimmed from" in report_text


@pytest.mark.asyncio
async def test_causal_search_raises_when_model_returns_no_content():
    model = EmptyResponseModel()
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

    with pytest.raises(CausalSearchError):
        await searcher.search("Why")


@pytest.mark.asyncio
async def test_final_context_includes_entities_section_with_row_cap():
    model = FakeChatModel(responses=["TOKEN " * 300, "Answer"])
    context_builder = RichContextBuilder()
    encoder = tiktoken.get_encoding("cl100k_base")

    final_context_config = {
        "include_causal_summary": True,
        "include_network_sections": ["entities"],
        "section_token_ratios": {"summary": 0.4, "entities": 0.6},
        "section_row_limits": {"entities": 2},
        "max_final_context_tokens": 300,
    }

    searcher = CausalSearch(
        model=model,
        context_builder=context_builder,
        token_encoder=encoder,
        model_params={},
        context_builder_params={"top_k_mapped_entities": 1},
        s_parameter=1,
        max_context_tokens=300,
        final_context_config=final_context_config,
    )

    await searcher.search("Check entities")

    summary_prompt = model.achat_calls[1]["history"][0]["content"]
    assert "-----Entities-----" in summary_prompt
    assert "Entity 1" in summary_prompt
    assert "Entity 2" in summary_prompt
    assert "Entity 3" not in summary_prompt


@pytest.mark.asyncio
async def test_ensure_full_summary_preserves_complete_report():
    summary_text = "TOKEN " * 50
    model = FakeChatModel(responses=[summary_text, "Answer"])
    context_builder = RichContextBuilder()
    encoder = tiktoken.get_encoding("cl100k_base")

    final_context_config = {
        "include_causal_summary": True,
        "ensure_full_summary": True,
        "include_network_sections": ["entities"],
        "section_token_ratios": {"summary": 0.1, "entities": 0.9},
        "section_row_limits": {"entities": 1},
        "max_final_context_tokens": 400,
    }

    searcher = CausalSearch(
        model=model,
        context_builder=context_builder,
        token_encoder=encoder,
        model_params={},
        context_builder_params={"top_k_mapped_entities": 1},
        s_parameter=1,
        max_context_tokens=400,
        final_context_config=final_context_config,
    )

    await searcher.search("Ensure summary")

    summary_prompt = model.achat_calls[1]["history"][0]["content"]
    summary_block = summary_prompt.split("-----Entities-----")[0]
    assert summary_block.count("TOKEN") == summary_text.count("TOKEN")
