"""Provider selection, embeddings fallback, the default vector store, web search and MCP tools.

Each test pins a bug fixed in 1.0.0 or a behavior the new code promises. Nothing here calls
a network service: LangChain's init_chat_model, the Hugging Face embeddings and the MCP
client are all replaced by small fakes.
"""

from __future__ import annotations

import asyncio
import sys
import types
from typing import Any

import pytest
from langchain_core.tools import BaseTool, StructuredTool

from agentic_architectures.config import Settings
from agentic_architectures.llm import factory
from agentic_architectures.llm.factory import get_embeddings as real_get_embeddings  # before the autouse stub
from agentic_architectures.llm.providers import PROVIDERS


def _settings(monkeypatch: pytest.MonkeyPatch, **values: Any) -> Settings:
    """Settings built from ``values`` only (no .env, no environment), installed everywhere."""
    for var in ("LLM_PROVIDER", "LLM_MODEL", "EMBEDDINGS_PROVIDER", "EMBEDDINGS_MODEL", "SEARCH_BACKEND"):
        monkeypatch.delenv(var, raising=False)
    s = Settings(_env_file=None, **values)  # type: ignore[call-arg]
    monkeypatch.setattr(factory, "settings", s)
    monkeypatch.setattr("agentic_architectures.tools.search.settings", s)
    return s


@pytest.fixture
def init_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Record every init_chat_model call instead of building a real model."""
    import langchain.chat_models as chat_models

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(chat_models, "init_chat_model", lambda **kw: calls.append(kw) or object())
    return calls


# --------------------------------------------------------------------- models


def test_each_provider_defaults_to_its_own_model(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _settings(monkeypatch, llm_provider="openai").llm_model == PROVIDERS["openai"].recommended_model
    assert _settings(monkeypatch, llm_provider="openai", llm_model="my-model").llm_model == "my-model"
    # The reasoning model the notebooks use follows the main model unless set.
    assert _settings(monkeypatch, llm_provider="openai", llm_model="my-model").llm_reasoning_model == "my-model"
    assert _settings(monkeypatch, llm_model="m", llm_reasoning_model="thinker").llm_reasoning_model == "thinker"


def test_another_provider_never_gets_the_configured_providers_model(monkeypatch, init_calls) -> None:
    _settings(monkeypatch, llm_provider="nebius", llm_model="some/nebius-model")
    factory.get_llm(provider="anthropic")  # 0.3.0 sent "some/nebius-model" to Anthropic
    assert init_calls[-1]["model"] == PROVIDERS["anthropic"].recommended_model
    assert init_calls[-1]["model_provider"] == "anthropic"


def test_google_uses_langchains_provider_name(monkeypatch, init_calls) -> None:
    _settings(monkeypatch, llm_provider="google")
    factory.get_llm()  # init_chat_model rejects "google": it calls the provider google_genai
    assert init_calls[-1]["model_provider"] == "google_genai"


def test_ollama_chat_uses_the_configured_base_url(monkeypatch, init_calls) -> None:
    _settings(monkeypatch, llm_provider="ollama", ollama_base_url="http://gpu-box:11434")
    factory.get_llm()
    assert init_calls[-1]["base_url"] == "http://gpu-box:11434"
    assert init_calls[-1]["num_ctx"] == 16384  # Ollama's 4,096 default silently truncates agent prompts
    factory.get_llm(base_url="http://elsewhere:11434")  # an explicit one wins
    assert init_calls[-1]["base_url"] == "http://elsewhere:11434"


def test_a_recommended_model_brings_the_settings_it_needs(monkeypatch, init_calls) -> None:
    _settings(monkeypatch, llm_provider="openai")
    factory.get_llm()  # gpt-6-luna calls tools on Chat Completions only with reasoning_effort none
    assert init_calls[-1]["reasoning_effort"] == "none"
    factory.get_llm(model="some-other-model")
    assert "reasoning_effort" not in init_calls[-1]
    factory.get_llm(reasoning_effort="high")
    assert init_calls[-1]["reasoning_effort"] == "high"


def test_a_provider_without_a_default_model_asks_for_one(monkeypatch, init_calls) -> None:
    _settings(monkeypatch, llm_provider="huggingface")
    with pytest.raises(ValueError, match="LLM_MODEL"):
        factory.get_llm()


def test_a_missing_integration_names_its_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    import langchain.chat_models as chat_models

    _settings(monkeypatch, llm_provider="mistralai")

    def missing(**_: Any) -> Any:
        raise ImportError("No module named 'langchain_mistralai'")

    monkeypatch.setattr(chat_models, "init_chat_model", missing)
    with pytest.raises(ImportError, match=r"agentic-architectures\[mistral\]"):
        factory.get_llm()


def test_an_empty_value_in_env_means_not_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """`.env.example` has KEY= for every key; empty must not count as a key or a provider."""
    monkeypatch.setenv("LLM_PROVIDER", "")
    monkeypatch.setenv("TAVILY_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.llm_provider == "nebius"
    assert s.tavily_api_key is None and s.api_key_for("openai") is None


def test_every_recommended_model_is_set() -> None:
    for name, info in PROVIDERS.items():
        assert info.recommended_model.strip(), name


# ----------------------------------------------------------------- embeddings


def test_embeddings_fall_back_to_a_local_model_for_a_provider_without_one(monkeypatch) -> None:
    _settings(monkeypatch, llm_provider="anthropic")

    class FakeHF:
        def __init__(self, model_name: str, **_: Any) -> None:
            self.model_name = model_name

    module = types.ModuleType("langchain_huggingface")
    module.HuggingFaceEmbeddings = FakeHF  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langchain_huggingface", module)
    emb = real_get_embeddings()  # 0.3.0 raised: the fallback never triggered
    assert isinstance(emb, FakeHF) and emb.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    with pytest.raises(ValueError, match="No embeddings"):
        real_get_embeddings(provider="anthropic")  # asked for by name: no silent substitute


# --------------------------------------------------------------- vector store


def test_the_default_vector_store_needs_no_extra_package() -> None:
    from langchain_core.documents import Document
    from langchain_core.embeddings.fake import DeterministicFakeEmbedding

    from agentic_architectures.memory.vector import get_vector_store

    docs = [Document(page_content=t) for t in ("comets are icy", "stars fuse hydrogen", "moons orbit planets")]
    store = get_vector_store(docs, embeddings=DeterministicFakeEmbedding(size=32), backend="memory")
    assert type(store).__name__ == "InMemoryVectorStore"
    hit = store.similarity_search("stars fuse hydrogen", k=1)[0]
    assert hit.page_content == "stars fuse hydrogen"
    empty = get_vector_store(embeddings=DeterministicFakeEmbedding(size=32), backend="memory")
    assert empty.similarity_search("anything", k=1) == []


# ----------------------------------------------------------------- web search


def test_tavily_without_a_key_fails_loudly_and_names_the_keyless_option(monkeypatch) -> None:
    pytest.importorskip("langchain_tavily")
    _settings(monkeypatch, search_backend="tavily", tavily_api_key=None)
    from agentic_architectures.tools.search import web_search_tool

    with pytest.raises(RuntimeError, match="SEARCH_BACKEND=mcp"):
        web_search_tool()


def test_the_mcp_search_backend_takes_the_servers_search_tool(monkeypatch) -> None:
    _settings(monkeypatch, search_backend="mcp", search_mcp_url="https://search.example/mcp")
    from agentic_architectures.tools import search

    def fake_mcp_tools(url: str) -> list[BaseTool]:
        assert url == "https://search.example/mcp"
        return [_tool("fetch_page"), _tool("web_search")]

    monkeypatch.setattr("agentic_architectures.tools.mcp.mcp_tools", fake_mcp_tools)
    assert search.web_search_tool().name == "web_search"


# ----------------------------------------------------------------- MCP tools


def _tool(name: str) -> StructuredTool:
    async def call(query: str) -> str:
        await asyncio.sleep(0)
        return f"{name}: {query}"

    return StructuredTool.from_function(coroutine=call, name=name, description=f"{name} tool")


def _fake_adapters(monkeypatch: pytest.MonkeyPatch, tools: list[BaseTool]) -> list[dict[str, Any]]:
    """Install a fake langchain_mcp_adapters.client; return the connections it was given."""
    seen: list[dict[str, Any]] = []

    class FakeClient:
        def __init__(self, connections: dict[str, Any]) -> None:
            seen.append(connections)

        async def get_tools(self) -> list[BaseTool]:
            return tools

    client = types.ModuleType("langchain_mcp_adapters.client")
    client.MultiServerMCPClient = FakeClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langchain_mcp_adapters", types.ModuleType("langchain_mcp_adapters"))
    monkeypatch.setitem(sys.modules, "langchain_mcp_adapters.client", client)
    return seen


def test_mcp_tools_are_callable_synchronously(monkeypatch: pytest.MonkeyPatch) -> None:
    from agentic_architectures.tools import mcp_tools

    seen = _fake_adapters(monkeypatch, [_tool("web_search")])
    [tool] = mcp_tools("https://search.example/mcp", headers={"Authorization": "Bearer k"})
    assert seen[0]["server"] == {
        "url": "https://search.example/mcp",
        "transport": "streamable_http",
        "headers": {"Authorization": "Bearer k"},
    }
    assert tool.invoke({"query": "comets"}) == "web_search: comets"
    assert asyncio.run(tool.ainvoke({"query": "moons"})) == "web_search: moons"


def test_mcp_tools_work_inside_a_running_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """A notebook has a loop running; asyncio.run inside a sync call would refuse."""
    from agentic_architectures.tools import mcp_tools

    _fake_adapters(monkeypatch, [_tool("web_search")])

    async def notebook_cell() -> str:
        [tool] = mcp_tools("https://search.example/mcp")  # sync API with a loop running
        return str(tool.invoke({"query": "stars"}))

    assert asyncio.run(notebook_cell()) == "web_search: stars"


def test_mcp_tools_by_name_and_with_none_offered(monkeypatch: pytest.MonkeyPatch) -> None:
    from agentic_architectures.tools import mcp_tools

    _fake_adapters(monkeypatch, [_tool("web_search"), _tool("fetch")])
    assert [t.name for t in mcp_tools("https://x/mcp", names=["fetch"])] == ["fetch"]
    with pytest.raises(ValueError, match="no tool named"):
        mcp_tools("https://x/mcp", names=["missing"])
    _fake_adapters(monkeypatch, [])
    with pytest.raises(ValueError, match="offers no tools"):
        mcp_tools("https://x/mcp")


# -------------------------------------------------------------- Mixture of Agents


def test_each_moa_layer_reads_only_the_previous_layer(structured_mock) -> None:
    from agentic_architectures.architectures import MixtureOfAgents

    proposal = {"refined_answer": "answer", "key_insight": "insight", "confidence_indicators": []}
    llm = structured_mock(structured_responses={"_MoAProposal": [proposal]}, text_responses=["final"])
    arch = MixtureOfAgents(llm=llm, num_layers=3, agents_per_layer=2)
    prompts: list[str] = []
    real_invoke = arch._proposer.invoke

    def recording_invoke(prompt: str, **kw: Any) -> Any:
        prompts.append(prompt)
        return real_invoke(prompt, **kw)

    arch._proposer.invoke = recording_invoke  # type: ignore[method-assign]

    def outputs(layer: int) -> list[dict[str, Any]]:
        return [
            {"layer": layer, "agent_idx": i, "persona": "p", "refined_answer": f"L{layer}-{i}", "key_insight": ""}
            for i in range(2)
        ]

    arch._layer_generate({"task": "t", "current_layer": 0, "layer_outputs": []})
    layer0, layer1 = outputs(0), outputs(1)
    prompts.clear()
    arch._layer_generate({"task": "t", "current_layer": 2, "layer_outputs": layer0 + layer1})
    assert all("L1-0" in p and "L0-0" not in p for p in prompts)  # layer 2 sees layer 1 only


def test_an_unknown_architecture_option_is_an_error(structured_mock) -> None:
    """0.3.0 stored unknown keywords and ignored them, so a misspelled option ran the default."""
    from agentic_architectures.architectures import CellularAutomata, Reflection

    llm = structured_mock()
    with pytest.raises(TypeError, match="grid_size"):
        CellularAutomata(llm=llm, rule_prompt="r", allowed_states=["a"], grid_size=(3, 3))
    with pytest.raises(TypeError, match="max_iteration"):
        Reflection(llm=llm, max_iteration=2)  # the option is max_iterations
    assert Reflection(llm=llm, max_iterations=2).max_iterations == 2


def test_swe_agent_asks_for_a_path_instead_of_failing_on_the_sandbox(structured_mock, tmp_path) -> None:
    from agentic_architectures.architectures import SWEAgent

    arch = SWEAgent(llm=structured_mock(), working_dir=tmp_path)
    for kind in ("read_file", "write_file", "run_check"):
        out = arch._execute({"last_action": {"action": kind, "path": " ", "content": "x", "answer": ""}})
        assert "needs a path" in out["observations"][0], kind
    assert list(tmp_path.iterdir()) == []  # nothing was written


def test_memgpt_rejects_an_empty_payload_and_shows_search_errors(structured_mock) -> None:
    from agentic_architectures.architectures import MemGPT

    arch = MemGPT(llm=structured_mock(), context_limit=5)
    before = arch.archival_count
    arch._execute_action({"last_action": {"action": "write_to_archival", "payload": ""}})
    arch._execute_action({"last_action": {"action": "search_archival", "payload": "  "}})
    assert arch.archival_count == before  # no empty document stored
    assert sum("needs" in c for c in arch.context_tier) == 2

    def broken(query: str, k: int = 4) -> list:
        raise RuntimeError("embeddings endpoint unreachable")

    arch.archival.search = broken  # type: ignore[method-assign]
    out = arch._execute_action({"last_action": {"action": "search_archival", "payload": "comets"}})
    assert "unreachable" in out["history"][0]["error"]
    assert any("[search error]" in c for c in arch.context_tier)  # used to be swallowed
