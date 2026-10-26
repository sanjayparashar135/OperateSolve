"""Context budgeting, sliding-window history, and conversation summarization."""

from __future__ import annotations

import pytest

from app.db.database import init_db
from app.db.repository import Repository
from app.services.chat_service import INSTRUCTIONS
from app.services.context.budget import ContextBudgetExceeded, resolve_budget
from app.services.context.tokens import estimate_tokens
from app.services.context.window import HistoryMessage, SUMMARY_PREFIX, build_context_plan
from app.services.llm.registry import MODEL_SPECS
from app.main import create_app
from tests.conftest import make_service, make_settings
from tests.fakes import FakeProvider
from tests.test_streaming import parse_sse


def _msg(position: int, role: str, content: str, message_id: str | None = None) -> HistoryMessage:
    return HistoryMessage(
        id=message_id or f"m{position}",
        position=position,
        role=role,  # type: ignore[arg-type]
        content=content,
    )


def _long(text: str, repeats: int) -> str:
    return (text + " ") * repeats


def _tight_settings(tmp_path, **overrides):
    values = {
        "context_max_input_tokens": 260,
        "context_token_safety_margin": 1.0,
        "context_reasoning_reserve_tokens": 0,
        "summarization_trigger_excluded_tokens": 30,
        "summarization_summary_budget_tokens": 40,
        "openai_max_output_tokens": 32,
    }
    values.update(overrides)
    return make_settings(tmp_path, **values)


async def _seed_long_chat(service, chat_id: str, *, rounds: int = 6, tag: str = "seed") -> None:
    blob = _long("runbook continuity detail", 35)
    for index in range(rounds):
        prepared = service.prepare_new_message(chat_id, f"{blob}-{index}", f"{tag}-{index}")
        async for _ in service.generate(prepared, None, "op"):
            pass


def test_short_conversation_keeps_full_history(tmp_path):
    settings = make_settings(tmp_path, context_max_input_tokens=24_000)
    luna = next(spec for spec in MODEL_SPECS if spec.key == "luna")
    budget = resolve_budget(luna, settings)
    messages = [
        _msg(1, "user", "Hello"),
        _msg(2, "assistant", "Hi"),
        _msg(3, "user", "How are you?"),
    ]
    plan = build_context_plan(instructions=INSTRUCTIONS, messages=messages, budget=budget)
    assert plan.truncated is False
    assert [message.content for message in plan.retained] == ["Hello", "Hi"]
    assert plan.excluded == ()
    assert plan.current_user.content == "How are you?"
    assert [turn.content for turn in plan.provider_turns()] == ["Hello", "Hi", "How are you?"]


def test_sliding_window_keeps_newest_complete_turns(tmp_path):
    settings = make_settings(
        tmp_path,
        context_max_input_tokens=70,
        context_token_safety_margin=1.0,
        context_reasoning_reserve_tokens=0,
        summarization_summary_budget_tokens=0,
        openai_max_output_tokens=16,
    )
    luna = next(spec for spec in MODEL_SPECS if spec.key == "luna")
    budget = resolve_budget(luna, settings)
    messages = [
        _msg(1, "user", _long("aaaaaaaa", 12)),
        _msg(2, "assistant", _long("bbbbbbbb", 12)),
        _msg(3, "user", "recent user"),
        _msg(4, "assistant", "recent assistant"),
        _msg(5, "user", "latest"),
    ]
    plan = build_context_plan(instructions="Sys.", messages=messages, budget=budget)
    assert plan.truncated is True
    assert {message.id for message in plan.excluded} >= {"m1", "m2"}
    assert [message.id for message in plan.retained] == ["m3", "m4"]
    assert plan.current_user.content == "latest"


def test_never_splits_a_complete_turn(tmp_path):
    settings = make_settings(tmp_path, context_max_input_tokens=180, context_token_safety_margin=1.0)
    luna = next(spec for spec in MODEL_SPECS if spec.key == "luna")
    budget = resolve_budget(luna, settings)
    messages = [
        _msg(1, "user", _long("old", 30)),
        _msg(2, "assistant", _long("old-answer", 30)),
        _msg(3, "user", _long("recent-user", 20)),
        _msg(4, "assistant", _long("recent-assistant", 20)),
        _msg(5, "user", "now"),
    ]
    plan = build_context_plan(instructions="Sys.", messages=messages, budget=budget)
    ids = {message.id for message in plan.retained}
    # Either both messages of a turn are kept, or neither.
    assert ("m3" in ids) == ("m4" in ids)
    assert ("m1" in ids) == ("m2" in ids)


def test_oversized_current_user_message_errors(tmp_path):
    settings = make_settings(
        tmp_path,
        context_max_input_tokens=40,
        context_token_safety_margin=1.0,
        context_reasoning_reserve_tokens=0,
        openai_max_output_tokens=16,
    )
    luna = next(spec for spec in MODEL_SPECS if spec.key == "luna")
    budget = resolve_budget(luna, settings)
    huge = "x" * 2_000
    with pytest.raises(ContextBudgetExceeded) as caught:
        build_context_plan(
            instructions="System.",
            messages=[_msg(1, "user", huge)],
            budget=budget,
        )
    assert "too large" in str(caught.value).lower()


def test_model_switch_changes_budget(tmp_path):
    settings = make_settings(tmp_path, context_max_input_tokens=1_000_000)
    luna = next(spec for spec in MODEL_SPECS if spec.key == "luna")
    nano = next(spec for spec in MODEL_SPECS if spec.key == "nano")
    luna_budget = resolve_budget(luna, settings)
    nano_budget = resolve_budget(nano, settings)
    assert luna_budget.input_budget > nano_budget.input_budget
    assert nano_budget.input_budget <= nano.max_input_tokens


@pytest.mark.asyncio
async def test_summarization_threshold_and_incremental_update(tmp_path):
    settings = _tight_settings(tmp_path)
    init_db(settings.database_path)
    repo = Repository(settings.database_path)
    provider = FakeProvider()
    provider.summary_text = "Facts: alpha\nGoals/preferences: none\nDecisions/constraints: none\nOpen questions/commitments: none\nTechnical details: none"
    service = make_service(settings, provider, repo)
    chat = repo.create_conversation(None)

    await _seed_long_chat(service, chat["id"], rounds=6, tag="seed")

    provider.complete_calls.clear()
    provider.summary_text = "Facts: merged-one\nGoals/preferences: none\nDecisions/constraints: none\nOpen questions/commitments: none\nTechnical details: none"
    prepared = service.prepare_new_message(chat["id"], "What next?", "ask-1")
    async for _ in service.generate(prepared, None, "op"):
        pass

    assert len(provider.complete_calls) == 1
    summary = repo.get_summary(chat["id"])
    assert summary is not None
    assert "merged-one" in summary["summary_text"]
    checkpoint = summary["covered_through_position"]

    # Next turn with room for the same window reuses the summary (no second call).
    provider.complete_calls.clear()
    provider.chunks = ["Follow ", "up"]
    prepared = service.prepare_new_message(chat["id"], "And then?", "ask-2")
    events = [event async for event in service.generate(prepared, None, "op")]
    assert len(provider.complete_calls) == 0
    assert repo.get_summary(chat["id"])["covered_through_position"] == checkpoint
    assert any(SUMMARY_PREFIX in turn.content for turn in provider.calls[-1]["turns"])
    assert "data: " in events[0]

    # Budget just above instructions+summary+current, but below the prior turn's retained pair.
    settings_tight = _tight_settings(
        tmp_path,
        database_path=settings.database_path,
        context_max_input_tokens=220,
        summarization_trigger_excluded_tokens=10,
        summarization_summary_budget_tokens=40,
    )
    provider.summary_text = "Facts: merged-two\nGoals/preferences: none\nDecisions/constraints: none\nOpen questions/commitments: none\nTechnical details: none"
    service = make_service(settings_tight, provider, repo)
    provider.complete_calls.clear()
    prepared = service.prepare_new_message(chat["id"], "Merge please", "ask-3")
    async for _ in service.generate(prepared, None, "op"):
        pass
    assert len(provider.complete_calls) == 1
    prompt = provider.complete_calls[0]["turns"][0].content
    assert "Previous summary:" in prompt
    assert "merged-one" in prompt
    assert "Newly excluded" in prompt
    updated = repo.get_summary(chat["id"])
    assert "merged-two" in updated["summary_text"]
    assert updated["covered_through_position"] >= checkpoint


@pytest.mark.asyncio
async def test_summarization_failure_falls_back_without_changing_chat_model(tmp_path):
    settings = _tight_settings(tmp_path)
    init_db(settings.database_path)
    repo = Repository(settings.database_path)
    provider = FakeProvider()
    provider.summary_fail = True
    service = make_service(settings, provider, repo)
    chat = repo.create_conversation(None)
    await _seed_long_chat(service, chat["id"], rounds=6, tag="fail-seed")

    provider.complete_calls.clear()
    provider.calls.clear()
    prepared = service.prepare_new_message(chat["id"], "Continue", "fail-ask")
    events = parse_sse("".join([chunk async for chunk in service.generate(prepared, None, "op")]))
    assert events[-1]["type"] == "done"
    assert provider.calls[-1]["model"] == "gpt-6-luna"
    assert repo.get_summary(chat["id"]) is None
    assert not any(SUMMARY_PREFIX in turn.content for turn in provider.calls[-1]["turns"])
    usage = repo.list_usage()
    assert any(row["purpose"] == "summarization" and row["status"] == "error" for row in usage)


@pytest.mark.asyncio
async def test_summarization_usage_is_recorded_separately(tmp_path):
    tight = _tight_settings(tmp_path)
    provider = FakeProvider()
    app = create_app(tight, provider=provider)
    from fastapi.testclient import TestClient

    blob = _long("usage seed continuity", 35)
    with TestClient(app) as test_client:
        chat_id = test_client.post("/api/chats", json={}).json()["id"]
        for index in range(6):
            test_client.post(
                f"/api/chats/{chat_id}/messages",
                json={"content": f"{blob}-{index}", "client_request_id": f"usage-seed-{index}"},
            )
        test_client.post(
            f"/api/chats/{chat_id}/messages",
            json={"content": "Summarize path", "client_request_id": "usage-ask"},
        )
        report = test_client.get("/api/usage").json()

    purposes = {row["purpose"] for row in report["recent"]}
    assert "chat" in purposes
    assert "summarization" in purposes
    summary_rows = [row for row in report["recent"] if row["purpose"] == "summarization"]
    assert summary_rows
    assert summary_rows[0]["usage_reported"] is True


@pytest.mark.asyncio
async def test_retry_and_idempotency_with_context_management(tmp_path):
    settings = _tight_settings(tmp_path)
    init_db(settings.database_path)
    repo = Repository(settings.database_path)
    provider = FakeProvider()
    service = make_service(settings, provider, repo)
    chat = repo.create_conversation(None)
    await _seed_long_chat(service, chat["id"], rounds=6, tag="idemp-seed")

    prepared = service.prepare_new_message(chat["id"], "Idempotent ask", "idemp-ask")
    async for _ in service.generate(prepared, None, "op"):
        pass
    first_prompt = provider.calls[-1]
    complete_count = len(provider.complete_calls)

    replay = service.prepare_new_message(chat["id"], "Idempotent ask", "idemp-ask")
    assert replay.replay is True
    async for _ in service.replay(replay):
        pass
    assert len(provider.calls) == len([c for c in provider.calls])  # unchanged by replay
    assert len(provider.complete_calls) == complete_count

    provider.fail = True
    prepared = service.prepare_new_message(chat["id"], "Retry me", "idemp-retry")
    events = parse_sse("".join([chunk async for chunk in service.generate(prepared, None, "op")]))
    assert events[-1]["type"] == "error"
    failed_id = events[-1]["message"]["id"]
    failed_prompt = provider.calls[-1]
    provider.fail = False
    retry = service.prepare_retry(chat["id"], failed_id)
    async for _ in service.generate(retry, None, "op"):
        pass
    assert provider.calls[-1]["turns"] == failed_prompt["turns"]
    assert provider.calls[-1]["instructions"] == failed_prompt["instructions"]
    assert first_prompt["instructions"] == INSTRUCTIONS


@pytest.mark.asyncio
async def test_summary_persists_across_service_restart(tmp_path):
    settings = _tight_settings(tmp_path)
    init_db(settings.database_path)
    repo = Repository(settings.database_path)
    provider = FakeProvider()
    provider.summary_text = "Facts: durable\nGoals/preferences: none\nDecisions/constraints: none\nOpen questions/commitments: none\nTechnical details: none"
    service = make_service(settings, provider, repo)
    chat = repo.create_conversation(None)
    await _seed_long_chat(service, chat["id"], rounds=6, tag="persist-seed")
    prepared = service.prepare_new_message(chat["id"], "Persist ask", "persist-ask")
    async for _ in service.generate(prepared, None, "op"):
        pass
    saved = repo.get_summary(chat["id"])
    assert saved is not None

    # New service instance, same SQLite file.
    provider2 = FakeProvider()
    service2 = make_service(settings, provider2, Repository(settings.database_path))
    provider2.complete_calls.clear()
    prepared = service2.prepare_new_message(chat["id"], "After restart", "persist-ask-2")
    async for _ in service2.generate(prepared, None, "op"):
        pass
    assert len(provider2.complete_calls) == 0
    assert any(SUMMARY_PREFIX in turn.content for turn in provider2.calls[-1]["turns"])
    assert "durable" in provider2.calls[-1]["turns"][0].content


def test_prompt_caching_compatible_prefix_with_summary(tmp_path):
    settings = make_settings(tmp_path, context_max_input_tokens=500, context_token_safety_margin=1.0)
    luna = next(spec for spec in MODEL_SPECS if spec.key == "luna")
    budget = resolve_budget(luna, settings)
    messages = [
        _msg(1, "user", "old"),
        _msg(2, "assistant", "ack"),
        _msg(3, "user", "mid"),
        _msg(4, "assistant", "ack2"),
        _msg(5, "user", "new"),
    ]
    summary = "Facts: old context\nGoals/preferences: none\nDecisions/constraints: none\nOpen questions/commitments: none\nTechnical details: none"
    plan = build_context_plan(
        instructions=INSTRUCTIONS,
        messages=messages,
        budget=budget,
        summary_text=summary,
        covered_through_position=2,
    )
    turns = plan.provider_turns()
    assert turns[0].role == "user"
    assert turns[0].content.startswith(SUMMARY_PREFIX)
    assert turns[1].role == "assistant"
    # No timestamps or request ids in the prompt.
    joined = "\n".join(turn.content for turn in turns)
    assert "request_id" not in joined
    assert "T00:" not in joined
    # Covered messages are not also present as raw history.
    assert all(message.position > 2 for message in plan.retained)


def test_token_estimate_is_local_and_monotonic():
    short = estimate_tokens("hi")
    long = estimate_tokens("hi " * 200)
    assert short >= 1
    assert long > short
