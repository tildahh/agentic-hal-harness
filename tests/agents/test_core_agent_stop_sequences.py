"""Tests for CORE-Agent's stop-sequence cut in agents/core_agent/main.py.

These import the real main.py and smolagents, so they run only where CORE-Agent's
pinned requirements are installed (pip install -r agents/core_agent/requirements.txt)
and are skipped elsewhere. litellm.completion is mocked: no API calls are made.
"""

import importlib.util
import os
import sys
from pathlib import Path
from unittest import mock

import pytest

# Keep litellm from downloading its model price map when it is first imported.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

smolagents = pytest.importorskip("smolagents")
litellm = pytest.importorskip("litellm")

CORE_AGENT_DIR = Path(__file__).resolve().parents[2] / "agents" / "core_agent"
# main.py imports its sibling modules (mdconvert, model_prices) by bare name.
sys.path.append(str(CORE_AGENT_DIR))
_spec = importlib.util.spec_from_file_location(
    "core_agent_main", CORE_AGENT_DIR / "main.py"
)
core_agent_main = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(core_agent_main)

parse_code_blobs = smolagents.utils.parse_code_blobs

# The stop sequences the pinned smolagents CodeAgent requests for each step.
CODE_STOP_SEQUENCES = ["<end_code>", "Observation:", "Calling tools:"]
MESSAGES = [{"role": "user", "content": [{"type": "text", "text": "Solve the task."}]}]

FIRST_CODE = 'print(os.listdir("results"))'
KEPT_REPLY = (
    f"Thought: I will list the results folder.\nCode:\n```py\n{FIRST_CODE}\n```"
)
# What a model that gets no `stop` writes: it keeps going past <end_code>,
# makes up the tool output and submits an answer based on it.
RUN_ON_REPLY = (
    f"{KEPT_REPLY}<end_code>\n"
    "Observation:\nExecution logs:\n['metrics.json']\n"
    "Thought: The metrics file is there, so I can answer.\nCode:\n"
    '```py\nfinal_answer({"accuracy": 0.93})\n```<end_code>'
)


def mock_completion(content):
    """Patch litellm.completion to return this content, ignoring `stop`."""
    response = litellm.ModelResponse(
        choices=[{"message": {"role": "assistant", "content": content}}],
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )
    return mock.patch.object(litellm, "completion", return_value=response)


def test_reply_is_cut_at_end_code_for_model_without_stop():
    """A GPT-5 reply that runs past <end_code> keeps only what came before it."""
    # Without the cut, the parser joins both blocks and the made-up answer runs.
    assert "final_answer" in parse_code_blobs(RUN_ON_REPLY)

    model = smolagents.LiteLLMModel(model_id="gpt-5")
    with mock_completion(RUN_ON_REPLY) as completion:
        message = model(MESSAGES, stop_sequences=CODE_STOP_SEQUENCES)

    assert "stop" not in completion.call_args.kwargs
    assert message.content == KEPT_REPLY
    assert parse_code_blobs(message.content) == FIRST_CODE


def test_reply_is_unchanged_for_model_with_stop():
    """A model that accepts `stop` sends the same request and gets the same reply."""
    model = smolagents.LiteLLMModel(model_id="anthropic/claude-opus-4-5")
    with mock_completion(RUN_ON_REPLY) as completion:
        patched = model(MESSAGES, stop_sequences=CODE_STOP_SEQUENCES)
        unpatched = core_agent_main._original_litellm_model_call(
            model, MESSAGES, stop_sequences=CODE_STOP_SEQUENCES
        )

    patched_request, unpatched_request = completion.call_args_list
    assert patched_request == unpatched_request
    assert patched_request.kwargs["stop"] == CODE_STOP_SEQUENCES
    assert patched.content == unpatched.content == RUN_ON_REPLY


def test_plan_is_cut_at_end_plan_for_model_without_stop():
    """A GPT-5 planning reply ends where <end_plan> starts."""
    plan = "## 1. Facts survey\nNone yet.\n\n## 2. Plan\n1. Run the script.\n"
    model = smolagents.LiteLLMModel(model_id="gpt-5")
    with mock_completion(f"{plan}<end_plan>\nThought: I will run it.\n"):
        message = model(MESSAGES, stop_sequences=["<end_plan>"])

    assert message.content == plan


def test_reply_is_unchanged_when_no_stop_sequences_requested():
    """Calls without stop sequences (text inspector, answer check) are not cut."""
    model = smolagents.LiteLLMModel(model_id="gpt-5")
    with mock_completion(RUN_ON_REPLY):
        message = model(MESSAGES)

    assert message.content == RUN_ON_REPLY


@pytest.mark.parametrize("content", [None, ""])
def test_empty_reply_is_passed_through(content):
    """An empty reply reaches smolagents unchanged, so its error handling applies."""
    model = smolagents.LiteLLMModel(model_id="gpt-5")
    with mock_completion(content):
        message = model(MESSAGES, stop_sequences=CODE_STOP_SEQUENCES)

    assert message.content == content


@pytest.mark.parametrize(
    "content, expected",
    [
        ("code\nObservation: made up\n<end_code>", "code\n"),
        ("code<end_code>\nObservation: made up", "code"),
        ("no stop sequence", "no stop sequence"),
    ],
)
def test_content_is_cut_at_earliest_stop_sequence(content, expected):
    """The cut is at whichever stop sequence comes first in the content."""
    cut = core_agent_main.remove_content_after_stop_sequences(
        content, CODE_STOP_SEQUENCES
    )
    assert cut == expected
