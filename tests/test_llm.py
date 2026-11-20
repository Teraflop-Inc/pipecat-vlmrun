from pipecat_vlmrun import VLMRunLLMService, gateway_messages
from pipecat_vlmrun.llm import DEFAULT_BASE_URL, DEFAULT_MODEL

SYSTEM = {"role": "system", "content": "You are helpful."}
USER = {"role": "user", "content": "Hi."}
ASSISTANT = {"role": "assistant", "content": "Hello."}


def test_a_plain_conversation_is_unchanged():
    messages = [SYSTEM, USER, ASSISTANT, USER]
    assert gateway_messages(messages) == messages


def test_a_second_leading_system_message_is_merged_into_the_first():
    shaped = gateway_messages([SYSTEM, {"role": "system", "content": "The user is Sam."}, USER])
    assert shaped == [{"role": "system", "content": "You are helpful.\n\nThe user is Sam."}, USER]


def test_a_later_system_message_becomes_a_user_message():
    note = {"role": "system", "content": "The user is Sam."}
    shaped = gateway_messages([SYSTEM, USER, ASSISTANT, note, USER])
    assert shaped == [
        SYSTEM,
        USER,
        ASSISTANT,
        {"role": "user", "content": "The user is Sam."},
        USER,
    ]


def test_a_system_prompt_alone_gets_a_placeholder_user_message():
    assert gateway_messages([SYSTEM]) == [SYSTEM, {"role": "user", "content": "(empty)"}]


def test_no_messages_gets_a_placeholder_user_message():
    assert gateway_messages([]) == [{"role": "user", "content": "(empty)"}]


def test_a_leading_system_message_with_image_parts_is_not_merged():
    parts = {"role": "system", "content": [{"type": "text", "text": "Look."}]}
    shaped = gateway_messages([SYSTEM, parts, USER])
    assert shaped == [SYSTEM, {**parts, "role": "user"}, USER]


def test_the_input_is_not_modified():
    messages = [SYSTEM, {"role": "system", "content": "More."}]
    before = [dict(message) for message in messages]
    gateway_messages(messages)
    assert messages == before


def test_service_defaults():
    llm = VLMRunLLMService(api_key="test-key")
    assert llm._settings.model == DEFAULT_MODEL
    assert str(llm._client.base_url).rstrip("/") == DEFAULT_BASE_URL
    assert llm._client.default_headers["X-VLMRun-Client"].startswith("pipecat-vlmrun/")
    assert llm.supports_developer_role is False, "Pipecat turns developer messages into user ones"


def test_service_settings_override_the_default_model():
    llm = VLMRunLLMService(
        api_key="test-key",
        settings=VLMRunLLMService.Settings(model="google/gemma4-26b-a4b-it"),
    )
    assert llm._settings.model == "google/gemma4-26b-a4b-it"


def test_request_messages_are_reshaped():
    llm = VLMRunLLMService(api_key="test-key")
    # Pipecat's adapter puts the system instruction first, ahead of the context's own messages.
    from_context = {
        "messages": [SYSTEM, {"role": "system", "content": "The user is Sam."}],
        "tools": [],
        "tool_choice": "none",
    }
    params = llm.build_chat_completion_params(from_context)
    assert params["messages"] == [
        {"role": "system", "content": "You are helpful.\n\nThe user is Sam."},
        {"role": "user", "content": "(empty)"},
    ]
    assert params["model"] == DEFAULT_MODEL
