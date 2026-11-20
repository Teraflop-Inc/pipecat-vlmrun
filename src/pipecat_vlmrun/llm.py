#
# Copyright (c) 2026, VLM Run
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""VLM Run gateway LLM service for Pipecat.

The gateway serves vision-language models through an OpenAI-compatible Chat Completions API,
so this is a thin subclass of Pipecat's ``OpenAILLMService``. It exists because the gateway is
stricter than OpenAI about message order, and Pipecat's own examples produce messages it rejects.
"""

from dataclasses import dataclass

from pipecat.adapters.llm.services.open_ai_adapter import OpenAILLMInvocationParams
from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.services.openai.llm import OpenAILLMService

from pipecat_vlmrun.version import CLIENT_HEADER, client_name

DEFAULT_BASE_URL = "https://gateway.vlm.run/v1/openai"
DEFAULT_MODEL = "qwen/qwen3.8-27b"

# The gateway needs a user message in every request. This stands in for one when a bot is
# started on a system prompt alone.
_EMPTY_USER_TURN = "(empty)"


@dataclass
class VLMRunLLMSettings(BaseOpenAILLMService.Settings):
    """Settings for VLMRunLLMService.

    The fields are those of ``OpenAILLMService.Settings``. Put gateway-only request fields,
    such as ``method`` or ``chat_template_kwargs``, in ``extra["extra_body"]``.
    """


class VLMRunLLMService(OpenAILLMService):
    """Chat completions on the VLM Run gateway.

    Images in the LLM context, such as camera frames added with
    ``LLMContext.create_image_message``, are sent to the model as image parts.

    The gateway rejects three message shapes that Pipecat bots produce, so this service
    reshapes them before each request:

    - A ``developer`` message becomes a ``user`` message.
    - A ``system`` message after the first is merged into the first when it directly follows
      it, and becomes a ``user`` message anywhere else.
    - A request with no ``user`` message gets a placeholder one.

    Example::

        llm = VLMRunLLMService(
            api_key=os.environ["VLMRUN_API_KEY"],
            settings=VLMRunLLMService.Settings(
                model="qwen/qwen3.8-27b",
                system_instruction="You are a helpful assistant.",
            ),
        )
    """

    # Pipecat converts "developer" messages to "user" for services that set this to False.
    supports_developer_role = False

    Settings = VLMRunLLMSettings
    _settings: Settings
    _requires_api_key = True

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the VLM Run LLM service.

        Args:
            api_key: VLM Run API key, from https://app.vlm.run.
            base_url: The gateway's OpenAI-compatible endpoint.
            settings: Runtime-updatable settings. The model defaults to ``qwen/qwen3.8-27b``.
            **kwargs: Additional arguments passed to ``OpenAILLMService``.
        """
        default_settings = self.Settings(model=DEFAULT_MODEL)
        if settings is not None:
            default_settings.apply_update(settings)

        headers = {CLIENT_HEADER: client_name(), **(kwargs.pop("default_headers", None) or {})}

        super().__init__(
            api_key=api_key,
            base_url=base_url,
            settings=default_settings,
            default_headers=headers,
            **kwargs,
        )

    def build_chat_completion_params(self, params_from_context: OpenAILLMInvocationParams) -> dict:
        """Build the request, with messages in the order the gateway accepts.

        Args:
            params_from_context: Parameters, derived from the LLM context.

        Returns:
            Parameters for the chat completion request.
        """
        params = super().build_chat_completion_params(params_from_context)
        params["messages"] = gateway_messages(list(params["messages"]))
        return params


def gateway_messages(messages: list[dict]) -> list[dict]:
    """Reorder a conversation so the gateway's chat models accept it.

    The gateway answers 400 "System message must be at the beginning" for a system message
    anywhere but first, including a second one right after the first. It answers 400 "No user
    query found in messages" for a request with no user message.

    Args:
        messages: Chat Completions messages, in conversation order.

    Returns:
        A new list. System messages that directly follow the first are merged into it, later
        ones become user messages, and a placeholder user message is added when there is none.
    """
    shaped: list[dict] = []
    for index, message in enumerate(messages):
        if message.get("role") == "system" and index > 0:
            leading = all(earlier.get("role") == "system" for earlier in messages[:index])
            first = shaped[0]
            if (
                leading
                and isinstance(message.get("content"), str)
                and isinstance(first.get("content"), str)
            ):
                shaped[0] = {**first, "content": f"{first['content']}\n\n{message['content']}"}
            else:
                shaped.append({**message, "role": "user"})
            continue
        shaped.append(message)

    if not any(message.get("role") == "user" for message in shaped):
        after_system = bool(shaped) and shaped[0].get("role") == "system"
        shaped.insert(1 if after_system else 0, {"role": "user", "content": _EMPTY_USER_TURN})
    return shaped
