import json
import logging
import os
import sys
import threading
import warnings
from collections import defaultdict
from contextlib import contextmanager
from typing import (
    Any,
    DefaultDict,
    Dict,
    List,
    Literal,
    Optional,
    Type,
    TypedDict,
    Union,
    cast,
)
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from crewai.events.types.llm_events import (
    LLMCallCompletedEvent,
    LLMCallFailedEvent,
    LLMCallStartedEvent,
    LLMCallType,
    LLMStreamChunkEvent,
)
from crewai.events.types.tool_usage_events import (
    ToolUsageStartedEvent,
    ToolUsageFinishedEvent,
    ToolUsageErrorEvent,
)
import openai


import io
from typing import TextIO

from crewai.llms.base_llm import BaseLLM
from crewai.utilities.events import crewai_event_bus
from crewai.utilities.exceptions.context_window_exceeding_exception import (
    LLMContextLengthExceededError,
)

load_dotenv()


class FilteredStream(io.TextIOBase):
    _lock = None

    def __init__(self, original_stream: TextIO):
        self._original_stream = original_stream
        self._lock = threading.Lock()

    def write(self, s: str) -> int:
        if not self._lock:
            self._lock = threading.Lock()

        with self._lock:
            lower_s = s.lower()

            # Skip common noisy LiteLLM banners and any other lines that contain "litellm"
            if (
                "litellm.info:" in lower_s
                or "Consider using a smaller input or implementing a text splitting strategy"
                in lower_s
            ):
                return 0

            return self._original_stream.write(s)

    def flush(self):
        with self._lock:
            return self._original_stream.flush()

    def __getattr__(self, name):
        """Delegate attribute access to the wrapped original stream.

        This ensures compatibility with libraries (e.g., Rich) that rely on
        attributes such as `encoding`, `isatty`, `buffer`, etc., which may not
        be explicitly defined on this proxy class.
        """
        return getattr(self._original_stream, name)

    # Delegate common properties/methods explicitly so they aren't shadowed by
    # the TextIOBase defaults (e.g., .encoding returns None by default, which
    # confuses Rich). These explicit pass-throughs ensure the wrapped Console
    # still sees a fully-featured stream.
    @property
    def encoding(self):
        return getattr(self._original_stream, "encoding", "utf-8")

    def isatty(self):
        return self._original_stream.isatty()

    def fileno(self):
        return self._original_stream.fileno()

    def writable(self):
        return True


# Apply the filtered stream globally so that any subsequent writes containing the filtered
# keywords (e.g., "litellm") are hidden from terminal output. We guard against double
# wrapping to ensure idempotency in environments where this module might be reloaded.
if not isinstance(sys.stdout, FilteredStream):
    sys.stdout = FilteredStream(sys.stdout)
if not isinstance(sys.stderr, FilteredStream):
    sys.stderr = FilteredStream(sys.stderr)


LLM_CONTEXT_WINDOW_SIZES = {
    # openai
    "gpt-4": 8192,
    "gpt-4o": 128000,
    "gpt-4o-mini": 200000,
    "gpt-4-turbo": 128000,
    "gpt-4.1": 1047576,  # Based on official docs
    "gpt-4.1-mini-2025-04-14": 1047576,
    "gpt-4.1-nano-2025-04-14": 1047576,
    "o1-preview": 128000,
    "o1-mini": 128000,
    "o3-mini": 200000,
    "o4-mini": 200000,
    # gemini
    "gemini-2.0-flash": 1048576,
    "gemini-2.0-flash-thinking-exp-01-21": 32768,
    "gemini-2.0-flash-lite-001": 1048576,
    "gemini-2.0-flash-001": 1048576,
    "gemini-2.5-flash-preview-04-17": 1048576,
    "gemini-2.5-pro-exp-03-25": 1048576,
    "gemini-1.5-pro": 2097152,
    "gemini-1.5-flash": 1048576,
    "gemini-1.5-flash-8b": 1048576,
    "gemini/gemma-3-1b-it": 32000,
    "gemini/gemma-3-4b-it": 128000,
    "gemini/gemma-3-12b-it": 128000,
    "gemini/gemma-3-27b-it": 128000,
    # deepseek
    "deepseek-chat": 128000,
    # groq
    "gemma2-9b-it": 8192,
    "gemma-7b-it": 8192,
    "llama3-groq-70b-8192-tool-use-preview": 8192,
    "llama3-groq-8b-8192-tool-use-preview": 8192,
    "llama-3.1-70b-versatile": 131072,
    "llama-3.1-8b-instant": 131072,
    "llama-3.2-1b-preview": 8192,
    "llama-3.2-3b-preview": 8192,
    "llama-3.2-11b-text-preview": 8192,
    "llama-3.2-90b-text-preview": 8192,
    "llama3-70b-8192": 8192,
    "llama3-8b-8192": 8192,
    "mixtral-8x7b-32768": 32768,
    "llama-3.3-70b-versatile": 128000,
    "llama-3.3-70b-instruct": 128000,
    # sambanova
    "Meta-Llama-3.3-70B-Instruct": 131072,
    "QwQ-32B-Preview": 8192,
    "Qwen2.5-72B-Instruct": 8192,
    "Qwen2.5-Coder-32B-Instruct": 8192,
    "Meta-Llama-3.1-405B-Instruct": 8192,
    "Meta-Llama-3.1-70B-Instruct": 131072,
    "Meta-Llama-3.1-8B-Instruct": 131072,
    "Llama-3.2-90B-Vision-Instruct": 16384,
    "Llama-3.2-11B-Vision-Instruct": 16384,
    "Meta-Llama-3.2-3B-Instruct": 4096,
    "Meta-Llama-3.2-1B-Instruct": 16384,
    # bedrock
    "us.amazon.nova-pro-v1:0": 300000,
    "us.amazon.nova-micro-v1:0": 128000,
    "us.amazon.nova-lite-v1:0": 300000,
    "us.anthropic.claude-3-5-sonnet-20240620-v1:0": 200000,
    "us.anthropic.claude-3-5-haiku-20241022-v1:0": 200000,
    "us.anthropic.claude-3-5-sonnet-20241022-v2:0": 200000,
    "us.anthropic.claude-3-7-sonnet-20250219-v1:0": 200000,
    "us.anthropic.claude-3-sonnet-20240229-v1:0": 200000,
    "us.anthropic.claude-3-opus-20240229-v1:0": 200000,
    "us.anthropic.claude-3-haiku-20240307-v1:0": 200000,
    "us.meta.llama3-2-11b-instruct-v1:0": 128000,
    "us.meta.llama3-2-3b-instruct-v1:0": 131000,
    "us.meta.llama3-2-90b-instruct-v1:0": 128000,
    "us.meta.llama3-2-1b-instruct-v1:0": 131000,
    "us.meta.llama3-1-8b-instruct-v1:0": 128000,
    "us.meta.llama3-1-70b-instruct-v1:0": 128000,
    "us.meta.llama3-3-70b-instruct-v1:0": 128000,
    "us.meta.llama3-1-405b-instruct-v1:0": 128000,
    "eu.anthropic.claude-3-5-sonnet-20240620-v1:0": 200000,
    "eu.anthropic.claude-3-sonnet-20240229-v1:0": 200000,
    "eu.anthropic.claude-3-haiku-20240307-v1:0": 200000,
    "eu.meta.llama3-2-3b-instruct-v1:0": 131000,
    "eu.meta.llama3-2-1b-instruct-v1:0": 131000,
    "apac.anthropic.claude-3-5-sonnet-20240620-v1:0": 200000,
    "apac.anthropic.claude-3-5-sonnet-20241022-v2:0": 200000,
    "apac.anthropic.claude-3-sonnet-20240229-v1:0": 200000,
    "apac.anthropic.claude-3-haiku-20240307-v1:0": 200000,
    "amazon.nova-pro-v1:0": 300000,
    "amazon.nova-micro-v1:0": 128000,
    "amazon.nova-lite-v1:0": 300000,
    "anthropic.claude-3-5-sonnet-20240620-v1:0": 200000,
    "anthropic.claude-3-5-haiku-20241022-v1:0": 200000,
    "anthropic.claude-3-5-sonnet-20241022-v2:0": 200000,
    "anthropic.claude-3-7-sonnet-20250219-v1:0": 200000,
    "anthropic.claude-3-sonnet-20240229-v1:0": 200000,
    "anthropic.claude-3-opus-20240229-v1:0": 200000,
    "anthropic.claude-3-haiku-20240307-v1:0": 200000,
    "anthropic.claude-v2:1": 200000,
    "anthropic.claude-v2": 100000,
    "anthropic.claude-instant-v1": 100000,
    "meta.llama3-1-405b-instruct-v1:0": 128000,
    "meta.llama3-1-70b-instruct-v1:0": 128000,
    "meta.llama3-1-8b-instruct-v1:0": 128000,
    "meta.llama3-70b-instruct-v1:0": 8000,
    "meta.llama3-8b-instruct-v1:0": 8000,
    "amazon.titan-text-lite-v1": 4000,
    "amazon.titan-text-express-v1": 8000,
    "cohere.command-text-v14": 4000,
    "ai21.j2-mid-v1": 8191,
    "ai21.j2-ultra-v1": 8191,
    "ai21.jamba-instruct-v1:0": 256000,
    "mistral.mistral-7b-instruct-v0:2": 32000,
    "mistral.mixtral-8x7b-instruct-v0:1": 32000,
    # mistral
    "mistral-tiny": 32768,
    "mistral-small-latest": 32768,
    "mistral-medium-latest": 32768,
    "mistral-large-latest": 32768,
    "mistral-large-2407": 32768,
    "mistral-large-2402": 32768,
    "mistral/mistral-tiny": 32768,
    "mistral/mistral-small-latest": 32768,
    "mistral/mistral-medium-latest": 32768,
    "mistral/mistral-large-latest": 32768,
    "mistral/mistral-large-2407": 32768,
    "mistral/mistral-large-2402": 32768,
}

DEFAULT_CONTEXT_WINDOW_SIZE = 8192
CONTEXT_WINDOW_USAGE_RATIO = 0.85


@contextmanager
def suppress_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        warnings.filterwarnings(
            "ignore", message="open_text is deprecated*", category=DeprecationWarning
        )

        yield


class Delta(TypedDict):
    content: Optional[str]
    role: Optional[str]


class StreamingChoices(TypedDict):
    delta: Delta
    index: int
    finish_reason: Optional[str]


class FunctionArgs(BaseModel):
    name: str = ""
    arguments: str = ""


class AccumulatedToolArgs(BaseModel):
    function: FunctionArgs = Field(default_factory=FunctionArgs)


class LocalLLM(BaseLLM):
    completion_cost: Optional[float] = None
    params: Dict[str, Any]
    client: openai.OpenAI
    def __init__(
        self,
        params,
    ):
        self.params = params
        self.client = openai.OpenAI(base_url=params.get("base_url", None), api_key=params.get("api_key", None))

    def _handle_non_streaming_response(
        self,
        params: Dict[str, Any],
        messages: List[Dict[str, Any]],
        tools,
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Any]] = None,
        from_task: Optional[Any] = None,
        from_agent: Optional[Any] = None,
    ) -> str | Any:
        """Handle a non-streaming response from the LLM.

        Args:
            params: Parameters for the completion call
            callbacks: Optional list of callback functions
            available_functions: Dict of available functions
            from_task: Optional Task that invoked the LLM
            from_agent: Optional Agent that invoked the LLM

        Returns:
            str: The response text
        """
        # --- 1) Make the completion call
        try:
            # Attempt to make the completion call, but catch context window errors
            # and convert them to our own exception type for consistent handling
            # across the codebase. This allows CrewAgentExecutor to handle context
            # length issues appropriately.
            response = self.client.chat.completions.create(
                model=params["model"],
                messages=messages,
                tools=tools,
                tool_choice="auto",
                stream=False,
            )

        except Exception as e:
            # Convert litellm's context window error to our own exception type
            # for consistent handling in the rest of the codebase
            raise
        # --- 2) Extract response message and content
        response_message = response.choices[0].message
        text_response = response_message.content or ""
        # --- 4) Check for tool calls
        tool_calls = getattr(response_message, "tool_calls", [])

        # --- 5) If no tool calls or no available functions, return the text response directly as long as there is a text response
        if (not tool_calls or not available_functions) and text_response:
            self._handle_emit_call_events(
                response=text_response,
                call_type=LLMCallType.LLM_CALL,
                from_task=from_task,
                from_agent=from_agent,
                messages=messages,
            )
            return text_response
        # --- 6) If there is no text response, no available functions, but there are tool calls, return the tool calls
        elif tool_calls and not available_functions and not text_response:
            return tool_calls

        # --- 7) Handle tool calls if present
        tool_result = self._handle_tool_call(
            tool_calls, available_functions, from_task, from_agent
        )
        if tool_result is not None:
            return tool_result
        # --- 8) If tool call handling didn't return a result, emit completion event and return text response
        self._handle_emit_call_events(
            response=text_response,
            call_type=LLMCallType.LLM_CALL,
            from_task=from_task,
            from_agent=from_agent,
            messages=messages,
        )
        return text_response

    def _handle_tool_call(
        self,
        tool_calls: List[Any],
        available_functions: Optional[Dict[str, Any]] = None,
        from_task: Optional[Any] = None,
        from_agent: Optional[Any] = None,
    ) -> Optional[str]:
        """Handle a tool call from the LLM.

        Args:
            tool_calls: List of tool calls from the LLM
            available_functions: Dict of available functions

        Returns:
            Optional[str]: The result of the tool call, or None if no tool call was made
        """
        # --- 1) Validate tool calls and available functions
        if not tool_calls or not available_functions:
            return None

        # --- 2) Extract function name from first tool call
        tool_call = tool_calls[0]
        function_name = tool_call.function.name
        function_args = {}  # Initialize to empty dict to avoid unbound variable

        # --- 3) Check if function is available
        if function_name in available_functions:
            try:
                # --- 3.1) Parse function arguments
                function_args = json.loads(tool_call.function.arguments)
                print(function_args)
                fn = available_functions[function_name]

                # --- 3.2) Execute function
                assert hasattr(crewai_event_bus, "emit")
                started_at = datetime.now()
                crewai_event_bus.emit(
                    self,
                    event=ToolUsageStartedEvent(
                        tool_name=function_name,
                        tool_args=function_args,
                        from_agent=from_agent,
                        from_task=from_task,
                    ),
                )

                result = fn(**function_args)
                crewai_event_bus.emit(
                    self,
                    event=ToolUsageFinishedEvent(
                        output=result,
                        tool_name=function_name,
                        tool_args=function_args,
                        started_at=started_at,
                        finished_at=datetime.now(),
                        from_task=from_task,
                        from_agent=from_agent,
                    ),
                )

                # --- 3.3) Emit success event
                self._handle_emit_call_events(
                    response=result,
                    call_type=LLMCallType.TOOL_CALL,
                    from_task=from_task,
                    from_agent=from_agent,
                )
                return result
            except Exception as e:
                # --- 3.4) Handle execution errors
                fn = available_functions.get(
                    function_name, lambda: None
                )  # Ensure fn is always a callable
                logging.error(f"Error executing function '{function_name}': {e}")
                assert hasattr(crewai_event_bus, "emit")
                crewai_event_bus.emit(
                    self,
                    event=LLMCallFailedEvent(error=f"Tool execution error: {str(e)}"),
                )
                crewai_event_bus.emit(
                    self,
                    event=ToolUsageErrorEvent(
                        tool_name=function_name,
                        tool_args=function_args,
                        error=f"Tool execution error: {str(e)}",
                    ),
                )
        return None

    def call(
        self,
        messages: Union[str, List[Dict[str, str]]],
        tools: Optional[List[dict]] = None,
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Any]] = None,
        from_task: Optional[Any] = None,
        from_agent: Optional[Any] = None,
    ) -> Union[str, Any]:
        """High-level LLM call method.

        Args:
            messages: Input messages for the LLM.
                     Can be a string or list of message dictionaries.
                     If string, it will be converted to a single user message.
                     If list, each dict must have 'role' and 'content' keys.
            tools: Optional list of tool schemas for function calling.
                  Each tool should define its name, description, and parameters.
            callbacks: Optional list of callback functions to be executed
                      during and after the LLM call.
            available_functions: Optional dict mapping function names to callables
                               that can be invoked by the LLM.
            from_task: Optional Task that invoked the LLM
            from_agent: Optional Agent that invoked the LLM

        Returns:
            Union[str, Any]: Either a text response from the LLM (str) or
                           the result of a tool function call (Any).

        Raises:
            TypeError: If messages format is invalid
            ValueError: If response format is not supported
            LLMContextLengthExceededException: If input exceeds model's context limit
        """
        # --- 1) Emit call started event
        assert hasattr(crewai_event_bus, "emit")
        crewai_event_bus.emit(
            self,
            event=LLMCallStartedEvent(
                messages=messages,
                tools=tools,
                callbacks=callbacks,
                available_functions=available_functions,
                from_task=from_task,
                from_agent=from_agent,
                model=self.params["model"],
            ),
        )

        # --- 3) Convert string messages to proper format if needed
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        # --- 4) Handle O1 model special case (system messages not supported)
        if "o1" in self.params["model"].lower():
            for message in messages:
                if message.get("role") == "system":
                    message["role"] = "assistant"
        # --- 5) Set up callbacks if provided
        with suppress_warnings():
            try:
                # --- 6) Prepare parameters for the completion call
                params = self.params
                return self._handle_non_streaming_response(
                    params, messages, tools, callbacks, available_functions, from_task, from_agent
                )
            except Exception as e:
                unsupported_stop = "Unsupported parameter" in str(
                    e
                ) and "'stop'" in str(e)

                if unsupported_stop:
                    if (
                        "additional_drop_params" in self.additional_params
                        and isinstance(
                            self.additional_params["additional_drop_params"], list
                        )
                    ):
                        self.additional_params["additional_drop_params"].append("stop")
                    else:
                        self.additional_params = {"additional_drop_params": ["stop"]}

                    logging.info("Retrying LLM call without the unsupported 'stop'")

                    return self.call(
                        messages,
                        tools=tools,
                        callbacks=callbacks,
                        available_functions=available_functions,
                        from_task=from_task,
                        from_agent=from_agent,
                    )

                assert hasattr(crewai_event_bus, "emit")
                crewai_event_bus.emit(
                    self,
                    event=LLMCallFailedEvent(
                        error=str(e), from_task=from_task, from_agent=from_agent
                    ),
                )
                raise

    def _handle_emit_call_events(
        self,
        response: Any,
        call_type: LLMCallType,
        from_task: Optional[Any] = None,
        from_agent: Optional[Any] = None,
        messages: str | list[dict[str, Any]] | None = None,
    ):
        """Handle the events for the LLM call.

        Args:
            response (str): The response from the LLM call.
            call_type (str): The type of call, either "tool_call" or "llm_call".
            from_task: Optional task object
            from_agent: Optional agent object
            messages: Optional messages object
        """
        assert hasattr(crewai_event_bus, "emit")
        crewai_event_bus.emit(
            self,
            event=LLMCallCompletedEvent(
                messages=messages,
                response=response,
                call_type=call_type,
                from_task=from_task,
                from_agent=from_agent,
                model=self.params["model"],
            ),
        )

    def _format_messages_for_provider(
        self, messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Format messages according to provider requirements.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
                     Can be empty or None.

        Returns:
            List of formatted messages according to provider requirements.
            For Anthropic models, ensures first message has 'user' role.

        Raises:
            TypeError: If messages is None or contains invalid message format.
        """
        if messages is None:
            raise TypeError("Messages cannot be None")

        # Validate message format first
        for msg in messages:
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                raise TypeError(
                    "Invalid message format. Each message must be a dict with 'role' and 'content' keys"
                )

        # Handle O1 models specially
        if "o1" in self.params["model"].lower():
            formatted_messages = []
            for msg in messages:
                # Convert system messages to assistant messages
                if msg["role"] == "system":
                    formatted_messages.append(
                        {"role": "assistant", "content": msg["content"]}
                    )
                else:
                    formatted_messages.append(msg)
            return formatted_messages

        # Handle Mistral models - they require the last message to have a role of 'user' or 'tool'
        if "mistral" in self.params["model"].lower():
            # Check if the last message has a role of 'assistant'
            if messages and messages[-1]["role"] == "assistant":
                return messages + [{"role": "user", "content": "Please continue."}]
            return messages

        # TODO: Remove this code after merging PR https://github.com/BerriAI/litellm/pull/10917
        # Ollama doesn't supports last message to be 'assistant'
        if (
            "ollama" in self.params["model"].lower()
            and messages
            and messages[-1]["role"] == "assistant"
        ):
            return messages + [{"role": "user", "content": ""}]

        # Handle Anthropic models
        if not self.is_anthropic:
            return messages

        # Anthropic requires messages to start with 'user' role
        if not messages or messages[0]["role"] == "system":
            # If first message is system or empty, add a placeholder user message
            return [{"role": "user", "content": "."}, *messages]

        return messages

    def supports_function_calling(self) -> bool:
        if self.params.get("supports_fn_call", True):
            return True
        else:
            return False

    def supports_stop_words(self) -> bool:
        if self.params.get("supports_stop_words", False):
            return True
        else:
            return False

    def get_context_window_size(self) -> int:
        """
        Returns the context window size, using 75% of the maximum to avoid
        cutting off messages mid-thread.

        Raises:
            ValueError: If a model's context window size is outside valid bounds (1024-2097152)
        """
        if self.context_window_size != 0:
            return self.context_window_size

        MIN_CONTEXT = 1024
        MAX_CONTEXT = 2097152  # Current max from gemini-1.5-pro

        # Validate all context window sizes
        for key, value in LLM_CONTEXT_WINDOW_SIZES.items():
            if value < MIN_CONTEXT or value > MAX_CONTEXT:
                raise ValueError(
                    f"Context window for {key} must be between {MIN_CONTEXT} and {MAX_CONTEXT}"
                )

        self.context_window_size = int(
            DEFAULT_CONTEXT_WINDOW_SIZE * CONTEXT_WINDOW_USAGE_RATIO
        )
        for key, value in LLM_CONTEXT_WINDOW_SIZES.items():
            if self.params["model"].startswith(key):
                self.context_window_size = int(value * CONTEXT_WINDOW_USAGE_RATIO)
        return self.context_window_size