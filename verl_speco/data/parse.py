# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
import json
import re
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, cast

import torch
from transformers import PreTrainedTokenizer

from .template import ChatTemplate

__all__ = [
    "GeneralParser",
    "HarmonyParser",
    "ThinkingParser",
    "clean_conversation_roles",
]

Conversation = List[Dict[str, Any]]

VALID_CHAT_ROLES = {"system", "user", "assistant", "tool"}

# Unknown roles are mapped to assistant once per role, across all conversations,
# so a noisy dataset does not flood the log.
_warned_roles: set[str] = set()


def clean_conversation_roles(
    conversation_items: List[Dict[str, Any]],
    *,
    strict: bool = False,
    sanitize: Any = None,
    stats: Dict[str, int] | None = None,
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Normalize roles and drop order-violating turns.

    Shared by :class:`GeneralParser` and the standalone TQ Producer. Unknown
    roles map to ``assistant`` and are warned once per role. In tolerant mode a
    bad turn is dropped and the conversation continues; ``strict`` restores the
    legacy warn-and-truncate behavior. ``sanitize`` optionally post-processes
    each kept turn (the parser's ``_sanitize_message``).
    """
    if stats is None:
        stats = {"mapped_roles": 0, "dropped_role_turns": 0}
    cleaned_items: List[Dict[str, Any]] = []
    prev_role: str | None = None
    for sentence in conversation_items:
        role = sentence.get("role")
        if role not in VALID_CHAT_ROLES:
            if strict:
                # Legacy behavior: keep the role and let the order checks and
                # template decide what happens to it.
                pass
            else:
                role_name = str(role)
                if role_name not in _warned_roles:
                    _warned_roles.add(role_name)
                    warnings.warn(f"Mapping unknown role '{role_name}' to 'assistant'")
                stats["mapped_roles"] += 1
                role = "assistant"

        violation: str | None = None
        if prev_role is None:
            # A single leading ``system`` turn is valid (the chat template
            # often needs it); the first non-system turn must still be ``user``.
            if role not in ("user", "system"):
                violation = (
                    "Conversation must start with a 'user' role (or one leading "
                    f"'system' turn), but found '{role}'."
                )
        elif prev_role == "system" and role != "user":
            violation = (
                "A leading 'system' message must be followed by a 'user' message, "
                f"but found '{role}'."
            )
        elif role == "tool" and prev_role not in ("assistant", "tool"):
            violation = (
                f"A 'tool' message must follow an 'assistant' or 'tool' message, "
                f"but was preceded by '{prev_role}'."
            )
        elif role == "assistant" and prev_role not in ("user", "tool"):
            violation = (
                f"An 'assistant' message must follow a 'user' or 'tool' message, "
                f"but was preceded by '{prev_role}'."
            )

        if violation is not None:
            if strict:
                warnings.warn(f"{violation} Conversation truncated.")
                break
            warnings.warn(f"{violation} Dropping the offending turn.")
            stats["dropped_role_turns"] += 1
            continue

        cleaned = dict(sentence, role=role)
        if sanitize is not None:
            cleaned = sanitize(cleaned)
        cleaned_items.append(cleaned)
        prev_role = role
    return cleaned_items, stats


class Parser(ABC):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        chat_template: ChatTemplate,
    ):
        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.standard_keys = {"role", "content", "tool_calls"}

    @abstractmethod
    def parse(
        self,
        conversation: "Conversation | str",
        max_length: int,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Parse the conversation into a list of tensors.

        Args:
            conversation: The conversation to parse.

        Returns:
            A list of tensors: [input_ids, loss_mask]
        """

    def _sanitize_message(self, message: dict) -> dict:
        """
        Clean up individual messages, handling the following issues:
        1. `tool_calls` is a string → Parse as a list
        2. `tool_calls[].function.arguments` is a string → Parse as a dictionary
        3. Non-standard fields (extra, etc.) in `tool_calls[]` → Remove
        """
        cleaned = {k: v for k, v in message.items() if k in self.standard_keys}

        # ===== handle tool_calls =====
        if "tool_calls" in cleaned:
            tool_calls = cleaned["tool_calls"]

            # tool_calls is a string → Parsing
            if isinstance(tool_calls, str):
                try:
                    tool_calls = json.loads(tool_calls)
                except json.JSONDecodeError:
                    warnings.warn(
                        "Failed to parse tool_calls JSON string, removing tool_calls"
                    )
                    cleaned.pop("tool_calls", None)
                    return cleaned

            # Clean each tool_call
            if isinstance(tool_calls, list):
                sanitized_tool_calls = []

                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        continue

                    # Only retain the standard fields: id, type, function
                    clean_tc = {
                        "id": tc.get("id", ""),
                        "type": tc.get("type", "function"),
                    }

                    # handle function
                    func = tc.get("function", {})
                    if isinstance(func, dict):
                        clean_func = {
                            "name": func.get("name", ""),
                        }

                        arguments = func.get("arguments", {})
                        if isinstance(arguments, str):
                            try:
                                arguments = json.loads(arguments)
                            except json.JSONDecodeError:
                                warnings.warn(
                                    f"Failed to parse arguments for tool '{clean_func['name']}': "
                                    f"{arguments[:100]}..."
                                )
                                arguments = {}

                        clean_func["arguments"] = arguments
                        clean_tc["function"] = clean_func

                    sanitized_tool_calls.append(clean_tc)

                cleaned["tool_calls"] = sanitized_tool_calls

        return cleaned


_harmony_encoding = None


class GeneralParser(Parser):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        chat_template: ChatTemplate,
        parser_strict_roles: bool = False,
    ):
        super().__init__(tokenizer, chat_template)
        self.system_prompt = chat_template.system_prompt
        self.user_message_separator = f"{chat_template.end_of_turn_token}"
        self.assistant_message_separator = f"{chat_template.assistant_header}"
        self.parser_strict_roles = bool(parser_strict_roles)
        self.role_cleanup_stats = {"mapped_roles": 0, "dropped_role_turns": 0}
        self.set_assistant_pattern(chat_template)

    def _clean_conversation_roles(
        self, conversation_items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Normalize roles, dropping order-violating turns unless strict."""
        cleaned, _ = clean_conversation_roles(
            conversation_items,
            strict=self.parser_strict_roles,
            sanitize=self._sanitize_message,
            stats=self.role_cleanup_stats,
        )
        return cleaned

    def format_role_cleanup_stats(self) -> str:
        stats = self.role_cleanup_stats
        return (
            f"mapped_roles={stats['mapped_roles']} "
            f"dropped_role_turns={stats['dropped_role_turns']}"
        )

    def apply_chat_template(self, messages, tool, **kwargs) -> str:
        conversation = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tool,
            **kwargs,
        )
        return conversation

    def set_assistant_pattern(self, chat_template: ChatTemplate):
        if chat_template.assistant_pattern_type == "longcat":
            self.assistant_pattern = (
                re.escape(self.assistant_message_separator)
                + r"([\s\S]*?(?:"
                + re.escape("[Round ")
                + r"\d+"
                + re.escape("] USER:")
                + "|$))"
            )
        else:
            end_of_turn_token = self.chat_template.end_of_turn_token or ""
            self.assistant_pattern = (
                re.escape(self.assistant_message_separator)
                + r"([\s\S]*?(?:"
                + re.escape(end_of_turn_token)
                + "|$))"
            )

    def parse(
        self,
        conversation: "Conversation | str",
        max_length: int,
        preformatted: bool = False,
        train_only_last_turn: bool = False,
        tool: List[Dict] = [],
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not preformatted:
            conversation_items = cast(Conversation, conversation)
            messages = []

            if conversation_items[0]["role"] == "system":
                warnings.warn(
                    "The first message is from system, we will use the system prompt from the data and ignore the system prompt from the template"
                )
                messages.append(
                    {"role": "system", "content": conversation_items[0]["content"]}
                )
                conversation_items = conversation_items[1:]
            else:
                if self.system_prompt:
                    messages.append({"role": "system", "content": self.system_prompt})

            conversation_items = self._clean_conversation_roles(conversation_items)
            messages.extend(conversation_items)
            try:
                conversation_text = self.apply_chat_template(
                    messages, tool=tool, **kwargs
                )
            except (ValueError, TypeError):
                # Fallback rendering for tokenizers without built-in chat_template
                warnings.warn(
                    "Tokenizer does not have a chat_template, using fallback rendering."
                )
                parts = []
                bos_token = getattr(self.tokenizer, "bos_token", None)
                user_header = self.chat_template.user_header or ""
                assistant_header = self.chat_template.assistant_header or ""
                end_of_turn = self.chat_template.end_of_turn_token or ""

                # Add BOS token at the start
                if bos_token:
                    parts.append(bos_token)

                for msg in messages:
                    if msg["role"] == "system":
                        parts.append(msg["content"])
                    elif msg["role"] == "user":
                        parts.append(f"{user_header}{msg['content']}")
                    elif msg["role"] == "assistant":
                        parts.append(f"{assistant_header}{msg['content']}{end_of_turn}")
                conversation_text = "".join(parts)
        else:
            conversation_text = str(conversation)

        if not self.tokenizer.pad_token_id:
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id

        # get input_ids
        encoding = self.tokenizer(
            conversation_text,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoding.input_ids[0]
        loss_mask = torch.zeros(len(input_ids), dtype=torch.long)

        matches = list(
            re.finditer(self.assistant_pattern, conversation_text, re.DOTALL)
        )
        if train_only_last_turn and matches:
            matches = [matches[-1]]  # Only keep the last match

        for match in matches:
            content_start_char = match.start(1)
            content_end_char = match.end(1)

            # --- Core Alternative Operation: Calculate Token Index Based on Prefix String Length ---
            # Encode the text "assistant start", the length of which is the position of the starting token.
            prefix_ids = self.tokenizer.encode(
                conversation_text[:content_start_char],
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )
            # Encodes the text "assistant end", the length of which is the position of the end token.
            full_ids = self.tokenizer.encode(
                conversation_text[:content_end_char],
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )

            start_token_idx = len(prefix_ids)
            end_token_idx = len(full_ids)

            # Handling out-of-bounds errors caused by truncation
            actual_start = min(start_token_idx, len(input_ids))
            actual_end = min(end_token_idx, len(input_ids))

            if actual_start < actual_end:
                loss_mask[actual_start:actual_end] = 1

        # Zero out loss_mask for ignore_tokens
        ignore_tokens = self.chat_template.ignore_token
        if ignore_tokens:
            for token_str in ignore_tokens:
                start = 0
                while True:
                    idx = conversation_text.find(token_str, start)
                    if idx == -1:
                        break
                    ignore_start_char = idx
                    ignore_end_char = idx + len(token_str)

                    prefix_ids = self.tokenizer.encode(
                        conversation_text[:ignore_start_char],
                        add_special_tokens=False,
                        truncation=True,
                        max_length=max_length,
                    )
                    full_ids = self.tokenizer.encode(
                        conversation_text[:ignore_end_char],
                        add_special_tokens=False,
                        truncation=True,
                        max_length=max_length,
                    )

                    start_token_idx = min(len(prefix_ids), len(input_ids))
                    end_token_idx = min(len(full_ids), len(input_ids))

                    if start_token_idx < end_token_idx:
                        loss_mask[start_token_idx:end_token_idx] = 0

                    start = ignore_end_char

        return input_ids, loss_mask


class HarmonyParser(Parser):
    def __init__(self, tokenizer: PreTrainedTokenizer, chat_template: ChatTemplate):
        super().__init__(tokenizer, chat_template)
        self.reasoning_levels = ["low", "medium", "high"]
        self.default_reasoning_level = "low"

    def build_single_turn_prompt(
        self,
        prompt_text: str,
        role: str,
        content: str,
    ) -> str:
        """Embed user message into the required prompt template."""
        if role == "system":
            prompt_text = f"<|start|>system<|message|>{content}<|end|>"
        elif role == "assistant_reasoning_effort":
            prompt_text = f"<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: 2025-06-28\n\nReasoning: {content.lower()}\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>"
        elif role == "user":
            prompt_text += f"<|start|>user<|message|>{content}<|end|>"
        elif role == "assistant_analysis":
            prompt_text += (
                f"<|start|>assistant<|channel|>analysis<|message|>{content}<|end|>"
            )
        elif role == "assistant_commentary":
            prompt_text += (
                f"<|start|>assistant<|channel|>commentary<|message|>{content}<|end|>"
            )
        elif role == "assistant_final":
            prompt_text += (
                f"<|start|>assistant<|channel|>final<|message|>{content}<|end|>"
            )
        else:
            raise ValueError(f"Unknown role: {role}")
        return prompt_text

    def parse(
        self,
        conversation: "Conversation | str",
        max_length: int,
        preformatted: bool = False,
        train_only_last_turn: bool = False,
        tool: List[Dict] = [],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # conversation = process_harmony_conversations(conversation)
        if not preformatted:
            conversation_items = cast(Conversation, conversation)
            prompt_text = ""
            for j, message in enumerate(conversation_items):
                if j == 0 and message["role"] not in (
                    "system",
                    "assistant_reasoning_effort",
                ):
                    prompt_text = self.build_single_turn_prompt(
                        prompt_text,
                        "assistant_reasoning_effort",
                        self.default_reasoning_level,
                    )
                prompt_text = self.build_single_turn_prompt(
                    prompt_text, message["role"], message["content"]
                )
            conversation_text = prompt_text
        else:
            conversation_text = str(conversation)

        if not self.tokenizer.pad_token_id:
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id

        encoding = self.tokenizer(
            conversation_text,
            return_offsets_mapping=True,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoding.input_ids[0]
        offsets = encoding.offset_mapping[0]
        loss_mask = torch.zeros(len(input_ids), dtype=torch.long)

        # Find spans of assistant responses using regex
        # We match `<|start|>assistant` and only extract the content following it.
        # This continues until `<|start|>user<|message|>` appears, or until the end of the string.
        pattern = re.compile(
            r"<\|start\|>assistant([\s\S]*?)(?=<\|start\|>user<\|message\|>|$)"
        )

        # Find all matching segments
        matches = list(pattern.finditer(conversation_text))
        if train_only_last_turn and matches:
            matches = [matches[-1]]  # Only keep the last match

        for match in matches:
            # match.start(0) is the start index of the full match (including `<|start|>assistant`)
            # match.start(1) is the start index of the first capture group (excluding `<|start|>assistant`)
            # match.end(1) is the end index of the content
            start_char = match.start(1)
            end_char = match.end(1)

            # Map character indices to token indices
            for idx, (ts, te) in enumerate(offsets):
                # Set mask to 1 only if the token's character range falls entirely within the "content area"
                if ts >= start_char and te <= end_char:
                    loss_mask[idx] = 1

        return input_ids, loss_mask


class ThinkingParser(GeneralParser):
    """
    Parser for thinking/reasoning models.

    This parser processes the entire conversation (not just the last turn).
    It handles reasoning_content and tool_calls in assistant messages.
    The loss mask covers from assistant_header to end_of_turn_token (inclusive).
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        chat_template: ChatTemplate,
        parser_strict_roles: bool = False,
    ):
        super().__init__(
            tokenizer, chat_template, parser_strict_roles=parser_strict_roles
        )
        self.standard_keys = {"role", "content", "tool_calls", "reasoning_content"}

    def apply_chat_template(self, messages, tool, **kwargs) -> str:
        """Apply chat template to all messages, handling reasoning_content and tool_calls."""
        conversation = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            add_special_tokens=False,
            tools=tool,
            **kwargs,
        )
        return conversation

    def parse(
        self,
        conversation: "Conversation | str",
        max_length: int,
        preformatted: bool = False,
        train_only_last_turn: bool = False,
        tool: List[Dict] = [],
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Parse conversation, processing all assistant turns for loss mask."""
        if self.chat_template.enable_thinking:
            kwargs["enable_thinking"] = True
        return super().parse(
            conversation, max_length, preformatted, train_only_last_turn, tool, **kwargs
        )
