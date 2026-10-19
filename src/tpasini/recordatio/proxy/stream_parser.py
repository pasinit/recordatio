"""SSE accumulators for Anthropic and OpenAI-compatible streaming APIs.

Each accumulator parses server-sent events from its respective streaming
format, accumulating the full assistant response text while raw chunks are
forwarded to the client unchanged.
"""

import json


class StreamAccumulator:
    """Accumulate an Anthropic streaming response into a full text turn.

    Feed raw SSE lines via ``feed(line)``.  When the stream is done,
    ``text`` holds the concatenated assistant message and ``tool_uses``
    holds any tool_use blocks emitted by the model.
    """

    def __init__(self) -> None:
        self._blocks: dict[int, list[str]] = {}
        self._tool_blocks: dict[int, dict] = {}  # index -> {id, name, chunks}
        self._thinking_blocks: dict[int, list[str]] = {}  # index -> chunks
        self._thinking_signatures: dict[int, str] = {}  # index -> signature
        self._block_types: dict[int, str] = {}  # index -> "text" | "tool_use" | "thinking"
        self.stop_reason: str | None = None

    def feed(self, line: bytes) -> None:
        """Process one raw SSE line (may be a ``data:`` line or blank)."""
        if not line.startswith(b"data:"):
            return
        payload = line[len(b"data:") :].strip()
        if payload in (b"[DONE]", b""):
            return
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            return

        event_type = event.get("type")
        if event_type == "message_delta":
            self.stop_reason = event.get("delta", {}).get("stop_reason") or self.stop_reason
        elif event_type == "content_block_start":
            index = event.get("index", 0)
            block = event.get("content_block", {})
            block_type = block.get("type", "text")
            self._block_types[index] = block_type
            if block_type == "tool_use":
                self._tool_blocks[index] = {
                    "id": block.get("id", ""),
                    "name": block.get("name", ""),
                    "chunks": [],
                }
            elif block_type == "thinking":
                self._thinking_blocks[index] = []
        elif event_type == "content_block_delta":
            index = event.get("index", 0)
            delta = event.get("delta", {})
            if delta.get("type") == "text_delta":
                self._blocks.setdefault(index, []).append(delta.get("text", ""))
            elif delta.get("type") == "input_json_delta" and index in self._tool_blocks:
                self._tool_blocks[index]["chunks"].append(delta.get("partial_json", ""))
            elif delta.get("type") == "thinking_delta" and index in self._thinking_blocks:
                self._thinking_blocks[index].append(delta.get("thinking", ""))
            elif delta.get("type") == "signature_delta" and index in self._thinking_blocks:
                self._thinking_signatures[index] = delta.get("signature", "")

    @property
    def text(self) -> str:
        """Full accumulated assistant text across all content blocks."""
        return "".join("".join(chunks) for _, chunks in sorted(self._blocks.items()))

    @property
    def tool_uses(self) -> list[dict]:
        """Completed tool_use blocks with parsed input, in block order."""
        return [self._parse_tool_block(block) for _, block in sorted(self._tool_blocks.items())]

    @property
    def content_blocks(self) -> list[dict]:
        """All content blocks (text and tool_use) interleaved in original index order.

        Used to faithfully reconstruct the assistant turn in conversation history,
        preserving any text preamble before tool_use blocks.
        """
        tool_uses_by_index = {
            index: self._parse_tool_block(block)
            for index, block in self._tool_blocks.items()
        }
        result = []
        all_indices = sorted(set(self._blocks) | set(self._tool_blocks) | set(self._thinking_blocks))
        for index in all_indices:
            if index in tool_uses_by_index:
                result.append({"type": "tool_use", **tool_uses_by_index[index]})
            elif index in self._thinking_blocks:
                thinking = "".join(self._thinking_blocks[index])
                signature = self._thinking_signatures.get(index, "")
                if thinking and signature:  # both required by the API
                    result.append({
                        "type": "thinking",
                        "thinking": thinking,
                        "signature": signature,
                    })
            elif index in self._blocks:
                text = "".join(self._blocks[index])
                if text:  # skip empty text blocks (e.g. content_block_start with no deltas)
                    result.append({"type": "text", "text": text})
        return result

    def _parse_tool_block(self, block: dict) -> dict:
        raw = "".join(block["chunks"])
        try:
            parsed_input = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            parsed_input = {}
        return {"id": block["id"], "name": block["name"], "input": parsed_input}


class OpenAIStreamAccumulator:
    """Accumulate an OpenAI-compatible streaming response (chat completions).

    Parses ``data: {...}`` lines with ``choices[].delta.content`` fields.
    Works for OpenAI, Groq, Mistral, Ollama, vLLM, and any other
    OpenAI-compatible provider.
    """

    def __init__(self) -> None:
        self._chunks: list[str] = []
        self.stop_reason: str | None = None

    def feed(self, line: bytes) -> None:
        if not line.startswith(b"data:"):
            return
        payload = line[len(b"data:"):].strip()
        if payload in (b"[DONE]", b""):
            return
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            return
        for choice in event.get("choices") or []:
            delta = choice.get("delta") or {}
            text = delta.get("content")
            if isinstance(text, str) and text:
                self._chunks.append(text)
            finish = choice.get("finish_reason")
            if finish:
                self.stop_reason = finish

    @property
    def text(self) -> str:
        return "".join(self._chunks)
