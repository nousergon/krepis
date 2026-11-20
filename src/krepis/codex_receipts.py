"""Bounded A6 decoder for model evidence in a native Responses SSE stream.

Feed only bytes read from the qualified upstream's successful, uncompressed
text/event-stream response. HTTP status, content type/encoding and TLS identity
are the egress boundary's responsibility. Request model names are never input.
Only LF and CRLF framing is accepted; bare CR is refused.
The observer does not modify the stream; it retains at most one bounded event.
A completed event is necessary but not sufficient: finish only after clean EOF.
"""
import json
from typing import Optional, cast


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate response field")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError("nonfinite response constant")


def _identity(value: object) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 512 or any(ord(c) < 32 for c in value):
        raise ValueError("missing or invalid observed identity")
    return value


class CodexReceiptDecoder:
    def __init__(self, *, max_event_bytes: int = 262_144, max_total_bytes: int = 2_097_152) -> None:
        if (type(max_event_bytes) is not int or type(max_total_bytes) is not int
                or not 1 <= max_event_bytes <= max_total_bytes <= 16_777_216):
            raise ValueError("bounded stream limits required")
        self.max_event_bytes = max_event_bytes
        self.max_total_bytes = max_total_bytes
        self._buffer = bytearray()
        self._event: list[bytes] = []
        self._event_name: Optional[str] = None
        self._event_bytes = 0
        self._total = 0
        self._failed = False
        self._finished = False
        self._done = False
        self._model: Optional[str] = None
        self._response_id: Optional[str] = None
        self._completed = False

    def feed(self, chunk: bytes) -> None:
        if self._failed or self._finished:
            raise ValueError("receipt decoder is closed")
        try:
            self._total += len(chunk)
            if self._total > self.max_total_bytes:
                raise ValueError("response exceeds receipt stream bound")
            # Iterate lines without retaining a caller-supplied oversized event.
            offset = 0
            while offset < len(chunk):
                newline = chunk.find(b'\n', offset)
                end = newline + 1 if newline >= 0 else len(chunk)
                if self._event_bytes + len(self._buffer) + end - offset > self.max_event_bytes:
                    raise ValueError("response exceeds receipt event bound")
                self._buffer.extend(chunk[offset:end])
                offset = end
                if not self._buffer.endswith(b'\n'):
                    continue
                line = bytes(self._buffer[:-1]).removesuffix(b'\r')
                if b'\r' in line:
                    raise ValueError('bare CR framing is unsupported')
                self._event_bytes += len(self._buffer)
                self._buffer.clear()
                if not line:
                    self._consume()
                    self._event.clear()
                    self._event_bytes = 0
                    self._event_name = None
                elif line.startswith(b'data:'):
                    data = line[5:]
                    self._event.append(data[1:] if data.startswith(b' ') else data)
                elif line.startswith(b'event:'):
                    name = line[6:].removeprefix(b' ').decode('utf-8')
                    if self._event_name is not None and name != self._event_name:
                        raise ValueError('conflicting event names')
                    self._event_name = name
                elif not (line.startswith(b':') or line.startswith(b'id:')):
                    raise ValueError("unsupported event stream field")
        except (ValueError, UnicodeError):
            self._failed = True
            self._buffer.clear()
            self._event.clear()
            raise ValueError("invalid native receipt stream") from None

    def _consume(self) -> None:
        if not self._event:
            return
        data = b'\n'.join(self._event)
        if data == b'[DONE]':
            if not self._completed or self._done or self._event_name is not None:
                raise ValueError("unexpected end marker")
            self._done = True
            return
        if self._completed or self._done:
            raise ValueError("data follows completed response")
        raw: object = json.loads(data.decode('utf-8'), object_pairs_hook=_unique, parse_constant=_reject_constant)
        if not isinstance(raw, dict):
            raise ValueError("response event must be an object")
        event = cast(dict[str, object], raw)
        kind = event.get('type')
        if not isinstance(kind, str):
            raise ValueError("missing response event type")
        if self._event_name is not None and self._event_name != kind:
            raise ValueError('contradictory named event')
        if kind in {'error', 'response.failed', 'response.incomplete'}:
            raise ValueError("upstream did not complete")
        response = event.get('response')
        if kind in {'response.created', 'response.in_progress', 'response.completed'}:
            if not isinstance(response, dict):
                raise ValueError("missing response metadata")
            metadata = cast(dict[str, object], response)
            if (metadata.get('status') in ('failed', 'incomplete', 'cancelled')
                    or metadata.get('error') is not None
                    or metadata.get('incomplete_details') is not None):
                raise ValueError('unsuccessful lifecycle metadata')
            observed_id = _identity(metadata.get('id'))
            if self._response_id is not None and observed_id != self._response_id:
                raise ValueError("conflicting response identities")
            self._response_id = observed_id
            if 'model' in metadata or kind == 'response.completed':
                model = _identity(metadata.get('model'))
                if self._model is not None and model != self._model:
                    raise ValueError("conflicting served models")
                self._model = model
            if kind == 'response.completed':
                if metadata.get('status') != 'completed' or metadata.get('error') is not None:
                    raise ValueError("unsuccessful completed response")
                self._completed = True

    def finish(self) -> tuple[str, str]:
        """Called after clean upstream EOF, never on disconnect or read failure."""
        self._finished = True
        if (self._failed or self._buffer or self._event or not self._completed
                or self._model is None or self._response_id is None):
            raise ValueError("complete observed response identity required")
        return self._model, self._response_id
