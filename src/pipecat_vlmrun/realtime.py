#
# Copyright (c) 2026, VLM Run
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Per-frame vision on the VLM Run realtime socket, as a Pipecat processor.

The socket is ``wss://gateway.vlm.run/v1/ws``. A session is opened with one ``session.create``
message that names the model. Every later message is one image, sent as raw bytes, and the
gateway answers each with a ``result``, a ``frame.dropped`` or an ``error``.

Pipecat has no base class or frame type for detection, segmentation, pose or OCR results, so
this module defines ``VLMRunResultFrame``.
"""

import asyncio
import io
import json
import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from PIL import Image
from pipecat.frames.frames import (
    CancelFrame,
    DataFrame,
    EndFrame,
    Frame,
    InputImageRawFrame,
    OutputImageRawFrame,
    StartFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import ConnectionClosed, InvalidStatus

from pipecat_vlmrun.overlay import annotate
from pipecat_vlmrun.version import CLIENT_HEADER, client_name

DEFAULT_WS_URL = "wss://gateway.vlm.run/v1/ws"

# Close codes, from https://docs.vlm.run/gateway/realtime.
_CLOSE_NORMAL = 1000
_CLOSE_REFUSED = 1008
# The reason the gateway gives when a session reaches its spend ceiling. Reopening would spend
# past a limit the account set, so the processor stops.
_REASON_SPENT = "spent"
# The longest single wait. A model that is starting up asks for about 240 seconds.
_MAX_RETRY_AFTER = 300.0
_MAX_BACKOFF = 8.0
_JPEG_FALLBACK_QUALITIES = (60, 40)


@dataclass
class VLMRunResultFrame(DataFrame):
    """One camera frame's result from a VLM Run realtime model.

    Parameters:
        model: Gateway model id the session was opened with.
        result: The model's JSON. Its ``object`` field names the payload type, for example
            ``img.detect.bboxes``, ``img.segment.masks``, ``img.pose.kpts``,
            ``img.pose.hands3d`` or ``doc.ocr.lines``. Coordinates are fractions of the
            frame's width and height. The realtime chat models return ``{"text": ...}``.
        size: Width and height of the camera frame the result was computed on.
        frame_index: The gateway's count of frames received on the session.
        gateway_ms: Model time the gateway reports for the frame.
        round_trip_ms: Time from sending the frame to receiving its reply.
        age_ms: Time from the camera frame entering the processor to this frame leaving it.
    """

    model: str
    result: dict[str, Any]
    size: tuple[int, int] | None = None
    frame_index: int | None = None
    gateway_ms: float | None = None
    round_trip_ms: float | None = None
    age_ms: float | None = None


@dataclass
class VLMRunRealtimeStats:
    """Counters for one processor, across every session it opened.

    Parameters:
        frames_seen: Camera frames that reached the processor.
        frames_sent: Frames sent to the gateway.
        frames_skipped: Frames replaced by a newer one before they could be sent.
        results: Results received.
        dropped: Frames the gateway dropped because they arrived too fast.
        errors: Frames that failed, including replies that timed out.
        sessions: Sessions opened.
        cost: Dollars billed, summed from the ``session.stats`` of sessions the processor
            closed itself. A session the gateway closed, for example when idle, is not counted.
        session: The latest ``session.created`` message.
        session_stats: The latest ``session.stats`` message.
    """

    frames_seen: int = 0
    frames_sent: int = 0
    frames_skipped: int = 0
    results: int = 0
    dropped: int = 0
    errors: int = 0
    sessions: int = 0
    cost: float = 0.0
    session: dict[str, Any] = field(default_factory=dict)
    session_stats: dict[str, Any] = field(default_factory=dict)


class _Refused(Exception):
    """The gateway will not open this session, and trying again will not help."""


class _TryLater(Exception):
    """The gateway cannot open a session now. ``retry_after`` is its hint, in seconds."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class VLMRunRealtimeProcessor(FrameProcessor):
    """Sends camera frames to a VLM Run realtime model and emits one result per frame read.

    Place it after ``transport.input()`` in a pipeline whose transport has
    ``video_in_enabled=True``. Camera frames pass through untouched. For each frame the model
    reads, the processor pushes a ``VLMRunResultFrame`` downstream, and with ``draw=True`` also
    an ``OutputImageRawFrame`` with the result drawn on the frame.

    Only the newest frame waits for the socket. While a read is in flight, older frames are
    skipped, so results never fall behind the camera.

    The session opens on the first camera frame, because the gateway bills every second a
    session is open. It is reopened when the gateway closes it: after 30 seconds without a
    frame, at the 10 minute session limit, or on a server restart. When the gateway asks for a
    wait, for example because the plan's sessions are full or the model is starting up, the
    processor waits and tries again. When the gateway refuses a session for good, or stays
    unavailable past ``max_retries``, the processor pushes an ``ErrorFrame`` and stops reading.
    Camera frames keep flowing.

    Example::

        detector = VLMRunRealtimeProcessor(
            api_key=os.environ["VLMRUN_API_KEY"],
            model="roboflow/rfdetr-large",
            draw=True,
        )
        pipeline = Pipeline([transport.input(), detector, transport.output()])
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        method: str | None = None,
        method_params: dict[str, Any] | None = None,
        prompt: str | None = None,
        url: str = DEFAULT_WS_URL,
        max_fps: float | None = None,
        video_source: str | None = None,
        draw: bool = False,
        jpeg_quality: int = 80,
        reply_timeout: float = 5.0,
        max_retries: int = 5,
        **kwargs,
    ):
        """Initialize the realtime processor.

        Args:
            api_key: VLM Run API key, from https://app.vlm.run.
            model: A gateway model that streams over the socket, for example
                ``roboflow/rfdetr-large``. ``GET /v1/openai/models`` lists them.
            method: The model's method, for example ``detect``, ``segment``, ``pose`` or
                ``ocr``. Defaults to the model's own default.
            method_params: Arguments for the method, for example ``{"prompt": "person"}``
                for ``facebook/sam3.1``.
            prompt: Text sent with every frame. Used by the realtime chat models.
            url: The gateway's realtime socket.
            max_fps: Upper bound on frames sent per second. The session's own limit also
                applies, and the lower of the two wins.
            video_source: Only read frames from this transport source, for example
                ``"camera"`` or ``"screenVideo"``. Defaults to every source.
            draw: Also push each frame with its result drawn on it.
            jpeg_quality: JPEG quality for frames sent to the gateway, 1 to 95.
            reply_timeout: Seconds to wait for a frame's reply before reopening the session.
            max_retries: Failed attempts in a row before the processor gives up. A model
                that is starting up can ask for a wait of about 4 minutes per attempt.
            **kwargs: Additional arguments passed to ``FrameProcessor``.
        """
        super().__init__(**kwargs)
        self._api_key = api_key
        self._url = url
        self._model = model
        self._method = method
        self._method_params = method_params
        self._prompt = prompt
        self._max_fps = max_fps
        self._video_source = video_source
        self._draw = draw
        self._jpeg_quality = jpeg_quality
        self._reply_timeout = reply_timeout
        self._max_retries = max_retries

        self._ws: Any = None
        self._interval = 0.0
        self._max_frame_bytes = 0
        self._latest: tuple[InputImageRawFrame, float] | None = None
        self._wake = asyncio.Event()
        self._reconfigured = False
        self._task: asyncio.Task | None = None
        self.stats = VLMRunRealtimeStats()

    @property
    def model(self) -> str:
        """The gateway model the next session will open with."""
        return self._model

    async def configure(
        self,
        *,
        model: str | None = None,
        method: str | None = None,
        method_params: dict[str, Any] | None = None,
        prompt: str | None = None,
    ) -> None:
        """Change what the processor reads. Takes effect on the next camera frame.

        A session is fixed to one model, method and prompt, so the current session is closed
        and a new one opens. Arguments left as None keep their current value.

        Args:
            model: A gateway model that streams over the socket.
            method: The model's method.
            method_params: Arguments for the method.
            prompt: Text sent with every frame.
        """
        if model is not None:
            self._model = model
        if method is not None:
            self._method = method
        if method_params is not None:
            self._method_params = method_params
        if prompt is not None:
            self._prompt = prompt
        self._reconfigured = True

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Queue camera frames for the gateway and pass every frame on.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            if self._task is None:
                self._task = self.create_task(self._run(), name="vlmrun-realtime")
            return

        if isinstance(frame, EndFrame | CancelFrame):
            await self._teardown()
        elif isinstance(frame, InputImageRawFrame) and self._wants(frame):
            self.stats.frames_seen += 1
            if self._latest is not None:
                self.stats.frames_skipped += 1
            self._latest = (frame, time.perf_counter())
            self._wake.set()

        await self.push_frame(frame, direction)

    async def cleanup(self):
        """Close the session and stop the reader. Runs on every exit path."""
        await super().cleanup()
        await self._teardown()

    def _wants(self, frame: InputImageRawFrame) -> bool:
        return self._video_source is None or frame.transport_source == self._video_source

    async def _teardown(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            await self.cancel_task(task)
        await self._close_session()
        self._latest = None
        self._wake.clear()

    async def _run(self) -> None:
        try:
            await self._read_frames()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            await self.push_error(f"{self} stopped: {error}", exception=error)

    async def _read_frames(self) -> None:
        failures = 0
        last_sent = 0.0
        while True:
            await self._wake.wait()

            if self._reconfigured:
                self._reconfigured = False
                await self._close_session()

            if self._ws is None:
                try:
                    await self._open_session()
                except _Refused as refused:
                    await self.push_error(f"{self} session refused: {refused}", exception=refused)
                    return
                except Exception as error:
                    failures += 1
                    if failures > self._max_retries:
                        await self.push_error(
                            f"{self} gave up after {failures} attempts: {error}", exception=error
                        )
                        return
                    delay = _backoff(failures, getattr(error, "retry_after", None))
                    logger.warning(
                        f"{self} could not open a session, retrying in {delay:g}s: {error}"
                    )
                    await asyncio.sleep(delay)
                    continue

            wait = self._interval - (time.perf_counter() - last_sent)
            if wait > 0:
                await asyncio.sleep(wait)
            self._wake.clear()
            pending, self._latest = self._latest, None
            if pending is None:
                continue
            frame, seen_at = pending

            jpeg = await asyncio.to_thread(_jpeg, frame, self._jpeg_quality, self._max_frame_bytes)
            if jpeg is None:
                self.stats.errors += 1
                logger.warning(f"{self} skipped a frame larger than the session's frame limit")
                continue

            last_sent = time.perf_counter()
            try:
                await self._ws.send(jpeg)
                self.stats.frames_sent += 1
                reply = await asyncio.wait_for(self._ws.recv(), self._reply_timeout)
            except ConnectionClosed as closed:
                self._ws = None
                code, reason = _close_info(closed)
                if code == _CLOSE_REFUSED or reason == _REASON_SPENT:
                    await self.push_error(f"{self} session ended: {reason or code}")
                    return
                if code != _CLOSE_NORMAL:
                    # 1000 is the gateway's idle or 10 minute close. Anything else is a fault.
                    failures += 1
                    if failures > self._max_retries:
                        await self.push_error(f"{self} session lost {failures} times: {reason}")
                        return
                    await asyncio.sleep(_backoff(failures, None))
                logger.debug(f"{self} session closed ({code} {reason}), reopening")
                self._retry(pending)
                continue
            except Exception as error:
                # A reply that arrives late would answer the wrong frame, so the session
                # cannot be reused after a timeout.
                self.stats.errors += 1
                await self._close_session()
                failures += 1
                if failures > self._max_retries:
                    await self.push_error(
                        f"{self} gave up after {failures} failed frames: {error}", exception=error
                    )
                    return
                logger.warning(f"{self} got no reply for a frame, reopening the session: {error}")
                continue

            now = time.perf_counter()
            event = json.loads(reply)
            kind = event.get("type")
            if kind == "frame.dropped":
                self.stats.dropped += 1
                continue
            if kind != "result":
                self.stats.errors += 1
                await self.push_error(f"{self} frame failed: {event.get('message') or event}")
                continue

            failures = 0
            self.stats.results += 1
            result = _content(event)
            await self.push_frame(
                VLMRunResultFrame(
                    model=self._model,
                    result=result,
                    size=frame.size,
                    frame_index=event.get("frame"),
                    gateway_ms=event.get("latency_ms"),
                    round_trip_ms=(now - last_sent) * 1000,
                    age_ms=(now - seen_at) * 1000,
                )
            )
            if self._draw:
                await self.push_frame(await asyncio.to_thread(_annotated, frame, result))

    def _retry(self, pending: tuple[InputImageRawFrame, float]) -> None:
        """Send a frame again on the new session, unless a newer one has arrived."""
        if self._latest is None:
            self._latest = pending
        self._wake.set()

    async def _open_session(self) -> None:
        opening: dict[str, Any] = {
            "type": "session.create",
            "model": self._model,
            "max_inflight": 1,
        }
        if self._method:
            opening["method"] = self._method
        if self._method_params:
            opening["method_params"] = self._method_params
        if self._prompt:
            opening["prompt"] = self._prompt

        try:
            ws = await ws_connect(
                self._url,
                additional_headers={
                    "Authorization": f"Bearer {self._api_key}",
                    CLIENT_HEADER: client_name(),
                },
                max_size=16 * 1024 * 1024,
                open_timeout=30,
            )
        except InvalidStatus as error:
            status = error.response.status_code
            if status in (401, 403):
                raise _Refused(f"HTTP {status}, check the VLM Run API key") from error
            raise

        try:
            await ws.send(json.dumps(opening))
            created = json.loads(await asyncio.wait_for(ws.recv(), 60))
        except ConnectionClosed as closed:
            # A full plan closes the socket before it answers session.create.
            code, reason = _close_info(closed)
            if code == _CLOSE_REFUSED:
                raise _Refused(reason or "refused") from closed
            raise _TryLater(reason or f"closed with code {code}") from closed
        except BaseException:
            await ws.close()
            raise

        if created.get("type") != "session.created":
            message = str(created.get("message") or created.get("error_type") or created)
            retry_after = created.get("retry_after")
            code = await _await_close(ws)
            if code == _CLOSE_REFUSED:
                raise _Refused(message)
            raise _TryLater(message, retry_after)

        rate = float(created.get("max_fps") or 0)
        if self._max_fps:
            rate = min(rate, self._max_fps) if rate else self._max_fps
        self._interval = 1.0 / rate if rate else 0.0
        self._max_frame_bytes = int(created.get("max_frame_bytes") or 0)
        self._ws = ws
        self.stats.sessions += 1
        self.stats.session = created
        logger.debug(f"{self} opened session {created.get('session_id')} on {self._model}")

    async def _close_session(self) -> None:
        """Close the open session, if any, and record what it cost."""
        ws, self._ws = self._ws, None
        if ws is None:
            return
        try:
            await ws.send(json.dumps({"type": "session.close"}))
            # A reply to a frame still in flight can arrive before session.stats.
            deadline = time.perf_counter() + 3.0
            while (remaining := deadline - time.perf_counter()) > 0:
                event = json.loads(await asyncio.wait_for(ws.recv(), remaining))
                if event.get("type") == "session.stats":
                    self.stats.session_stats = event
                    self.stats.cost += float(event.get("cost") or 0)
                    break
        except Exception:
            # The session is going away either way; the gateway stops billing when the
            # socket closes.
            pass
        finally:
            await ws.close()


def _backoff(failures: int, retry_after: float | None) -> float:
    """Seconds to wait before the next attempt. The gateway's own hint wins when it gives one."""
    if retry_after:
        return min(float(retry_after), _MAX_RETRY_AFTER)
    return min(0.5 * 2 ** (failures - 1), _MAX_BACKOFF)


def _close_info(closed: ConnectionClosed) -> tuple[int | None, str]:
    frame = closed.rcvd
    return (frame.code, frame.reason) if frame is not None else (None, "")


async def _await_close(ws: Any) -> int | None:
    """Wait briefly for the close that follows a refused session, and return its code."""
    try:
        await asyncio.wait_for(ws.recv(), 2)
    except ConnectionClosed as closed:
        return _close_info(closed)[0]
    except Exception:
        pass
    await ws.close()
    return None


def _jpeg(frame: InputImageRawFrame, quality: int, max_bytes: int) -> bytes | None:
    """Encode a camera frame, lowering quality until it fits the session's frame limit."""
    image = Image.frombytes(frame.format or "RGB", frame.size, frame.image).convert("RGB")
    for attempt in (quality, *_JPEG_FALLBACK_QUALITIES):
        out = io.BytesIO()
        image.save(out, format="JPEG", quality=attempt)
        data = out.getvalue()
        if not max_bytes or len(data) <= max_bytes:
            return data
    return None


def _content(event: dict[str, Any]) -> dict[str, Any]:
    """The model's reply for one frame, as a dict."""
    content = event["response"]["choices"][0]["message"]["content"]
    try:
        parsed = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return {"text": content}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def _annotated(frame: InputImageRawFrame, result: dict[str, Any]) -> OutputImageRawFrame:
    image = Image.frombytes(frame.format or "RGB", frame.size, frame.image)
    drawn = annotate(image, result)
    return OutputImageRawFrame(image=drawn.tobytes(), size=drawn.size, format="RGB")
