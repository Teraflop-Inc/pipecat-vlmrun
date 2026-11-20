"""A fake VLM Run realtime gateway and a small pipeline harness."""

import asyncio
import json
from dataclasses import dataclass, field

import pytest
import pytest_asyncio
from PIL import Image
from pipecat.frames.frames import (
    EndFrame,
    Frame,
    InputImageRawFrame,
    OutputImageRawFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.workers.runner import WorkerRunner
from websockets.asyncio.server import serve

from pipecat_vlmrun import VLMRunResultFrame

API_KEY = "test-key"
DETECTION = {
    "object": "img.detect.bboxes",
    "items": [{"bbox_xywh": [0.1, 0.2, 0.3, 0.4], "label": "person", "score": 0.9}],
}


@dataclass
class FakeGateway:
    """Scripted stand-in for wss://gateway.vlm.run/v1/ws."""

    content: object = field(default_factory=lambda: dict(DETECTION))
    max_fps: float = 1000.0
    reply_delay: float = 0.0
    # Connections, counted from the first, that are closed with 1013 before session.create.
    full_for: int = 0
    # Answer session.create with this error and close with this code.
    refuse: tuple[str, int, float | None] | None = None
    # Per connection index: (results before closing, close code, close reason).
    close_after: dict[int, tuple[int, int, str]] = field(default_factory=dict)
    # Connections, by index, that never answer a frame.
    silent: set[int] = field(default_factory=set)
    drop_frames: bool = False

    url: str = ""
    connections: int = 0
    openings: list[dict] = field(default_factory=list)
    frames: int = 0
    closes: int = 0
    headers: list[dict] = field(default_factory=list)

    def process_request(self, connection, request):
        if request.headers.get("Authorization") != f"Bearer {API_KEY}":
            return connection.respond(401, '{"detail":"Invalid API Key"}')
        return None

    async def handle(self, ws):
        index = self.connections
        self.connections += 1
        self.headers.append(dict(ws.request.headers))
        if index < self.full_for:
            await ws.close(1013, "The plan holds 2 WebSocket session(s) at once")
            return
        opening = json.loads(await ws.recv())
        self.openings.append(opening)
        if self.refuse is not None:
            message, code, retry_after = self.refuse
            await ws.send(
                json.dumps({"type": "error", "message": message, "retry_after": retry_after})
            )
            await ws.close(code, "")
            return
        await ws.send(
            json.dumps(
                {
                    "type": "session.created",
                    "session_id": f"s{index}",
                    "model": opening["model"],
                    "max_fps": self.max_fps,
                    "max_inflight": 1,
                    "max_frame_bytes": 2 * 1024 * 1024,
                }
            )
        )
        results = 0
        async for message in ws:
            if isinstance(message, str):
                if json.loads(message).get("type") == "session.close":
                    self.closes += 1
                    await ws.send(
                        json.dumps({"type": "session.stats", "reads": results, "cost": 0.001})
                    )
                    await ws.close(1000, "")
                    return
                continue
            self.frames += 1
            if index in self.silent:
                continue
            if self.reply_delay:
                await asyncio.sleep(self.reply_delay)
            if self.drop_frames:
                await ws.send(
                    json.dumps({"type": "frame.dropped", "frame": self.frames, "reason": "busy"})
                )
                continue
            content = self.content if isinstance(self.content, str) else json.dumps(self.content)
            await ws.send(
                json.dumps(
                    {
                        "type": "result",
                        "frame": results + 1,
                        "latency_ms": 5.0,
                        "response": {"choices": [{"message": {"content": content}}]},
                    }
                )
            )
            results += 1
            if index in self.close_after and results >= self.close_after[index][0]:
                _, code, reason = self.close_after[index]
                await ws.close(code, reason)
                return


@pytest_asyncio.fixture
async def gateway():
    fake = FakeGateway()
    async with serve(fake.handle, "127.0.0.1", 0, process_request=fake.process_request) as server:
        port = server.sockets[0].getsockname()[1]
        fake.url = f"ws://127.0.0.1:{port}"
        yield fake


@pytest.fixture(autouse=True)
def fast_backoff(monkeypatch):
    monkeypatch.setattr("pipecat_vlmrun.realtime._backoff", lambda failures, retry_after: 0.01)


def camera_frame(size=(64, 48), color=(10, 120, 200), source=None) -> InputImageRawFrame:
    image = Image.new("RGB", size, color)
    frame = InputImageRawFrame(image=image.tobytes(), size=size, format="RGB")
    frame.transport_source = source
    return frame


class Sink(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.results: list[VLMRunResultFrame] = []
        self.drawn: list[OutputImageRawFrame] = []
        self.camera = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, VLMRunResultFrame):
            self.results.append(frame)
        elif isinstance(frame, OutputImageRawFrame):
            self.drawn.append(frame)
        elif isinstance(frame, InputImageRawFrame):
            self.camera += 1
        await self.push_frame(frame, direction)


@dataclass
class Run:
    sink: Sink
    errors: list[str]


async def run_pipeline(processor, frames, *, gap=0.02, settle=0.3, during=None) -> Run:
    """Feed camera frames through [processor, sink] and return what came out."""
    sink = Sink()
    worker = PipelineWorker(
        Pipeline([processor, sink]), cancel_on_idle_timeout=False, params=PipelineParams()
    )
    errors: list[str] = []
    started = asyncio.Event()

    @worker.event_handler("on_pipeline_started")
    async def _on_started(worker, frame):
        started.set()

    @worker.event_handler("on_pipeline_error")
    async def _on_error(worker, frame):
        errors.append(str(frame.error))

    async def feed():
        await asyncio.wait_for(started.wait(), 10)
        for index, frame in enumerate(frames):
            await worker.queue_frame(frame)
            await asyncio.sleep(gap)
            if during is not None:
                await during(index)
        await asyncio.sleep(settle)
        await worker.queue_frame(EndFrame())

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    await asyncio.wait_for(asyncio.gather(runner.run(), feed()), 30)
    return Run(sink=sink, errors=errors)
