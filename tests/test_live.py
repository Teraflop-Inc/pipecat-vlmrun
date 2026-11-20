"""Tests against the real VLM Run gateway. They spend a few cents of credit.

Run with: VLMRUN_API_KEY=... uv run pytest -m live
"""

import asyncio
import io
import os
import time
import urllib.request

import pytest
from conftest import run_pipeline
from PIL import Image
from pipecat.frames.frames import (
    EndFrame,
    ErrorFrame,
    Frame,
    InputImageRawFrame,
    LLMFullResponseEndFrame,
    LLMRunFrame,
    LLMTextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.workers.runner import WorkerRunner

from pipecat_vlmrun import VLMRunLLMService, VLMRunRealtimeProcessor

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.environ.get("VLMRUN_API_KEY"), reason="VLMRUN_API_KEY is not set"),
]

# The sample image VLM Run's own realtime docs use.
IMAGE_URL = (
    "https://storage.googleapis.com/vlm-data-public-prod/hub/examples/image.agent/"
    "lunch-skyscraper.jpg"
)
SYSTEM = "You are a helpful assistant in a voice conversation. Answer in one short sentence."
# Qwen thinks before it answers unless told not to, which adds seconds to the first word.
NO_THINKING = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}

REALTIME_MODELS = [
    ("roboflow/rfdetr-large", None, None, None, "img.detect.bboxes"),
    ("roboflow/rfdetr-seg-large", None, None, None, "img.segment.masks"),
    ("facebook/sam3.1", "segment", {"prompt": "person"}, None, "img.segment.masks"),
    ("usyd-community/vitpose-plus-large", None, None, None, "img.pose.kpts"),
    ("geopavlakos/hamer", None, None, None, "img.pose.hands3d"),
    ("paddleocr/pp-ocrv6", None, None, None, "doc.ocr.lines"),
    ("pp-ocrv6:realtime", None, None, None, "doc.ocr.lines"),
    ("google/gemma4-e2b-it:realtime", None, None, "Describe the scene in one sentence.", None),
    ("qwen/qwen3.5-0.8b:realtime", None, None, "Describe the scene in one sentence.", None),
]


@pytest.fixture(scope="module")
def photo() -> Image.Image:
    with urllib.request.urlopen(IMAGE_URL, timeout=30) as response:
        image = Image.open(io.BytesIO(response.read())).convert("RGB")
    image.thumbnail((640, 640))
    return image


def camera_frames(photo: Image.Image, count: int) -> list[InputImageRawFrame]:
    data = photo.tobytes()
    return [InputImageRawFrame(image=data, size=photo.size, format="RGB") for _ in range(count)]


@pytest.mark.parametrize(
    "model,method,method_params,prompt,payload",
    REALTIME_MODELS,
    ids=[case[0] for case in REALTIME_MODELS],
)
async def test_realtime_model(photo, model, method, method_params, prompt, payload):
    processor = VLMRunRealtimeProcessor(
        api_key=os.environ["VLMRUN_API_KEY"],
        model=model,
        method=method,
        method_params=method_params,
        prompt=prompt,
        draw=True,
        reply_timeout=20,
    )
    run = await run_pipeline(processor, camera_frames(photo, 20), gap=0.1, settle=2.0)

    assert run.errors == []
    assert processor.stats.results >= 1
    assert len(run.sink.results) == processor.stats.results
    assert len(run.sink.drawn) == len(run.sink.results)
    result = run.sink.results[-1].result
    if payload is None:
        assert isinstance(result.get("text"), str) and result["text"].strip()
    else:
        assert result["object"] == payload
        assert isinstance(result["items"], list)
    assert processor.stats.session_stats.get("type") == "session.stats", "the session was closed"
    trips = sorted(frame.round_trip_ms for frame in run.sink.results)
    print(
        f"\n{model}: {processor.stats.results} results from 20 frames, "
        f"round trip {trips[len(trips) // 2]:.0f} ms median, ${processor.stats.cost:.4f}"
    )


class _Recorder(FrameProcessor):
    def __init__(self, turn: dict, watch: FrameDirection):
        super().__init__(enable_direct_mode=True)
        self._turn = turn
        self._watch = watch

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if direction == self._watch:
            if isinstance(frame, LLMTextFrame):
                self._turn.setdefault("first_text", time.perf_counter())
                self._turn["text"] = self._turn.get("text", "") + frame.text
            elif isinstance(frame, LLMFullResponseEndFrame):
                self._turn["done"].set()
            elif isinstance(frame, ErrorFrame):
                self._turn["error"] = str(frame.error)
        await self.push_frame(frame, direction)


async def ask(llm: VLMRunLLMService, messages: list[dict]) -> dict:
    """One turn through [user aggregator, llm, assistant aggregator], as Pipecat's examples do."""
    turn: dict = {"done": asyncio.Event()}
    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context)
    pipeline = Pipeline(
        [
            _Recorder(turn, FrameDirection.UPSTREAM),
            user_aggregator,
            llm,
            _Recorder(turn, FrameDirection.DOWNSTREAM),
            assistant_aggregator,
        ]
    )
    worker = PipelineWorker(pipeline, cancel_on_idle_timeout=False, params=PipelineParams())
    started = asyncio.Event()

    @worker.event_handler("on_pipeline_started")
    async def _on_started(worker, frame):
        started.set()

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    running = asyncio.create_task(runner.run())
    await asyncio.wait_for(started.wait(), 10)
    context.set_messages(messages)
    await worker.queue_frames([LLMRunFrame()])
    try:
        await asyncio.wait_for(turn["done"].wait(), 90)
    except TimeoutError:
        turn.setdefault("error", "no end of response within 90 s")
    await asyncio.sleep(0.05)
    await worker.queue_frame(EndFrame())
    await asyncio.wait_for(running, 20)
    return turn


def chat(model: str = "qwen/qwen3.8-27b") -> VLMRunLLMService:
    extra = NO_THINKING if model.startswith("qwen/") else {}
    settings = VLMRunLLMService.Settings(
        model=model, system_instruction=SYSTEM, max_tokens=64, extra=extra
    )
    return VLMRunLLMService(api_key=os.environ["VLMRUN_API_KEY"], settings=settings)


# Each case is a way Pipecat's own examples start or steer a bot.
CHAT_CASES = {
    "developer kickoff": [{"role": "developer", "content": "Please introduce yourself."}],
    "system instruction only": [],
    "system message first": [
        {"role": "system", "content": "The user's name is Sam."},
        {"role": "user", "content": "What is my name?"},
    ],
    "system message mid-conversation": [
        {"role": "user", "content": "Hi."},
        {"role": "assistant", "content": "Hello."},
        {"role": "system", "content": "The user's name is Sam."},
        {"role": "user", "content": "What is my name?"},
    ],
    "plain user turn": [{"role": "user", "content": "Say hello."}],
}


@pytest.mark.parametrize("name", CHAT_CASES)
async def test_chat_accepts_the_ways_pipecat_starts_a_bot(name):
    turn = await ask(chat(), [dict(message) for message in CHAT_CASES[name]])
    assert turn.get("error") is None
    assert turn.get("text", "").strip()


@pytest.mark.parametrize("model", ["qwen/qwen3.8-27b", "google/gemma4-26b-a4b-it"])
async def test_chat_describes_a_camera_frame(photo, model):
    message = await LLMContext.create_image_message(
        image=photo.tobytes(),
        format="RGB",
        size=photo.size,
        text="How many people are in this picture? Answer with a number.",
    )
    turn = await ask(chat(model), [message])
    assert turn.get("error") is None
    assert any(character.isdigit() for character in turn.get("text", "")), turn.get("text")
