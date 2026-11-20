#
# Copyright (c) 2026, VLM Run
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Run a VLM Run realtime model on your camera and send the video back with the result drawn on.

Needs only a VLM Run API key:

    VLMRUN_API_KEY=... uv run examples/realtime-vision.py

Then open http://localhost:7860/client and allow the camera.

Pick the model with VLMRUN_REALTIME_MODEL. The realtime models are:

    roboflow/rfdetr-large               object detection (the default)
    roboflow/rfdetr-seg-large           instance segmentation
    facebook/sam3.1                     segmentation by prompt, set VLMRUN_REALTIME_PROMPT
    usyd-community/vitpose-plus-large   body pose
    geopavlakos/hamer                   hand pose
    paddleocr/pp-ocrv6                  text in the scene
    google/gemma4-e2b-it:realtime       one-line description, set VLMRUN_REALTIME_PROMPT
    qwen/qwen3.5-0.8b:realtime          one-line description, set VLMRUN_REALTIME_PROMPT
"""

import os

from dotenv import load_dotenv
from loguru import logger
from pipecat.frames.frames import Frame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.transports.base_transport import BaseTransport, TransportParams, VideoInSourceParams
from pipecat.workers.runner import WorkerRunner

from pipecat_vlmrun import VLMRunRealtimeProcessor, VLMRunResultFrame

load_dotenv(override=True)

MODEL = os.getenv("VLMRUN_REALTIME_MODEL", "roboflow/rfdetr-large")
PROMPT = os.getenv("VLMRUN_REALTIME_PROMPT")

VIDEO = dict(
    video_in_enabled=True,
    video_in_sources={"camera": VideoInSourceParams(framerate=30)},
    video_out_enabled=True,
    video_out_is_live=True,
    video_out_width=1280,
    video_out_height=720,
)

# We use lambdas to defer transport parameter creation until the transport
# type is selected at runtime.
transport_params = {
    "webrtc": lambda: TransportParams(**VIDEO),
}


class ResultLogger(FrameProcessor):
    """Logs what the model saw, about once a second."""

    def __init__(self):
        super().__init__()
        self._count = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, VLMRunResultFrame):
            self._count += 1
            if self._count % 20 == 1:
                logger.info(
                    f"{frame.model} in {frame.round_trip_ms:.0f} ms: {describe(frame.result)}"
                )
        await self.push_frame(frame, direction)


def describe(result: dict) -> str:
    if "text" in result:
        return result["text"]
    items = result.get("items") or []
    names = [item.get("label") or item.get("text") or item.get("side") or "item" for item in items]
    return ", ".join(names) or "nothing"


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot with {MODEL}")

    vision = VLMRunRealtimeProcessor(
        api_key=os.environ["VLMRUN_API_KEY"],
        model=MODEL,
        # SAM 3.1 segments what its prompt names; the chat models are asked the prompt.
        method_params={"prompt": PROMPT or "person"} if MODEL.startswith("facebook/sam") else None,
        prompt=PROMPT if "realtime" in MODEL else None,
        video_source="camera",
        draw=True,
    )

    pipeline = Pipeline(
        [
            transport.input(),  # Camera frames from the user
            vision,  # One result per frame the model reads, plus the frame with it drawn on
            ResultLogger(),
            transport.output(),  # The drawn frames, back to the user
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    await runner.add_workers(worker)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected. Session totals: {vision.stats}")
        await runner.cancel()

    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
