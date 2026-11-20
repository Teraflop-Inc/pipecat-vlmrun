# pipecat-vlmrun

[VLM Run](https://vlm.run) vision models for [Pipecat](https://github.com/pipecat-ai/pipecat).

The [VLM Run gateway](https://docs.vlm.run/gateway/introduction) serves vision models behind one
API key. This package gives a Pipecat bot two ways to use them:

- **`VLMRunLLMService`**: a chat model that can see. Use it as the LLM in a voice bot, and put
  camera frames or images in the conversation.
- **`VLMRunRealtimeProcessor`**: a per-frame vision model on the bot's live camera. It reads
  every frame it can and emits detections, masks, poses, text or a one-line description.

This integration is maintained by VLM Run.

## Installation

```bash
uv add pipecat-vlmrun
```

or

```bash
pip install pipecat-vlmrun
```

Get an API key at [app.vlm.run](https://app.vlm.run) and set it as `VLMRUN_API_KEY`.

## Chat: `VLMRunLLMService`

A drop-in LLM service. It talks to the gateway's OpenAI-compatible Chat Completions API.

```python
import os

from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat_vlmrun import VLMRunLLMService

llm = VLMRunLLMService(
    api_key=os.environ["VLMRUN_API_KEY"],
    settings=VLMRunLLMService.Settings(
        model="qwen/qwen3.8-27b",
        system_instruction="You are a helpful assistant in a voice conversation.",
    ),
)

context = LLMContext()
user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context)

pipeline = Pipeline(
    [
        transport.input(),
        stt,
        user_aggregator,
        llm,
        tts,
        transport.output(),
        assistant_aggregator,
    ]
)
```

To ask about an image or a camera frame, add it to the context:

```python
message = await LLMContext.create_image_message(
    image=frame.image, format=frame.format, size=frame.size, text="What do you see?"
)
context.add_message(message)
await worker.queue_frames([LLMRunFrame()])
```

VLM Run has no speech models that stream, so a voice bot still needs a speech-to-text and a
text-to-speech service from another provider.

### Why not `OpenAILLMService` with a `base_url`?

The gateway is stricter than OpenAI about message order, and it answers `400` to messages that
Pipecat bots commonly send. `VLMRunLLMService` reshapes them before each request:

| Pipecat sends | The gateway says | This service |
| --- | --- | --- |
| A `developer` message, which Pipecat's examples use to start a bot | `400` | Sends it as a `user` message |
| A `system` message after the first | `400` "System message must be at the beginning" | Merges it into the first when it directly follows it, otherwise sends it as a `user` message |
| No `user` message, for example a bot started on its system prompt alone | `400` "No user query found in messages" | Adds a placeholder `user` message |

### Models

The default is `qwen/qwen3.8-27b`, which supports tool calling. `google/gemma4-26b-a4b-it` is
faster to its first word but has no tool calling. `GET https://gateway.vlm.run/v1/openai/models`
lists every model.

Qwen thinks before it answers unless told not to, which delays the first word. For a voice bot,
turn it off:

```python
VLMRunLLMService.Settings(
    model="qwen/qwen3.8-27b",
    extra={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
)
```

Other gateway-only request fields, such as `method`, also go in `extra["extra_body"]`.

## Realtime: `VLMRunRealtimeProcessor`

Put it after `transport.input()` in a pipeline whose transport has `video_in_enabled=True`.

```python
import os

from pipecat.pipeline.pipeline import Pipeline
from pipecat_vlmrun import VLMRunRealtimeProcessor

detector = VLMRunRealtimeProcessor(
    api_key=os.environ["VLMRUN_API_KEY"],
    model="roboflow/rfdetr-large",
    video_source="camera",
    draw=True,
)

pipeline = Pipeline([transport.input(), detector, transport.output()])
```

For each frame the model reads, the processor pushes a `VLMRunResultFrame` downstream. With
`draw=True` it also pushes the frame with the result drawn on it, as an `OutputImageRawFrame`
that `transport.output()` sends back to the user. Camera frames pass through untouched.

Read results in any processor placed after it:

```python
from pipecat.processors.frame_processor import FrameProcessor
from pipecat_vlmrun import VLMRunResultFrame


class PersonCounter(FrameProcessor):
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, VLMRunResultFrame):
            people = [item for item in frame.result["items"] if item["label"] == "person"]
            print(len(people), "people,", round(frame.round_trip_ms), "ms")
        await self.push_frame(frame, direction)
```

### Realtime models

| Model | What each frame returns | `result["object"]` |
| --- | --- | --- |
| `roboflow/rfdetr-large` | Object boxes with labels | `img.detect.bboxes` |
| `roboflow/rfdetr-seg-large` | Object boxes and instance masks | `img.segment.masks` |
| `facebook/sam3.1` | Masks for what `method_params={"prompt": "..."}` names | `img.segment.masks` |
| `usyd-community/vitpose-plus-large` | Body keypoints per person | `img.pose.kpts` |
| `geopavlakos/hamer` | Hand keypoints, 2D and 3D | `img.pose.hands3d` |
| `paddleocr/pp-ocrv6` (alias `pp-ocrv6:realtime`) | Text lines with boxes | `doc.ocr.lines` |
| `google/gemma4-e2b-it:realtime` | An answer to `prompt`, as `{"text": ...}` | none |
| `qwen/qwen3.5-0.8b:realtime` | An answer to `prompt`, as `{"text": ...}` | none |

Boxes are `bbox_xywh` and points are `kpts_xy`, both as fractions of the frame's width and
height. The gateway's catalog changes; `GET /v1/openai/models` is the source of truth.

### Result frame

`VLMRunResultFrame` fields:

| Field | Meaning |
| --- | --- |
| `model` | The gateway model id |
| `result` | The model's JSON for the frame |
| `size` | Width and height of the frame it was computed on |
| `frame_index` | The gateway's frame count for the session |
| `gateway_ms` | Model time the gateway reports |
| `round_trip_ms` | Sending the frame to receiving the reply |
| `age_ms` | The camera frame entering the processor to the result leaving it |

### Options

| Argument | Default | Meaning |
| --- | --- | --- |
| `model` | required | A gateway model that streams over the socket |
| `method` | the model's default | For example `detect`, `segment`, `pose`, `ocr` |
| `method_params` | none | Arguments for the method |
| `prompt` | none | Text sent with every frame, for the realtime chat models |
| `max_fps` | the session's limit | Upper bound on frames sent per second |
| `video_source` | every source | Only read frames from this source, for example `"camera"` |
| `draw` | `False` | Also push each frame with its result drawn on it |
| `jpeg_quality` | `80` | JPEG quality of frames sent to the gateway |
| `reply_timeout` | `5.0` | Seconds to wait for a reply before reopening the session |
| `max_retries` | `5` | Failed attempts in a row before the processor gives up |

`await processor.configure(model=..., method=..., method_params=..., prompt=...)` changes what
it reads while the bot is running. The change takes effect on the next camera frame.

`processor.stats` counts frames seen, sent and skipped, results, errors and sessions.

### Sessions, limits and billing

- The gateway bills every second a realtime session is open, so the session opens on the
  first camera frame and closes when the pipeline ends.
- Only the newest frame waits for the socket. Older frames are skipped while a read is in
  flight, so results do not fall behind the camera.
- The gateway closes a session after 30 seconds without a frame, and at 10 minutes. The
  processor opens a new one on the next frame.
- A plan holds a fixed number of sessions at once. When it is full, the processor retries
  with backoff, up to `max_retries`.
- When the gateway refuses a session for good (unknown model, bad key, spend limit), or
  stays unavailable past `max_retries`, the processor pushes an `ErrorFrame` and stops
  reading. Camera frames keep flowing.

See [Realtime](https://docs.vlm.run/gateway/realtime) in the VLM Run docs for limits and prices.

## Examples

```bash
git clone https://github.com/vlm-run/pipecat-vlmrun
cd pipecat-vlmrun
uv sync --extra examples
cp .env.example .env   # then fill in the keys
```

**`examples/realtime-vision.py`** runs a realtime model on your camera and sends the video back
with the result drawn on. It needs only a VLM Run key.

```bash
uv run examples/realtime-vision.py
```

Open http://localhost:7860/client, turn the camera on, and click Connect. Pick another model
with `VLMRUN_REALTIME_MODEL`, for example:

```bash
VLMRUN_REALTIME_MODEL=usyd-community/vitpose-plus-large uv run examples/realtime-vision.py
VLMRUN_REALTIME_MODEL=facebook/sam3.1 VLMRUN_REALTIME_PROMPT=cup uv run examples/realtime-vision.py
```

**`examples/vision-chat.py`** is a voice bot that describes an image and then talks about it.
It is Pipecat's own `vision-openai.py` example with the LLM swapped for `VLMRunLLMService`. It
also needs `DEEPGRAM_API_KEY` and `CARTESIA_API_KEY` for speech.

```bash
uv run examples/vision-chat.py
```

## Tests

```bash
uv sync
uv run pytest                 # unit tests, against a fake gateway, no key needed
uv run pytest -m live -s      # against the real gateway, needs VLMRUN_API_KEY, costs a few cents
```

## Compatibility

Tested with Pipecat v1.16.0 on Python 3.12.

## License

BSD 2-Clause. See [LICENSE](LICENSE).
