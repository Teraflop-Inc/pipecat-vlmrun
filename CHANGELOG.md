# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0]

### Added

- `VLMRunLLMService`: chat completions on the VLM Run gateway, with images in the LLM context.
  Reshapes `developer` messages, system messages after the first, and requests with no user
  message, which the gateway rejects.
- `VLMRunRealtimeProcessor`: per-frame vision on the gateway's realtime socket, emitting a
  `VLMRunResultFrame` per frame read. Works with `roboflow/rfdetr-large`,
  `roboflow/rfdetr-seg-large`, `facebook/sam3.1`, `usyd-community/vitpose-plus-large`,
  `geopavlakos/hamer`, `paddleocr/pp-ocrv6`, `google/gemma4-e2b-it:realtime` and
  `qwen/qwen3.5-0.8b:realtime`.
- Sessions reopen after the gateway's idle and 10 minute closes, retry with backoff when a
  plan's sessions are full, and wait for a model that is starting up.
- `annotate`: draws boxes, masks, keypoints, text lines and captions on a frame.
- Example: `realtime-vision.py`.
