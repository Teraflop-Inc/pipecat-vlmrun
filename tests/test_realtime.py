import pytest
from conftest import API_KEY, camera_frame, run_pipeline

from pipecat_vlmrun import VLMRunRealtimeProcessor


def processor(gateway, **kwargs):
    kwargs.setdefault("model", "roboflow/rfdetr-large")
    kwargs.setdefault("api_key", API_KEY)
    return VLMRunRealtimeProcessor(url=gateway.url, **kwargs)


async def test_each_frame_read_gives_a_result(gateway):
    proc = processor(gateway, method="detect")
    run = await run_pipeline(proc, [camera_frame() for _ in range(5)], gap=0.05)

    assert run.errors == []
    assert run.sink.camera == 5, "camera frames pass through untouched"
    assert len(run.sink.results) == 5
    first = run.sink.results[0]
    assert first.model == "roboflow/rfdetr-large"
    assert first.result["object"] == "img.detect.bboxes"
    assert first.result["items"][0]["label"] == "person"
    assert first.size == (64, 48)
    assert first.frame_index == 1
    assert first.gateway_ms == 5.0
    assert first.round_trip_ms > 0 and first.age_ms >= first.round_trip_ms
    assert gateway.openings[0] == {
        "type": "session.create",
        "model": "roboflow/rfdetr-large",
        "max_inflight": 1,
        "method": "detect",
    }


async def test_session_opens_on_the_first_frame_and_is_closed_at_the_end(gateway):
    proc = processor(gateway)
    run = await run_pipeline(proc, [])
    assert gateway.connections == 0, "no frames, no billed session"

    proc = processor(gateway)
    run = await run_pipeline(proc, [camera_frame()])
    assert gateway.connections == 1
    assert gateway.closes == 1, "session.close was sent"
    assert proc.stats.cost == pytest.approx(0.001)
    assert proc.stats.session_stats["type"] == "session.stats"
    assert run.errors == []


async def test_identifies_itself_to_the_gateway(gateway):
    await run_pipeline(processor(gateway), [camera_frame()])
    assert gateway.headers[0]["x-vlmrun-client"].startswith("pipecat-vlmrun/")


async def test_method_params_and_prompt_are_sent(gateway):
    proc = processor(
        gateway,
        model="facebook/sam3.1",
        method="segment",
        method_params={"prompt": "person"},
        prompt="what is here",
    )
    await run_pipeline(proc, [camera_frame()])
    opening = gateway.openings[0]
    assert opening["method_params"] == {"prompt": "person"}
    assert opening["prompt"] == "what is here"


async def test_draw_pushes_an_annotated_frame(gateway):
    proc = processor(gateway, draw=True)
    run = await run_pipeline(proc, [camera_frame()], settle=0.5)
    assert len(run.sink.drawn) == 1
    drawn = run.sink.drawn[0]
    assert drawn.size == (64, 48)
    assert drawn.image != camera_frame().image, "the box was drawn"


async def test_only_the_newest_frame_waits(gateway):
    gateway.reply_delay = 0.1
    proc = processor(gateway)
    run = await run_pipeline(proc, [camera_frame() for _ in range(20)], gap=0.01, settle=0.5)
    assert proc.stats.frames_seen == 20
    assert proc.stats.frames_skipped > 0
    assert 0 < len(run.sink.results) < 20
    assert proc.stats.frames_sent + proc.stats.frames_skipped == 20


async def test_max_fps_paces_frames(gateway):
    proc = processor(gateway, max_fps=10)
    run = await run_pipeline(proc, [camera_frame() for _ in range(30)], gap=0.01, settle=0.1)
    assert len(run.sink.results) <= 6, "about 0.4 s at 10 fps"


async def test_text_reply_becomes_a_text_result(gateway):
    gateway.content = "A person at a desk."
    run = await run_pipeline(
        processor(gateway, model="google/gemma4-e2b-it:realtime"), [camera_frame()]
    )
    assert run.sink.results[0].result == {"text": "A person at a desk."}


async def test_video_source_filters_frames(gateway):
    proc = processor(gateway, video_source="camera")
    frames = [camera_frame(source="screenVideo"), camera_frame(source="camera")]
    run = await run_pipeline(proc, frames, gap=0.1)
    assert run.sink.camera == 2
    assert proc.stats.frames_seen == 1
    assert len(run.sink.results) == 1


async def test_reopens_after_the_gateway_closes_an_idle_session(gateway):
    gateway.close_after = {0: (1, 1000, "idle")}
    proc = processor(gateway)
    run = await run_pipeline(proc, [camera_frame() for _ in range(4)], gap=0.1)
    assert run.errors == []
    assert proc.stats.sessions == 2
    assert len(run.sink.results) == 4, "the frame that hit the closed socket was sent again"


async def test_reopens_after_a_server_restart(gateway):
    gateway.close_after = {0: (1, 1012, "service restart")}
    proc = processor(gateway)
    run = await run_pipeline(proc, [camera_frame() for _ in range(3)], gap=0.1)
    assert run.errors == []
    assert proc.stats.sessions == 2
    assert len(run.sink.results) == 3


async def test_waits_out_a_full_plan(gateway):
    gateway.full_for = 2
    proc = processor(gateway)
    run = await run_pipeline(proc, [camera_frame() for _ in range(3)], gap=0.1)
    assert run.errors == []
    assert gateway.connections == 3
    assert len(run.sink.results) >= 1


async def test_gives_up_on_a_plan_that_stays_full(gateway):
    gateway.full_for = 100
    proc = processor(gateway, max_retries=2)
    run = await run_pipeline(proc, [camera_frame() for _ in range(3)], gap=0.1)
    assert gateway.connections == 3, "the first attempt and two retries"
    assert len(run.errors) == 1 and "gave up after 3 attempts" in run.errors[0]
    assert run.sink.camera == 3, "camera frames keep flowing"


async def test_a_refused_session_is_not_retried(gateway):
    gateway.refuse = ("Model 'nope' not found.", 1008, None)
    proc = processor(gateway, model="nope")
    run = await run_pipeline(proc, [camera_frame() for _ in range(3)], gap=0.05)
    assert gateway.connections == 1
    assert len(run.errors) == 1 and "Model 'nope' not found." in run.errors[0]
    assert run.sink.results == []
    assert run.sink.camera == 3


async def test_a_long_retry_after_is_reported_not_waited_out(gateway):
    gateway.refuse = ("model is starting up. Retry in about 240 s.", 1013, 240.0)
    proc = processor(gateway)
    run = await run_pipeline(proc, [camera_frame()])
    assert gateway.connections == 1
    assert "starting up" in run.errors[0]


async def test_a_bad_key_is_not_retried(gateway):
    proc = processor(gateway, api_key="wrong")
    run = await run_pipeline(proc, [camera_frame() for _ in range(2)], gap=0.05)
    assert gateway.connections == 0, "refused at the HTTP upgrade"
    assert len(run.errors) == 1 and "API key" in run.errors[0]


async def test_a_reply_that_never_comes_reopens_the_session(gateway):
    gateway.silent = {0}
    proc = processor(gateway, reply_timeout=0.1)
    run = await run_pipeline(proc, [camera_frame() for _ in range(6)], gap=0.1)
    assert run.errors == []
    assert proc.stats.sessions == 2
    assert proc.stats.errors == 1
    assert len(run.sink.results) >= 1


async def test_stops_at_the_spend_ceiling(gateway):
    gateway.close_after = {0: (1, 1000, "spent")}
    proc = processor(gateway)
    run = await run_pipeline(proc, [camera_frame() for _ in range(4)], gap=0.1)
    assert gateway.connections == 1, "did not reopen past the spend limit"
    assert len(run.errors) == 1 and "spent" in run.errors[0]


async def test_dropped_frames_are_counted(gateway):
    gateway.drop_frames = True
    proc = processor(gateway)
    run = await run_pipeline(proc, [camera_frame() for _ in range(3)], gap=0.05)
    assert run.sink.results == []
    assert proc.stats.dropped == 3
    assert run.errors == []


async def test_configure_opens_a_new_session_on_the_new_model(gateway):
    proc = processor(gateway)

    async def switch(index):
        if index == 1:
            await proc.configure(model="usyd-community/vitpose-plus-large", method="pose")

    run = await run_pipeline(proc, [camera_frame() for _ in range(4)], gap=0.1, during=switch)
    assert [opening["model"] for opening in gateway.openings] == [
        "roboflow/rfdetr-large",
        "usyd-community/vitpose-plus-large",
    ]
    assert gateway.openings[1]["method"] == "pose"
    assert run.sink.results[-1].model == "usyd-community/vitpose-plus-large"
    assert gateway.closes == 2, "both sessions were closed, so neither keeps billing"
