from app.timing import TimingRecorder


def test_timing_recorder_records_named_blocks() -> None:
    recorder = TimingRecorder()

    with recorder.measure("retrieval"):
        pass
    timings = recorder.finish()

    assert "retrieval" in timings
    assert "total" in timings
