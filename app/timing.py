from time import perf_counter


class TimingRecorder:
    def __init__(self) -> None:
        self._total_start = perf_counter()
        self.timings: dict[str, float] = {}

    def measure(self, name: str):
        return _TimingBlock(self, name)

    def add(self, name: str, seconds: float) -> None:
        self.timings[name] = round(self.timings.get(name, 0.0) + seconds, 4)

    def set(self, name: str, value: float) -> None:
        self.timings[name] = round(value, 4)

    def finish(self) -> dict[str, float]:
        self.timings["total"] = round(perf_counter() - self._total_start, 4)
        return dict(self.timings)


class _TimingBlock:
    def __init__(self, recorder: TimingRecorder, name: str) -> None:
        self.recorder = recorder
        self.name = name
        self.start = 0.0

    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.recorder.add(self.name, perf_counter() - self.start)
