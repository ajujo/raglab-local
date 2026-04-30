"""Performance timer for RAG-Lab pipeline phases.

Provides a simple context-manager-like timer for measuring latency
of individual pipeline phases.
"""

import time
from typing import Dict


class PhaseTimer:
    """Measures execution time for named pipeline phases."""

    def __init__(self):
        self._timers: Dict[str, float] = {}
        self._current_phase: str | None = None
        self._start_time: float = 0.0

    def start(self, phase_name: str) -> None:
        """Start timing a specific phase."""
        if self._current_phase is not None:
            self.stop()
        self._current_phase = phase_name
        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop the current timer and record the duration.

        Returns:
            Duration in seconds.
        """
        if self._current_phase is None:
            return 0.0
        duration = time.perf_counter() - self._start_time
        self._timers[self._current_phase] = duration
        self._current_phase = None
        return duration

    def get_duration(self, phase_name: str) -> float:
        """Get the duration of a specific phase."""
        return self._timers.get(phase_name, 0.0)

    def get_all_durations(self) -> Dict[str, float]:
        """Get all recorded phase durations."""
        return dict(self._timers)

    def total_duration(self) -> float:
        """Get the sum of all phase durations."""
        return sum(self._timers.values())

    def reset(self) -> None:
        """Clear all recorded timers."""
        self._timers.clear()
        self._current_phase = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self._current_phase is not None:
            self.stop()
