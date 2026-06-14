"""Global in-memory structures shared across all MAS agents.

All four agents run inside a single Twisted reactor process
(`MASStrategy.run()` launches them with one `reactor.run()`), so a
module-level dict imported by reference is shared reliably with **no
network hop** — unlike FIPA-ACL messages, which PADE delivers
fire-and-forget (at-most-once, silently dropped on connection failure).

This is why control-plane ground truth lives here, not in messages:

- `FRAME_BUFFER`      — numpy image arrays keyed by frame_id (data plane).
- `CAPTURE_MANIFEST`   — authoritative `{animal_id: total_frames_captured}`,
                        written directly by CaptureAgent. The watchdog uses
                        this (not the lossy `pipeline-complete` message) to
                        know which animals must end up with a weight.
- `CAPTURE_DONE_TS`    — wall-clock `time.time()` set by CaptureAgent at FINISH.
- `QUEUE_STATS`        — mirror of each agent's pending counters, used only as
                        a progress hint and for queue-depth observability
                        (never as a decision gate — see PredictWeightAgent).
"""

import threading
from typing import Any

# --- Data plane -----------------------------------------------------------
# Keys are string frame identifiers; values are numpy ndarrays.
FRAME_BUFFER: dict[str, Any] = {}

# --- Control plane ground truth (in-process, reliable) --------------------
CAPTURE_MANIFEST: dict[int, int] = {}
CAPTURE_DONE_TS: float | None = None

# --- Observability / progress hint ----------------------------------------
# Each agent mirrors its own pending counters here under QUEUE_STATS_LOCK in a
# short critical section (outside its own self._lock, after computing the value).
# The PredictWeightAgent watchdog reads this only to refresh `last_progress_ts`
# (upstream busy) and ResourceManager writes it to queue_depth.csv. No decision
# depends on these values.
QUEUE_STATS: dict[str, Any] = {
    "frame_buffer_size": 0,
    "pending_enhance": 0,
    "pending_eval": 0,
    "pending_inference": 0,
    "finalized_animals": 0,
}
QUEUE_STATS_LOCK = threading.Lock()


def update_queue_stat(key: str, value: Any) -> None:
    """Thread-safe helper to mirror a single counter into QUEUE_STATS."""
    with QUEUE_STATS_LOCK:
        QUEUE_STATS[key] = value
