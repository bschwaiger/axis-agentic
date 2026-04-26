"""
Event bus for the AXIS Agentic Cockpit.

A thread-safe queue that the pipeline writes to and the SSE endpoint reads from.
Events are JSON-serializable dicts with a "type" field.
"""
from __future__ import annotations

import json
import queue
import threading
import time
from typing import Any

# Singleton event queue — multiple SSE clients each get their own view
_subscribers: list[queue.Queue] = []
_lock = threading.Lock()

# Whether cockpit mode is active (set by cockpit app on startup)
cockpit_enabled = False

# Pipeline control flags
pipeline_stop_requested = False
pipeline_paused = threading.Event()
pipeline_paused.set()  # starts unpaused (set = not blocking)

# Pending approval futures — the pipeline blocks on these
_approval_events: dict[str, threading.Event] = {}
_approval_values: dict[str, Any] = {}


def request_stop():
    """Signal the pipeline to stop after current step."""
    global pipeline_stop_requested
    pipeline_stop_requested = True
    # Also unblock pause so the stop can take effect
    pipeline_paused.set()


def request_pause():
    """Pause the pipeline — it will block before next step."""
    pipeline_paused.clear()


def request_resume():
    """Resume a paused pipeline."""
    pipeline_paused.set()


def reset_controls():
    """Reset stop/pause state for a new pipeline run."""
    global pipeline_stop_requested
    pipeline_stop_requested = False
    pipeline_paused.set()


def check_stop() -> bool:
    """Check if stop was requested. Call between pipeline steps."""
    return pipeline_stop_requested


def wait_if_paused():
    """Block if paused. Call between pipeline steps."""
    pipeline_paused.wait()


def enable():
    """Enable cockpit mode globally."""
    global cockpit_enabled
    cockpit_enabled = True


def is_enabled() -> bool:
    return cockpit_enabled


def subscribe() -> queue.Queue:
    """Create a new subscriber queue for SSE streaming."""
    q: queue.Queue = queue.Queue(maxsize=500)
    with _lock:
        _subscribers.append(q)
    return q


def unsubscribe(q: queue.Queue):
    """Remove a subscriber queue."""
    with _lock:
        if q in _subscribers:
            _subscribers.remove(q)


def emit(event_type: str, **data):
    """Emit an event to all subscribers. No-op if cockpit is disabled."""
    if not cockpit_enabled:
        return
    event = {"type": event_type, "ts": time.time(), **data}
    with _lock:
        dead = []
        for q in _subscribers:
            try:
                q.put_nowait(event)
            except queue.Full:
                dead.append(q)
        for q in dead:
            _subscribers.remove(q)


def wait_for_approval(key: str, timeout: float = 600.0) -> Any:
    """Block until the cockpit UI posts an approval for this key.

    Returns the approval payload, or None on timeout.
    Used for matrix checkpoint and query approval.
    """
    evt = threading.Event()
    _approval_events[key] = evt
    if evt.wait(timeout=timeout):
        value = _approval_values.pop(key, None)
        _approval_events.pop(key, None)
        return value
    _approval_events.pop(key, None)
    return None


def post_approval(key: str, value: Any = True):
    """Called by the HTTP handler to unblock a waiting approval."""
    _approval_values[key] = value
    evt = _approval_events.get(key)
    if evt:
        evt.set()


def format_sse(event: dict) -> str:
    """Format a dict as an SSE data frame."""
    return f"data: {json.dumps(event)}\n\n"
