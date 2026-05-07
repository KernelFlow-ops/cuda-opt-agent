"""CLI interrupt handling."""

from __future__ import annotations

import os
import signal
import sys
import time
from types import FrameType

FORCE_EXIT_WINDOW_SECONDS = 2.0
_last_sigint_at = 0.0


def install_interrupt_handler() -> None:
    """Install a two-stage Ctrl+C handler.

    First Ctrl+C raises KeyboardInterrupt so the graph can save state. A second
    Ctrl+C within FORCE_EXIT_WINDOW_SECONDS exits immediately.
    """
    signal.signal(signal.SIGINT, _handle_sigint)


def _handle_sigint(signum: int, frame: FrameType | None) -> None:
    global _last_sigint_at
    now = time.monotonic()
    if now - _last_sigint_at <= FORCE_EXIT_WINDOW_SECONDS:
        print("\nForce exit requested. Terminating immediately.", file=sys.stderr, flush=True)
        os._exit(130)

    _last_sigint_at = now
    print(
        "\nCtrl+C received: saving state and pausing. Press Ctrl+C again within "
        f"{FORCE_EXIT_WINDOW_SECONDS:g}s to force exit.",
        file=sys.stderr,
        flush=True,
    )
    raise KeyboardInterrupt


def _reset_interrupt_state_for_tests() -> None:
    global _last_sigint_at
    _last_sigint_at = 0.0


