"""JoyCon-driven episode control for lerobot-record.

Drives the same three-flag ``events`` dict that :func:`init_keyboard_listener`
produces, but sets the flags from JoyCon button presses received over DexComm
(Zenoh) instead of the keyboard. This lets an operator holding the exo drive
episodes headlessly, with no focused TTY.

Mapping and enablement are read from the environment; see
``vega_module/.docs/JOYCON_EPISODE_CONTROL.md``.
"""

from __future__ import annotations

import logging
import os
import time
from threading import Lock

from lerobot.utils.keyboard_input import apply_recording_control

logger = logging.getLogger(__name__)

JOYCON_TOPIC = "exo/joycon"

_ACTION_TO_CONTROL = {
    "next": "right",
    "rerecord": "left",
    "stop": "esc",
}

_DEFAULT_BUTTONS = {
    "next": "left_right",
    "rerecord": "left_left",
    "stop": "left_down",
}

_DEBOUNCE_S = 0.2


def episode_control_from_env() -> str:
    """Return the configured episode-control backend (``"joycon"`` or ``"tty"``)."""
    return os.environ.get("EPISODE_CONTROL", "tty").strip().lower()


def mapping_from_env() -> dict[str, str]:
    """Read the action -> ``<side>_<button>`` mapping from the environment."""
    return {
        "next": os.environ.get("EPISODE_BTN_NEXT", _DEFAULT_BUTTONS["next"]).strip().lower(),
        "rerecord": os.environ.get("EPISODE_BTN_RERECORD", _DEFAULT_BUTTONS["rerecord"]).strip().lower(),
        "stop": os.environ.get("EPISODE_BTN_STOP", _DEFAULT_BUTTONS["stop"]).strip().lower(),
    }


def _parse_button(spec: str) -> tuple[str, str]:
    """Split a ``<side>_<button>`` spec into ``(side, button)``."""
    side, _, button = spec.partition("_")
    if side not in ("left", "right") or not button:
        raise ValueError(
            f"Invalid JoyCon button spec {spec!r}; expected '<left|right>_<button>', e.g. 'left_right'."
        )
    return side, button


class _JoyConListener:
    """Subscribes to the JoyCon topic and sets the episode ``events`` flags.

    Edge-detects each mapped button (buttons stream held-down at ~40 Hz) so one
    physical press produces exactly one action, with a short debounce. Exposes
    ``.stop()`` so the record loop's ``finally`` can tear it down like the
    keyboard listener.
    """

    def __init__(self, mapping: dict[str, str], events: dict):
        self._events = events
        self._actions = {
            action: _parse_button(spec) for action, spec in mapping.items()
        }
        self._prev: dict[str, bool] = {action: False for action in self._actions}
        self._last_fire: dict[str, float] = {action: 0.0 for action in self._actions}
        self._lock = Lock()

        from dexcomm import Node
        from dexcomm.codecs import DictDataCodec

        self._node = Node(name="lerobot_joycon_episode", namespace="")
        self._subscriber = self._node.create_subscriber(
            JOYCON_TOPIC, callback=self._on_joycon, decoder=DictDataCodec.decode
        )
        controls = ", ".join(f"{spec}={action}" for action, spec in mapping.items())
        logger.info("JoyCon episode control active on '%s' (%s)", JOYCON_TOPIC, controls)

    def _on_joycon(self, data: dict) -> None:
        with self._lock:
            now = time.monotonic()
            for action, (side, button) in self._actions.items():
                pressed = bool(data.get(side, {}).get("buttons", {}).get(button, False))
                rising = pressed and not self._prev[action]
                self._prev[action] = pressed
                if rising and (now - self._last_fire[action]) >= _DEBOUNCE_S:
                    self._last_fire[action] = now
                    apply_recording_control(_ACTION_TO_CONTROL[action], self._events)

    def stop(self) -> None:
        close = getattr(self._subscriber, "undeclare", None) or getattr(self._subscriber, "close", None)
        if close is not None:
            try:
                close()
            except Exception as err:
                logger.warning("Could not close the JoyCon subscriber: %s", err)
        self._subscriber = None
        self._node = None


def init_joycon_listener(mapping: dict[str, str] | None = None):
    """Initialize JoyCon episode control.

    Mirrors :func:`init_keyboard_listener`'s contract: returns ``(listener,
    events)`` where ``events`` is the ``{exit_early, rerecord_episode,
    stop_recording}`` dict mutated in place, and ``listener`` exposes ``.stop()``.
    """
    events = {
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,
    }
    listener = _JoyConListener(mapping or mapping_from_env(), events)
    return listener, events
