# Copyright: Ankitects Pty Ltd and contributors
# License: GNU AGPL, version 3 or later; http://www.gnu.org/licenses/agpl.html

"""Opt-in: schedule RWKV through the collection backend's native path.

With ANKI_RWKV_NATIVE=1, the desktop loads the RWKV runtime into the Rust
collection backend (the RwkvNative* RPCs) instead of the Python reviewer
backend. The backend then applies RWKV-Curve answer states, fills the answer
metadata, advances the model state on every answer and scores RWKV-Instant
queues itself: the same code the iOS client runs, so both schedule alike.

Python-only features (RWKV card info rows, stats-graph scores, workload
simulation and forecasts, the answer-time estimate) are unavailable on this
path, as they are when no RWKV backend is configured.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from anki import scheduler_pb2

logger = logging.getLogger(__name__)

NATIVE_ENV = "ANKI_RWKV_NATIVE"
STATE_DIR = "rwkv-native-state"


def enabled() -> bool:
    return os.environ.get(NATIVE_ENV) == "1"


def load_request(mw: Any) -> scheduler_pb2.RwkvNativeLoadRequest | None:
    """The profile's models, as the Python backend would load them."""
    from aqt import rwkv_model
    from aqt import rwkv_scheduler as scheduler
    from aqt.rwkv_precision import state_precision_for_profile

    profile = getattr(getattr(mw, "pm", None), "profile", None)
    reviewer = getattr(mw, "reviewer", None) or type("R", (), {"mw": mw})()
    state = scheduler._rwkv_collection_config_state(reviewer)
    if not state.review_enabled:
        return None
    models = rwkv_model.runtime_models(
        rwkv_model.model_for_profile(profile),
        instant_used=state.instant_enabled,
        curve_used=state.curve_enabled,
        single_model=rwkv_model.single_model_for_profile(profile),
    )
    model_path = scheduler._embedded_rwkv_model_file(models.model)
    if model_path is None:
        return None
    request = scheduler_pb2.RwkvNativeLoadRequest(
        state_precision=state_precision_for_profile(profile),
        state_dir=str(Path(mw.pm.profileFolder()) / STATE_DIR),
    )
    if models.curve_model is None:
        request.instant_model_path = str(model_path)
        request.instant_model = models.model
        return request
    request.curve_model_path = str(model_path)
    request.curve_model = models.model
    if models.instant_model is not None:
        instant_path = scheduler._embedded_rwkv_model_file(models.instant_model)
        if instant_path is not None:
            request.instant_model_path = str(instant_path)
            request.instant_model = models.instant_model
    return request


_loaded_for: tuple[int, bytes] | None = None


def configure(mw: Any) -> bool:
    """Loads the native runtime for the open collection; True when loaded.

    Reloads only when the profile's models or the collection changed. The
    state syncs with the review history before its first use.
    """
    global _loaded_for
    col = getattr(mw, "col", None)
    if col is None:
        return False
    try:
        request = load_request(mw)
        if request is None:
            if _loaded_for is not None:
                col._backend.rwkv_native_unload()
                _loaded_for = None
            return False
        key = (id(col), request.SerializeToString())
        if _loaded_for == key and col._backend.rwkv_native_get_status().loaded:
            return True
        col._backend.rwkv_native_load(request)
        _loaded_for = key
        _register_hooks()
        return True
    except Exception:
        logger.exception("failed to load the native RWKV runtime")
        return False


def checkpoint(mw: Any) -> None:
    col = getattr(mw, "col", None)
    if col is None or _loaded_for is None:
        return
    try:
        col._backend.rwkv_native_checkpoint()
    except Exception:
        logger.exception("failed to save the native RWKV state")


_hooks_registered = False


def _register_hooks() -> None:
    global _hooks_registered
    if _hooks_registered:
        return
    import aqt
    from aqt import gui_hooks

    def on_close() -> None:
        global _loaded_for
        checkpoint(aqt.mw)
        _loaded_for = None

    gui_hooks.profile_will_close.append(on_close)
    _hooks_registered = True
