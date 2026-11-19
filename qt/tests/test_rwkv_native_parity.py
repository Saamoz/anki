# Copyright: Ankitects Pty Ltd and contributors
# License: GNU AGPL, version 3 or later; http://www.gnu.org/licenses/agpl.html

"""Parity between the desktop's Python RWKV path and the native Rust path.

The desktop replays history and builds show-time queries in
`aqt.rwkv_scheduler`; clients without Python (the iOS app) use the
collection backend's `RwkvNative*` RPCs instead. Both must give the same
recall, intervals and S90 for the same collection, so desktop and phone
schedule alike.
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from anki import scheduler_pb2
from anki.collection import Collection
from anki.decks import DeckId, UpdateDeckConfigs
from aqt import rwkv_scheduler as scheduler

MODELS = Path(__file__).parents[1] / "aqt/rwkv_inference"
DAY_MS = 86_400_000

pytestmark = pytest.mark.skipif(
    not (MODELS / "sm92/weights.f32.bin").exists(),
    reason="bundled RWKV weights are required",
)


def _synthetic_collection(path: Path, cards: int, seed: int) -> tuple[Collection, list[int]]:
    """A collection with a plausible review history over the past ~300 days."""
    col = Collection(str(path))
    # Reviews must fall after the creation day, or their day offsets clamp.
    col.db.execute("update col set crt = ?", int(time.time()) - 400 * 86_400)
    col.close()
    col = Collection(str(path))
    col.set_config("fsrs", True)
    config = col.decks.get_deck_configs_for_update(DeckId(1)).all_config[0].config
    config.config.rwkv_review_enabled = True
    config.config.reviews_per_day = 9999
    col.decks.update_deck_configs(UpdateDeckConfigs(target_deck_id=1, configs=[config]))

    rng = random.Random(seed)
    notetype = col.models.current()
    assert notetype is not None
    card_ids = []
    for index in range(cards):
        note = col.new_note(notetype)
        note.fields[0] = f"card {index}"
        col.add_note(note, DeckId(1))
        card_ids.append(note.cards()[0].id)

    now = int(time.time() * 1000)
    rows = []
    for index, card_id in enumerate(card_ids):
        at = now - 300 * DAY_MS + index * 3_600_000
        interval = 1
        for review in range(12):
            ease = 1 if review > 0 and rng.random() < 0.1 else rng.choice((3, 3, 3, 2, 4))
            kind = 0 if review == 0 else (2 if ease == 1 else 1)
            rows.append((at, card_id, ease, interval, kind, rng.randint(2_000, 15_000)))
            interval = 1 if ease == 1 else max(1, int(interval * rng.uniform(1.8, 3.0)))
            at += interval * DAY_MS + rng.randint(0, 40_000_000)
            if at > now - DAY_MS:
                break
    rows.sort()
    last = 0
    for at, card_id, ease, interval, kind, taken in rows:
        at = max(at, last + 1)
        last = at
        col.db.execute(
            "insert into revlog (id, cid, usn, ease, ivl, lastIvl, factor, time, type)"
            " values (?, ?, 0, ?, ?, 0, 2500, ?, ?)",
            at,
            card_id,
            ease,
            interval,
            taken,
            kind,
        )
    today = col.sched.today
    for card_id in card_ids:
        col.db.execute(
            "update cards set type = 2, queue = 2, ivl = 5, due = ? where id = ?",
            today - card_id % 3,
            card_id,
        )
    return col, card_ids


def _python_predictions(
    col: Collection, card_ids: list[int], precision: str = "f32"
) -> dict[int, object]:
    """The desktop path: Python replay inputs, the embedded runtime, and
    `rwkv_review_input` queries."""
    from aqt.rwkv_srs_benchmark import EmbeddedRwkvReviewerBackend

    reviewer = SimpleNamespace(mw=SimpleNamespace(col=col))
    history = scheduler._historical_rwkv_review_inputs(reviewer)
    backend = EmbeddedRwkvReviewerBackend(
        model_path=MODELS / "sm92/weights.f32.bin",
        state_precision=precision,
        model="sm92",
    )
    backend.warm_up(history.reviews, review_ids=history.review_ids)
    out = {}
    for card_id in card_ids:
        card = col.get_card(card_id)
        reviewer._v3 = SimpleNamespace(states=col._backend.get_scheduling_states(card_id))
        out[card_id] = backend.predict_review_curve(reviewer=reviewer, card=card)
    return out


def _native_load(
    col: Collection, state_dir: Path, precision: str = "f32"
) -> scheduler_pb2.RwkvNativeStatus:
    return col._backend.rwkv_native_load(
        scheduler_pb2.RwkvNativeLoadRequest(
            curve_model_path=str(MODELS / "sm92/weights.f32.bin"),
            curve_model="sm92",
            state_precision=precision,
            state_dir=str(state_dir),
        )
    )


def _assert_parity(python: dict[int, object], native: dict[int, object]) -> None:
    deviation = {"r": 0.0, "interval": 0.0, "s90": 0.0}
    for card_id, expected in python.items():
        actual = native[card_id]
        deviation["r"] = max(deviation["r"], abs(actual.retrievability - expected.retrievability))
        for field, index in (("again", 0), ("hard", 1), ("good", 2), ("easy", 3)):
            for key, values, overrides in (
                ("interval", actual.intervals, expected.interval_overrides),
                ("s90", actual.s90s, expected.s90_overrides),
            ):
                value = getattr(overrides, field)
                if value:
                    deviation[key] = max(deviation[key], abs(values[index] - value) / value)
    print(f"max |dR|={deviation['r']:.2e} rel interval={deviation['interval']:.2e} rel s90={deviation['s90']:.2e}")
    for card_id, expected in python.items():
        actual = native[card_id]
        assert actual.available
        # Queries measure seconds since the last review at slightly
        # different moments, so allow for that.
        assert actual.retrievability == pytest.approx(expected.retrievability, abs=2e-4)
        for field, index in (("again", 0), ("hard", 1), ("good", 2), ("easy", 3)):
            interval = getattr(expected.interval_overrides, field)
            s90 = getattr(expected.s90_overrides, field)
            # Short intervals move by the skew itself: allow a few seconds.
            if interval is not None:
                assert actual.intervals[index] == pytest.approx(
                    interval, rel=2e-3, abs=3 / 86_400
                )
            if s90 is not None:
                assert actual.s90s[index] == pytest.approx(s90, rel=2e-3, abs=3 / 86_400)


@pytest.mark.parametrize(
    "seed,precision", [(1, "f32"), (2, "f32"), (3, "cards_int4_notes_int8")]
)
def test_native_rwkv_matches_desktop_python(
    tmp_path: Path, seed: int, precision: str
) -> None:
    col, card_ids = _synthetic_collection(tmp_path / "collection.anki2", 40, seed)
    try:
        python = _python_predictions(col, card_ids, precision)

        status = _native_load(col, tmp_path / "state", precision)
        assert status.loaded and status.has_curve
        status = col._backend.rwkv_native_sync(force_full_rebuild=False)
        assert status.ready and status.last_sync_full_rebuild
        native = {cid: col._backend.rwkv_native_card_prediction(cid=cid) for cid in card_ids}
        _assert_parity(python, native)

        # New reviews: the native path replays only them; the desktop path
        # rebuilds from scratch. Both must still agree.
        rng = random.Random(seed + 100)
        last = col.db.scalar("select max(id) from revlog")
        for offset in range(1, 9):
            col.db.execute(
                "insert into revlog (id, cid, usn, ease, ivl, lastIvl, factor, time, type)"
                " values (?, ?, 0, 3, 7, 3, 2500, 6000, 1)",
                last + offset * 60_000,
                rng.choice(card_ids),
            )
        status = col._backend.rwkv_native_sync(force_full_rebuild=False)
        assert not status.last_sync_full_rebuild
        assert status.last_sync_replayed_reviews == 8
        python = _python_predictions(col, card_ids, precision)
        native = {cid: col._backend.rwkv_native_card_prediction(cid=cid) for cid in card_ids}
        _assert_parity(python, native)
    finally:
        col.close()


def test_native_curve_states_use_rwkv_intervals(tmp_path: Path) -> None:
    col, card_ids = _synthetic_collection(tmp_path / "collection.anki2", 20, 3)
    try:
        fsrs_states = col._backend.get_scheduling_states(card_ids[0])
        _native_load(col, tmp_path / "state")
        col._backend.rwkv_native_sync(force_full_rebuild=False)
        states = col._backend.get_scheduling_states(card_ids[0])
        prediction = col._backend.rwkv_native_card_prediction(cid=card_ids[0])
        good = states.good.normal
        assert good.WhichOneof("kind") in ("review", "relearning")
        review = good.review if good.WhichOneof("kind") == "review" else good.relearning.review
        assert review.memory_state.stability == pytest.approx(prediction.s90s[2], rel=1e-3)
        assert states != fsrs_states
    finally:
        col.close()
