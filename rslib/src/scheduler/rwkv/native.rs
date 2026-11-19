// Copyright: Ankitects Pty Ltd and contributors
// License: GNU AGPL, version 3 or later; http://www.gnu.org/licenses/agpl.html

//! Native RWKV scheduling inside the collection backend.
//!
//! The desktop drives RWKV from Python (`qt/aqt/rwkv_scheduler.py`). This
//! module ports the parts a review client needs into Rust so that any client
//! of the protobuf backend (the iOS app in particular, and the desktop behind
//! a flag) gets the same behaviour without that layer:
//!
//! - loading the runtime (one model, or a two-head model with an
//!   instant-only model beside it);
//! - replaying the canonical review history, reusing the resident state or
//!   the newest persisted checkpoint when the history only grew;
//! - persisting state in the engine's checkpoint store;
//! - RWKV-Curve answer states for the card shown, and the answer metadata
//!   (`rwkv_s90`, `rwkv_retrievability`, `rwkv_review_kind`);
//! - RWKV-Instant review-queue scores;
//! - a calibration summary from the predictions recorded during replay.
//!
//! Model state always equals a replay of the canonical history: after an
//! answer, the new revlog row is replayed exactly as a later rebuild would
//! replay it, so a rebuild never disagrees with live updates.

use std::collections::HashMap;
use std::collections::HashSet;
use std::path::PathBuf;
use std::time::Instant;

use anki_proto::scheduler::RwkvHistoricalReviewFingerprintRequest;
use anki_proto::scheduler::RwkvNativeCalibrationResponse;
use anki_proto::scheduler::RwkvNativeCardPredictionResponse;
use anki_proto::scheduler::RwkvNativeLoadRequest;
use anki_proto::scheduler::RwkvNativeStatus;
use anki_proto::scheduler::RwkvReviewInputRowsForDeckReviewQueueRequest;
use rusqlite::OpenFlags;
use rusqlite::OptionalExtension;
use sha2::Digest;
use sha2::Sha256;

use super::rwkv_empty_history_hash;
use super::rwkv_history_hash_after_review;
use super::rwkv_history_hash_hex;
use super::RwkvHistoricalFingerprintReview;
use super::RwkvHistoricalReplayOptions;
use crate::card::Card;
use crate::card::FsrsMemoryState;
use crate::collection::RwkvReviewQueueScoreEntry;
use crate::deckconfig::DeckConfig;
use crate::prelude::*;
use crate::rwkv::ModelKind;
use crate::rwkv::ReviewInput;
use crate::rwkv::ReviewState as EngineReviewState;
use crate::rwkv::RwkvRuntime;
use crate::rwkv::StatePrecision;
use crate::scheduler::answering::CardAnswer;
use crate::scheduler::answering::Rating;
use crate::scheduler::states::CardState;
use crate::scheduler::states::FilteredState;
use crate::scheduler::states::NormalState;
use crate::scheduler::states::SchedulingStates;
use crate::scheduler::timing::SchedTimingToday;
use crate::storage::RwkvReviewRetrievabilityCacheRow;
use crate::storage::RwkvReviewRetrievabilitySampleRole;

/// Bumped when replay or persistence semantics change, so older stores are
/// rebuilt instead of reused.
const NATIVE_SCHEMA: &str = "anki-rwkv-native-v1";
const STORE_FILE: &str = "rwkv-native-state.sqlite3";
const STORE_BUILD_FILE: &str = ".rwkv-native-state-building.sqlite3";
const DEFAULT_CHECKPOINT_EVERY_ANSWERS: u32 = 50;
/// Delta segments before the store is rewritten as one full segment.
const COMPACT_AFTER_SEGMENTS: i64 = 48;
/// Reviews replayed per engine call; progress is reported between calls.
const REPLAY_CHUNK: usize = 16_384;
/// Calibration rows written by this module.
pub(crate) const NATIVE_PREDICTION_SOURCE: &str = "rwkv_native";
const DEFAULT_TARGET_RETENTION: f32 = 0.9;
const MAX_INTERVAL_DAYS: u32 = 36_500;

/// The desktop's RwkvReviewState: learning start, learning, review,
/// relearning and filtered.
mod review_state {
    pub(super) const LEARN_START: i64 = 0;
    pub(super) const LEARNING: i64 = 1;
    pub(super) const REVIEW: i64 = 2;
    pub(super) const RELEARNING: i64 = 3;
    pub(super) const FILTERED: i64 = 4;
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum RwkvRebuildStage {
    #[default]
    History,
    Replay,
    Saving,
}

impl RwkvRebuildStage {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            RwkvRebuildStage::History => "history",
            RwkvRebuildStage::Replay => "replay",
            RwkvRebuildStage::Saving => "saving",
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct RwkvRebuildProgress {
    pub stage: RwkvRebuildStage,
    pub current: u32,
    pub total: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct HistoryIdentity {
    last_review_id: i64,
    review_count: u64,
    history_hash: String,
}

/// Cheap summary of the tables the replay reads. A change means the history
/// (or its identities) may have changed and the state must be synced before
/// it is used.
#[derive(Debug, Clone, PartialEq, Eq)]
struct HistoryMarker {
    revlog_max_id: i64,
    revlog_count: i64,
    cards_count: i64,
    cards_max_mod: i64,
    decks_max_mod: i64,
    configs_max_mod: i64,
    usn: i32,
}

#[derive(Debug, Clone, Copy, Default)]
struct SyncReport {
    replayed: u64,
    full_rebuild: bool,
    restored_checkpoint: bool,
    millis: u32,
}

#[derive(Debug, Clone)]
struct CachedPrediction {
    /// The history the prediction was made from.
    review_count: u64,
    retrievability: f32,
    s90s: [Option<f32>; 4],
    curve_applied: bool,
}

pub(crate) struct NativeRwkv {
    runtime: RwkvRuntime,
    curve_model: String,
    instant_model: String,
    paired: bool,
    has_curve: bool,
    precision: String,
    state_dir: PathBuf,
    /// Reloaded for a full rebuild: resetting a used runtime keeps its
    /// feature encodings, so it would not match a fresh replay.
    load_request: RwkvNativeLoadRequest,
    replay_key: String,
    resident: Option<HistoryIdentity>,
    /// `resident`'s hash as bytes, which the next review's hash chains from.
    resident_hash: [u8; 32],
    checkpoint: Option<(i64, HistoryIdentity)>,
    segments_in_store: i64,
    answers_since_checkpoint: u32,
    checkpoint_every: u32,
    marker: Option<HistoryMarker>,
    last_sync: SyncReport,
    last_error: String,
    predictions: HashMap<CardId, CachedPrediction>,
    /// Decks whose Instant scores were computed for the current history.
    scored_decks: HashSet<DeckId>,
}

impl std::fmt::Debug for NativeRwkv {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeRwkv")
            .field("curve_model", &self.curve_model)
            .field("instant_model", &self.instant_model)
            .field("resident", &self.resident)
            .field("checkpoint", &self.checkpoint)
            .finish_non_exhaustive()
    }
}

/// Which RWKV schedulers a preset uses.
#[derive(Debug, Clone, Copy, Default)]
struct PresetUse {
    curve: bool,
    instant: bool,
}

impl PresetUse {
    fn of(config: &DeckConfig) -> Self {
        PresetUse {
            curve: config.inner.rwkv_review_enabled,
            instant: config.inner.rwkv_review_instant_order_enabled,
        }
    }

    fn active(self) -> bool {
        self.curve || self.instant
    }
}

/// One canonical history, with the hash after every review so any earlier
/// identity can be checked as a prefix without another traversal.
struct CanonicalHistory {
    inputs: Vec<ReviewInput>,
    review_ids: Vec<i64>,
    hashes: Vec<[u8; 32]>,
    identity: HistoryIdentity,
}

impl CanonicalHistory {
    /// The number of leading reviews `identity` covers, if it is a prefix.
    fn prefix_len(&self, identity: &HistoryIdentity) -> Option<usize> {
        let count = usize::try_from(identity.review_count).ok()?;
        if count == 0 {
            return (identity.history_hash == rwkv_history_hash_hex(rwkv_empty_history_hash()))
                .then_some(0);
        }
        if count > self.inputs.len() {
            return None;
        }
        (self.review_ids[count - 1] == identity.last_review_id
            && rwkv_history_hash_hex(self.hashes[count - 1]) == identity.history_hash)
            .then_some(count)
    }
}

fn invalid(message: impl Into<String>) -> AnkiError {
    AnkiError::InvalidInput {
        source: snafu::FromString::without_source(message.into()),
    }
}

fn io_err(err: std::io::Error) -> AnkiError {
    invalid(format!("RWKV: {err}"))
}

fn replay_key(curve: &str, instant: &str, precision: &str) -> String {
    let mut digest = Sha256::new();
    digest.update(NATIVE_SCHEMA.as_bytes());
    for part in [curve, instant, precision] {
        digest.update([0]);
        digest.update(part.as_bytes());
    }
    rwkv_history_hash_hex(digest.finalize().into())
}

/// The desktop's `_stable_preset_id`: decimal ids are deck-config ids; other
/// (add-on) ids map to the first 63 bits of an 8-byte BLAKE2b digest.
pub(crate) fn stable_addon_preset_id(preset_id: &str) -> i64 {
    if !preset_id.is_empty() && preset_id.bytes().all(|b| b.is_ascii_digit()) {
        if let Ok(id) = preset_id.parse() {
            return id;
        }
    }
    use blake2::digest::Update;
    use blake2::digest::VariableOutput;
    let mut hasher = blake2::Blake2bVar::new(8).expect("valid BLAKE2b output size");
    hasher.update(preset_id.as_bytes());
    let mut out = [0u8; 8];
    hasher
        .finalize_variable(&mut out)
        .expect("output matches requested size");
    (u64::from_be_bytes(out) & ((1 << 63) - 1)) as i64
}

fn home_deck_id(card: &Card) -> DeckId {
    if card.original_deck_id.0 != 0 {
        card.original_deck_id
    } else {
        card.deck_id
    }
}

fn rating_ease(rating: Rating) -> u8 {
    match rating {
        Rating::Again => 1,
        Rating::Hard => 2,
        Rating::Good => 3,
        Rating::Easy => 4,
    }
}

/// One replayed review (`rwkv_history.review_input_from_row`).
fn replay_input(review: &RwkvHistoricalFingerprintReview) -> ReviewInput {
    let row = review.row;
    ReviewInput {
        card_id: row.card_id,
        note_id: Some(row.note_id),
        deck_id: Some(row.deck_id),
        preset_id: Some(review.stable_preset_id),
        is_query: false,
        ease: Some(row.ease as u8),
        duration_millis: Some(row.duration_millis),
        card_type: Some(if row.is_learning_start {
            review_state::LEARN_START
        } else {
            row.review_kind + 1
        }),
        day_offset: Some(review.day_offset),
        current_elapsed_days: Some(review.elapsed_days),
        current_elapsed_seconds: Some(review.elapsed_seconds),
        target_retentions: [None; 4],
        enforce_grade_order: true,
    }
}

/// `_rwkv_review_state_for_scheduling_state`. Every scheduling state maps
/// to a review state, so the card-type fallback is never needed here.
fn base_review_state(current: CardState) -> i64 {
    match current {
        CardState::Filtered(_) => review_state::FILTERED,
        CardState::Normal(NormalState::New(_)) => review_state::LEARN_START,
        CardState::Normal(NormalState::Learning(_)) => review_state::LEARNING,
        CardState::Normal(NormalState::Review(_)) => review_state::REVIEW,
        CardState::Normal(NormalState::Relearning(_)) => review_state::RELEARNING,
    }
}

/// `_rwkv_raw_review_kind`: the revlog kind written for a review state.
fn raw_review_kind(state: i64) -> Option<u32> {
    match state {
        review_state::LEARN_START | review_state::LEARNING => Some(0),
        review_state::REVIEW => Some(1),
        review_state::RELEARNING => Some(2),
        review_state::FILTERED => Some(3),
        _ => None,
    }
}

/// The state's normal kind and elapsed time (`_scheduling_state_elapsed`).
fn state_elapsed(current: CardState) -> (Option<i64>, Option<i64>) {
    match current {
        CardState::Normal(NormalState::Review(review)) => (Some(review.elapsed_days as i64), None),
        CardState::Normal(NormalState::Learning(learning)) => {
            (None, Some(learning.elapsed_secs as i64))
        }
        CardState::Normal(NormalState::Relearning(relearning)) => (
            Some(relearning.review.elapsed_days as i64),
            Some(relearning.learning.elapsed_secs as i64),
        ),
        _ => (None, None),
    }
}

/// `_historical_review_day_offset` / `rwkv_historical_day_offset`.
fn day_offset_for_millis(millis: i64, timing: &SchedTimingToday) -> i64 {
    super::rwkv_historical_day_offset(millis, timing)
}

/// `_rwkv_factor_zero_rescheduled_learning_kind`: a rescheduling filtered
/// answer of a new or learning card that stays learning without a memory
/// state is logged with factor 0 and would vanish from replay; label it with
/// the card's original kind instead.
fn factor_zero_rescheduled_learning_state(current: CardState, next: CardState) -> Option<i64> {
    let (
        CardState::Filtered(FilteredState::Rescheduling(current)),
        CardState::Filtered(FilteredState::Rescheduling(next)),
    ) = (current, next)
    else {
        return None;
    };
    let original = match current.original_state {
        NormalState::New(_) => review_state::LEARN_START,
        NormalState::Learning(_) => review_state::LEARNING,
        _ => return None,
    };
    match next.original_state {
        NormalState::Learning(learning) if learning.memory_state.is_none() => Some(original),
        _ => None,
    }
}

fn apply_s90(state: &mut CardState, s90: f32) {
    let review = match state {
        CardState::Normal(NormalState::Review(review)) => review,
        CardState::Normal(NormalState::Relearning(relearning)) => &mut relearning.review,
        _ => return,
    };
    let memory = review.memory_state.get_or_insert(FsrsMemoryState {
        stability: s90,
        stability_internal: s90,
        stability_fast: None,
        difficulty: 0.0,
    });
    if memory.difficulty <= 0.0 {
        memory.difficulty = 5.0;
    }
    memory.stability = s90;
}

/// Builds a runtime with no history: one model, or a two-head model with an
/// instant-only model beside it. Returns the runtime, the two model names
/// and whether they are paired.
fn load_runtime(
    req: &RwkvNativeLoadRequest,
    precision: &str,
) -> Result<(RwkvRuntime, String, String, bool)> {
    let parsed_precision = StatePrecision::parse(precision).map_err(io_err)?;
    let curve_path = req.curve_model_path.clone().filter(|p| !p.is_empty());
    let instant_path = req.instant_model_path.clone().filter(|p| !p.is_empty());
    let instant_name = req.instant_model.clone().unwrap_or_default();
    let loaded = match (curve_path, instant_path) {
        (Some(curve_path), Some(instant_path)) => {
            let curve_kind = ModelKind::parse(&req.curve_model).map_err(io_err)?;
            let instant_kind = ModelKind::parse(&instant_name).map_err(io_err)?;
            (
                RwkvRuntime::load_paired(
                    curve_path.into(),
                    instant_path.into(),
                    DEFAULT_TARGET_RETENTION,
                    MAX_INTERVAL_DAYS,
                    parsed_precision,
                    curve_kind,
                    instant_kind,
                )
                .map_err(io_err)?,
                req.curve_model.clone(),
                instant_name,
                true,
            )
        }
        (Some(curve_path), None) => {
            let kind = ModelKind::parse(&req.curve_model).map_err(io_err)?;
            (
                RwkvRuntime::load_with_precision(
                    curve_path.into(),
                    DEFAULT_TARGET_RETENTION,
                    MAX_INTERVAL_DAYS,
                    parsed_precision,
                    kind,
                )
                .map_err(io_err)?,
                req.curve_model.clone(),
                String::new(),
                false,
            )
        }
        (None, Some(instant_path)) => {
            let kind = ModelKind::parse(&instant_name).map_err(io_err)?;
            (
                RwkvRuntime::load_with_precision(
                    instant_path.into(),
                    DEFAULT_TARGET_RETENTION,
                    MAX_INTERVAL_DAYS,
                    parsed_precision,
                    kind,
                )
                .map_err(io_err)?,
                String::new(),
                instant_name,
                false,
            )
        }
        (None, None) => return Err(invalid("an RWKV model file is required")),
    };
    Ok(loaded)
}

impl NativeRwkv {
    fn status(&self, col_counts: (u32, u32)) -> RwkvNativeStatus {
        RwkvNativeStatus {
            loaded: true,
            ready: self.resident.is_some(),
            curve_model: self.curve_model.clone(),
            instant_model: self.instant_model.clone(),
            paired: self.paired,
            has_curve: self.has_curve,
            review_count: self.resident.as_ref().map_or(0, |id| id.review_count),
            last_review_id: self.resident.as_ref().map_or(0, |id| id.last_review_id),
            last_sync_replayed_reviews: self.last_sync.replayed,
            last_sync_full_rebuild: self.last_sync.full_rebuild,
            last_sync_restored_checkpoint: self.last_sync.restored_checkpoint,
            last_sync_millis: self.last_sync.millis,
            checkpoint_review_count: self.checkpoint.as_ref().map_or(0, |cp| cp.1.review_count),
            curve_presets: col_counts.0,
            instant_presets: col_counts.1,
            state_precision: self.precision.clone(),
            last_error: self.last_error.clone(),
        }
    }

    fn store_path(&self) -> PathBuf {
        self.state_dir.join(STORE_FILE)
    }

    /// The newest segment of the store, if it belongs to this replay key.
    fn read_latest_checkpoint(&mut self) -> Option<(i64, HistoryIdentity)> {
        let path = self.store_path();
        if !path.exists() {
            return None;
        }
        let read = || -> rusqlite::Result<Option<(i64, HistoryIdentity, String, i64)>> {
            let db = rusqlite::Connection::open_with_flags(
                &path,
                OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
            )?;
            let count: i64 = db.query_row("select count(*) from segments", [], |r| r.get(0))?;
            db.query_row(
                "select id, last_review_id, review_count, history_hash, replay_key
                 from segments order by id desc limit 1",
                [],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        HistoryIdentity {
                            last_review_id: row.get(1)?,
                            review_count: row.get::<_, i64>(2)?.max(0) as u64,
                            history_hash: row.get(3)?,
                        },
                        row.get::<_, String>(4)?,
                        count,
                    ))
                },
            )
            .optional()
        };
        match read() {
            Ok(Some((id, identity, key, count))) if key == self.replay_key => {
                self.segments_in_store = count;
                Some((id, identity))
            }
            Ok(_) => None,
            Err(err) => {
                tracing::warn!(?err, "unreadable RWKV native state store");
                None
            }
        }
    }

    /// Writes the resident state: a delta onto the newest segment, or a
    /// fresh store holding one full segment.
    fn write_checkpoint(&mut self, full: bool) -> Result<()> {
        let Some(identity) = self.resident.clone() else {
            return Ok(());
        };
        std::fs::create_dir_all(&self.state_dir)?;
        let full =
            full || self.checkpoint.is_none() || self.segments_in_store >= COMPACT_AFTER_SEGMENTS;
        let generation = self.replay_key.clone();
        let review_count = i64::try_from(identity.review_count).unwrap_or(i64::MAX);
        let segment_id = if full {
            let build_path = self.state_dir.join(STORE_BUILD_FILE);
            self.runtime
                .finish_warm_up_state_checkpoints()
                .map_err(io_err)?;
            let _ = std::fs::remove_file(&build_path);
            let id = self
                .runtime
                .write_warm_up_state_checkpoint(
                    build_path.clone(),
                    &generation,
                    None,
                    identity.last_review_id,
                    review_count,
                    &identity.history_hash,
                    &self.replay_key,
                    &[],
                    &[],
                    &[],
                    true,
                    true,
                )
                .map_err(io_err)?;
            self.runtime
                .finish_warm_up_state_checkpoints()
                .map_err(io_err)?;
            std::fs::rename(&build_path, self.store_path())?;
            self.segments_in_store = 1;
            id
        } else {
            let parent = self.checkpoint.as_ref().map(|cp| cp.0);
            let id = self
                .runtime
                .write_warm_up_state_checkpoint(
                    self.store_path(),
                    &generation,
                    parent,
                    identity.last_review_id,
                    review_count,
                    &identity.history_hash,
                    &self.replay_key,
                    &[],
                    &[],
                    &[],
                    false,
                    true,
                )
                .map_err(io_err)?;
            self.segments_in_store += 1;
            id
        };
        self.checkpoint = Some((segment_id, identity));
        self.answers_since_checkpoint = 0;
        Ok(())
    }

    fn restore_checkpoint(&mut self, segment_id: i64) -> Result<()> {
        self.runtime
            .finish_warm_up_state_checkpoints()
            .map_err(io_err)?;
        self.runtime
            .restore_warm_up_state_checkpoint(self.store_path(), &self.replay_key, segment_id)
            .map_err(io_err)
    }
}

impl Collection {
    fn rwkv_native_preset_counts(&self) -> Result<(u32, u32)> {
        let mut counts = (0, 0);
        for config in self.storage.all_deck_config()? {
            let used = PresetUse::of(&config);
            counts.0 += used.curve as u32;
            counts.1 += used.instant as u32;
        }
        Ok(counts)
    }

    pub(crate) fn rwkv_native_status(&self) -> Result<RwkvNativeStatus> {
        match self.state.rwkv_native.as_deref() {
            Some(native) => Ok(native.status(self.rwkv_native_preset_counts()?)),
            None => Ok(RwkvNativeStatus::default()),
        }
    }

    pub(crate) fn rwkv_native_load(
        &mut self,
        req: RwkvNativeLoadRequest,
    ) -> Result<RwkvNativeStatus> {
        self.rwkv_native_unload();
        let precision = if req.state_precision.is_empty() {
            "cards_int4_notes_int8".to_string()
        } else {
            req.state_precision.clone()
        };
        require!(!req.state_dir.is_empty(), "RWKV state folder is required");
        let (runtime, curve_name, instant_name, paired) = load_runtime(&req, &precision)?;
        let has_curve = runtime.has_curve();
        let mut native = NativeRwkv {
            runtime,
            replay_key: replay_key(&curve_name, &instant_name, &precision),
            curve_model: curve_name,
            instant_model: instant_name,
            paired,
            has_curve,
            precision,
            state_dir: PathBuf::from(&req.state_dir),
            load_request: req.clone(),
            resident: None,
            resident_hash: rwkv_empty_history_hash(),
            checkpoint: None,
            segments_in_store: 0,
            answers_since_checkpoint: 0,
            checkpoint_every: if req.checkpoint_every_answers == 0 {
                DEFAULT_CHECKPOINT_EVERY_ANSWERS
            } else {
                req.checkpoint_every_answers
            },
            marker: None,
            last_sync: SyncReport::default(),
            last_error: String::new(),
            predictions: HashMap::new(),
            scored_decks: HashSet::new(),
        };
        native.checkpoint = native.read_latest_checkpoint();
        self.state.rwkv_native = Some(Box::new(native));
        self.rwkv_native_status()
    }

    pub(crate) fn rwkv_native_unload(&mut self) {
        if let Some(mut native) = self.state.rwkv_native.take() {
            if let Err(err) = native.runtime.finish_warm_up_state_checkpoints() {
                tracing::warn!(?err, "closing RWKV native state store");
            }
            self.rwkv_native_clear_scores(&mut native);
        }
    }

    fn rwkv_native_clear_scores(&mut self, native: &mut NativeRwkv) {
        self.rwkv_native_invalidate_scores(native);
    }

    pub(crate) fn rwkv_native_loaded(&self) -> bool {
        self.state.rwkv_native.is_some()
    }

    /// Runs `f` with the native runtime taken out of the collection state, so
    /// it can use the collection freely.
    fn with_rwkv_native<T>(
        &mut self,
        f: impl FnOnce(&mut Collection, &mut NativeRwkv) -> Result<T>,
    ) -> Result<Option<T>> {
        let Some(mut native) = self.state.rwkv_native.take() else {
            return Ok(None);
        };
        let result = f(self, &mut native);
        if let Err(err) = &result {
            native.last_error = err.to_string();
        }
        self.state.rwkv_native = Some(native);
        result.map(Some)
    }

    fn rwkv_native_marker(&self) -> Result<HistoryMarker> {
        Ok(self.storage.db.query_row(
            "select
               (select coalesce(max(id), 0) from revlog),
               (select count() from revlog),
               (select count() from cards),
               (select coalesce(max(mod), 0) from cards),
               (select coalesce(max(mtime_secs), 0) from decks),
               (select coalesce(max(mtime_secs), 0) from deck_config),
               (select usn from col)",
            [],
            |row| {
                Ok(HistoryMarker {
                    revlog_max_id: row.get(0)?,
                    revlog_count: row.get(1)?,
                    cards_count: row.get(2)?,
                    cards_max_mod: row.get(3)?,
                    decks_max_mod: row.get(4)?,
                    configs_max_mod: row.get(5)?,
                    usn: row.get(6)?,
                })
            },
        )?)
    }

    fn rwkv_native_fingerprint_request(&self) -> Result<RwkvHistoricalReviewFingerprintRequest> {
        let configs = self.storage.all_deck_config()?;
        let mut dynamic_preset_replay = configs.iter().any(|config| {
            PresetUse::of(config).active() && config.inner.rwkv_review_dynamic_preset_replay
        });
        if let Some(setting) = self.get_config_optional::<bool, _>("rwkvDynamicPresetReplay") {
            dynamic_preset_replay = setting;
        }
        let mut stable_preset_ids = HashMap::new();
        if let Some(overlay) = self.get_config_optional::<serde_json::Value, _>("fsrsPresetOverlay")
        {
            for (key, field) in [
                ("presets", "id"),
                ("rules", "preset_id"),
                ("simulator_rules", "preset_id"),
            ] {
                for item in overlay
                    .get(key)
                    .and_then(|items| items.as_array())
                    .into_iter()
                    .flatten()
                {
                    if let Some(id) = item.get(field).and_then(|id| id.as_str()) {
                        if !id.is_empty() {
                            stable_preset_ids.insert(id.to_string(), stable_addon_preset_id(id));
                        }
                    }
                }
            }
        }
        Ok(RwkvHistoricalReviewFingerprintRequest {
            ignored_review_ids: vec![],
            dynamic_preset_replay,
            stable_preset_ids,
            // Every preset uses creation time for a first review; absent
            // entries already default to that.
            first_review_uses_creation_by_config_id: HashMap::new(),
            expected_identity: None,
        })
    }

    /// The canonical replay history (`rwkv_history.review_input_from_row`).
    fn rwkv_native_history(&mut self) -> Result<CanonicalHistory> {
        let request = self.rwkv_native_fingerprint_request()?;
        let mut inputs = Vec::new();
        let mut review_ids = Vec::new();
        let mut hashes = Vec::new();
        let fingerprint = self.visit_rwkv_historical_reviews(
            request,
            RwkvHistoricalReplayOptions::default(),
            |review, hash, _| {
                review_ids.push(review.row.review_id);
                hashes.push(hash);
                inputs.push(replay_input(&review));
            },
        )?;
        Ok(CanonicalHistory {
            inputs,
            review_ids,
            hashes,
            identity: HistoryIdentity {
                last_review_id: fingerprint.last_review_id,
                review_count: fingerprint.review_count,
                history_hash: fingerprint.history_hash,
            },
        })
    }

    pub(crate) fn rwkv_native_sync(&mut self, force_full: bool) -> Result<RwkvNativeStatus> {
        require!(self.rwkv_native_loaded(), "RWKV is not loaded");
        self.with_rwkv_native(|col, native| col.rwkv_native_sync_inner(native, force_full))?;
        self.rwkv_native_status()
    }

    /// Syncs only when the history may have changed since the last sync.
    fn rwkv_native_ensure_synced(&mut self, native: &mut NativeRwkv) -> Result<()> {
        if native.resident.is_some() && native.marker.as_ref() == Some(&self.rwkv_native_marker()?)
        {
            return Ok(());
        }
        self.rwkv_native_sync_inner(native, false)
    }

    fn rwkv_native_sync_inner(&mut self, native: &mut NativeRwkv, force_full: bool) -> Result<()> {
        let started = Instant::now();
        let mut progress = self.new_progress_handler::<RwkvRebuildProgress>();
        progress.update(false, |p| p.stage = RwkvRebuildStage::History)?;
        let marker = self.rwkv_native_marker()?;
        let history = self.rwkv_native_history()?;
        let total = history.inputs.len();

        let mut report = SyncReport::default();
        let resident_prefix = native
            .resident
            .as_ref()
            .filter(|_| !force_full)
            .and_then(|identity| history.prefix_len(identity));
        let start = if let Some(start) = resident_prefix {
            start
        } else if let Some(start) = native
            .checkpoint
            .clone()
            .filter(|_| !force_full)
            .and_then(|(segment, identity)| Some((segment, history.prefix_len(&identity)?)))
            .and_then(
                |(segment, start)| match native.restore_checkpoint(segment) {
                    Ok(()) => Some(start),
                    Err(err) => {
                        tracing::warn!(?err, "RWKV checkpoint restore failed; rebuilding");
                        None
                    }
                },
            )
        {
            report.restored_checkpoint = true;
            start
        } else {
            native
                .runtime
                .finish_warm_up_state_checkpoints()
                .map_err(io_err)?;
            native.runtime = load_runtime(&native.load_request, &native.precision)?.0;
            native.checkpoint = None;
            native.segments_in_store = 0;
            report.full_rebuild = true;
            0
        };
        // Any failure below leaves the resident state between histories.
        native.resident = None;
        native.predictions.clear();

        let mut predictions = Vec::new();
        let mut done = start;
        while done < total {
            let end = (done + REPLAY_CHUNK).min(total);
            progress.update(false, |p| {
                p.stage = RwkvRebuildStage::Replay;
                p.current = (done - start) as u32;
                p.total = (total - start) as u32;
            })?;
            let chunk = history.inputs[done..end].to_vec();
            for (index, prediction) in native
                .runtime
                .warm_up_reviews(chunk, true)
                .map_err(io_err)?
            {
                predictions.push(RwkvReviewRetrievabilityCacheRow {
                    revlog_id: RevlogId(history.review_ids[done + index]),
                    prediction,
                    sample_role: RwkvReviewRetrievabilitySampleRole::FinalFit,
                    fold_index: -1,
                });
            }
            done = end;
        }
        report.replayed = (total - start) as u64;
        native.resident = Some(history.identity.clone());
        native.resident_hash = history
            .hashes
            .last()
            .copied()
            .unwrap_or_else(rwkv_empty_history_hash);
        native.marker = Some(marker);
        native.last_error.clear();
        if report.full_rebuild {
            // Older rows would mix predictions of another model or history.
            self.storage.prune_rwkv_review_retrievability_rows(
                NATIVE_PREDICTION_SOURCE,
                &history
                    .review_ids
                    .iter()
                    .map(|id| {
                        (
                            RevlogId(*id),
                            RwkvReviewRetrievabilitySampleRole::FinalFit,
                            -1,
                        )
                    })
                    .collect::<Vec<_>>(),
                &[],
            )?;
        }
        self.storage
            .set_rwkv_review_retrievability_predictions(&predictions, NATIVE_PREDICTION_SOURCE)?;

        native.answers_since_checkpoint += report.replayed.min(u32::MAX as u64) as u32;
        if report.full_rebuild || native.answers_since_checkpoint >= native.checkpoint_every {
            progress.update(false, |p| p.stage = RwkvRebuildStage::Saving)?;
            native.write_checkpoint(report.full_rebuild)?;
        }
        if report.replayed > 0 || report.full_rebuild {
            self.rwkv_native_invalidate_scores(native);
        }
        report.millis = started.elapsed().as_millis().min(u32::MAX as u128) as u32;
        native.last_sync = report;
        tracing::debug!(
            reviews = total,
            replayed = report.replayed,
            full = report.full_rebuild,
            restored = report.restored_checkpoint,
            elapsed_ms = report.millis,
            "RWKV native sync"
        );
        Ok(())
    }

    pub(crate) fn rwkv_native_checkpoint(&mut self) -> Result<RwkvNativeStatus> {
        self.with_rwkv_native(|_, native| native.write_checkpoint(false))?;
        self.rwkv_native_status()
    }

    fn rwkv_native_invalidate_scores(&mut self, native: &mut NativeRwkv) {
        if !native.scored_decks.is_empty() {
            native.scored_decks.clear();
            // Dropping the scores rebuilds the queue, which rescores it.
            if let Ok(deck) = self.get_current_deck() {
                if let Err(err) = self.set_rwkv_review_queue_score_entries(deck.id, HashMap::new())
                {
                    tracing::warn!(?err, "clearing RWKV Instant scores");
                }
            }
        }
    }

    fn rwkv_native_config_for_card(&self, card: &Card) -> Result<Option<DeckConfig>> {
        let deck_id = home_deck_id(card);
        let Some(deck) = self.storage.get_deck(deck_id)? else {
            return Ok(None);
        };
        let Some(config_id) = deck.config_id() else {
            return Ok(None);
        };
        self.storage.get_deck_config(config_id)
    }

    /// `_latest_eligible_review_for_card`: id, ease and kind of the card's
    /// newest answer that replay keeps.
    fn rwkv_native_latest_eligible_review(
        &self,
        card_id: CardId,
    ) -> Result<Option<(i64, i64, i64)>> {
        Ok(self
            .storage
            .db
            .query_row(
                "select r.id, r.ease, r.type from revlog r join cards c on c.id = r.cid
                 where r.cid = ?1
                   and r.ease between 1 and 4 and r.type in (0, 1, 2, 3, 4, 5)
                   and not (r.type = 3 and r.factor = 0)
                   and r.id > coalesce((select max(id) from revlog
                       where cid = ?1 and type = 4 and ease = 0 and factor = 0), 0)
                   and r.id > coalesce(
                     (select case when json_valid(cast(val as text)) then
                        case when json_type(cast(val as text), '$.after') = 'integer'
                        then max(json_extract(cast(val as text), '$.after'), 0) end end
                      from config where key = 'rwkvHistoryReset_' || c.id),
                     case when json_valid(c.data) then
                       case when json_type(c.data, '$.rwkv_reset_after') = 'integer'
                       then max(json_extract(c.data, '$.rwkv_reset_after'), 0) else 0 end
                     else 0 end)
                 order by r.id desc limit 1",
                [card_id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .optional()?)
    }

    /// `_rwkv_review_state_for_live_context`: a review card answered again on
    /// the day of an earlier answer is a relearning or filtered repeat.
    fn rwkv_native_live_review_state(
        &self,
        card_id: CardId,
        base: i64,
        answered_at_millis: i64,
        timing: &SchedTimingToday,
    ) -> Result<i64> {
        if base != review_state::REVIEW {
            return Ok(base);
        }
        let Some((previous_id, previous_ease, previous_kind)) =
            self.rwkv_native_latest_eligible_review(card_id)?
        else {
            return Ok(base);
        };
        if day_offset_for_millis(previous_id, timing)
            != day_offset_for_millis(answered_at_millis, timing)
        {
            return Ok(base);
        }
        Ok(match previous_kind {
            1 if previous_ease == 1 => review_state::RELEARNING,
            1 => review_state::FILTERED,
            2 if matches!(previous_ease, 1 | 2) => review_state::RELEARNING,
            2 => review_state::FILTERED,
            0 => review_state::FILTERED,
            _ => base,
        })
    }

    /// The query a show-time prediction uses (`rwkv_review_input` with no
    /// ease). `card` must have its RWKV last review time populated.
    fn rwkv_native_query_input(
        &mut self,
        card: &Card,
        current: CardState,
        config: &DeckConfig,
        dynamic_retentions: Option<[f32; 4]>,
        timing: &SchedTimingToday,
    ) -> Result<(ReviewInput, f32)> {
        let preset = self.fsrs_preset_for_card(card)?;
        let preset_id =
            super::rwkv_stable_preset_id(&preset.id, &HashMap::new()).unwrap_or_else(|_| {
                match &preset.id {
                    crate::scheduler::fsrs::preset::FsrsPresetId::Addon(id) => {
                        stable_addon_preset_id(id)
                    }
                    crate::scheduler::fsrs::preset::FsrsPresetId::DeckConfig(id) => id.0,
                }
            });
        let desired_retention = if preset.desired_retention > 0.0 && preset.desired_retention <= 1.0
        {
            preset.desired_retention
        } else {
            DEFAULT_TARGET_RETENTION
        };
        let base = base_review_state(current);
        let now_millis = timing.now.0 * 1000;
        let review_state = self.rwkv_native_live_review_state(card.id, base, now_millis, timing)?;
        let (mut elapsed_days, mut elapsed_seconds) = state_elapsed(current);
        if matches!(current, CardState::Normal(NormalState::New(_))) && elapsed_seconds.is_none() {
            // The first query measures time since the card was created.
            let seconds = (timing.now.0 - card.id.0 / 1000).max(0);
            elapsed_seconds = Some(seconds);
            elapsed_days = Some(seconds / 86_400);
        }
        if matches!(
            review_state,
            review_state::REVIEW | review_state::RELEARNING | review_state::FILTERED
        ) {
            // `_elapsed_since_card_last_review`.
            let last = card.last_review_time.filter(|time| {
                card.rwkv_history_reset_after
                    .is_none_or(|cutoff| time.0.saturating_mul(1000) > cutoff)
            });
            let last = match last {
                Some(last) => Some(last.0),
                None => self
                    .rwkv_native_latest_eligible_review(card.id)?
                    .map(|(id, _, _)| id / 1000),
            };
            elapsed_days = last.map(|last| (timing.next_day_at.0 - last).max(0) / 86_400);
            elapsed_seconds = last.map(|last| (timing.now.0 - last).max(0));
        }
        let mut target_retentions = [Some(desired_retention); 4];
        if let Some(dynamic) = dynamic_retentions {
            if dynamic.iter().all(|r| *r > 0.0 && *r <= 1.0) {
                target_retentions = dynamic.map(Some);
            }
        }
        Ok((
            ReviewInput {
                card_id: card.id.0,
                note_id: Some(card.note_id.0),
                deck_id: Some(home_deck_id(card).0),
                preset_id: Some(preset_id),
                is_query: true,
                ease: None,
                duration_millis: None,
                card_type: Some(review_state),
                day_offset: Some(timing.days_elapsed as i64),
                current_elapsed_days: elapsed_days,
                current_elapsed_seconds: elapsed_seconds,
                target_retentions,
                enforce_grade_order: config.inner.rwkv_review_enforce_grade_order,
            },
            desired_retention,
        ))
    }

    /// Predicts for one card from the resident state.
    fn rwkv_native_predict(
        &mut self,
        native: &mut NativeRwkv,
        card: &Card,
        states: &SchedulingStates,
        config: &DeckConfig,
    ) -> Result<(crate::rwkv::ReviewOutput, f32)> {
        let timing = self.timing_today()?;
        let mut card = card.clone();
        self.populate_rwkv_last_review_times(std::slice::from_mut(&mut card))?;
        let dynamic = states
            .dynamic_desired_retention_enabled
            .then_some(states.dynamic_desired_retentions)
            .flatten();
        let (input, target) =
            self.rwkv_native_query_input(&card, states.current, config, dynamic, &timing)?;
        let state = native.runtime.warm_up_state(&input).map_err(io_err)?;
        let output = native
            .runtime
            .review(
                input,
                EngineReviewState {
                    card: state.card.as_deref(),
                    deck: state.deck.as_deref(),
                    note: state.note.as_deref(),
                    preset: state.preset.as_deref(),
                    global: state.global.as_deref(),
                },
            )
            .map_err(io_err)?;
        Ok((output, target))
    }

    /// Replaces a card's answer states with RWKV-Curve's when its preset uses
    /// RWKV-Curve and the native runtime is ready. Errors leave the states
    /// unchanged: scheduling falls back to FSRS rather than failing.
    pub(crate) fn rwkv_native_apply_curve_states(
        &mut self,
        card: &Card,
        states: SchedulingStates,
    ) -> Result<SchedulingStates> {
        if !self.rwkv_native_loaded() {
            return Ok(states);
        }
        let Some(config) = self.rwkv_native_config_for_card(card)? else {
            return Ok(states);
        };
        let used = PresetUse::of(&config);
        if !used.active() {
            return Ok(states);
        }
        let card = card.clone();
        let result = self.with_rwkv_native(|col, native| {
            col.rwkv_native_ensure_synced(native)?;
            let (output, _) = col.rwkv_native_predict(native, &card, &states, &config)?;
            let review_count = native.resident.as_ref().map_or(0, |id| id.review_count);
            let mut curve_applied = false;
            let mut updated = states.clone();
            if used.curve && native.has_curve && output.intervals.iter().any(Option::is_some) {
                updated = col.scheduling_states_with_intervals(card.id, output.intervals)?;
                for (state, (interval, s90)) in [
                    &mut updated.again,
                    &mut updated.hard,
                    &mut updated.good,
                    &mut updated.easy,
                ]
                .into_iter()
                .zip(output.intervals.iter().zip(output.s90s.iter()))
                {
                    if let (Some(_), Some(s90)) = (interval, s90) {
                        if s90.is_finite() && *s90 > 0.0 {
                            apply_s90(state, *s90);
                        }
                    }
                }
                curve_applied = true;
            }
            native.predictions.insert(
                card.id,
                CachedPrediction {
                    review_count,
                    retrievability: output.retrievability,
                    s90s: output.s90s,
                    curve_applied,
                },
            );
            Ok(updated)
        });
        match result {
            Ok(Some(updated)) => Ok(updated),
            Ok(None) => Ok(states),
            Err(err) => {
                tracing::warn!(?err, "RWKV-Curve states unavailable; using FSRS");
                Ok(states)
            }
        }
    }

    /// Fills the RWKV fields of an answer the client left empty, before the
    /// answer is applied (`set_answer_rwkv_metadata`).
    pub(crate) fn rwkv_native_fill_answer(&mut self, answer: &mut CardAnswer) -> Result<()> {
        // Fields the client already set are kept: the desktop sets the
        // review kind itself, and shares this code for the rest.
        if !self.rwkv_native_loaded()
            || (answer.rwkv_s90.is_some()
                && answer.rwkv_retrievability.is_some()
                && answer.rwkv_review_kind.is_some())
        {
            return Ok(());
        }
        let Some(card) = self.storage.get_card(answer.card_id)? else {
            return Ok(());
        };
        let Some(config) = self.rwkv_native_config_for_card(&card)? else {
            return Ok(());
        };
        if !PresetUse::of(&config).active() {
            return Ok(());
        }
        let timing = self.timing_today()?;
        let answered_at = if answer.answered_at.0 > 0 {
            answer.answered_at.0
        } else {
            timing.now.0 * 1000
        };
        let mut base = base_review_state(answer.current_state);
        if let Some(original) =
            factor_zero_rescheduled_learning_state(answer.current_state, answer.new_state)
        {
            base = original;
        }
        if answer.rwkv_review_kind.is_none() {
            let review_state =
                self.rwkv_native_live_review_state(card.id, base, answered_at, &timing)?;
            answer.rwkv_review_kind = raw_review_kind(review_state);
        }

        let ease = rating_ease(answer.rating);
        let cached = |col: &Collection| {
            col.state.rwkv_native.as_ref().and_then(|native| {
                let review_count = native.resident.as_ref()?.review_count;
                native
                    .predictions
                    .get(&card.id)
                    .filter(|prediction| prediction.review_count == review_count)
                    .cloned()
            })
        };
        let mut prediction = cached(self);
        if prediction.is_none() && answer.rwkv_s90.is_none() {
            // The client answered with states RWKV did not make (a card deep
            // in a prefetched queue): predict now, and answer with
            // RWKV-Curve's state for the chosen rating instead.
            let (fsrs_states, card) = self.get_scheduling_states_inner(card.clone(), None)?;
            if fsrs_states.current == answer.current_state {
                let states = self.rwkv_native_apply_curve_states(&card, fsrs_states)?;
                prediction = cached(self);
                if prediction.as_ref().is_some_and(|p| p.curve_applied) {
                    answer.new_state = match answer.rating {
                        Rating::Again => states.again,
                        Rating::Hard => states.hard,
                        Rating::Good => states.good,
                        Rating::Easy => states.easy,
                    };
                }
            }
        }
        if let Some(prediction) = prediction {
            if answer.rwkv_retrievability.is_none()
                && prediction.retrievability.is_finite()
                && (0.0..=1.0).contains(&prediction.retrievability)
            {
                answer.rwkv_retrievability = Some(prediction.retrievability);
            }
            if prediction.curve_applied && answer.rwkv_s90.is_none() {
                if let Some(s90) = prediction.s90s[(ease - 1) as usize] {
                    if s90.is_finite() && s90 > 0.0 {
                        answer.rwkv_s90 = Some(s90);
                    }
                }
            }
        }
        Ok(())
    }

    /// Whether the history has not changed since the last sync, so the
    /// coming answer can be replayed on its own.
    pub(crate) fn rwkv_native_unchanged_since_sync(&self) -> bool {
        let Some(native) = self.state.rwkv_native.as_deref() else {
            return false;
        };
        native.resident.is_some()
            && native.marker.is_some()
            && self.rwkv_native_marker().ok().as_ref() == native.marker.as_ref()
    }

    /// Replays the new answer once it is saved. With an unchanged history
    /// before the answer, only the answered card's new reviews are read;
    /// otherwise the whole history is synced. A failure only marks the state
    /// stale; it never undoes the answer.
    pub(crate) fn rwkv_native_after_answer(&mut self, card_id: CardId, unchanged_before: bool) {
        let result = self.with_rwkv_native(|col, native| {
            native.predictions.remove(&card_id);
            if unchanged_before && col.rwkv_native_fast_forward(native, card_id)? {
                return Ok(());
            }
            col.rwkv_native_sync_inner(native, false)
        });
        if let Err(err) = result {
            tracing::warn!(?err, "RWKV state not advanced after answer");
            if let Some(native) = self.state.rwkv_native.as_mut() {
                native.resident = None;
            }
        }
    }

    /// Appends one card's new reviews to the resident state, when they are
    /// the only change since the last sync and start no learning sequence
    /// (which could drop that card's earlier reviews from replay). Returns
    /// false when a full sync is needed instead.
    fn rwkv_native_fast_forward(
        &mut self,
        native: &mut NativeRwkv,
        card_id: CardId,
    ) -> Result<bool> {
        let started = Instant::now();
        let (Some(resident), Some(before)) = (native.resident.clone(), native.marker.clone())
        else {
            return Ok(false);
        };
        let after = self.rwkv_native_marker()?;
        if after.cards_count != before.cards_count
            || after.decks_max_mod != before.decks_max_mod
            || after.configs_max_mod != before.configs_max_mod
            || after.usn != before.usn
            || after.revlog_count < before.revlog_count
        {
            return Ok(false);
        }
        let added = (after.revlog_count - before.revlog_count) as usize;
        let request = self.rwkv_native_fingerprint_request()?;
        let mut hash = native.resident_hash;
        let mut inputs = Vec::new();
        let mut review_ids = Vec::new();
        let mut starts_learning = false;
        let _ = self.visit_rwkv_historical_reviews(
            request,
            RwkvHistoricalReplayOptions {
                only_card: Some(card_id),
                ..Default::default()
            },
            |review, _, _| {
                if review.row.review_id > resident.last_review_id {
                    starts_learning |= review.row.is_learning_start;
                    hash = rwkv_history_hash_after_review(hash, &review);
                    review_ids.push(review.row.review_id);
                    inputs.push(replay_input(&review));
                }
            },
        )?;
        if starts_learning || inputs.len() != added {
            return Ok(false);
        }
        let replayed = inputs.len();
        if replayed > 0 {
            native.resident = None;
            let predictions = native
                .runtime
                .warm_up_reviews(inputs, true)
                .map_err(io_err)?
                .into_iter()
                .map(|(index, prediction)| RwkvReviewRetrievabilityCacheRow {
                    revlog_id: RevlogId(review_ids[index]),
                    prediction,
                    sample_role: RwkvReviewRetrievabilitySampleRole::FinalFit,
                    fold_index: -1,
                })
                .collect::<Vec<_>>();
            self.storage.set_rwkv_review_retrievability_predictions(
                &predictions,
                NATIVE_PREDICTION_SOURCE,
            )?;
            native.resident = Some(HistoryIdentity {
                last_review_id: *review_ids.last().unwrap(),
                review_count: resident.review_count + replayed as u64,
                history_hash: rwkv_history_hash_hex(hash),
            });
            native.resident_hash = hash;
            native.answers_since_checkpoint += replayed as u32;
            if native.answers_since_checkpoint >= native.checkpoint_every {
                native.write_checkpoint(false)?;
            }
            self.rwkv_native_invalidate_scores(native);
        }
        native.marker = Some(after);
        native.last_sync = SyncReport {
            replayed: replayed as u64,
            millis: started.elapsed().as_millis().min(u32::MAX as u128) as u32,
            ..Default::default()
        };
        Ok(true)
    }

    /// Scores the current deck's review candidates for RWKV-Instant before
    /// its queue is built, when a preset of the deck uses RWKV-Instant.
    pub(crate) fn rwkv_native_prepare_queue_scores(&mut self, deck: &Deck) -> Result<()> {
        if !self.rwkv_native_loaded() {
            return Ok(());
        }
        let Some(config_id) = deck.config_id() else {
            return Ok(());
        };
        let Some(config) = self.storage.get_deck_config(config_id)? else {
            return Ok(());
        };
        if !config.inner.rwkv_review_instant_order_enabled {
            return Ok(());
        }
        let deck_id = deck.id;
        let days_elapsed = self.timing_today()?.days_elapsed;
        let result = self.with_rwkv_native(|col, native| {
            col.rwkv_native_ensure_synced(native)?;
            if native.scored_decks.contains(&deck_id)
                && col
                    .rwkv_review_queue_scores(deck_id, days_elapsed)
                    .is_some()
            {
                return Ok(());
            }
            let scores = col.rwkv_native_instant_scores(native, deck, &config)?;
            col.set_rwkv_review_queue_score_entries(deck_id, scores)?;
            native.scored_decks.insert(deck_id);
            Ok(())
        });
        if let Err(err) = result {
            tracing::warn!(?err, "RWKV-Instant scores unavailable; using due order");
        }
        Ok(())
    }

    fn rwkv_native_instant_scores(
        &mut self,
        native: &mut NativeRwkv,
        deck: &Deck,
        config: &DeckConfig,
    ) -> Result<HashMap<CardId, RwkvReviewQueueScoreEntry>> {
        let rows = self.rwkv_review_input_rows_for_deck_review_queue(
            RwkvReviewInputRowsForDeckReviewQueueRequest {
                deck_id: deck.id.0,
                include_disabled_decks: false,
                include_new_cards: false,
            },
        )?;
        let mut inputs = Vec::with_capacity(rows.rows.len());
        let mut targets = Vec::with_capacity(rows.rows.len());
        for row in &rows.rows {
            let state = match (
                row.current_state_kind.as_str(),
                row.current_normal_state_kind.as_str(),
            ) {
                ("filtered", _) => review_state::FILTERED,
                (_, "new") => review_state::LEARN_START,
                (_, "learning") => review_state::LEARNING,
                (_, "review") => review_state::REVIEW,
                (_, "relearning") => review_state::RELEARNING,
                _ => row.card_type as i64,
            };
            let target = if row.target_retention > 0.0 && row.target_retention <= 1.0 {
                row.target_retention
            } else {
                DEFAULT_TARGET_RETENTION
            };
            targets.push(target);
            inputs.push(ReviewInput {
                card_id: row.card_id,
                note_id: Some(row.note_id),
                deck_id: Some(row.deck_id),
                preset_id: (!row.preset_id.is_empty())
                    .then(|| stable_addon_preset_id(&row.preset_id)),
                is_query: true,
                ease: None,
                duration_millis: None,
                card_type: Some(state),
                day_offset: Some(row.day_offset as i64),
                current_elapsed_days: row.current_elapsed_days.map(i64::from),
                current_elapsed_seconds: row.current_elapsed_seconds.map(i64::from),
                target_retentions: [Some(target); 4],
                enforce_grade_order: row.enforce_grade_order.unwrap_or(true),
            });
        }
        let card_ids: Vec<_> = inputs.iter().map(|input| CardId(input.card_id)).collect();
        let predictions = if inputs.is_empty() {
            vec![]
        } else {
            native
                .runtime
                .predict_retrievability_many_from_warm_up(inputs)
                .map_err(io_err)?
        };
        let min_intervening = config.inner.rwkv_review_min_intervening_reviews;
        let intervening = if min_intervening > 0 {
            let deck_ids = self.storage.deck_id_with_children(deck)?;
            self.storage
                .recent_rwkv_intervening_reviews(&deck_ids, min_intervening)?
                .into_iter()
                .filter(|(_, reviews)| *reviews < min_intervening)
                .collect()
        } else {
            HashMap::new()
        };
        Ok(card_ids
            .into_iter()
            .zip(predictions)
            .zip(targets)
            .filter(|((_, r), _)| r.is_finite())
            .map(|((card_id, retrievability), target)| {
                (
                    card_id,
                    RwkvReviewQueueScoreEntry {
                        retrievability,
                        intervening_reviews: intervening.get(&card_id).copied(),
                        target_retention: Some(target),
                    },
                )
            })
            .collect())
    }

    pub(crate) fn rwkv_native_card_prediction(
        &mut self,
        card_id: CardId,
    ) -> Result<RwkvNativeCardPredictionResponse> {
        let card = self.storage.get_card(card_id)?.or_not_found(card_id)?;
        let Some(config) = self.rwkv_native_config_for_card(&card)? else {
            return Ok(Default::default());
        };
        let used = PresetUse::of(&config);
        let (states, card) = self.get_scheduling_states_inner(card, None)?;
        let result = self.with_rwkv_native(|col, native| {
            col.rwkv_native_ensure_synced(native)?;
            col.rwkv_native_predict(native, &card, &states, &config)
        })?;
        let Some((output, target)) = result else {
            return Ok(Default::default());
        };
        let flat = |values: [Option<f32>; 4]| values.iter().map(|v| v.unwrap_or(0.0)).collect();
        Ok(RwkvNativeCardPredictionResponse {
            available: true,
            curve_enabled: used.curve,
            instant_enabled: used.instant,
            retrievability: output.retrievability,
            curve_retrievability: output.curve_retrievability,
            current_s90: output.current_s90,
            intervals: flat(output.intervals),
            s90s: flat(output.s90s),
            button_probabilities: output.button_probabilities.to_vec(),
            target_retention: target,
        })
    }

    /// Bucketed calibration of the predictions recorded during replay: the
    /// model's recall prediction before each answer against its outcome.
    pub(crate) fn rwkv_native_calibration(
        &mut self,
        buckets: u32,
    ) -> Result<RwkvNativeCalibrationResponse> {
        let buckets = buckets.clamp(2, 50) as usize;
        let rows: Vec<(f32, bool)> = {
            let sql = "select p.prediction, r.ease
                 from retrievability_cache.search_stats_rwkv_review_retrievability p
                 join revlog r on r.id = p.revlog_id
                 where p.source = ?1 and p.sample_role = 'final_fit'
                   and r.ease between 1 and 4 and r.type in (1, 2, 3)";
            let mut stmt = self.storage.db.prepare(sql)?;
            let rows = stmt
                .query_map([NATIVE_PREDICTION_SOURCE], |row| {
                    Ok((row.get::<_, f64>(0)? as f32, row.get::<_, i64>(1)? > 1))
                })?
                .collect::<std::result::Result<Vec<_>, _>>()?;
            rows
        };
        let mut sums = vec![(0f64, 0f64, 0u32); buckets];
        let mut log_loss = 0f64;
        let (mut mean_p, mut mean_y) = (0f64, 0f64);
        for (p, recalled) in &rows {
            let p = p.clamp(1e-6, 1.0 - 1e-6) as f64;
            let y = if *recalled { 1.0 } else { 0.0 };
            let bucket = ((p * buckets as f64) as usize).min(buckets - 1);
            sums[bucket].0 += p;
            sums[bucket].1 += y;
            sums[bucket].2 += 1;
            log_loss -= y * p.ln() + (1.0 - y) * (1.0 - p).ln();
            mean_p += p;
            mean_y += y;
        }
        let total = rows.len();
        let n = total.max(1) as f64;
        let mut squared = 0f64;
        let out_buckets = sums
            .iter()
            .enumerate()
            .map(|(index, (p, y, count))| {
                let c = (*count).max(1) as f64;
                if *count > 0 {
                    squared += (p / c - y / c).powi(2) * *count as f64;
                }
                anki_proto::scheduler::rwkv_native_calibration_response::Bucket {
                    predicted_min: index as f32 / buckets as f32,
                    predicted_max: (index + 1) as f32 / buckets as f32,
                    mean_predicted: (p / c) as f32,
                    observed: (y / c) as f32,
                    count: *count,
                }
            })
            .collect();
        Ok(RwkvNativeCalibrationResponse {
            buckets: out_buckets,
            total: total as u32,
            log_loss: (log_loss / n) as f32,
            rmse_bins: (squared / n).sqrt() as f32,
            mean_predicted: (mean_p / n) as f32,
            mean_observed: (mean_y / n) as f32,
        })
    }
}

#[cfg(test)]
mod test {
    use std::path::Path;

    use super::*;
    use crate::card::CardQueue;
    use crate::card::CardType;
    use crate::config::BoolKey;
    use crate::deckconfig::DeckConfigId;
    use crate::revlog::RevlogEntry;
    use crate::revlog::RevlogReviewKind;
    use crate::tests::NoteAdder;

    const DAY_MS: i64 = 86_400_000;

    fn weights(model: &str) -> Option<String> {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../qt/aqt/rwkv_inference")
            .join(model)
            .join("weights.f32.bin");
        path.exists().then(|| path.to_string_lossy().into_owned())
    }

    /// A deterministic sequence, so failures reproduce.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 33
        }
    }

    struct Fixture {
        col: Collection,
        cards: Vec<Card>,
        rng: Lcg,
        next_review_id: i64,
    }

    /// Cards with a synthetic review history over the past ~200 days, every
    /// card a due review card, and the default preset on `scheduler`.
    fn fixture(cards: usize, reviews_per_card: usize, curve: bool, instant: bool) -> Fixture {
        let mut col = Collection::new();
        // Reviews must fall after the collection's creation day.
        col.storage
            .set_creation_stamp(TimestampSecs(TimestampSecs::now().0 - 400 * 86_400))
            .unwrap();
        col.clear_study_queues();
        col.set_config_bool(BoolKey::Fsrs, true, false).unwrap();
        let mut config = col
            .storage
            .get_deck_config(DeckConfigId(1))
            .unwrap()
            .unwrap();
        config.inner.rwkv_review_enabled = curve;
        config.inner.rwkv_review_instant_order_enabled = instant;
        config.inner.rwkv_review_enforce_grade_order = true;
        config.inner.reviews_per_day = 9999;
        col.storage.update_deck_conf(&config).unwrap();
        let mut ids = Vec::new();
        for _ in 0..cards {
            let note = NoteAdder::basic(&mut col).add(&mut col);
            ids.extend(col.storage.card_ids_of_notes(&[note.id]).unwrap());
        }
        let now = TimestampMillis::now().0;
        let mut rng = Lcg(42);
        let start = now - 200 * DAY_MS;
        let mut rows = Vec::new();
        for (index, card_id) in ids.iter().enumerate() {
            // A plausible learner: mostly Good, intervals growing ~2.5x, and
            // an occasional lapse that resets them.
            let mut at = start + (index as i64) * 3_600_000;
            let mut interval = 1i64;
            for review in 0..reviews_per_card {
                let ease = if review > 0 && rng.next().is_multiple_of(12) {
                    1
                } else {
                    3
                };
                rows.push((at, *card_id, ease, review == 0));
                interval = if ease == 1 {
                    1
                } else {
                    (interval * 5 / 2).max(1)
                };
                at += DAY_MS * interval + (rng.next() % 50_000) as i64;
                if at > now - DAY_MS {
                    break;
                }
            }
        }
        rows.sort_by_key(|row| row.0);
        let mut last = 0;
        for (mut id, card_id, ease, first) in rows {
            if id <= last {
                id = last + 1;
            }
            last = id;
            col.storage
                .add_revlog_entry(
                    &RevlogEntry {
                        id: RevlogId(id),
                        cid: card_id,
                        usn: Usn(0),
                        button_chosen: ease,
                        interval: if first { -600 } else { 3 },
                        ease_factor: 2_500,
                        taken_millis: 4_000 + (id % 7_000) as u32,
                        review_kind: if first {
                            RevlogReviewKind::Learning
                        } else {
                            RevlogReviewKind::Review
                        },
                        ..Default::default()
                    },
                    false,
                )
                .unwrap();
        }
        let today = col.timing_today().unwrap().days_elapsed as i32;
        let mut out = Vec::new();
        for card_id in ids {
            let mut card = col.storage.get_card(card_id).unwrap().unwrap();
            card.ctype = CardType::Review;
            card.queue = CardQueue::Review;
            card.interval = 5;
            card.due = today - (card_id.0 % 3) as i32;
            card.memory_state = Some(FsrsMemoryState {
                stability: 5.0,
                stability_internal: 5.0,
                stability_fast: None,
                difficulty: 5.0,
            });
            col.storage.update_card(&card).unwrap();
            out.push(card);
        }
        Fixture {
            col,
            cards: out,
            rng,
            next_review_id: last + 1,
        }
    }

    impl Fixture {
        fn add_reviews(&mut self, count: usize) {
            for _ in 0..count {
                let card = &self.cards[(self.rng.next() as usize) % self.cards.len()];
                self.col
                    .storage
                    .add_revlog_entry(
                        &RevlogEntry {
                            id: RevlogId(self.next_review_id),
                            cid: card.id,
                            usn: Usn(0),
                            button_chosen: 3,
                            interval: 4,
                            ease_factor: 2_500,
                            taken_millis: 5_000,
                            review_kind: RevlogReviewKind::Review,
                            ..Default::default()
                        },
                        false,
                    )
                    .unwrap();
                self.next_review_id += 1_000;
            }
        }
    }

    fn load(col: &mut Collection, dir: &Path, instant: bool) -> RwkvNativeStatus {
        col.rwkv_native_load(RwkvNativeLoadRequest {
            curve_model_path: weights("sm92"),
            curve_model: "sm92".into(),
            instant_model_path: if instant { weights("sm98") } else { None },
            instant_model: instant.then(|| "sm98".into()),
            state_precision: "f32".into(),
            state_dir: dir.to_string_lossy().into_owned(),
            checkpoint_every_answers: 5,
        })
        .unwrap()
    }

    fn predictions(col: &mut Collection, cards: &[Card]) -> Vec<RwkvNativeCardPredictionResponse> {
        cards
            .iter()
            .map(|card| col.rwkv_native_card_prediction(card.id).unwrap())
            .collect()
    }

    fn assert_same(a: &[RwkvNativeCardPredictionResponse], b: &[RwkvNativeCardPredictionResponse]) {
        assert_eq!(a.len(), b.len());
        for (a, b) in a.iter().zip(b) {
            assert!(a.available && b.available);
            assert!(
                (a.retrievability - b.retrievability).abs() < 2e-5,
                "R {} vs {}",
                a.retrievability,
                b.retrievability
            );
            for (x, y) in a.intervals.iter().zip(&b.intervals) {
                assert!(
                    (x - y).abs() <= 1e-3 * x.abs().max(1.0),
                    "interval {x} vs {y}"
                );
            }
            for (x, y) in a.s90s.iter().zip(&b.s90s) {
                assert!((x - y).abs() <= 1e-3 * x.abs().max(1.0), "s90 {x} vs {y}");
            }
        }
    }

    /// The reference: a fresh runtime that replays the whole history.
    fn fresh_predictions(
        fix: &mut Fixture,
        instant: bool,
    ) -> Vec<RwkvNativeCardPredictionResponse> {
        let dir = tempfile::tempdir().unwrap();
        let mut native = fix.col.state.rwkv_native.take();
        let _ = load(&mut fix.col, dir.path(), instant);
        let status = fix.col.rwkv_native_sync(true).unwrap();
        assert!(status.last_sync_full_rebuild);
        let out = predictions(&mut fix.col, &fix.cards.clone());
        fix.col.rwkv_native_unload();
        std::mem::swap(&mut fix.col.state.rwkv_native, &mut native);
        out
    }

    #[test]
    fn stable_preset_ids_match_the_desktop() {
        assert_eq!(stable_addon_preset_id("1234"), 1234);
        // python: int.from_bytes(hashlib.blake2b(b"abc", digest_size=8)
        //   .digest(), "big") & ((1 << 63) - 1)
        assert_eq!(stable_addon_preset_id("abc"), 6_393_727_014_797_677_913);
    }

    #[test]
    fn live_state_maps_to_revlog_kinds() {
        assert_eq!(raw_review_kind(review_state::LEARN_START), Some(0));
        assert_eq!(raw_review_kind(review_state::LEARNING), Some(0));
        assert_eq!(raw_review_kind(review_state::REVIEW), Some(1));
        assert_eq!(raw_review_kind(review_state::RELEARNING), Some(2));
        assert_eq!(raw_review_kind(review_state::FILTERED), Some(3));
    }

    #[test]
    fn incremental_replay_matches_a_full_rebuild() {
        if weights("sm92").is_none() {
            return;
        }
        for instant in [false, true] {
            let mut fix = fixture(30, 10, true, instant);
            let dir = tempfile::tempdir().unwrap();
            let _ = load(&mut fix.col, dir.path(), instant);
            let first = fix.col.rwkv_native_sync(false).unwrap();
            assert!(first.ready && first.last_sync_full_rebuild);
            assert_eq!(first.paired, instant);

            fix.add_reviews(7);
            let second = fix.col.rwkv_native_sync(false).unwrap();
            assert!(!second.last_sync_full_rebuild && !second.last_sync_restored_checkpoint);
            assert_eq!(second.last_sync_replayed_reviews, 7);
            assert_eq!(second.review_count, first.review_count + 7);

            let incremental = predictions(&mut fix.col, &fix.cards.clone());
            let reference = fresh_predictions(&mut fix, instant);
            assert_same(&incremental, &reference);
        }
    }

    #[test]
    fn checkpoint_restores_without_replay() {
        if weights("sm92").is_none() {
            return;
        }
        let mut fix = fixture(25, 8, true, false);
        let dir = tempfile::tempdir().unwrap();
        let _ = load(&mut fix.col, dir.path(), false);
        let _ = fix.col.rwkv_native_sync(false).unwrap();
        fix.add_reviews(3);
        let _ = fix.col.rwkv_native_sync(false).unwrap();
        let _ = fix.col.rwkv_native_checkpoint().unwrap();
        let before = predictions(&mut fix.col, &fix.cards.clone());
        fix.col.rwkv_native_unload();

        let _ = load(&mut fix.col, dir.path(), false);
        let status = fix.col.rwkv_native_sync(false).unwrap();
        assert!(status.last_sync_restored_checkpoint, "{status:?}");
        assert_eq!(status.last_sync_replayed_reviews, 0);
        assert_same(&before, &predictions(&mut fix.col, &fix.cards.clone()));

        // Reviews after the checkpoint replay on top of it.
        fix.col.rwkv_native_unload();
        fix.add_reviews(4);
        let _ = load(&mut fix.col, dir.path(), false);
        let status = fix.col.rwkv_native_sync(false).unwrap();
        assert!(status.last_sync_restored_checkpoint);
        assert_eq!(status.last_sync_replayed_reviews, 4);
        let restored = predictions(&mut fix.col, &fix.cards.clone());
        assert_same(&restored, &fresh_predictions(&mut fix, false));
    }

    #[test]
    fn changed_history_rebuilds() {
        if weights("sm92").is_none() {
            return;
        }
        let mut fix = fixture(20, 8, true, false);
        let dir = tempfile::tempdir().unwrap();
        let _ = load(&mut fix.col, dir.path(), false);
        let _ = fix.col.rwkv_native_sync(false).unwrap();
        // Remove an early review, as an undo or a sync from another device
        // can rewrite history before the newest review.
        let early: i64 = fix
            .col
            .storage
            .db
            .query_row(
                "select id from revlog order by id limit 1 offset 5",
                [],
                |r| r.get(0),
            )
            .unwrap();
        fix.col
            .storage
            .db
            .execute("delete from revlog where id = ?", [early])
            .unwrap();
        let status = fix.col.rwkv_native_sync(false).unwrap();
        assert!(status.last_sync_full_rebuild, "{status:?}");
        let rebuilt = predictions(&mut fix.col, &fix.cards.clone());
        assert_same(&rebuilt, &fresh_predictions(&mut fix, false));
    }

    #[test]
    fn curve_states_and_answer_metadata() {
        if weights("sm92").is_none() {
            return;
        }
        let mut fix = fixture(20, 8, true, false);
        let dir = tempfile::tempdir().unwrap();
        let _ = load(&mut fix.col, dir.path(), false);
        let _ = fix.col.rwkv_native_sync(false).unwrap();

        let queued = fix.col.get_next_card().unwrap().unwrap();
        let card_id = queued.card.id;
        let prediction = fix.col.rwkv_native_card_prediction(card_id).unwrap();
        assert!(prediction.available && prediction.curve_enabled);
        let states = queued.states.unwrap();
        let CardState::Normal(NormalState::Review(good)) = states.good else {
            panic!("good is a review state: {:?}", states.good);
        };
        // The scheduler turns the unrounded interval into whole days.
        let predicted_good = prediction.intervals[2];
        assert!(
            (good.scheduled_days as f32 - predicted_good).abs() <= predicted_good * 0.35 + 1.5,
            "good {} vs RWKV {}",
            good.scheduled_days,
            predicted_good
        );
        // Both are predictions for the same card a moment apart.
        let stability = good.memory_state.unwrap().stability;
        assert!((stability - prediction.s90s[2]).abs() <= 1e-3 * prediction.s90s[2]);

        let before = fix.col.rwkv_native_status().unwrap().review_count;
        fix.col.answer_good();
        let status = fix.col.rwkv_native_status().unwrap();
        assert_eq!(status.review_count, before + 1);
        assert_eq!(status.last_sync_replayed_reviews, 1);
        assert!(!status.last_sync_full_rebuild);
        let (kind, stability): (i64, f32) = fix
            .col
            .storage
            .db
            .query_row(
                "select r.type, json_extract(c.data, '$.s') from revlog r join cards c on c.id = r.cid
                 where r.cid = ? order by r.id desc limit 1",
                [card_id],
                |r| Ok((r.get(0)?, r.get::<_, f64>(1)? as f32)),
            )
            .unwrap();
        assert_eq!(kind, 1);
        assert!((stability - prediction.s90s[2]).abs() < 0.01 * prediction.s90s[2]);

        // The answer replays as a rebuild would replay it. The answered card
        // is left out: its query depends on the seconds since its answer,
        // which differ between the two predictions.
        let others: Vec<_> = fix
            .cards
            .iter()
            .filter(|c| c.id != card_id)
            .cloned()
            .collect();
        let live = predictions(&mut fix.col, &others);
        let reference = fresh_predictions(&mut fix, false);
        let reference: Vec<_> = fix
            .cards
            .iter()
            .zip(reference)
            .filter(|(c, _)| c.id != card_id)
            .map(|(_, p)| p)
            .collect();
        assert_same(&live, &reference);
    }

    #[test]
    fn answers_with_rwkv_states_for_cards_deep_in_the_queue() {
        if weights("sm92").is_none() {
            return;
        }
        let mut fix = fixture(20, 8, true, false);
        let dir = tempfile::tempdir().unwrap();
        let _ = load(&mut fix.col, dir.path(), false);
        let _ = fix.col.rwkv_native_sync(false).unwrap();

        let queued = fix.col.get_queued_cards(5, false, false).unwrap();
        let deep = queued.cards[3].clone();
        let states = deep.states.unwrap();
        let prediction = fix.col.rwkv_native_card_prediction(deep.card.id).unwrap();
        // Beyond the head of the queue the states are FSRS's. Answering such
        // a card (off the queue, as only the top card can be answered from
        // it) still schedules with RWKV.
        let fsrs_good = states.good;
        fix.col
            .answer_card(&mut CardAnswer {
                card_id: deep.card.id,
                current_state: states.current,
                new_state: fsrs_good,
                rating: Rating::Good,
                answered_at: TimestampMillis::now(),
                milliseconds_taken: 3_000,
                custom_data: None,
                desired_retention_override: None,
                desired_retention_override_is_final: false,
                rwkv_s90: None,
                rwkv_retrievability: None,
                rwkv_review_kind: None,
                from_queue: false,
            })
            .unwrap();
        let card = fix.col.storage.get_card(deep.card.id).unwrap().unwrap();
        let stability = card.memory_state.unwrap().stability;
        assert!(
            (stability - prediction.s90s[2]).abs() <= 1e-2 * prediction.s90s[2],
            "answered with RWKV's S90 {} rather than FSRS: {stability}",
            prediction.s90s[2]
        );
    }

    #[test]
    fn instant_scores_order_the_queue() {
        if weights("sm92").is_none() || weights("sm98").is_none() {
            return;
        }
        let mut fix = fixture(20, 8, true, true);
        let dir = tempfile::tempdir().unwrap();
        let _ = load(&mut fix.col, dir.path(), true);
        let _ = fix.col.rwkv_native_sync(false).unwrap();
        let queued = fix.col.get_queued_cards(5, false, true).unwrap();
        assert!(!queued.cards.is_empty());
        let days = fix.col.timing_today().unwrap().days_elapsed;
        let deck = fix.col.get_current_deck().unwrap();
        let scores = fix
            .col
            .rwkv_review_queue_scores(deck.id, days)
            .expect("scores set");
        assert_eq!(scores.len(), fix.cards.len());
        fix.col.answer_good();
        // An answer changes the state, so the queue is scored again.
        fix.col.get_queued_cards(1, false, true).unwrap();
        assert!(fix.col.rwkv_review_queue_scores(deck.id, days).is_some());
    }

    /// `cargo test --release -p anki --lib rwkv::native::test::sync_timing -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn sync_timing() {
        if weights("sm92").is_none() {
            return;
        }
        let cards: usize = std::env::var("RWKV_BENCH_CARDS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(5_000);
        let mut fix = fixture(cards, 40, true, false);
        let reviews: i64 = fix
            .col
            .storage
            .db
            .query_row("select count() from revlog", [], |r| r.get(0))
            .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let _ = load(&mut fix.col, dir.path(), false);
        let t = Instant::now();
        let status = fix.col.rwkv_native_sync(false).unwrap();
        eprintln!(
            "reviews={reviews} full rebuild: {:?} ({} ms reported)",
            t.elapsed(),
            status.last_sync_millis
        );
        let t = Instant::now();
        let _ = fix.col.rwkv_native_sync(false).unwrap();
        eprintln!("no-op sync (traversal only): {:?}", t.elapsed());
        fix.add_reviews(1);
        let t = Instant::now();
        let _ = fix.col.rwkv_native_sync(false).unwrap();
        eprintln!("incremental sync, 1 review: {:?}", t.elapsed());
        // An answer on an unchanged history takes the card-scoped path.
        let card_id = fix.cards[3].id;
        fix.col
            .storage
            .add_revlog_entry(
                &RevlogEntry {
                    id: RevlogId(fix.next_review_id),
                    cid: card_id,
                    usn: Usn(0),
                    button_chosen: 3,
                    interval: 4,
                    ease_factor: 2_500,
                    taken_millis: 5_000,
                    review_kind: RevlogReviewKind::Review,
                    ..Default::default()
                },
                false,
            )
            .unwrap();
        let t = Instant::now();
        fix.col.rwkv_native_after_answer(card_id, true);
        let status = fix.col.rwkv_native_status().unwrap();
        eprintln!(
            "answer fast path: {:?} (replayed {}, full {})",
            t.elapsed(),
            status.last_sync_replayed_reviews,
            status.last_sync_full_rebuild
        );
        let fast = predictions(&mut fix.col, &fix.cards[..50]);
        let _ = fix.col.rwkv_native_sync(true).unwrap();
        assert_same(&fast, &predictions(&mut fix.col, &fix.cards[..50]));
        let t = Instant::now();
        let _ = fix.col.rwkv_native_checkpoint().unwrap();
        eprintln!("delta checkpoint: {:?}", t.elapsed());
        let size = std::fs::metadata(dir.path().join(STORE_FILE))
            .map(|m| m.len())
            .unwrap_or(0);
        eprintln!("store size: {} MB", size / 1_000_000);
        let t = Instant::now();
        let card = fix.cards[0].id;
        let _ = fix.col.rwkv_native_card_prediction(card).unwrap();
        eprintln!("one card prediction: {:?}", t.elapsed());
    }
}
