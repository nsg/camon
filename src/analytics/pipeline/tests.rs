use super::*;

use super::decoder_slot::{DECODER_RESTART_BACKOFF, DECODER_SPAWN_BACKOFF_MAX};
use super::framing::MIN_CROP_FRACTION;
use super::sampling::{
    frames_per_segment, halve_past, pick_four, sample_indices, thin_evenly,
    FILMSTRIP_ACCUMULATOR_CAP, RUN_FRAME_ACCUMULATOR_CAP,
};
use super::skips::SKIP_REPORT_INTERVAL;
use crate::analytics::motion::MotionBox;
use crate::analytics::motion_settings::{SettingsUpdate, TunerMode, MASK_COLS};
use std::sync::atomic::AtomicU32;

fn detector_with_masks() -> MotionDetector {
    const W: usize = 64;
    const H: usize = 48;
    let mut detector = MotionDetector::new(16.0, 4.0);
    let still = vec![60u8; W * H];
    for _ in 0..20 {
        detector.process_frame(&still, W, H);
    }
    let mut moved = still.clone();
    for y in 8..24 {
        for x in 8..24 {
            moved[y * W + x] = 220;
        }
    }
    detector.process_frame(&moved, W, H);
    detector
}

#[test]
fn debug_maps_are_only_encoded_while_somebody_is_watching() {
    let store = MotionStore::new(&["watched".to_string(), "idle".to_string()]);
    let detector = detector_with_masks();
    for kind in MapKind::ALL {
        store.mark_map_requested_ago("watched", kind, Duration::ZERO);
        store.mark_map_requested_ago("idle", kind, Duration::from_secs(600));
    }

    publish_debug_maps(&store, "watched", &detector);
    publish_debug_maps(&store, "idle", &detector);

    for kind in MapKind::ALL {
        assert!(
            store.get_map("watched", kind).is_some(),
            "{} was not published for the camera being watched",
            kind.as_str()
        );
        assert!(
            store.get_map("idle", kind).is_none(),
            "{} was encoded for a camera nobody is watching",
            kind.as_str()
        );
    }
}

#[test]
fn debug_map_demand_is_tracked_per_stage() {
    let store = MotionStore::new(&["cam".to_string()]);
    let detector = detector_with_masks();
    store.mark_map_requested_ago("cam", MapKind::Background, Duration::ZERO);

    publish_debug_maps(&store, "cam", &detector);

    assert!(store.get_map("cam", MapKind::Background).is_some());
    for kind in MapKind::ALL
        .into_iter()
        .filter(|k| *k != MapKind::Background)
    {
        assert!(
            store.get_map("cam", kind).is_none(),
            "{} rode along on a request for another stage",
            kind.as_str()
        );
    }
}

#[test]
fn publishing_a_map_does_not_renew_its_own_demand() {
    let store = MotionStore::new(&["cam".to_string()]);
    let detector = detector_with_masks();
    store.mark_map_requested_ago("cam", MapKind::Stability, Duration::ZERO);

    publish_debug_maps(&store, "cam", &detector);
    store.mark_map_requested_ago("cam", MapKind::Stability, Duration::from_secs(600));

    assert!(
        !store.map_wanted("cam", MapKind::Stability),
        "publishing latched the gate open, so the encode never stops"
    );
}

#[test]
fn the_raw_and_no_shadow_stages_share_one_encode() {
    let store = MotionStore::new(&["cam".to_string()]);
    let detector = detector_with_masks();
    store.mark_map_requested_ago("cam", MapKind::RawMog2, Duration::ZERO);
    store.mark_map_requested_ago("cam", MapKind::NoShadow, Duration::ZERO);

    publish_debug_maps(&store, "cam", &detector);

    let raw = store.get_map("cam", MapKind::RawMog2);
    assert!(raw.is_some());
    assert_eq!(raw, store.get_map("cam", MapKind::NoShadow));
}

#[test]
fn the_raw_and_no_shadow_stages_are_still_gated_apart() {
    let store = MotionStore::new(&["cam".to_string()]);
    let detector = detector_with_masks();

    store.mark_map_requested_ago("cam", MapKind::NoShadow, Duration::ZERO);
    publish_debug_maps(&store, "cam", &detector);
    assert!(store.get_map("cam", MapKind::NoShadow).is_some());
    assert!(
        store.get_map("cam", MapKind::RawMog2).is_none(),
        "raw was filled by a request for no-shadow"
    );

    let store = MotionStore::new(&["cam".to_string()]);
    store.mark_map_requested_ago("cam", MapKind::RawMog2, Duration::ZERO);
    publish_debug_maps(&store, "cam", &detector);
    assert!(store.get_map("cam", MapKind::RawMog2).is_some());
    assert!(
        store.get_map("cam", MapKind::NoShadow).is_none(),
        "no-shadow was filled by a request for raw"
    );
}

#[test]
fn decoder_restart_backoff_ends_when_shutdown_is_requested() {
    let shutdown = Arc::new(AtomicBool::new(false));
    let signaller = Arc::clone(&shutdown);
    std::thread::spawn(move || {
        thread::sleep(Duration::from_millis(20));
        signaller.store(true, Ordering::Relaxed);
    });

    let started = Instant::now();
    sleep_unless_shutdown(DECODER_RESTART_BACKOFF, &shutdown);
    assert!(
        started.elapsed() < DECODER_RESTART_BACKOFF / 2,
        "backoff outlived the shutdown request"
    );
}

#[test]
fn decoder_restart_backoff_is_skipped_when_shutdown_is_already_requested() {
    let started = Instant::now();
    sleep_unless_shutdown(DECODER_RESTART_BACKOFF, &AtomicBool::new(true));
    assert!(started.elapsed() < Duration::from_millis(100));
}

#[test]
fn decoder_restart_backoff_runs_to_completion_without_a_shutdown() {
    let started = Instant::now();
    sleep_unless_shutdown(Duration::from_millis(250), &AtomicBool::new(false));
    assert!(started.elapsed() >= Duration::from_millis(250));
}

const TEST_RETRY: RetrySchedule = RetrySchedule {
    start: Duration::from_millis(5),
    max: Duration::from_millis(20),
};

fn failing_build(attempts: &AtomicU32) -> impl FnMut() -> Result<(), &'static str> + '_ {
    || {
        attempts.fetch_add(1, Ordering::Relaxed);
        Err("no ffmpeg")
    }
}

#[test]
fn analyzer_construction_is_retried_until_it_succeeds() {
    let attempts = AtomicU32::new(0);
    let built = build_with_retry(
        "cam",
        "motion analyzer",
        &AtomicBool::new(false),
        TEST_RETRY,
        || match attempts.fetch_add(1, Ordering::Relaxed) {
            0 | 1 => Err("no ffmpeg"),
            _ => Ok("analyzer"),
        },
    );
    assert_eq!(built, Some("analyzer"));
    assert_eq!(attempts.load(Ordering::Relaxed), 3);
}

#[test]
fn analyzer_construction_retry_ends_when_shutdown_is_requested() {
    let shutdown = Arc::new(AtomicBool::new(false));
    let signaller = Arc::clone(&shutdown);
    std::thread::spawn(move || {
        thread::sleep(Duration::from_millis(50));
        signaller.store(true, Ordering::Relaxed);
    });

    let attempts = AtomicU32::new(0);
    let started = Instant::now();
    let built = build_with_retry(
        "cam",
        "motion analyzer",
        &shutdown,
        TEST_RETRY,
        failing_build(&attempts),
    );
    assert!(built.is_none());
    assert!(
        attempts.load(Ordering::Relaxed) > 1,
        "gave up after a single attempt instead of retrying"
    );
    assert!(
        started.elapsed() < Duration::from_secs(1),
        "retry loop outlived the shutdown request"
    );
}

#[test]
fn a_decoder_that_never_spawns_backs_off_and_stops_repeating_itself() {
    let mut retry = DecoderSpawnRetry::new(DECODER_SPAWN_SCHEDULE);
    let mut reported = Vec::new();
    let mut delays = Vec::new();
    for _ in 0..200 {
        let (delay, report) = retry.failed();
        delays.push(delay);
        if let Some(attempts) = report {
            reported.push(attempts);
        }
    }

    assert_eq!(&reported[..5], &[1, 2, 4, 8, 16]);
    assert!(
        reported.len() < 15,
        "200 failures produced {} lines",
        reported.len()
    );
    let last = *delays.last().unwrap();
    assert!(
        last >= DECODER_SPAWN_BACKOFF_MAX * 4 / 5 && last <= DECODER_SPAWN_BACKOFF_MAX * 6 / 5,
        "{last:?} is not near the ceiling"
    );
    assert!(delays[0] < delays[3], "backoff did not widen");
}

#[test]
fn a_successful_spawn_clears_the_backoff_and_the_streak() {
    let mut retry = DecoderSpawnRetry::new(DECODER_SPAWN_SCHEDULE);
    for _ in 0..20 {
        retry.failed();
    }
    retry.succeeded();

    let (delay, report) = retry.failed();
    assert_eq!(report, Some(1), "escalation did not reset");
    assert!(delay <= DECODER_SPAWN_SCHEDULE.start * 6 / 5, "{delay:?}");
}

#[test]
fn two_cameras_failing_together_do_not_retry_in_lockstep() {
    let delays: std::collections::HashSet<Duration> = (0..16)
        .map(|_| DecoderSpawnRetry::new(DECODER_SPAWN_SCHEDULE).failed().0)
        .collect();
    assert!(delays.len() > 1, "every camera drew the same delay");
}

#[test]
fn analyzer_construction_is_not_attempted_once_shutdown_is_requested() {
    let attempts = AtomicU32::new(0);
    let built = build_with_retry(
        "cam",
        "motion analyzer",
        &AtomicBool::new(true),
        DECODER_SPAWN_SCHEDULE,
        failing_build(&attempts),
    );
    assert!(built.is_none());
    assert_eq!(attempts.load(Ordering::Relaxed), 0);
}

struct CountedChild {
    alive: bool,
    generation: u32,
}

fn running() -> AtomicBool {
    AtomicBool::new(false)
}

fn ensure_counted(
    slot: &mut LongLived<CountedChild>,
    stop: Option<&AtomicBool>,
    forks: &mut u32,
) -> bool {
    ensure_long_lived(
        slot,
        stop,
        "cam",
        |child| child.alive,
        || {
            *forks += 1;
            Ok(CountedChild {
                alive: true,
                generation: *forks,
            })
        },
    )
}

#[test]
fn a_crop_decoder_is_forked_once_and_kept_across_batches() {
    let stop = running();
    let mut slot = LongLived::default();
    let mut forks = 0;
    let mut children = Vec::new();
    for _ in 0..5 {
        assert!(ensure_counted(&mut slot, Some(&stop), &mut forks));
        let child = slot.decoder.as_ref().expect("the batch got no decoder");
        children.push(child.generation);
        slot.primed_with = PRIMING_SEGMENTS;
    }

    assert_eq!(forks, 1, "the crop decoder was forked per batch");
    assert_eq!(children, [1; 5], "the batches ran on different children");
    assert!(
        slot.primed(),
        "the camera lost what its child had already been told"
    );
}

#[test]
fn a_batch_hands_the_crop_decoder_back_to_its_camera() {
    let dir = tempfile::tempdir().unwrap();
    let ctx = test_context("cam", dir.path());
    let mut analyzer = MotionAnalyzer::with_decoder(ctx, FrameDecoder::dead());
    let (crop, _frames) = CropDecoder::detached();
    analyzer.crop_decoder = LongLived {
        decoder: Some(crop),
        primed_with: PRIMING_SEGMENTS,
    };

    analyzer.process_motion_runs(
        vec![MotionSegment {
            seq: 1,
            data: Arc::new(Vec::new()),
            duration_ns: SEC,
        }],
        None,
    );

    assert!(
        analyzer.crop_decoder.decoder.is_some(),
        "the batch kept the camera's crop decoder instead of giving it back"
    );
    assert!(
        analyzer.crop_decoder.primed(),
        "the camera lost what its child had already been told"
    );
}

#[test]
fn a_respawned_crop_decoder_is_primed_again_and_only_then() {
    let stop = running();
    let window = Some(&stop);
    let mut slot = LongLived::default();
    let mut forks = 0;

    assert!(ensure_counted(&mut slot, window, &mut forks));
    assert!(
        !slot.primed(),
        "a fresh child arrived claiming to be primed"
    );
    slot.primed_with = PRIMING_SEGMENTS;

    assert!(ensure_counted(&mut slot, window, &mut forks));
    assert!(slot.primed(), "re-primed a child that never died");
    assert_eq!(forks, 1);

    slot.decoder.as_mut().unwrap().alive = false;
    assert!(ensure_counted(&mut slot, window, &mut forks));
    assert_eq!(forks, 2, "a dead child was not replaced");
    assert!(!slot.primed(), "the replacement child was left unprimed");
}

#[test]
fn the_drain_uses_a_running_crop_decoder_and_forks_no_other() {
    let mut forks = 0;

    let mut empty = LongLived::default();
    assert!(!ensure_counted(&mut empty, None, &mut forks));

    let mut dead = LongLived {
        decoder: Some(CountedChild {
            alive: false,
            generation: 0,
        }),
        primed_with: PRIMING_SEGMENTS,
    };
    assert!(!ensure_counted(&mut dead, None, &mut forks));
    assert_eq!(forks, 0, "the drain forked a crop decoder");

    let mut running_child = LongLived {
        decoder: Some(CountedChild {
            alive: true,
            generation: 0,
        }),
        primed_with: PRIMING_SEGMENTS,
    };
    assert!(
        ensure_counted(&mut running_child, None, &mut forks),
        "the drain refused a decoder it was already holding"
    );
    assert_eq!(forks, 0);
}

#[test]
fn a_stop_requested_after_the_pass_began_forks_no_crop_decoder() {
    let stop = running();
    let window = Some(&stop);
    assert!(
        !stop.load(Ordering::Relaxed),
        "nothing had been requested yet"
    );

    stop.store(true, Ordering::Relaxed);

    let mut slot = LongLived::default();
    let mut forks = 0;
    assert!(
        !ensure_counted(&mut slot, window, &mut forks),
        "a batch ran on a decoder forked after the stop"
    );
    assert_eq!(forks, 0, "a stop mid-pass still forked a crop decoder");
    assert!(slot.decoder.is_none());
}

#[test]
fn priming_decodes_with_the_hot_buffer_lock_already_released() {
    let dir = tempfile::tempdir().unwrap();
    let ctx = test_context("cam", dir.path());
    let buffer = Arc::clone(&ctx.buffer);
    {
        let mut buf = buffer.write_recover();
        for seq in 0..5 {
            buf.push(gop(seq));
        }
    }
    let analyzer = MotionAnalyzer::with_decoder(ctx, FrameDecoder::dead());

    let mut slot = LongLived {
        decoder: Some(CountedChild {
            alive: true,
            generation: 1,
        }),
        primed_with: 0,
    };
    let mut primed: Vec<(Arc<Vec<u8>>, u64)> = Vec::new();
    analyzer.prime_with(&mut slot, 3, |_, data, duration_ns| {
        assert!(
            buffer.try_write().is_ok(),
            "the hot buffer was still locked while a priming segment was decoded"
        );
        primed.push((Arc::clone(data), duration_ns));
    });

    assert_eq!(primed.len(), PRIMING_SEGMENTS as usize);
    assert_eq!(primed[0].1, SEC);
    let resident = buffer.read_recover();
    assert!(
        Arc::ptr_eq(
            &primed[0].0,
            &resident.get_segment_by_sequence(0).expect("seq 0").data
        ),
        "the priming segments were copied out from under the lock"
    );
    drop(resident);

    assert!(slot.primed());
    analyzer.prime_with(&mut slot, 4, |_, _, _| {
        panic!("primed a child that had already been primed")
    });
}

#[test]
fn a_child_primed_on_part_of_the_window_tries_again_on_the_next_run() {
    let dir = tempfile::tempdir().unwrap();
    let ctx = test_context("cam", dir.path());
    let buffer = Arc::clone(&ctx.buffer);
    {
        let mut buf = buffer.write_recover();
        for seq in 0..33 {
            buf.push(gop(seq));
        }
    }
    let first_resident = buffer.read_recover().first_sequence();
    assert!(
        first_resident > 0,
        "nothing was evicted, so nothing is partial"
    );
    let analyzer = MotionAnalyzer::with_decoder(ctx, FrameDecoder::dead());

    let mut slot = LongLived {
        decoder: Some(CountedChild {
            alive: true,
            generation: 1,
        }),
        primed_with: 0,
    };

    let mut fed = 0;
    analyzer.prime_with(&mut slot, PRIMING_SEGMENTS - 1, |_, _, _| fed += 1);
    assert_eq!(fed, 0, "primed with segments the buffer never held");
    assert!(!slot.primed(), "a child that saw nothing was called primed");

    let partial = first_resident + PRIMING_SEGMENTS - 1;
    analyzer.prime_with(&mut slot, partial, |_, _, _| fed += 1);
    assert_eq!(fed, PRIMING_SEGMENTS - 1, "the resident part was not fed");
    assert!(
        !slot.primed(),
        "a child that got part of the window was called primed"
    );

    analyzer.prime_with(&mut slot, partial + 1, |_, _, _| fed += 1);
    assert_eq!(
        fed,
        2 * PRIMING_SEGMENTS - 1,
        "the run with the whole window did not prime"
    );
    assert!(slot.primed());
}

#[test]
fn a_buffer_too_short_for_the_priming_window_still_primes_its_child() {
    let dir = tempfile::tempdir().unwrap();
    let mut ctx = test_context("cam", dir.path());
    ctx.buffer = HotBuffer::new("cam".to_string(), 2);
    let buffer = Arc::clone(&ctx.buffer);
    let analyzer = MotionAnalyzer::with_decoder(ctx, FrameDecoder::dead());

    let mut slot = LongLived {
        decoder: Some(CountedChild {
            alive: true,
            generation: 1,
        }),
        primed_with: 0,
    };

    let mut fed = 0;
    let mut fed_by_run = Vec::new();
    for seq in 0..8 {
        buffer.write_recover().push(gop(seq));
        let before = fed;
        analyzer.prime_with(&mut slot, seq, |_, _, _| fed += 1);
        fed_by_run.push(fed - before);
    }

    assert!(
        slot.primed(),
        "a camera with a buffer this short never got its child past the probe"
    );
    assert_eq!(
        fed, PRIMING_SEGMENTS,
        "the child was fed {fed} segments to spend a {PRIMING_SEGMENTS}-segment probe"
    );
    assert_eq!(
        fed_by_run.iter().sum::<u64>(),
        PRIMING_SEGMENTS,
        "runs kept feeding a child that was already primed: {fed_by_run:?}"
    );
}

#[tokio::test]
async fn the_debug_overlay_frame_is_encoded_only_while_the_view_is_open() {
    let dir = tempfile::tempdir().unwrap();
    let debug_store = DetectionDebugStore::new(&["cam".to_string()]);
    let (detect_tx, queue) = crate::analytics::detect_worker::detect_queue(None);
    let mut ctx = test_context("cam", dir.path());
    ctx.debug_store = Some(debug_store.clone());
    ctx.detect_tx = Some(detect_tx);
    let mut analyzer = MotionAnalyzer::with_decoder(ctx, FrameDecoder::dead());

    let frames = || {
        vec![(
            RgbFrame {
                data: vec![0u8; 4 * 4 * 3],
                width: 4,
                height: 4,
            },
            Some(FULL_FRAME),
        )]
    };
    let segment = |seq| MotionSegment {
        seq,
        data: Arc::new(Vec::new()),
        duration_ns: 1,
    };

    analyzer.process_run_frames(vec![segment(0)], frames());
    let job = queue.recv().await.expect("a crop job per run");
    assert!(
        job.full_frame_jpeg.is_none(),
        "encoded a debug overlay for a view nobody has open"
    );
    assert_eq!(job.crop_jpegs.len(), 1);

    debug_store.list("cam");
    analyzer.process_run_frames(vec![segment(1)], frames());
    let job = queue.recv().await.expect("a crop job per run");
    assert!(
        job.full_frame_jpeg.is_some(),
        "the view is open and its overlay has no frame to draw on"
    );

    debug_store.mark_requested_ago("cam", Duration::from_secs(600));
    analyzer.process_run_frames(vec![segment(2)], frames());
    let job = queue.recv().await.expect("a crop job per run");
    assert!(
        job.full_frame_jpeg.is_none(),
        "kept encoding after the viewer left"
    );
}

#[test]
fn the_analyzer_tick_gives_back_what_an_ended_session_was_holding() {
    let dir = tempfile::tempdir().unwrap();
    let debug_store = DetectionDebugStore::new(&["cam".to_string()]);
    let mut ctx = test_context("cam", dir.path());
    ctx.debug_store = Some(debug_store.clone());
    let mut analyzer = MotionAnalyzer::with_decoder(ctx, FrameDecoder::dead());
    let stopping = AtomicBool::new(true);

    debug_store.list("cam");
    debug_store.insert(
        "cam",
        vec![Arc::new(vec![0xaa])],
        Vec::new(),
        "test-model".to_string(),
        0,
        Some(Arc::new(vec![0xbb])),
        Vec::new(),
        None,
        Vec::new(),
    );
    assert_eq!(debug_store.stored("cam"), 1);

    analyzer.tick(&stopping);
    assert_eq!(
        debug_store.stored("cam"),
        1,
        "took the frames away from a viewer who is still there"
    );

    debug_store.mark_requested_ago("cam", Duration::from_secs(600));
    analyzer.tick(&stopping);
    assert_eq!(
        debug_store.stored("cam"),
        0,
        "the viewer left and the frames stayed resident"
    );
}

#[test]
fn a_pass_that_gives_up_on_its_decoder_still_empties_the_crop_channel() {
    let dir = tempfile::tempdir().unwrap();
    let ctx = test_context("cam", dir.path());
    let mut analyzer = MotionAnalyzer::with_decoder(ctx, FrameDecoder::dead());
    let (crop, frames) = CropDecoder::detached();
    analyzer.crop_decoder = LongLived {
        decoder: Some(crop),
        primed_with: PRIMING_SEGMENTS,
    };
    let fill = |frames: &std::sync::mpsc::SyncSender<Vec<u8>>| {
        for _ in 0..4 {
            frames.try_send(vec![0u8; 8]).expect("the channel has room");
        }
    };

    fill(&frames);
    assert!(
        !analyzer.tick(&AtomicBool::new(true)),
        "the pass was supposed to give up before analyzing anything"
    );
    assert_eq!(
        analyzer.release_idle_crop_frames(),
        0,
        "a pass that gave up at the decoder left the crop decoder's frames pinned"
    );

    fill(&frames);
    assert_eq!(analyzer.release_idle_crop_frames(), 4);
}

#[test]
fn control_plane_runs_when_decoder_is_dead() {
    let dir = tempfile::tempdir().unwrap();
    let ctx = test_context("cam", dir.path());
    let settings = ctx.motion_settings.clone();
    let tuner_store = ctx.tuner_store.clone();
    let mut analyzer = MotionAnalyzer::with_decoder(ctx, FrameDecoder::dead());
    let mut learned = vec![0.0; MASK_CELLS];
    learned[4] = 300.0;
    analyzer
        .tuner
        .load_state(&crate::analytics::tuner::TunerState {
            version: 2,
            learned,
            last_change: vec![None; MASK_CELLS],
        });
    settings
        .update(
            "cam",
            SettingsUpdate {
                tuner_mode: Some(TunerMode::Shadow),
                ..Default::default()
            },
        )
        .unwrap();
    assert!(tuner_store.request_reset("cam"));
    analyzer.decoder_retry = DecoderSpawnRetry::new(RetrySchedule {
        start: Duration::ZERO,
        max: Duration::ZERO,
    });

    assert!(!analyzer
        .tick_with_decoder_spawn(&running(), || { Err(std::io::Error::other("no ffmpeg")) }));

    let snapshot = tuner_store
        .get("cam")
        .expect("control plane did not publish");
    assert_eq!(snapshot.mode, TunerMode::Shadow);
    assert!(snapshot.learned.iter().all(|&value| value == 0.0));
    assert!(!tuner_store.take_reset("cam"), "reset was not consumed");
}

#[test]
fn failed_tuner_save_is_retried() {
    let dir = tempfile::tempdir().unwrap();
    let ctx = test_context("cam", dir.path());
    let tuner_store = ctx.tuner_store.clone();
    let mut analyzer = MotionAnalyzer::with_decoder(ctx, FrameDecoder::dead());
    let blocked_parent = dir.path().join("not-a-directory");
    std::fs::write(&blocked_parent, b"file").unwrap();
    analyzer.tuner_state_path = blocked_parent.join("motion_tuner.json");
    assert!(tuner_store.request_reset("cam"));
    let now = Instant::now();

    analyzer.control_plane(now);
    assert!(analyzer.tuner_dirty, "failed save cleared the dirty bit");
    analyzer.control_plane(now + Duration::from_secs(1));
    assert!(
        analyzer.tuner_dirty,
        "dirty state was retried on the per-tick path"
    );

    std::fs::remove_file(&blocked_parent).unwrap();
    std::fs::create_dir(&blocked_parent).unwrap();
    analyzer.control_plane(now + Duration::from_secs(61));
    assert!(!analyzer.tuner_dirty, "successful retry stayed dirty");
    assert!(
        analyzer.tuner_state_path.exists(),
        "successful retry did not persist state"
    );
}

#[test]
fn frames_arriving_during_the_respawn_backoff_are_released_before_the_pass_returns() {
    let dir = tempfile::tempdir().unwrap();
    let ctx = test_context("cam", dir.path());
    let mut analyzer = MotionAnalyzer::with_decoder(ctx, FrameDecoder::dead());
    let (crop, frames) = CropDecoder::detached();
    analyzer.crop_decoder = LongLived {
        decoder: Some(crop),
        primed_with: PRIMING_SEGMENTS,
    };
    const BACKOFF: Duration = Duration::from_millis(800);
    analyzer.decoder_retry = DecoderSpawnRetry::new(RetrySchedule {
        start: BACKOFF,
        max: BACKOFF,
    });

    let reader = std::thread::spawn(move || {
        std::thread::sleep(POLL_INTERVAL / 2);
        for _ in 0..4 {
            frames.try_send(vec![0u8; 8]).expect("the channel has room");
        }
    });

    let stop = running();
    assert!(
        !analyzer.ensure_decoder_alive_with(&stop, || Err(std::io::Error::other("no ffmpeg"))),
        "a fork that failed must not report a live decoder"
    );
    reader.join().expect("reader thread panicked");

    assert_eq!(
        analyzer.release_idle_crop_frames(),
        0,
        "frames handed over during the backoff stayed pinned until the next pass"
    );
}

fn test_context(camera_id: &str, data_dir: &std::path::Path) -> AnalyzerContext {
    let ids = [camera_id.to_string()];
    AnalyzerContext {
        camera_id: camera_id.to_string(),
        buffer: HotBuffer::new(camera_id.to_string(), 30),
        motion_store: MotionStore::new(&ids),
        detection_store: None,
        debug_store: None,
        detect_tx: None,
        event_registry: None,
        config: AnalyticsConfig::default(),
        motion_settings: MotionSettingsStore::new(
            &ids,
            data_dir,
            DEFAULT_VAR_THRESHOLD,
            DEFAULT_MIN_CONTOUR_AREA,
        ),
        tuner_store: crate::analytics::tuner::TunerStore::with_params(
            &ids,
            crate::analytics::tuner::TunerParams::default(),
        ),
        data_dir: data_dir.to_path_buf(),
        event_tx: None,
        mqtt_tx: None,
        pre_padding_ns: 0,
        post_padding: Duration::from_secs(10),
        max_event_duration: Duration::from_secs(120),
    }
}

#[test]
fn motion_bbox_marks_every_spanned_cell() {
    let mut cells = [false; MASK_CELLS];
    mark_cells(
        &[MotionBox {
            x: 19,
            y: 5,
            width: 2,
            height: 10,
        }],
        ANALYSIS_WIDTH as usize,
        ANALYSIS_HEIGHT as usize,
        &mut cells,
    );

    assert!(cells[0]);
    assert!(cells[1]);
    assert_eq!(cells.iter().filter(|&&marked| marked).count(), 2);
}

#[derive(Clone, Default)]
struct AbandonedCounts(Arc<std::sync::Mutex<Vec<Option<u64>>>>);

impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for AbandonedCounts {
    fn on_event(
        &self,
        event: &tracing::Event<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        struct Field(Option<Option<u64>>);
        impl tracing::field::Visit for Field {
            fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
                if field.name() == "segments_abandoned" {
                    let rendered = format!("{value:?}");
                    self.0 = Some(rendered.parse().ok());
                }
            }
            fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
                if field.name() == "segments_abandoned" {
                    self.0 = Some(Some(value));
                }
            }
        }
        let mut field = Field(None);
        event.record(&mut field);
        if let Some(count) = field.0 {
            self.0.lock().expect("capture poisoned").push(count);
        }
    }
}

fn abandonments(body: impl FnOnce()) -> Vec<Option<u64>> {
    use tracing_subscriber::layer::SubscriberExt;
    let counts = AbandonedCounts::default();
    let subscriber = tracing_subscriber::registry().with(counts.clone());
    tracing::subscriber::with_default(subscriber, body);
    let captured = counts.0.lock().expect("capture poisoned").clone();
    captured
}

const SEC: u64 = 1_000_000_000;

fn gop(index: u64) -> crate::buffer::GopSegment {
    crate::buffer::GopSegment {
        start_pts: index * SEC,
        duration_ns: SEC,
        data: Arc::new(vec![0x47; 188]),
        frame_count: 1,
    }
}

fn analyzer_with_a_dead_decoder(
    dir: &std::path::Path,
    analyzed_through: u64,
) -> (
    MotionAnalyzer,
    Arc<RwLock<HotBuffer>>,
    tokio::sync::mpsc::Receiver<WriterMessage>,
) {
    let (tx, rx) = tokio::sync::mpsc::channel(4);
    let mut ctx = test_context("cam", dir);
    ctx.event_tx = Some(tx);
    let buffer = Arc::clone(&ctx.buffer);
    {
        let mut buf = buffer.write_recover();
        for seq in 0..=analyzed_through {
            buf.push(gop(seq));
        }
    }

    let mut analyzer = MotionAnalyzer::with_decoder(ctx, FrameDecoder::dead());
    let now = Instant::now();
    for seq in 0..=analyzed_through {
        analyzer.run_tracker.observe(seq, true, now);
    }
    analyzer.last_processed = analyzed_through + 1;
    analyzer.observed_sequences = true;
    (analyzer, buffer, rx)
}

fn flushed_event(
    rx: &mut tokio::sync::mpsc::Receiver<WriterMessage>,
) -> crate::buffer::warm::FinishedEvent {
    match rx.try_recv().expect("no event was flushed at all") {
        WriterMessage::Event(event) => event,
        WriterMessage::Upgrade(_) => panic!("the flush sent an upgrade"),
    }
}

#[test]
fn a_dead_decoder_waits_out_the_camera_and_keeps_the_tail_it_cannot_score() {
    let dir = tempfile::tempdir().unwrap();
    let (mut analyzer, buffer, mut rx) = analyzer_with_a_dead_decoder(dir.path(), 1);

    let camera = Arc::clone(&buffer);
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(300));
        camera.write_recover().push(gop(2));
        camera.write_recover().seal();
    });

    let started = Instant::now();
    analyzer.drain_tail(DrainGate::starting_at(started, TAIL_DRAIN_BOUND));
    analyzer.flush_open_run();

    assert!(
        started.elapsed() < TAIL_DRAIN_BOUND,
        "the analyzer sat out its whole bound instead of following the watermark"
    );
    assert_eq!(
        flushed_event(&mut rx).segments.len(),
        3,
        "the event stopped short of the GOP the camera pushed on its way out"
    );
}

#[test]
fn a_dead_decoder_still_stops_at_its_drain_bound_and_says_what_went_unscored() {
    const BOUND: Duration = Duration::from_millis(300);

    let dir = tempfile::tempdir().unwrap();
    let (mut analyzer, buffer, mut rx) = analyzer_with_a_dead_decoder(dir.path(), 1);
    buffer.write_recover().push(gop(2));
    buffer.write_recover().push(gop(3));
    buffer.write_recover().push(gop(4));
    buffer.write_recover().seal_provisionally();
    buffer.write_recover().push(gop(5));

    let started = Instant::now();
    let reported = abandonments(|| {
        analyzer.drain_tail(DrainGate::starting_at(started, BOUND));
    });
    analyzer.flush_open_run();

    assert!(started.elapsed() >= BOUND, "the wait was not the bound's");
    assert!(
        started.elapsed() < BOUND * 10,
        "a camera that never finished held the analyzer past its bound"
    );
    assert_eq!(
        reported,
        vec![Some(3)],
        "the abandonment reported from the position that ended the wait, not from where \
         scoring stopped"
    );
    assert_eq!(
        flushed_event(&mut rx).segments.len(),
        6,
        "the flush left behind footage that was already in the buffer"
    );
}

#[test]
fn a_batch_arriving_inside_the_drain_forks_no_crop_decoder() {
    let dir = tempfile::tempdir().unwrap();
    let (mut analyzer, buffer, _rx) = analyzer_with_a_dead_decoder(dir.path(), 1);

    buffer.write_recover().seal();
    analyzer.drain_tail(DrainGate::starting_at(Instant::now(), TAIL_DRAIN_BOUND));

    analyzer.process_motion_runs(
        vec![MotionSegment {
            seq: 1,
            data: Arc::new(Vec::new()),
            duration_ns: SEC,
        }],
        None,
    );
    assert!(
        analyzer.crop_decoder.decoder.is_none(),
        "the drain forked a crop decoder"
    );
}

fn analyzer_with_a_registry(
    dir: &std::path::Path,
    analyzed_through: u64,
    slots: usize,
) -> (
    MotionAnalyzer,
    EventRegistry,
    Arc<RwLock<HotBuffer>>,
    tokio::sync::mpsc::Sender<WriterMessage>,
    tokio::sync::mpsc::Receiver<WriterMessage>,
) {
    let (tx, rx) = tokio::sync::mpsc::channel(slots);
    let registry = EventRegistry::new(&["cam".to_string()]);
    let mut ctx = test_context("cam", dir);
    ctx.event_tx = Some(tx.clone());
    ctx.event_registry = Some(registry.clone());
    ctx.detection_store = Some(DetectionStore::new(&["cam".to_string()]));
    let buffer = Arc::clone(&ctx.buffer);
    {
        let mut buf = buffer.write_recover();
        for seq in 0..=analyzed_through {
            buf.push(gop(seq));
        }
    }

    let mut analyzer = MotionAnalyzer::with_decoder(ctx, FrameDecoder::dead());
    let now = Instant::now();
    for seq in 0..=analyzed_through {
        analyzer.run_tracker.observe(seq, true, now);
    }
    analyzer.last_processed = analyzed_through + 1;
    analyzer.observed_sequences = true;
    (analyzer, registry, buffer, tx, rx)
}

fn analyzer_with_a_motion_sensor(
    dir: &std::path::Path,
) -> (MotionAnalyzer, tokio::sync::mpsc::Receiver<MqttEvent>) {
    let (tx, rx) = tokio::sync::mpsc::channel(8);
    let mut ctx = test_context("cam", dir);
    ctx.mqtt_tx = Some(tx);
    (MotionAnalyzer::with_decoder(ctx, FrameDecoder::dead()), rx)
}

fn sensor_events(rx: &mut tokio::sync::mpsc::Receiver<MqttEvent>) -> Vec<&'static str> {
    let mut seen = Vec::new();
    while let Ok(event) = rx.try_recv() {
        seen.push(match event {
            MqttEvent::MotionStart { .. } => "start",
            MqttEvent::MotionEnd { .. } => "end",
            _ => "other",
        });
    }
    seen
}

#[test]
fn the_shutdown_flush_clears_the_sensor_for_a_run_that_was_only_pending() {
    let dir = tempfile::tempdir().unwrap();
    let (mut analyzer, mut rx) = analyzer_with_a_motion_sensor(dir.path());
    let t0 = Instant::now();
    let cap = Duration::from_secs(120);
    analyzer.observe_run(0, true, t0);
    analyzer.observe_run(1, true, t0 + cap - Duration::from_secs(2));
    assert!(analyzer.observe_run(2, false, t0 + cap).is_some());
    assert_eq!(
        sensor_events(&mut rx),
        vec!["start"],
        "the cap boundary is event bookkeeping and must not touch the sensor"
    );

    analyzer.flush_open_run();
    assert_eq!(sensor_events(&mut rx), vec!["end"]);
}

#[test]
fn a_run_replaced_after_its_window_elapses_ends_the_sensor_before_starting_it_again() {
    let dir = tempfile::tempdir().unwrap();
    let (mut analyzer, mut rx) = analyzer_with_a_motion_sensor(dir.path());
    let t0 = Instant::now();
    let cap = Duration::from_secs(120);
    let post = Duration::from_secs(10);
    let last_motion = t0 + cap - Duration::from_secs(2);
    analyzer.observe_run(0, true, t0);
    analyzer.observe_run(1, true, last_motion);
    assert!(analyzer.observe_run(2, false, t0 + cap).is_some());
    assert_eq!(sensor_events(&mut rx), vec!["start"]);

    assert_eq!(
        analyzer.observe_run(3, true, last_motion + post + Duration::from_secs(1)),
        None
    );
    assert_eq!(sensor_events(&mut rx), vec!["end", "start"]);
}

#[test]
fn a_cap_boundary_in_the_middle_of_motion_is_silent_on_the_sensor() {
    let dir = tempfile::tempdir().unwrap();
    let (mut analyzer, mut rx) = analyzer_with_a_motion_sensor(dir.path());
    let t0 = Instant::now();
    let cap = Duration::from_secs(120);
    let post = Duration::from_secs(10);
    analyzer.observe_run(0, true, t0);
    assert!(analyzer.observe_run(1, true, t0 + cap).is_some());
    assert_eq!(sensor_events(&mut rx), vec!["start"]);

    analyzer.observe_run(2, false, t0 + cap + post + Duration::from_secs(1));
    assert_eq!(sensor_events(&mut rx), vec!["end"]);
}

fn person() -> crate::storage::Verdict {
    crate::storage::Verdict {
        object_classes: vec!["person".to_string()],
        detections: vec![crate::storage::event_index::DetectionDetail {
            class: "person".to_string(),
            confidence: 0.9,
        }],
        backend: "ollama".to_string(),
        model: "test-model".to_string(),
    }
}

const PATIENCE: Duration = Duration::from_secs(10);

fn wait_for(mut reached: impl FnMut() -> bool, what_went_wrong: &str) {
    let waited = Instant::now();
    while !reached() {
        assert!(waited.elapsed() < PATIENCE, "{what_went_wrong}");
        std::thread::yield_now();
    }
}

async fn next_message(
    rx: &mut tokio::sync::mpsc::Receiver<WriterMessage>,
    expected: &str,
) -> WriterMessage {
    match tokio::time::timeout(PATIENCE, rx.recv()).await {
        Ok(Some(message)) => message,
        Ok(None) => panic!("the writer channel closed before {expected}"),
        Err(_) => panic!("{expected} never arrived"),
    }
}

fn filler() -> WriterMessage {
    WriterMessage::Upgrade(EventUpgrade::for_event(
        UpgradeTarget {
            start_pts_ns: u64::MAX,
            duration_ms: 0,
            continues: false,
        },
        person(),
    ))
}

#[tokio::test]
async fn a_verdict_landing_while_the_analyzer_holds_the_write_still_upgrades_the_event() {
    let dir = tempfile::tempdir().unwrap();
    let (mut analyzer, registry, buffer, spare, mut rx) =
        analyzer_with_a_registry(dir.path(), 2, 1);
    spare.try_send(filler()).expect("the channel starts empty");

    let run = analyzer
        .run_tracker
        .flush(Some(2))
        .expect("the analyzer carries an open run");
    let before_the_store_read = buffer.write_recover();
    let flushing = std::thread::spawn(move || analyzer.emit_event(run, None));

    wait_for(
        || registry.held("cam") > 0,
        "no record was open while the analyzer was still short of the detection store: a \
         verdict landing between the read and the handoff would have nothing to land on",
    );

    let targets = registry.deliver_verdict("cam", &[0, 1, 2], &person());
    assert!(
        targets.is_empty(),
        "an upgrade was sent for an event whose write is not in the channel yet: it would \
         reach the writer first and find no file"
    );
    drop(before_the_store_read);

    assert!(
        !flushing.is_finished(),
        "the analyzer got its write away before the verdict landed; the window this test \
         needs was never open"
    );

    next_message(&mut rx, "the filler message").await;
    let event = match next_message(&mut rx, "the event").await {
        WriterMessage::Event(event) => event,
        WriterMessage::Upgrade(_) => panic!("the upgrade overtook the write it belongs to"),
    };
    assert!(
        !event.has_objects,
        "the assembly saw detections the test never stored; the upgrade below would prove \
         nothing"
    );
    match next_message(&mut rx, "the upgrade the verdict earned").await {
        WriterMessage::Upgrade(upgrade) => {
            assert_eq!(upgrade.start_pts_ns, event.first_pts);
            assert_eq!(upgrade.duration_ms, event.duration_ms() as u32);
            assert_eq!(upgrade.object_classes, vec!["person".to_string()]);
        }
        WriterMessage::Event(_) => panic!("a second event was written"),
    }
    flushing.join().expect("the analyzer panicked");
}

#[test]
fn a_verdict_landing_after_the_write_is_queued_upgrades_the_event_it_names() {
    let dir = tempfile::tempdir().unwrap();
    let (mut analyzer, registry, _buffer, _spare, mut rx) =
        analyzer_with_a_registry(dir.path(), 2, 4);

    analyzer.flush_open_run();
    let event = flushed_event(&mut rx);

    assert_eq!(
        registry.deliver_verdict("cam", &[0, 1, 2], &person()),
        [UpgradeTarget {
            start_pts_ns: event.first_pts,
            duration_ms: event.duration_ms() as u32,
            continues: false,
        }]
    );
}

#[test]
fn an_event_whose_write_never_left_leaves_no_record_behind() {
    let dir = tempfile::tempdir().unwrap();
    let (mut analyzer, registry, _buffer, spare, rx) = analyzer_with_a_registry(dir.path(), 2, 4);
    drop(rx);
    drop(spare);

    analyzer.flush_open_run();

    assert_eq!(
        registry.held("cam"),
        0,
        "a record outlived the event whose write was lost"
    );
    assert!(registry
        .deliver_verdict("cam", &[0, 1, 2], &person())
        .is_empty());
}

#[test]
fn a_flush_at_shutdown_neither_waits_for_a_verdict_nor_drops_the_write() {
    let dir = tempfile::tempdir().unwrap();
    let (mut analyzer, registry, _buffer, _spare, mut rx) =
        analyzer_with_a_registry(dir.path(), 2, 4);
    let _never_answered = registry.expect_verdict("cam", &[0, 1, 2]);

    let started = Instant::now();
    analyzer.flush_open_run();

    assert!(
        started.elapsed() < TAIL_DRAIN_BOUND,
        "the flush waited on a verdict instead of getting the recording away"
    );
    assert_eq!(
        flushed_event(&mut rx).segments.len(),
        3,
        "the event the drain exists to save never reached the writer"
    );
    assert_eq!(
        registry.held("cam"),
        1,
        "the record was dropped while its verdict was still outstanding"
    );
}

#[tokio::test]
async fn an_upgrade_with_no_writer_left_to_take_it_is_dropped_not_waited_on() {
    let dir = tempfile::tempdir().unwrap();
    let (mut analyzer, registry, _buffer, spare, mut rx) =
        analyzer_with_a_registry(dir.path(), 2, 1);
    spare.try_send(filler()).expect("the channel starts empty");

    let flushing = std::thread::spawn(move || analyzer.flush_open_run());
    wait_for(
        || registry.held("cam") > 0,
        "no record was opened for the event",
    );
    assert!(
        registry
            .deliver_verdict("cam", &[0, 1, 2], &person())
            .is_empty(),
        "an upgrade was sent for an event whose write is not in the channel yet"
    );

    next_message(&mut rx, "the filler message").await;
    wait_for(|| rx.len() == 1, "the write never reached the writer");

    rx.close();
    wait_for(
        || flushing.is_finished(),
        "the analyzer wedged on an upgrade no writer was left to take",
    );
    flushing.join().expect("the analyzer panicked");

    match rx
        .recv()
        .await
        .expect("the write was lost with the upgrade")
    {
        WriterMessage::Event(event) => assert_eq!(event.segments.len(), 3),
        WriterMessage::Upgrade(_) => panic!("the upgrade was queued after all"),
    }
    assert!(rx.recv().await.is_none());
}

#[tokio::test]
async fn spawn_analyzer_stops_at_once_when_shutdown_is_already_requested() {
    let dir = tempfile::tempdir().unwrap();
    let started = Instant::now();
    let handle = spawn_analyzer(
        test_context("cam", dir.path()),
        Arc::new(AtomicBool::new(true)),
    );
    handle.await.expect("analyzer task panicked");
    assert!(started.elapsed() < DECODER_SPAWN_SCHEDULE.start);
}

#[tokio::test]
#[ignore]
async fn spawn_analyzer_builds_a_real_analyzer_and_stops_on_shutdown() {
    let dir = tempfile::tempdir().unwrap();
    let shutdown = Arc::new(AtomicBool::new(false));
    let ctx = test_context("cam", dir.path());
    let buffer = Arc::clone(&ctx.buffer);
    let handle = spawn_analyzer(ctx, Arc::clone(&shutdown));

    tokio::time::sleep(Duration::from_millis(500)).await;
    assert!(!handle.is_finished(), "analyzer stopped on its own");

    shutdown.store(true, Ordering::Relaxed);
    buffer.write_recover().seal();
    tokio::time::timeout(Duration::from_secs(5), handle)
        .await
        .expect("analyzer did not stop")
        .expect("analyzer task panicked");
}

#[tokio::test]
#[ignore]
async fn an_analyzer_keeps_going_until_the_camera_publishes_its_watermark() {
    let dir = tempfile::tempdir().unwrap();
    let shutdown = Arc::new(AtomicBool::new(false));
    let ctx = test_context("cam", dir.path());
    let buffer = Arc::clone(&ctx.buffer);
    let handle = spawn_analyzer(ctx, Arc::clone(&shutdown));
    tokio::time::sleep(Duration::from_millis(500)).await;

    shutdown.store(true, Ordering::Relaxed);
    tokio::time::sleep(Duration::from_secs(2)).await;
    assert!(
        !handle.is_finished(),
        "the analyzer exited on the flag alone, before the camera had finished"
    );

    buffer.write_recover().seal();
    tokio::time::timeout(Duration::from_secs(5), handle)
        .await
        .expect("the analyzer ignored the watermark and sat out its whole drain bound")
        .expect("analyzer task panicked");
}

#[test]
#[ignore]
fn two_motion_batches_are_extracted_through_one_crop_decoder() {
    const SEGMENTS: usize = 8;
    let dir = tempfile::tempdir().unwrap();
    let recorded = crate::analytics::decoder::tests::recorded_segments(SEGMENTS);
    let ctx = test_context("cam", dir.path());
    let buffer = Arc::clone(&ctx.buffer);
    {
        let mut buf = buffer.write_recover();
        for (i, data) in recorded.iter().enumerate() {
            buf.push(crate::buffer::GopSegment {
                start_pts: i as u64 * SEC,
                duration_ns: SEC,
                data: Arc::clone(data),
                frame_count: 25,
            });
        }
    }
    let mut analyzer = MotionAnalyzer::with_decoder(ctx, FrameDecoder::dead());
    analyzer.frame_use = FrameUse::Thumbnails;

    let batch = |seq: u64| {
        vec![MotionSegment {
            seq,
            data: Arc::clone(&recorded[seq as usize]),
            duration_ns: SEC,
        }]
    };

    let stop = running();
    analyzer.process_motion_runs(batch(3), Some(&stop));
    let first = analyzer
        .crop_decoder
        .decoder
        .as_ref()
        .expect("the first batch forked no crop decoder")
        .child_id();
    assert!(first.is_some(), "the crop decoder has no child");
    assert!(
        analyzer.crop_decoder.primed(),
        "the first batch never primed the child it forked"
    );

    analyzer.process_motion_runs(batch(7), Some(&stop));
    let second = analyzer
        .crop_decoder
        .decoder
        .as_ref()
        .expect("the second batch lost the crop decoder")
        .child_id();
    assert_eq!(
        first, second,
        "the second batch forked a crop decoder of its own"
    );
}

#[test]
#[ignore]
fn a_batch_leaves_frames_behind_and_the_next_pass_takes_them() {
    const SEGMENTS: usize = 16;
    let dir = tempfile::tempdir().unwrap();
    let recorded = crate::analytics::decoder::tests::recorded_segments(SEGMENTS);
    let ctx = test_context("cam", dir.path());
    let buffer = Arc::clone(&ctx.buffer);
    {
        let mut buf = buffer.write_recover();
        for (i, data) in recorded.iter().enumerate() {
            buf.push(crate::buffer::GopSegment {
                start_pts: i as u64 * SEC,
                duration_ns: SEC,
                data: Arc::clone(data),
                frame_count: 25,
            });
        }
    }
    let mut analyzer = MotionAnalyzer::with_decoder(ctx, FrameDecoder::dead());
    analyzer.frame_use = FrameUse::Thumbnails;
    let stop = running();
    let batch = |from: u64| {
        (from..from + 4)
            .map(|seq| MotionSegment {
                seq,
                data: Arc::clone(&recorded[seq as usize]),
                duration_ns: SEC,
            })
            .collect::<Vec<_>>()
    };
    let settle = || std::thread::sleep(Duration::from_millis(500));

    analyzer.process_motion_runs(batch(3), Some(&stop));
    analyzer.process_motion_runs(batch(7), Some(&stop));
    settle();
    assert!(
        analyzer.release_idle_crop_frames() > 0,
        "the batches left nothing behind, so there is nothing here to release"
    );

    analyzer.process_motion_runs(batch(11), Some(&stop));
    settle();
    assert!(
        !analyzer.tick(&AtomicBool::new(true)),
        "the pass was supposed to give up before analyzing anything"
    );
    assert_eq!(
        analyzer.release_idle_crop_frames(),
        0,
        "the pass left the last batch's frames resident"
    );
}

#[test]
fn normalize_rect_maps_to_unit_coords() {
    let r = MotionBox {
        x: 80,
        y: 60,
        width: 160,
        height: 120,
    };
    let n = normalize_rect(r, 320, 240);
    assert!((n.x - 0.25).abs() < 0.01);
    assert!((n.y - 0.25).abs() < 0.01);
    assert!((n.w - 0.50).abs() < 0.01);
    assert!((n.h - 0.50).abs() < 0.01);
}

#[test]
fn union_rects_empty_returns_none() {
    assert!(union_rects_padded(&[], 0.2).is_none());
}

#[test]
fn union_rects_single_rect_with_padding() {
    let r = NormalizedRect {
        x: 0.4,
        y: 0.4,
        w: 0.2,
        h: 0.2,
    };
    let u = union_rects_padded(&[r], 0.2).unwrap();
    assert!((u.x - 0.36).abs() < 0.01);
    assert!((u.w - 0.28).abs() < 0.01);
}

#[test]
fn union_rects_clamps_to_bounds() {
    let r = NormalizedRect {
        x: 0.0,
        y: 0.0,
        w: 0.1,
        h: 0.1,
    };
    let u = union_rects_padded(&[r], 0.5).unwrap();
    assert!(u.x >= 0.0);
    assert!(u.y >= 0.0);
    assert!(u.x + u.w <= 1.0);
    assert!(u.y + u.h <= 1.0);
}

#[test]
fn union_rects_merges_two_rects() {
    let rects = vec![
        NormalizedRect {
            x: 0.1,
            y: 0.1,
            w: 0.2,
            h: 0.2,
        },
        NormalizedRect {
            x: 0.6,
            y: 0.6,
            w: 0.2,
            h: 0.2,
        },
    ];
    let u = union_rects_padded(&rects, 0.0).unwrap();
    assert!((u.x - 0.1).abs() < 0.01);
    assert!((u.y - 0.1).abs() < 0.01);
    assert!((u.w - 0.7).abs() < 0.01);
    assert!((u.h - 0.7).abs() < 0.01);
}

#[test]
fn union_rects_enforces_minimum_size() {
    let r = NormalizedRect {
        x: 0.5,
        y: 0.5,
        w: 0.01,
        h: 0.01,
    };
    let u = union_rects_padded(&[r], 0.0).unwrap();
    assert!(u.w >= MIN_CROP_FRACTION);
    assert!(u.h >= MIN_CROP_FRACTION);
}

fn coordinate_frame() -> RgbFrame {
    let (width, height) = (200usize, 100usize);
    let mut data = Vec::with_capacity(width * height * 3);
    for row in 0..height {
        for col in 0..width {
            data.extend_from_slice(&[(col % 256) as u8, row as u8, 0]);
        }
    }
    RgbFrame {
        data,
        width,
        height,
    }
}

#[test]
fn crop_frame_extracts_region() {
    let frame = coordinate_frame();
    let region = NormalizedRect {
        x: 0.25,
        y: 0.25,
        w: 0.5,
        h: 0.5,
    };
    let cropped = crop_frame(&frame, &region).unwrap();
    assert_eq!(cropped.width, 100);
    assert_eq!(cropped.height, 50);
    assert_eq!(cropped.data.len(), 100 * 50 * 3);
    assert_eq!(&cropped.data[0..3], &[50, 25, 0]);
    let last = cropped.data.len() - 3;
    assert_eq!(&cropped.data[last..], &[149, 74, 0]);
}

#[test]
fn crop_frame_clamps_at_edge() {
    let frame = coordinate_frame();
    let region = NormalizedRect {
        x: 0.8,
        y: 0.8,
        w: 0.5,
        h: 0.5,
    };
    let cropped = crop_frame(&frame, &region).unwrap();
    assert_eq!(cropped.width, 40);
    assert_eq!(cropped.height, 20);
    assert_eq!(&cropped.data[0..3], &[160, 80, 0]);
}

#[test]
fn crop_frame_fully_outside_is_none() {
    let frame = coordinate_frame();
    let region = NormalizedRect {
        x: 1.0,
        y: 0.0,
        w: 0.5,
        h: 0.5,
    };
    assert!(crop_frame(&frame, &region).is_none());
}

#[test]
fn crop_frame_empty_frame_is_none() {
    let frame = RgbFrame {
        data: Vec::new(),
        width: 0,
        height: 0,
    };
    let region = NormalizedRect {
        x: 0.0,
        y: 0.0,
        w: 1.0,
        h: 1.0,
    };
    assert!(crop_frame(&frame, &region).is_none());
}

#[test]
fn rgb_jpeg_round_trips_through_image_crate() {
    let frame = coordinate_frame();
    let jpeg = rgb_jpeg(&frame).unwrap();
    assert_eq!(&jpeg[0..2], &[0xff, 0xd8]);
    let decoded = image::load_from_memory(&jpeg).unwrap();
    assert_eq!(decoded.width(), 200);
    assert_eq!(decoded.height(), 100);
}

#[test]
fn rgb_jpeg_rejects_mismatched_buffer() {
    let frame = RgbFrame {
        data: vec![0; 10],
        width: 200,
        height: 100,
    };
    assert!(rgb_jpeg(&frame).is_none());
}

#[test]
fn gray_jpeg_round_trips_through_image_crate() {
    let (w, h) = (32usize, 16usize);
    let data: Vec<u8> = (0..w * h).map(|i| (i % 256) as u8).collect();
    let jpeg = gray_jpeg((&data, w, h)).unwrap();
    assert_eq!(&jpeg[0..2], &[0xff, 0xd8]);
    let decoded = image::load_from_memory(&jpeg).unwrap();
    assert_eq!(decoded.width(), 32);
    assert_eq!(decoded.height(), 16);
}

#[test]
fn gray_jpeg_rejects_bad_dimensions() {
    assert!(gray_jpeg((&[0u8; 10], 3, 3)).is_none());
    assert!(gray_jpeg((&[], 0, 0)).is_none());
}

fn white_frame(width: usize, height: usize) -> RgbFrame {
    RgbFrame {
        data: vec![255u8; width * height * 3],
        width,
        height,
    }
}

fn empty_mask() -> Vec<bool> {
    vec![false; MASK_CELLS]
}

fn is_black(frame: &RgbFrame, col: usize, row: usize) -> bool {
    let i = (row * frame.width + col) * 3;
    frame.data[i] == 0 && frame.data[i + 1] == 0 && frame.data[i + 2] == 0
}

#[test]
fn detection_mask_noop_when_empty() {
    let mut frame = white_frame(160, 120);
    apply_detection_mask(&mut frame, FULL_FRAME, &empty_mask());
    assert!(frame.data.iter().all(|&b| b == 255), "frame untouched");
}

#[test]
fn detection_mask_full_frame_blacks_exact_cell() {
    let mut frame = white_frame(160, 120);
    let mut mask = empty_mask();
    mask[0] = true;
    mask[2 * MASK_COLS + 3] = true;
    apply_detection_mask(&mut frame, FULL_FRAME, &mask);

    assert!(is_black(&frame, 0, 0));
    assert!(is_black(&frame, 9, 9));
    assert!(!is_black(&frame, 10, 0));
    assert!(!is_black(&frame, 0, 10));

    assert!(is_black(&frame, 30, 20));
    assert!(is_black(&frame, 39, 29));
    assert!(!is_black(&frame, 29, 20));
    assert!(!is_black(&frame, 40, 20));
    assert!(!is_black(&frame, 30, 19));
}

#[test]
fn detection_mask_intersects_partial_crop() {
    let crop = NormalizedRect {
        x: 0.5,
        y: 0.0,
        w: 0.5,
        h: 1.0,
    };
    let mut frame = white_frame(80, 120);
    let mut mask = empty_mask();
    mask[0] = true;
    mask[8] = true;
    apply_detection_mask(&mut frame, crop, &mask);

    assert!(is_black(&frame, 0, 0));
    assert!(is_black(&frame, 9, 0));
    assert!(!is_black(&frame, 10, 0));
    assert!(!is_black(&frame, 79, 0));
}

#[test]
fn detection_mask_full_frame_crop_matches_uncropped() {
    let mut frame = white_frame(160, 120);
    let mut mask = empty_mask();
    mask[MASK_COLS + 1] = true; // col 1, row 1 => x=10..20, y=10..20
    let spanning = NormalizedRect {
        x: 0.0,
        y: 0.0,
        w: 1.0,
        h: 1.0,
    };
    apply_detection_mask(&mut frame, spanning, &mask);
    assert!(is_black(&frame, 10, 10));
    assert!(is_black(&frame, 19, 19));
    assert!(!is_black(&frame, 9, 10));
    assert!(!is_black(&frame, 20, 20));
}

fn frames(tags: &[u8]) -> Vec<Vec<u8>> {
    tags.iter().map(|&t| vec![t]).collect()
}

fn tags(frames: &[Vec<u8>]) -> Vec<u8> {
    frames.iter().map(|f| f[0]).collect()
}

#[test]
fn frames_are_extracted_for_events_without_object_detection() {
    assert_eq!(FrameUse::of(true, false), FrameUse::Thumbnails);
    assert_eq!(FrameUse::of(true, true), FrameUse::Detection);
    assert_eq!(FrameUse::of(false, true), FrameUse::Detection);
    assert_eq!(FrameUse::of(false, false), FrameUse::None);
}

#[test]
fn thumbnails_only_decode_smaller_frames() {
    assert_eq!(FrameUse::Detection.crop_size(), DETECTION_CROP_SIZE);
    assert_eq!(FrameUse::Thumbnails.crop_size(), THUMBNAIL_CROP_SIZE);
    assert!(THUMBNAIL_CROP_SIZE.0 < DETECTION_CROP_SIZE.0);
}

#[test]
fn run_filmstrip_accumulates_across_batches() {
    let mut strip = RunFilmstrip::default();
    strip.push(frames(&[1, 2]));
    strip.push(frames(&[3]));
    let taken = strip.take().unwrap();
    assert_eq!(tags(&taken), vec![1, 2, 3]);
}

#[test]
fn run_filmstrip_take_resets_and_is_none_when_empty() {
    let mut strip = RunFilmstrip::default();
    assert!(strip.take().is_none());
    strip.push(frames(&[1]));
    assert!(strip.take().is_some());
    assert!(strip.take().is_none());
}

#[test]
fn run_filmstrip_halves_past_the_cap() {
    let mut strip = RunFilmstrip::default();
    for batch in 0..6u8 {
        strip.push(frames(&[batch * 2, batch * 2 + 1]));
    }
    assert!(strip.frames.len() <= FILMSTRIP_ACCUMULATOR_CAP);
    let taken = strip.take().unwrap();
    assert_eq!(taken.len(), FILMSTRIP_FRAMES);
    assert_eq!(taken[0], vec![0]);
    assert!(taken[FILMSTRIP_FRAMES - 1][0] >= 8);
}

#[test]
fn run_filmstrip_subsamples_spread_over_the_run() {
    let mut strip = RunFilmstrip::default();
    strip.push(frames(&[0, 1, 2, 3, 4, 5]));
    let taken = strip.take().unwrap();
    assert_eq!(tags(&taken), vec![0, 2, 4, 5]);
}

#[test]
fn run_filmstrip_close_does_not_steal_the_next_runs_frames() {
    let mut strip = RunFilmstrip::default();
    strip.push(frames(&[1, 2]));
    let closed = strip.take().unwrap();
    strip.push(frames(&[3]));
    assert_eq!(tags(&closed), vec![1, 2]);
    assert_eq!(tags(&strip.take().unwrap()), vec![3]);
}

#[test]
fn pick_four_keeps_four_frames_spread_over_the_run() {
    fn tagged(n: usize) -> Vec<(RgbFrame, Option<NormalizedRect>)> {
        (0..n)
            .map(|i| {
                let frame = RgbFrame {
                    data: vec![i as u8; 3],
                    width: 1,
                    height: 1,
                };
                let crop = NormalizedRect {
                    x: i as f32,
                    y: 0.0,
                    w: 1.0,
                    h: 1.0,
                };
                (frame, Some(crop))
            })
            .collect()
    }
    fn picked(frames: Vec<(RgbFrame, Option<NormalizedRect>)>) -> Vec<u8> {
        frames
            .iter()
            .map(|(frame, crop)| {
                assert_eq!(crop.unwrap().x, frame.data[0] as f32, "tag follows frame");
                frame.data[0]
            })
            .collect()
    }

    assert_eq!(picked(pick_four(tagged(4))), vec![0, 1, 2, 3]);
    assert_eq!(picked(pick_four(tagged(5))), vec![0, 1, 3, 4]);
    assert_eq!(picked(pick_four(tagged(6))), vec![0, 2, 4, 5]);
    assert_eq!(picked(pick_four(tagged(12))), vec![0, 4, 8, 11]);
}

#[test]
fn frames_per_segment_covers_the_final_pick_and_one_spare() {
    assert_eq!(frames_per_segment(1), FILMSTRIP_FRAMES + 1);
    assert_eq!(frames_per_segment(2), 3);
    assert_eq!(frames_per_segment(3), 3);
    assert_eq!(frames_per_segment(4), 2);
    assert_eq!(frames_per_segment(0), FILMSTRIP_FRAMES + 1);
}

#[test]
fn halve_past_never_settles_above_its_cap() {
    for cap in 1..8usize {
        let mut acc: Vec<usize> = Vec::new();
        for i in 0..1000usize {
            acc.push(i);
            assert!(acc.len() <= cap + 1, "cap {cap} peaked at {}", acc.len());
            halve_past(&mut acc, cap);
            assert!(acc.len() <= cap, "cap {cap} settled at {}", acc.len());
        }
        assert_eq!(acc[0], 0, "the first entry always survives");
    }
}

#[test]
fn thin_evenly_keeps_both_ends_and_spreads_the_rest() {
    fn picked(n: usize, keep: usize) -> Vec<usize> {
        thin_evenly((0..n).collect(), keep)
    }
    assert_eq!(picked(5, 1), vec![0]);
    assert_eq!(picked(5, 2), vec![0, 4]);
    assert_eq!(picked(5, 4), vec![0, 1, 2, 4]);
    assert_eq!(picked(10, 2), vec![0, 9]);
    assert_eq!(picked(10, 4), vec![0, 3, 6, 9]);
    assert_eq!(picked(3, 4), vec![0, 1, 2]);
    assert!(picked(0, 4).is_empty());
    assert!(picked(5, 0).is_empty());
}

#[test]
fn the_run_accumulator_cannot_outgrow_its_cap() {
    for run_len in 1..64usize {
        let sampled = sample_indices(run_len).len();
        let held = sampled * frames_per_segment(sampled);
        assert!(
            held <= RUN_FRAME_ACCUMULATOR_CAP,
            "a run of {run_len} segments holds {held} frames"
        );
    }
}

fn run_selection(run_len: usize, frames: impl Fn(u64) -> usize) -> Vec<(u8, u8)> {
    let run: Vec<MotionSegment> = (0..run_len as u64)
        .map(|seq| MotionSegment {
            seq,
            data: Arc::new(vec![seq as u8]),
            duration_ns: 1_000_000_000,
        })
        .collect();
    let crops: HashMap<u64, NormalizedRect> = (0..run_len as u64)
        .map(|seq| {
            let crop = NormalizedRect {
                x: seq as f32,
                y: 0.0,
                w: 1.0,
                h: 1.0,
            };
            (seq, crop)
        })
        .collect();

    sample_run_frames(&run, &crops, 1, 1, |data, _duration_ns, sink| {
        let seq = data[0];
        for i in 0..frames(seq as u64) {
            sink(vec![seq, i as u8, 0]);
        }
    })
    .iter()
    .map(|(frame, crop)| {
        assert_eq!(
            crop.expect("a sampled segment always carries its crop").x,
            frame.data[0] as f32,
            "crop tag followed the wrong segment"
        );
        (frame.data[0], frame.data[1])
    })
    .collect()
}

#[test]
fn a_run_is_reduced_to_four_frames_of_its_own_motion() {
    assert_eq!(
        run_selection(9, |_| 5),
        vec![(0, 0), (3, 0), (6, 4), (8, 4)]
    );
    assert_eq!(
        run_selection(4, |_| 5),
        vec![(0, 0), (1, 0), (2, 4), (3, 4)]
    );
    assert_eq!(
        run_selection(3, |_| 5),
        vec![(0, 0), (1, 0), (2, 0), (2, 4)]
    );
    assert_eq!(
        run_selection(2, |_| 5),
        vec![(0, 0), (0, 4), (1, 2), (1, 4)]
    );
    assert_eq!(
        run_selection(1, |_| 5),
        vec![(0, 0), (0, 1), (0, 3), (0, 4)]
    );
    assert_eq!(run_selection(1, |_| 2), vec![(0, 0), (0, 1)]);
}

#[test]
fn a_segment_that_decoded_short_costs_a_frame_not_its_place_in_the_strip() {
    let picked = run_selection(9, |seq| if seq == 3 { 1 } else { 5 });
    assert_eq!(picked, vec![(0, 0), (3, 0), (6, 4), (8, 4)]);
}

#[test]
fn a_high_sample_fps_moves_the_frames_picked_but_not_how_many() {
    assert_eq!(
        run_selection(9, |_| 60),
        vec![(0, 0), (3, 0), (6, 58), (8, 58)]
    );
}

#[test]
fn zero_frame_tripwire_resets_on_any_decoded_frame() {
    let mut tripwire = ZeroFrameTripwire::default();
    for _ in 0..BLIND_DECODER_STREAK - 1 {
        assert!(!tripwire.observe(0));
    }
    assert!(!tripwire.observe(1), "a decoded frame clears the streak");
    for _ in 0..BLIND_DECODER_STREAK - 1 {
        assert!(!tripwire.observe(0), "streak restarts from zero");
    }
}

#[test]
fn zero_frame_tripwire_trips_at_the_threshold_and_rearms() {
    let mut tripwire = ZeroFrameTripwire::default();
    for _ in 0..BLIND_DECODER_STREAK - 1 {
        assert!(!tripwire.observe(0));
    }
    assert!(
        tripwire.observe(0),
        "trips on the threshold-th empty decode"
    );
    for _ in 0..BLIND_DECODER_STREAK - 1 {
        assert!(!tripwire.observe(0));
    }
    assert!(tripwire.observe(0));
}

#[test]
fn zero_frame_tripwire_reset_clears_a_partial_streak() {
    let mut tripwire = ZeroFrameTripwire::default();
    for _ in 0..BLIND_DECODER_STREAK - 1 {
        assert!(!tripwire.observe(0));
    }
    tripwire.reset();
    assert!(
        !tripwire.observe(0),
        "reset streak needs the full run again"
    );
}

#[test]
fn skipped_segments_reports_the_dropped_range() {
    let skipped = SkippedSegments::between(10, 14).unwrap();
    assert_eq!(
        skipped,
        SkippedSegments {
            count: 4,
            from_seq: 10,
            to_seq: 13,
        }
    );
    let one = SkippedSegments::between(10, 11).unwrap();
    assert_eq!(one.count, 1);
    assert_eq!((one.from_seq, one.to_seq), (10, 10));
}

#[test]
fn skipped_segments_is_none_when_the_analyzer_kept_up() {
    assert_eq!(SkippedSegments::between(10, 10), None);
    assert_eq!(SkippedSegments::between(12, 10), None);
    assert_eq!(SkippedSegments::between(0, 0), None);
}

#[test]
fn skipped_segments_reports_what_vanished_mid_collection() {
    assert_eq!(
        SkippedSegments::of(&[10, 11, 12, 13, 14]),
        Some(SkippedSegments {
            count: 5,
            from_seq: 10,
            to_seq: 14,
        })
    );
    let scattered = SkippedSegments::of(&[10, 14]).unwrap();
    assert_eq!(scattered.count, 2);
    assert_eq!((scattered.from_seq, scattered.to_seq), (10, 14));
    assert_eq!(SkippedSegments::of(&[]), None);
}

#[test]
fn skip_reporter_coalesces_a_chronically_behind_analyzer() {
    let mut reporter = SkipReporter::default();
    let t0 = Instant::now();
    let first = reporter
        .record(SkippedSegments::between(0, 3).unwrap(), t0)
        .unwrap();
    assert_eq!(first.count, 3);

    let mut polls = 0;
    for tick in 1..200u64 {
        let at = t0 + Duration::from_millis(200 * tick);
        let seq = 3 + tick;
        if reporter
            .record(SkippedSegments::between(seq, seq + 1).unwrap(), at)
            .is_some()
        {
            polls += 1;
        }
    }
    assert_eq!(polls, 1);
}

#[test]
fn skip_reporter_totals_everything_it_held_back() {
    let mut reporter = SkipReporter::default();
    let t0 = Instant::now();
    reporter.record(SkippedSegments::between(0, 1).unwrap(), t0);
    assert!(reporter
        .record(SkippedSegments::between(1, 3).unwrap(), t0)
        .is_none());
    assert!(reporter
        .record(SkippedSegments::between(3, 6).unwrap(), t0)
        .is_none());
    let total = reporter
        .record(
            SkippedSegments::between(6, 7).unwrap(),
            t0 + SKIP_REPORT_INTERVAL,
        )
        .unwrap();
    assert_eq!(
        total,
        SkippedSegments {
            count: 6,
            from_seq: 1,
            to_seq: 6,
        }
    );
}

#[test]
fn motion_verdict_needs_the_threshold() {
    let scored = |score| SegmentAnalysis {
        score,
        crop: None,
        motion_rects: Vec::new(),
        motion_cells: [false; MASK_CELLS],
    };
    assert!(!scored(0.0).has_motion());
    assert!(!scored(MOTION_THRESHOLD - 0.001).has_motion());
    assert!(scored(MOTION_THRESHOLD).has_motion());
}

#[test]
fn unanalyzed_segments_do_not_end_an_event() {
    const POST: Duration = Duration::from_secs(10);
    const CAP: Duration = Duration::from_secs(300);
    let segments: Vec<_> = (0..20).map(|seq| pending(seq, SECOND_NS)).collect();
    let times = batch_instants(&segments, Instant::now());
    let motion = |seq: u64| seq == 0 || seq == 16;

    let mut tracker = RunTracker::new(POST, CAP);
    let closed: Vec<_> = segments
        .iter()
        .zip(&times)
        .filter(|(seg, _)| motion(seg.seq))
        .filter_map(|(seg, &at)| tracker.observe(seg.seq, true, at))
        .collect();
    assert!(closed.is_empty(), "unanalyzed footage ended the event");
    let run = tracker.flush(None).unwrap();
    assert_eq!(run.first_motion_seq, 0);
    assert_eq!(run.last_seq, 16);

    let mut as_quiet = RunTracker::new(POST, CAP);
    let closed: Vec<_> = segments
        .iter()
        .zip(&times)
        .filter_map(|(seg, &at)| as_quiet.observe(seg.seq, motion(seg.seq), at))
        .collect();
    assert_eq!(closed.len(), 1);
    assert_eq!(closed[0].last_seq, 10);
    assert_eq!(as_quiet.flush(None).unwrap().first_motion_seq, 16);
}

fn pending(seq: u64, duration_ns: u64) -> PendingSegment {
    PendingSegment {
        seq,
        data: Arc::new(Vec::new()),
        start_pts: seq * duration_ns,
        duration_ns,
    }
}

const SECOND_NS: u64 = 1_000_000_000;

#[test]
fn batch_instants_space_segments_by_their_media_duration() {
    let now = Instant::now();
    let segments = vec![
        pending(0, SECOND_NS),
        pending(1, 2 * SECOND_NS),
        pending(2, 0),
    ];
    let times = batch_instants(&segments, now);
    assert_eq!(times[0], now - Duration::from_secs(2));
    assert_eq!(times[1], now);
    assert_eq!(times[2], now);
}

#[test]
fn batch_instants_of_a_single_segment_is_now() {
    let now = Instant::now();
    assert_eq!(batch_instants(&[pending(7, SECOND_NS)], now), vec![now]);
    assert!(batch_instants(&[], now).is_empty());
}

#[test]
fn batch_instants_never_run_past_now() {
    let now = Instant::now();
    let segments = vec![
        pending(0, u64::MAX),
        pending(1, u64::MAX),
        pending(2, SECOND_NS),
        pending(3, SECOND_NS),
    ];
    let times = batch_instants(&segments, now);
    assert!(times.iter().all(|&at| at <= now), "instant past now");
    assert!(
        times.windows(2).all(|w| w[0] <= w[1]),
        "capture order reversed"
    );
    assert_eq!(*times.last().unwrap(), now);
    assert_eq!(times[2], now - Duration::from_secs(1));
}

#[test]
fn duration_cap_fires_inside_a_backlogged_batch() {
    const POST: Duration = Duration::from_secs(10);
    const CAP: Duration = Duration::from_secs(30);
    let segments: Vec<_> = (0..90).map(|seq| pending(seq, SECOND_NS)).collect();

    let mut tracker = RunTracker::new(POST, CAP);
    let closed: Vec<_> = segments
        .iter()
        .zip(batch_instants(&segments, Instant::now()))
        .filter_map(|(seg, at)| tracker.observe(seg.seq, true, at))
        .collect();

    assert_eq!(closed.len(), 2, "two chunks close, the third stays open");
    assert_eq!(closed[0].first_motion_seq, 0);
    assert_eq!(closed[0].last_seq, 29);
    assert!(!closed[0].continues);
    assert_eq!(closed[1].first_motion_seq, 30);
    assert_eq!(closed[1].last_seq, 59);
    assert!(closed[1].continues);
    assert!(tracker.is_open());

    let mut shared = RunTracker::new(POST, CAP);
    let now = Instant::now();
    assert!(segments
        .iter()
        .all(|seg| shared.observe(seg.seq, true, now).is_none()));
}

#[test]
fn post_padding_elapses_inside_a_backlogged_batch() {
    const POST: Duration = Duration::from_secs(10);
    const CAP: Duration = Duration::from_secs(300);
    let segments: Vec<_> = (0..60).map(|seq| pending(seq, SECOND_NS)).collect();

    let mut tracker = RunTracker::new(POST, CAP);
    let closed: Vec<_> = segments
        .iter()
        .zip(batch_instants(&segments, Instant::now()))
        .filter_map(|(seg, at)| tracker.observe(seg.seq, seg.seq == 0, at))
        .collect();

    assert_eq!(closed.len(), 1, "the run closes within the batch");
    assert_eq!(closed[0].first_motion_seq, 0);
    assert_eq!(closed[0].last_seq, 10);
    assert!(!tracker.is_open());
}

#[test]
fn detection_mask_ignores_wrong_length() {
    let mut frame = white_frame(160, 120);
    apply_detection_mask(&mut frame, FULL_FRAME, &[true, false, true]);
    assert!(frame.data.iter().all(|&b| b == 255));
}
