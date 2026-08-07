use super::*;

const INTERVAL: Duration = Duration::from_secs(5);
const HOLD: Duration = Duration::from_secs(60);

fn state() -> SensorState {
    SensorState::new(INTERVAL, HOLD)
}

#[test]
fn slugify_folds_non_alphanumerics() {
    assert_eq!(slugify("Front Door"), "front_door");
    assert_eq!(slugify("front-door"), "front_door");
    assert_eq!(slugify("CAM.1"), "cam_1");
    assert_eq!(slugify("yard"), "yard");
}

#[test]
fn capitalize_uppercases_first_char_only() {
    assert_eq!(capitalize("person"), "Person");
    assert_eq!(capitalize("delivery van"), "Delivery van");
    assert_eq!(capitalize(""), "");
}

#[test]
fn occupancy_turns_on_and_holds_then_expires() {
    let mut s = state();
    let t0 = Instant::now();

    // A sighting turns the sensor ON.
    assert!(s.record_sighting("yard", "person", t0));
    assert!(s.is_occupied("yard", "person"));

    // Within the hold-off nothing expires.
    assert!(s
        .expire_occupancy(t0 + HOLD - Duration::from_secs(1))
        .is_empty());
    assert!(s.is_occupied("yard", "person"));

    // At the hold-off it turns OFF.
    let expired = s.expire_occupancy(t0 + HOLD);
    assert_eq!(expired, vec![("yard".to_string(), "person".to_string())]);
    assert!(!s.is_occupied("yard", "person"));
}

#[test]
fn new_sighting_extends_the_hold() {
    let mut s = state();
    let t0 = Instant::now();
    s.record_sighting("yard", "person", t0);

    // A second sighting inside the window is not a fresh transition but
    // does restart the countdown.
    let later = t0 + HOLD - Duration::from_secs(1);
    assert!(!s.record_sighting("yard", "person", later));
    assert!(s.expire_occupancy(t0 + HOLD).is_empty());
    assert!(s.is_occupied("yard", "person"));

    // The hold now runs from the newer sighting.
    assert_eq!(s.expire_occupancy(later + HOLD).len(), 1);
    assert!(!s.is_occupied("yard", "person"));
}

#[test]
fn occupancy_re_arms_after_expiring() {
    let mut s = state();
    let t0 = Instant::now();
    s.record_sighting("yard", "person", t0);
    s.expire_occupancy(t0 + HOLD);
    // OFF, so the next sighting is a fresh OFF -> ON transition again.
    let t1 = t0 + HOLD + Duration::from_secs(30);
    assert!(s.record_sighting("yard", "person", t1));
    assert!(s.is_occupied("yard", "person"));
    assert!(s
        .expire_occupancy(t1 + HOLD - Duration::from_secs(1))
        .is_empty());
}

#[test]
fn occupancy_is_tracked_per_camera_and_class() {
    let mut s = state();
    let t0 = Instant::now();
    s.record_sighting("yard", "person", t0);
    s.record_sighting("yard", "car", t0 + Duration::from_secs(30));
    s.record_sighting("gate", "person", t0 + Duration::from_secs(30));

    // Only the oldest pair expires at t0 + HOLD.
    let expired = s.expire_occupancy(t0 + HOLD);
    assert_eq!(expired, vec![("yard".to_string(), "person".to_string())]);
    assert!(s.is_occupied("yard", "car"));
    assert!(s.is_occupied("gate", "person"));
    assert!(!s.is_occupied("yard", "person"));
}

#[test]
fn snapshots_are_due_immediately_then_on_the_interval() {
    let mut s = state();
    let t0 = Instant::now();
    // No motion: nothing to snapshot.
    assert!(s.due_snapshots(t0).is_empty());

    assert!(s.motion_start("yard"));
    // A freshly opened run is due at once.
    assert_eq!(s.due_snapshots(t0), vec!["yard".to_string()]);
    // ...and not again until the interval elapses.
    assert!(s
        .due_snapshots(t0 + INTERVAL - Duration::from_millis(1))
        .is_empty());
    assert_eq!(s.due_snapshots(t0 + INTERVAL), vec!["yard".to_string()]);
}

#[test]
fn motion_end_stops_the_snapshot_cadence() {
    let mut s = state();
    let t0 = Instant::now();
    s.motion_start("yard");
    s.due_snapshots(t0);
    assert!(s.motion_end("yard"));
    assert!(s.due_snapshots(t0 + INTERVAL * 10).is_empty());
    // A second end is a no-op — the caller must not publish OFF twice.
    assert!(!s.motion_end("yard"));
}

#[test]
fn duplicate_motion_start_is_ignored() {
    let mut s = state();
    assert!(s.motion_start("yard"));
    assert!(!s.motion_start("yard"));
    assert_eq!(s.motion_active.len(), 1);
}

#[test]
fn discovery_payloads_match_expected_json() {
    let config = MqttConfig {
        topic_prefix: "camon".to_string(),
        discovery_prefix: "homeassistant".to_string(),
        ..MqttConfig::default()
    };
    let topics = Topics::new(&config);
    let payloads = discovery_payloads(&topics, "Front Door", &["person".to_string()]);

    let device = serde_json::json!({
        "identifiers": ["camon_front_door"],
        "name": "Camon Front Door",
        "manufacturer": "camon",
        "sw_version": env!("CAMON_VERSION"),
    });

    assert_eq!(payloads.len(), 4);

    assert_eq!(
        payloads[0].0,
        "homeassistant/camera/camon_front_door/config"
    );
    assert_eq!(
        payloads[0].1,
        serde_json::json!({
            "name": "Snapshot",
            "unique_id": "camon_front_door_snapshot",
            "topic": "camon/Front Door/snapshot",
            "availability_topic": "camon/availability",
            "device": device,
        })
    );

    assert_eq!(
        payloads[1].0,
        "homeassistant/binary_sensor/camon_front_door_motion/config"
    );
    assert_eq!(
        payloads[1].1,
        serde_json::json!({
            "name": "Motion",
            "unique_id": "camon_front_door_motion",
            "state_topic": "camon/Front Door/motion",
            "device_class": "motion",
            "availability_topic": "camon/availability",
            "device": device,
        })
    );

    assert_eq!(
        payloads[2].0,
        "homeassistant/binary_sensor/camon_front_door_occupancy_person/config"
    );
    assert_eq!(
        payloads[2].1,
        serde_json::json!({
            "name": "Person occupancy",
            "unique_id": "camon_front_door_occupancy_person",
            "state_topic": "camon/Front Door/occupancy/person",
            "device_class": "occupancy",
            "availability_topic": "camon/availability",
            "device": device,
        })
    );

    assert_eq!(
        payloads[3].0,
        "homeassistant/camera/camon_front_door_occupancy_person/config"
    );
    assert_eq!(
        payloads[3].1,
        serde_json::json!({
            "name": "Person snapshot",
            "unique_id": "camon_front_door_occupancy_person_snapshot",
            "topic": "camon/Front Door/occupancy/person/snapshot",
            "availability_topic": "camon/availability",
            "device": device,
        })
    );
}

#[test]
fn no_classes_means_no_occupancy_entities() {
    let topics = Topics::new(&MqttConfig::default());
    let payloads = discovery_payloads(&topics, "yard", &[]);
    assert_eq!(payloads.len(), 2);
}

#[test]
fn every_class_adds_a_sensor_and_a_snapshot_camera() {
    let topics = Topics::new(&MqttConfig::default());
    let classes = ["person".to_string(), "cat".to_string()];
    let payloads = discovery_payloads(&topics, "yard", &classes);
    assert_eq!(payloads.len(), 2 + 2 * classes.len());
    // Unique ids must stay distinct across components.
    let ids: HashSet<&str> = payloads
        .iter()
        .map(|(_, payload)| payload["unique_id"].as_str().unwrap())
        .collect();
    assert_eq!(ids.len(), payloads.len());
}

#[test]
fn topics_tolerate_a_trailing_slash_in_the_prefix() {
    let config = MqttConfig {
        topic_prefix: "camon/".to_string(),
        discovery_prefix: "homeassistant/".to_string(),
        ..MqttConfig::default()
    };
    let topics = Topics::new(&config);
    assert_eq!(topics.availability(), "camon/availability");
    assert_eq!(topics.motion("yard"), "camon/yard/motion");
    assert_eq!(topics.occupancy("yard", "car"), "camon/yard/occupancy/car");
    assert_eq!(
        topics.occupancy_snapshot("yard", "car"),
        "camon/yard/occupancy/car/snapshot"
    );
    assert_eq!(topics.snapshot("yard"), "camon/yard/snapshot");
    assert_eq!(
        topics.discovery("camera", "camon_yard"),
        "homeassistant/camera/camon_yard/config"
    );
}

#[test]
fn reconnect_publishes_an_explicit_state_for_every_entity() {
    let topics = Topics::new(&MqttConfig::default());
    let cameras = ["yard".to_string(), "gate".to_string()];
    let classes = ["person".to_string(), "cat".to_string()];
    let mut s = state();
    s.motion_start("yard");
    s.record_sighting("gate", "cat", Instant::now());

    let payloads = state_payloads(&topics, &s, &cameras, &classes);
    assert_eq!(payloads.len(), cameras.len() * (1 + classes.len()));

    let by_topic: HashMap<&str, &str> = payloads
        .iter()
        .map(|(topic, payload)| (topic.as_str(), *payload))
        .collect();
    assert_eq!(by_topic["camon/yard/motion"], "ON");
    assert_eq!(by_topic["camon/gate/occupancy/cat"], "ON");
    // The point of the enumeration: everything that is not ON says so,
    // instead of leaving whatever the broker still has retained.
    assert_eq!(by_topic["camon/gate/motion"], "OFF");
    assert_eq!(by_topic["camon/yard/occupancy/cat"], "OFF");
    assert_eq!(by_topic["camon/yard/occupancy/person"], "OFF");
    assert_eq!(by_topic["camon/gate/occupancy/person"], "OFF");
}

#[test]
fn republished_topics_are_exactly_the_discovered_state_topics() {
    let topics = Topics::new(&MqttConfig::default());
    let cameras = ["Front Door".to_string(), "yard".to_string()];
    let classes = ["person".to_string()];

    let discovered: HashSet<String> = cameras
        .iter()
        .flat_map(|camera_id| discovery_payloads(&topics, camera_id, &classes))
        .filter_map(|(_, payload)| payload["state_topic"].as_str().map(str::to_string))
        .collect();
    let republished: HashSet<String> = state_payloads(&topics, &state(), &cameras, &classes)
        .into_iter()
        .map(|(topic, _)| topic)
        .collect();

    assert_eq!(discovered, republished);
}

fn bridge_context(cameras: &[&str], classes: &[&str]) -> BridgeContext {
    BridgeContext {
        config: MqttConfig::default(),
        buffers: Arc::new(HashMap::new()),
        camera_ids: cameras.iter().map(|c| c.to_string()).collect(),
        classes: classes.iter().map(|c| c.to_string()).collect(),
        entities_path: None,
        shutdown: Arc::new(AtomicBool::new(false)),
    }
}

/// An in-flight decode that has not reported anything yet, for the tests
/// that care about the task rather than about what it produced.
fn snapshot_task(handle: tokio::task::JoinHandle<()>) -> SnapshotTask {
    SnapshotTask {
        handle,
        decoded: Arc::new(AtomicBool::new(false)),
        run: 0,
    }
}

/// A bridge that remembers nothing across restarts, which is every test
/// that is not about the memory itself.
fn no_memory(ctx: &BridgeContext) -> EntityMemory {
    EntityMemory::load(&Topics::new(&ctx.config), ctx)
}

/// The set a run with nothing remembered announces: the config's own.
fn announced_for(ctx: &BridgeContext) -> EntityRecord {
    EntityRecord::current(None, &Topics::new(&ctx.config), ctx)
}

fn capacity_for(ctx: &BridgeContext) -> usize {
    request_queue_capacity(ctx.camera_ids.len(), ctx.classes.len(), 0)
}

/// A client whose event loop is never polled, so its request queue only
/// ever fills. That is precisely the state a reconnect after an outage
/// finds it in. The event loop comes back too: dropping it closes the
/// channel, which fails publishes for a reason a live bridge never has.
fn unpolled_client(capacity: usize) -> (AsyncClient, rumqttc::EventLoop) {
    let options = MqttOptions::new("camon-test", "127.0.0.1", 1883);
    AsyncClient::new(options, capacity)
}

#[test]
fn the_burst_is_clears_then_discovery_then_states_then_availability() {
    let topics = Topics::new(&MqttConfig::default());
    let ctx = bridge_context(&["yard", "gate"], &["person"]);
    let orphans = vec![
        "camon/old/snapshot".to_string(),
        "homeassistant/camera/camon_old/config".to_string(),
    ];
    let burst = reconnect_burst(&topics, &state(), &announced_for(&ctx), &orphans);

    // The clears lead, and they are empty payloads: that is both how Home
    // Assistant is told to forget a discovered entity and how a retained
    // state stops being held. Anything after them says something.
    assert_eq!(burst[0], (orphans[0].clone(), Vec::new()));
    assert_eq!(burst[1], (orphans[1].clone(), Vec::new()));
    assert!(burst[2..].iter().all(|(_, payload)| !payload.is_empty()));

    let discovery = burst
        .iter()
        .rposition(|(topic, _)| topic.starts_with("homeassistant/"))
        .unwrap();
    let first_state = burst
        .iter()
        .position(|(topic, _)| topic.ends_with("/motion"))
        .unwrap();
    assert!(discovery < first_state);
    // The availability flip is the last thing queued, never the first —
    // publishing it before the clears is exactly what resurrects an entity
    // the config dropped.
    let (topic, payload) = burst.last().unwrap();
    assert_eq!(topic, "camon/availability");
    assert_eq!(payload, b"online");

    // One clear per orphaned topic, then per camera two discovery payloads
    // plus two per class, one state plus one per class. Then the marker.
    let (cameras, classes) = (ctx.camera_ids.len(), ctx.classes.len());
    assert_eq!(burst.len(), orphans.len() + cameras * (3 + 3 * classes) + 1);
}

#[test]
fn a_retried_burst_carries_the_state_of_the_retry_not_the_failure() {
    let topics = Topics::new(&MqttConfig::default());
    let ctx = bridge_context(&["yard"], &[]);
    let mut s = state();
    s.motion_start("yard");

    let motion = |burst: Vec<(String, Vec<u8>)>| {
        burst
            .into_iter()
            .find(|(topic, _)| topic == "camon/yard/motion")
            .map(|(_, payload)| payload)
            .unwrap()
    };
    assert_eq!(
        motion(reconnect_burst(&topics, &s, &announced_for(&ctx), &[])),
        b"ON"
    );

    // The run ends while the burst is still owed to the broker. Rebuilding
    // is what keeps the retry from re-asserting the value that has gone.
    s.motion_end("yard");
    assert_eq!(
        motion(reconnect_burst(&topics, &s, &announced_for(&ctx), &[])),
        b"OFF"
    );
}

#[test]
fn the_queue_always_has_room_for_a_whole_burst() {
    let topics = Topics::new(&MqttConfig::default());
    // Well past any supported install, and past the fifteen-camera default
    // -class shape that does not fit a fixed 256.
    // The orphan counts are a whole previous config's worth of entities:
    // clears ride in the same all-or-nothing burst, so the queue has to
    // hold them too.
    for (cameras, classes, orphans) in [(1, 0, 0), (15, 5, 0), (15, 5, 180), (64, 16, 1)] {
        let camera_ids: Vec<String> = (0..cameras).map(|i| format!("cam{i}")).collect();
        let class_names: Vec<String> = (0..classes).map(|i| format!("class{i}")).collect();
        let orphan_topics: Vec<String> = (0..orphans)
            .map(|i| format!("camon/gone{i}/motion"))
            .collect();
        let ctx = BridgeContext {
            config: MqttConfig::default(),
            buffers: Arc::new(HashMap::new()),
            camera_ids,
            classes: class_names,
            entities_path: None,
            shutdown: Arc::new(AtomicBool::new(false)),
        };

        // The formula the queue is sized from must be the burst that is
        // actually built, or the sizing means nothing.
        let burst = reconnect_burst(&topics, &state(), &announced_for(&ctx), &orphan_topics);
        assert_eq!(burst.len(), burst_len(cameras, classes, orphans));

        // All-or-nothing publishing plus a burst that cannot fit is a
        // permanent retry loop that never reaches `online`.
        let (client, _eventloop) =
            unpolled_client(request_queue_capacity(cameras, classes, orphans));
        assert!(publish_burst(&client, burst));
    }
}

#[test]
fn a_rejected_burst_goes_out_once_the_queue_drains() {
    let topics = Topics::new(&MqttConfig::default());
    let ctx = bridge_context(&["yard", "gate"], &["person"]);
    let burst = reconnect_burst(&topics, &state(), &announced_for(&ctx), &[]);

    // One message already queued leaves the burst a slot short of fitting.
    let (client, mut eventloop) = unpolled_client(burst.len());
    assert_eq!(publish_state(&client, "camon/filler", "x"), Published::Yes);
    assert!(!publish_burst(&client, burst.clone()));

    // What the event loop does when the connection comes back: take
    // everything the channel is holding. Recovery has to happen on this
    // queue, so the retry runs against the same client rather than a fresh
    // one that was never full.
    eventloop.clean();
    assert!(publish_burst(&client, burst));
}

/// A record for the broker `MqttConfig::default()` names, which is the one
/// every `bridge_context` here announces to.
fn record(prefix: &str, cameras: &[&str], classes: &[&str]) -> EntityRecord {
    EntityRecord {
        version: ENTITY_RECORD_VERSION,
        broker: broker_id(&MqttConfig::default()),
        topic_prefix: prefix.to_string(),
        discovery_prefix: "homeassistant".to_string(),
        cameras: cameras.iter().map(|c| c.to_string()).collect(),
        classes: classes.iter().map(|c| c.to_string()).collect(),
        pending_clears: Vec::new(),
    }
}

#[test]
fn every_announced_entity_has_both_of_its_retained_topics() {
    let topics = Topics::new(&MqttConfig::default());
    let classes = ["person".to_string()];
    let announced = retained_topics(&topics, "yard", &classes);
    let discovery = discovery_payloads(&topics, "yard", &classes);

    // The discovery document and the topic it points at, for every entity:
    // clearing only the first would leave the broker holding the state —
    // including a sighting crop, which is hundreds of KiB of it.
    assert_eq!(announced.len(), 2 * discovery.len());
    for (topic, payload) in discovery {
        assert!(announced.contains(&topic));
        let state = payload["state_topic"]
            .as_str()
            .or_else(|| payload["topic"].as_str())
            .unwrap();
        assert!(announced.contains(&state.to_string()));
    }
}

#[test]
fn a_removed_camera_takes_every_retained_topic_with_it() {
    let orphans = orphaned_topics(
        &record("camon", &["yard", "gate"], &["person"]),
        &record("camon", &["yard"], &["person"]),
    );
    assert_eq!(
        orphans,
        vec![
            "camon/gate/motion",
            "camon/gate/occupancy/person",
            "camon/gate/occupancy/person/snapshot",
            "camon/gate/snapshot",
            "homeassistant/binary_sensor/camon_gate_motion/config",
            "homeassistant/binary_sensor/camon_gate_occupancy_person/config",
            "homeassistant/camera/camon_gate/config",
            "homeassistant/camera/camon_gate_occupancy_person/config",
        ]
    );
}

#[test]
fn a_removed_class_is_cleared_for_every_camera() {
    let orphans = orphaned_topics(
        &record("camon", &["yard", "gate"], &["person", "cat"]),
        &record("camon", &["yard", "gate"], &["person"]),
    );
    // Two entities per camera per class, two retained topics each.
    assert_eq!(orphans.len(), 2 * 2 * 2);
    assert!(orphans.iter().all(|topic| topic.contains("cat")));
    assert!(orphans.contains(&"camon/yard/occupancy/cat/snapshot".to_string()));
    assert!(orphans
        .contains(&"homeassistant/binary_sensor/camon_gate_occupancy_cat/config".to_string()));
}

#[test]
fn object_detection_off_for_one_run_keeps_announcing_its_classes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mqtt_entities.json");
    let topics = Topics::new(&MqttConfig::default());
    save_record(&path, &record("camon", &["yard"], &["person"])).unwrap();

    // No classes this run: object detection is off, or its client could
    // not be built. Neither is the operator dropping the entity, and being
    // wrong costs them its history and its place on a dashboard.
    let mut ctx = bridge_context(&["yard"], &[]);
    ctx.entities_path = Some(path.clone());
    let memory = EntityMemory::load(&topics, &ctx);
    assert_eq!(memory.announced.classes, vec!["person".to_string()]);
    assert!(memory.orphans.is_empty());

    // Carrying them only through the diff would be worse than useless: the
    // burst would omit them and then publish `online`, which is exactly
    // how the retained ON of an occupancy sensor comes back to life. So
    // they are announced — discovered, and restated for what they are.
    let burst = reconnect_burst(&topics, &state(), &memory.announced, &memory.orphans);
    assert!(burst.iter().any(
        |(topic, _)| topic == "homeassistant/binary_sensor/camon_yard_occupancy_person/config"
    ));
    assert!(burst
        .iter()
        .any(|(topic, payload)| topic == "camon/yard/occupancy/person" && payload == b"OFF"));

    // ...and the queue is sized from that same set, not from the empty
    // class list the config handed in.
    let (client, _eventloop) = unpolled_client(request_queue_capacity(
        memory.announced.cameras.len(),
        memory.announced.classes.len(),
        memory.orphans.len(),
    ));
    assert!(publish_burst(&client, burst));

    // The camera dimension still answers for itself while they are carried,
    // and a camera added during such a run is recorded with the classes it
    // was actually announced with.
    let mut ctx = bridge_context(&["front", "yard"], &[]);
    ctx.entities_path = Some(path);
    let memory = EntityMemory::load(&topics, &ctx);
    assert!(memory.orphans.is_empty());
    let burst = reconnect_burst(&topics, &state(), &memory.announced, &memory.orphans);
    assert!(burst.iter().any(
        |(topic, _)| topic == "homeassistant/binary_sensor/camon_front_occupancy_person/config"
    ));
}

#[test]
fn a_moved_topic_prefix_orphans_the_state_it_left_behind() {
    let previous = record("camon", &["yard"], &[]);
    let orphans = orphaned_topics(&previous, &record("nvr", &["yard"], &[]));
    assert!(orphans.contains(&"camon/yard/motion".to_string()));
    assert!(orphans.contains(&"camon/yard/snapshot".to_string()));
    // Shared by every entity rather than owned by one, so it is only ever
    // orphaned by the prefix itself moving — and nothing under the new
    // prefix is cleared, since the burst is about to publish it.
    assert!(orphans.contains(&"camon/availability".to_string()));
    assert!(orphans.iter().all(|topic| !topic.starts_with("nvr/")));

    // A config that only gained a camera has orphaned nothing at all, the
    // availability topic included.
    assert!(orphaned_topics(&previous, &record("camon", &["yard", "gate"], &[])).is_empty());
}

#[test]
fn nothing_is_cleared_without_a_record_this_build_and_broker_own() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mqtt_entities.json");
    let topics = Topics::new(&MqttConfig::default());
    let mut ctx = bridge_context(&["yard"], &["person"]);
    ctx.entities_path = Some(path.clone());
    let previous = record("camon", &["yard", "gate"], &["person"]);
    // What that record costs `gate` when it *is* acted on, so every case
    // below asserts a refusal rather than an empty diff.
    save_record(&path, &previous).unwrap();
    assert_eq!(EntityMemory::load(&topics, &ctx).orphans.len(), 8);

    // A first start knows nothing about the past, which is not evidence
    // that anything was removed.
    std::fs::remove_file(&path).unwrap();
    let memory = EntityMemory::load(&topics, &ctx);
    assert!(memory.orphans.is_empty());
    assert!(memory.on_disk.is_none());

    // Neither is a record camon cannot read...
    std::fs::write(&path, b"{ not json").unwrap();
    assert!(EntityMemory::load(&topics, &ctx).orphans.is_empty());

    // ...nor one carrying fields this build does not know, whatever else
    // it gets right: an unrecognized document is not deletion authority.
    let mut json = serde_json::to_value(&previous).unwrap();
    json["entities"] = serde_json::json!(["camon_gate_motion"]);
    std::fs::write(&path, serde_json::to_vec(&json).unwrap()).unwrap();
    assert!(EntityMemory::load(&topics, &ctx).orphans.is_empty());

    // ...nor one written to a format version this build does not speak.
    let mut json = serde_json::to_value(&previous).unwrap();
    json["version"] = serde_json::json!(ENTITY_RECORD_VERSION + 1);
    std::fs::write(&path, serde_json::to_vec(&json).unwrap()).unwrap();
    assert!(EntityMemory::load(&topics, &ctx).orphans.is_empty());
}

#[test]
fn a_record_from_another_broker_is_not_deletion_authority() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mqtt_entities.json");
    let mut elsewhere = record("camon", &["yard", "gate"], &["person"]);
    elsewhere.broker = "other.example:1883".to_string();
    save_record(&path, &elsewhere).unwrap();

    // Same data dir, same prefixes, a different broker: `gate`'s entities
    // over there say nothing about the entities over here, which may never
    // have existed — or may belong to the camon still serving them.
    let topics = Topics::new(&MqttConfig::default());
    let mut ctx = bridge_context(&["yard"], &[]);
    ctx.entities_path = Some(path);
    let memory = EntityMemory::load(&topics, &ctx);
    assert!(memory.orphans.is_empty());
    assert_eq!(memory.announced.broker, "localhost:1883");
    // Nor is it a reason to announce that broker's classes here: a record
    // camon will not delete from is not one it will publish from either.
    assert!(memory.announced.classes.is_empty());
}

#[test]
fn a_clear_stays_owed_until_a_disconnect_proves_it_went_out() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mqtt_entities.json");
    let topics = Topics::new(&MqttConfig::default());
    save_record(&path, &record("camon", &["yard", "gate"], &["person"])).unwrap();
    let mut ctx = bridge_context(&["yard"], &["person"]);
    ctx.entities_path = Some(path.clone());

    let mut memory = EntityMemory::load(&topics, &ctx);
    assert_eq!(memory.orphans.len(), 8);

    let mut link = Link {
        connected: true,
        republish_pending: true,
        ..Link::default()
    };
    let mut tasks = HashMap::new();
    let mut s = state();

    // A burst the queue refuses is evidence of nothing: the record on disk
    // still describes the run that announced `gate`.
    let (full, _full_loop) = unpolled_client(1);
    on_tick(
        &full,
        &topics,
        &mut s,
        &ctx,
        &mut tasks,
        &mut link,
        &mut memory,
        false,
    );
    assert!(link.republish_pending);
    assert_eq!(load_record(&path).unwrap().cameras, ["yard", "gate"]);

    // Accepted, so the announced set is recorded — with the clears marked
    // still owed, because `try_publish` only queued them.
    let (roomy, _roomy_loop) = unpolled_client(request_queue_capacity(1, 1, 8));
    on_tick(
        &roomy,
        &topics,
        &mut s,
        &ctx,
        &mut tasks,
        &mut link,
        &mut memory,
        false,
    );
    assert!(!link.republish_pending);
    let written = load_record(&path).unwrap();
    assert_eq!(written.cameras, ["yard"]);
    assert_eq!(written.pending_clears, memory.orphans);

    // A process killed here — queued, never written to the socket — finds
    // the same clears owed on the next start instead of a forgotten ghost.
    assert_eq!(EntityMemory::load(&topics, &ctx).orphans, memory.orphans);

    // The disconnect is what proves the socket took them.
    memory.note_clears_flushed();
    assert!(load_record(&path).unwrap().pending_clears.is_empty());
    assert!(EntityMemory::load(&topics, &ctx).orphans.is_empty());
}

#[test]
fn an_owed_clear_is_dropped_when_its_topic_is_announced_again() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mqtt_entities.json");
    let topics = Topics::new(&MqttConfig::default());
    let mut previous = record("camon", &["yard"], &["person"]);
    // Removed last run, put back before this one. Clearing what the burst
    // is about to publish would delete a live entity.
    previous.pending_clears = vec![
        "camon/yard/motion".to_string(),
        "camon/gate/motion".to_string(),
    ];
    save_record(&path, &previous).unwrap();

    let mut ctx = bridge_context(&["yard"], &["person"]);
    ctx.entities_path = Some(path);
    assert_eq!(
        EntityMemory::load(&topics, &ctx).orphans,
        vec!["camon/gate/motion".to_string()]
    );
}

#[test]
fn a_record_that_cannot_be_written_stays_owed() {
    let dir = tempfile::tempdir().unwrap();
    // A file where the directory has to be, so every write against this
    // path fails the way a read-only or full disk does.
    let blocked = dir.path().join("blocked");
    std::fs::write(&blocked, b"not a directory").unwrap();
    let path = blocked.join("mqtt_entities.json");

    let topics = Topics::new(&MqttConfig::default());
    let mut ctx = bridge_context(&["yard"], &["person"]);
    ctx.entities_path = Some(path.clone());
    let mut memory = EntityMemory::load(&topics, &ctx);

    memory.note_burst_accepted();
    // Nothing reached disk, so nothing may be believed to be there: a run
    // that announced entities and recorded none of it turns the operator's
    // next removal into a ghost nothing cleans up.
    assert!(memory.on_disk.is_none());

    std::fs::remove_file(&blocked).unwrap();
    memory.note_burst_accepted();
    assert_eq!(memory.on_disk, load_record(&path));
    assert_eq!(load_record(&path).unwrap().cameras, ["yard"]);
}

#[tokio::test]
async fn a_dropped_state_publish_asks_for_a_full_republish() {
    let topics = Topics::new(&MqttConfig::default());
    let ctx = bridge_context(&["yard"], &["person"]);
    // Zero hold, so the sighting recorded here expires on the next tick.
    let mut s = SensorState::new(INTERVAL, Duration::ZERO);
    s.record_sighting("yard", "person", Instant::now());

    let (client, _eventloop) = unpolled_client(1);
    assert_eq!(publish_state(&client, "camon/filler", "x"), Published::Yes);

    let mut link = Link {
        connected: true,
        republish_pending: false,
        ..Link::default()
    };
    let mut tasks = HashMap::new();
    on_tick(
        &client,
        &topics,
        &mut s,
        &ctx,
        &mut tasks,
        &mut link,
        &mut no_memory(&ctx),
        false,
    );

    // The OFF had nowhere to go. Nothing else would ever re-assert it, so
    // the tick has to come back with the whole state or that sensor stays
    // ON in Home Assistant for good.
    assert!(link.republish_pending);
}

#[test]
fn a_topic_that_can_never_be_published_stalls_nothing() {
    let topics = Topics::new(&MqttConfig::default());
    // `Config::validate` rejects this shape while the bridge is enabled;
    // the bridge defends itself anyway, because retrying it forever would
    // mean `online` never goes out.
    let ctx = bridge_context(&["ya+rd"], &[]);
    let burst = reconnect_burst(&topics, &state(), &announced_for(&ctx), &[]);
    let (client, _eventloop) = unpolled_client(capacity_for(&ctx));
    assert!(publish_burst(&client, burst));

    assert_eq!(
        publish_state(&client, "camon/ya+rd/motion", "OFF"),
        Published::ImpossibleTopic
    );
    // ...and it must not ask for a retry that could not change anything.
    let mut link = Link::default();
    link.note(Published::ImpossibleTopic);
    assert!(!link.republish_pending);
    link.note(Published::QueueFull);
    assert!(link.republish_pending);
}

#[test]
fn a_full_queue_fails_the_whole_burst_including_availability() {
    let topics = Topics::new(&MqttConfig::default());
    let ctx = bridge_context(&["yard"], &["person"]);
    let (client, _eventloop) = unpolled_client(2);
    // Fill it, exactly as an outage does.
    assert!(!publish_burst(
        &client,
        reconnect_burst(&topics, &state(), &announced_for(&ctx), &[])
    ));
    // Nothing drains it, so a retry gets nowhere either — and reports so
    // rather than leaving `online` unsent and the entities unavailable.
    assert!(!publish_burst(
        &client,
        reconnect_burst(&topics, &state(), &announced_for(&ctx), &[])
    ));
}

#[tokio::test]
async fn the_tick_retries_the_burst_until_the_queue_takes_it() {
    let topics = Topics::new(&MqttConfig::default());
    let ctx = bridge_context(&["yard"], &["person"]);
    let mut s = state();
    let mut tasks = HashMap::new();
    let mut link = Link {
        connected: true,
        republish_pending: true,
        ..Link::default()
    };

    let (small, _small_loop) = unpolled_client(2);
    on_tick(
        &small,
        &topics,
        &mut s,
        &ctx,
        &mut tasks,
        &mut link,
        &mut no_memory(&ctx),
        false,
    );
    // Still owed to the broker: the flag is what brings the next tick back
    // here, instead of leaving `online` unsent on a healthy connection.
    assert!(link.republish_pending);

    let (roomy, _roomy_loop) = unpolled_client(capacity_for(&ctx));
    on_tick(
        &roomy,
        &topics,
        &mut s,
        &ctx,
        &mut tasks,
        &mut link,
        &mut no_memory(&ctx),
        false,
    );
    assert!(!link.republish_pending);

    // A tick while disconnected must not spend the retry on a queue that
    // nothing is draining. Against a queue with room to spare, so the flag
    // can only still be pending because the guard held.
    link.republish_pending = true;
    link.connected = false;
    let (idle, _idle_loop) = unpolled_client(capacity_for(&ctx));
    on_tick(
        &idle,
        &topics,
        &mut s,
        &ctx,
        &mut tasks,
        &mut link,
        &mut no_memory(&ctx),
        false,
    );
    assert!(link.republish_pending);
}

#[tokio::test]
async fn snapshots_are_not_queued_while_disconnected() {
    let topics = Topics::new(&MqttConfig::default());
    let buffer = HotBuffer::new("yard".to_string(), 60);
    {
        let mut buf = buffer.write_recover();
        for i in 0..2 {
            buf.push(crate::buffer::GopSegment {
                start_pts: i * 1_000_000_000,
                duration_ns: 1_000_000_000,
                data: Arc::new(vec![0u8; 16]),
                frame_count: 1,
            });
        }
    }
    let mut ctx = bridge_context(&["yard"], &[]);
    ctx.buffers = Arc::new(HashMap::from([("yard".to_string(), buffer)]));

    let (client, _eventloop) = unpolled_client(capacity_for(&ctx));
    let mut tasks = HashMap::new();
    let mut link = Link::default();
    spawn_snapshot(&client, &topics, &ctx, "yard", 0, &mut tasks, &link);
    // Disconnected: no decode, so no JPEG parked in a queue nothing is
    // draining until long after it mattered.
    assert!(tasks.is_empty());

    link.connected = true;
    spawn_snapshot(&client, &topics, &ctx, "yard", 0, &mut tasks, &link);
    assert_eq!(tasks.len(), 1);
    for task in tasks.values() {
        task.handle.abort();
    }
}

/// A pid is dead once `/proc` has lost it or reports it as a zombie —
/// `kill_on_drop` reaps asynchronously, so the zombie window is normal.
fn process_dead(pid: u32) -> bool {
    match std::fs::read_to_string(format!("/proc/{pid}/stat")) {
        Ok(stat) => stat
            .rsplit(')')
            .next()
            .is_some_and(|rest| rest.trim_start().starts_with('Z')),
        Err(_) => true,
    }
}

#[tokio::test]
async fn a_wedged_decode_is_bounded_and_kills_its_child() {
    let dir = tempfile::tempdir().unwrap();
    let pidfile = dir.path().join("pid");
    let mut command = tokio::process::Command::new("sh");
    command.arg("-c").arg(format!(
        "echo $$ > {}; exec sleep 60",
        pidfile.to_str().unwrap()
    ));

    let started = std::time::Instant::now();
    let out = piped_decode(command, b"unread", 16, Duration::from_millis(200)).await;
    assert!(out.is_none());
    assert!(started.elapsed() < Duration::from_secs(5));

    // Written by the child before it execs `sleep`; on a loaded machine it
    // may not be there the moment the decode gives up.
    let mut pid = None;
    for _ in 0..100 {
        if let Ok(written) = std::fs::read_to_string(&pidfile) {
            if let Ok(parsed) = written.trim().parse::<u32>() {
                pid = Some(parsed);
                break;
            }
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let pid = pid.expect("child never recorded its pid");
    for _ in 0..100 {
        if process_dead(pid) {
            return;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("child {pid} outlived the decode that owned it");
}

/// Only a closed descriptor is an EOF. `shutdown` on a child's stdin
/// flushes and returns, so a decoder handed a segment it must probe before
/// it can emit anything waits for input that is never coming — the whole
/// decode timeout, no frame, for every camera, every time. `cat` says it
/// without needing an ffmpeg: it echoes its input and exits when the input
/// ends, so an answer at all is the end of the input having arrived.
#[tokio::test]
async fn a_decode_is_told_where_its_input_ends() {
    let out = piped_decode(
        tokio::process::Command::new("cat"),
        b"hello",
        16,
        Duration::from_secs(5),
    )
    .await;
    assert_eq!(
        out.as_deref(),
        Some(&b"hello"[..]),
        "the child never saw the end of its input"
    );
}

#[tokio::test]
async fn a_decode_that_produces_nothing_is_not_an_error() {
    let mut command = tokio::process::Command::new("sh");
    command.arg("-c").arg("exit 0");
    let out = piped_decode(command, b"unread", 16, Duration::from_secs(5)).await;
    assert_eq!(out, Some(Vec::new()));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn shutdown_does_not_wait_on_a_decode_that_ignores_cancellation() {
    // A task with no await point at all, which `abort` cannot reach. Real
    // snapshot tasks are not this shape — they await the ffmpeg pipe and
    // then the encode, and abort lands there in microseconds — so this is
    // the bound's backstop case rather than its everyday one.
    let mut tasks = HashMap::new();
    tasks.insert(
        "yard".to_string(),
        snapshot_task(tokio::spawn(async {
            std::thread::sleep(Duration::from_secs(1))
        })),
    );

    let started = std::time::Instant::now();
    abort_snapshots(tasks).await;
    assert!(started.elapsed() < Duration::from_millis(900));
}

#[tokio::test]
async fn shutdown_joins_cancellable_decodes_at_once() {
    let mut tasks = HashMap::new();
    tasks.insert(
        "yard".to_string(),
        snapshot_task(tokio::spawn(async {
            tokio::time::sleep(Duration::from_secs(60)).await
        })),
    );

    let started = std::time::Instant::now();
    abort_snapshots(tasks).await;
    assert!(started.elapsed() < SNAPSHOT_ABORT_JOIN);
}

/// An event loop pointed at a socket that accepts every connection and
/// closes it immediately, so the CONNECT is met with an EOF and `poll()`
/// fails at once and keeps failing — the state the pacing delay exists for.
/// A listener rather than a port left free: a free port is only free until
/// something else takes it, and this test must not depend on what else is
/// running. Everything comes back so the caller can hold it for the length
/// of the test — dropping the client would end the event loop for a reason
/// unrelated to the connection, and dropping the accept task would put the
/// port back to being merely free.
async fn eventloop_against_a_closing_socket(
) -> (AsyncClient, rumqttc::EventLoop, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let accepts = tokio::spawn(async move {
        while let Ok((stream, _)) = listener.accept().await {
            drop(stream);
        }
    });
    let (client, eventloop) =
        AsyncClient::new(MqttOptions::new("camon-test", "127.0.0.1", port), 1);
    (client, eventloop, accepts)
}

#[tokio::test]
async fn a_connection_failure_reaches_the_bridge_and_paces_interruptibly() {
    let (_client, eventloop, accepts) = eventloop_against_a_closing_socket().await;
    let (tx, mut events) = tokio::sync::mpsc::channel(LINK_EVENT_CAPACITY);
    let stop = Arc::new(tokio::sync::Notify::new());
    let task = tokio::spawn(run_eventloop(eventloop, tx, Arc::clone(&stop)));

    // The bridge hears about the failure rather than being left to infer
    // it from a `poll()` it no longer performs.
    assert_eq!(events.recv().await, Some(LinkEvent::Disconnected));

    // The task is now inside RECONNECT_DELAY, which shutdown must not have
    // to wait out. `Ok(Ok(()))` is the task having returned on the signal;
    // a delay that ignored it would still be sleeping at the timeout.
    stop.notify_one();
    let ended = tokio::time::timeout(Duration::from_secs(1), task).await;
    assert!(
        matches!(ended, Ok(Ok(()))),
        "the reconnect delay outlived the stop signal"
    );
    accepts.abort();
}

/// The flag says a stop has begun; the producers dropping their senders is
/// what says there is nothing left to receive. Both arms of the bridge's
/// loop ask this one question, so neither can drift into answering it on
/// its own.
#[test]
fn the_bridge_stops_only_once_its_producers_have_gone() {
    let stopping = AtomicBool::new(true);
    assert!(
        !bridge_is_done(false, &stopping),
        "the bridge stopped while its analyzers were still draining"
    );
    assert!(bridge_is_done(true, &stopping));
    // Producers gone without a stop is the channel closing under a running
    // camon, which the availability marker must not be torn down for.
    assert!(!bridge_is_done(true, &AtomicBool::new(false)));
}

/// A `MotionEnd` arriving during the drain is an analyzer flushing its open
/// run on the way out, and the bridge now stays to receive it. What it must
/// not do is answer it by forking a decode: every snapshot spawns an ffmpeg,
/// and a stop is the one time this process is trying to get every child it
/// already has to exit. The state transition still goes out — that is the
/// message that clears Home Assistant's motion sensor, and losing it is the
/// whole reason the bridge stayed.
#[tokio::test]
async fn a_transition_during_the_drain_is_published_but_forks_no_snapshot() {
    use crate::buffer::{GopSegment, HotBuffer};

    let buffer = HotBuffer::new("yard".to_string(), 30);
    buffer.write_recover().push(GopSegment {
        start_pts: 0,
        duration_ns: 1_000_000_000,
        data: Arc::new(vec![0x47; 188]),
        frame_count: 1,
    });
    let mut ctx = bridge_context(&["yard"], &[]);
    ctx.buffers = Arc::new(HashMap::from([("yard".to_string(), buffer)]));

    let topics = Topics::new(&ctx.config);
    let (client, _loop) = unpolled_client(capacity_for(&ctx));
    let mut state = state();
    let mut link = Link {
        connected: true,
        republish_pending: false,
        ..Link::default()
    };
    let mut tasks = HashMap::new();

    // Both spawn sites, because guarding one of two is the bug this fixes.
    let mut deliver = |event, state: &mut SensorState, tasks: &mut HashMap<_, _>| {
        handle_event(
            event, &client, &topics, state, &ctx, tasks, &mut link, true, // stopping
        );
    };

    deliver(
        MqttEvent::MotionStart {
            camera_id: "yard".to_string(),
        },
        &mut state,
        &mut tasks,
    );
    assert!(
        state.has_motion("yard"),
        "the ON transition was not applied"
    );
    assert!(tasks.is_empty(), "a MotionStart forked a snapshot decode");

    deliver(
        MqttEvent::MotionEnd {
            camera_id: "yard".to_string(),
        },
        &mut state,
        &mut tasks,
    );
    // Everything a live camon would publish for these transitions still
    // went out; only the forks were skipped.
    assert!(
        !state.has_motion("yard"),
        "the OFF transition was not applied"
    );
    assert!(
        tasks.is_empty(),
        "the drain's last MotionEnd forked a snapshot decode"
    );
}

/// The regression the phased stop nearly introduced. Phase 2 lets an
/// analyzer keep working for up to `TAIL_DRAIN_BOUND` past the flag, and
/// the last thing it does before exiting is flush its open run and send the
/// `MotionEnd` that clears Home Assistant's motion sensor. A bridge that
/// stopped receiving on the flag alone stopped one tick in, and that
/// transition went nowhere — Home Assistant would hold movement until camon
/// came back.
///
/// Real time, not paused: the bridge shares a runtime with an event loop
/// reconnecting to a broker that is not there, and a virtual clock in the
/// middle of that never settles enough to advance.
#[tokio::test]
async fn the_bridge_keeps_receiving_while_the_analyzers_are_still_draining() {
    let ctx = bridge_context(&["yard"], &[]);
    ctx.shutdown.store(true, Ordering::Relaxed); // a stop signal, just now
    let (tx, rx) = tokio::sync::mpsc::channel(1);
    let bridge = tokio::spawn(run_bridge(ctx, rx));

    // Past the tick the bridge used to stop on, and well short of the
    // bound a real analyzer has.
    tokio::time::sleep(Duration::from_millis(1_500)).await;
    let motion_end = MqttEvent::MotionEnd {
        camera_id: "yard".to_string(),
    };
    tx.send(motion_end.clone())
        .await
        .expect("the bridge dropped the channel while its analyzers were still draining");

    // Received, not merely queued behind a loop that has stopped reading.
    // The channel holds one, so this second send can only complete once
    // the bridge has taken the first — a loop that has broken out leaves
    // it here until the receiver is dropped, and then fails it.
    tokio::time::timeout(Duration::from_secs(5), tx.send(motion_end))
        .await
        .expect("the bridge stopped consuming while its analyzers were still draining")
        .expect("the bridge dropped the channel while its analyzers were still draining");

    // The analyzers exit; now the bridge has nothing left to serve.
    drop(tx);
    tokio::time::timeout(Duration::from_secs(10), bridge)
        .await
        .expect("the bridge did not stop once its producers were gone")
        .expect("bridge task panicked");
}

/// A poller that dies is a task death, not a broker outage — and the
/// difference is the whole reason the bridge is a fatal-policy task.
///
/// Outages are the eventloop task's own business: it retries, paced, for
/// ever, and the bridge just watches the connection edges go by. Its
/// *absence* is unrecoverable from in here, because `poll()` is not
/// cancellation-safe and was moved onto that task precisely so nothing
/// could race it — there is no way to poll it back from the bridge loop. A
/// bridge that carried on would tick for ever, publishing into a queue
/// nobody drains, with Home Assistant holding whatever it last heard and
/// camon looking perfectly healthy. So it ends, and the supervisor named it
/// mqtt-bridge on the way past.
///
/// Real time, not paused: a rumqttc event loop reconnecting to a broker
/// that is not there never lets a virtual clock settle.
#[tokio::test]
async fn a_dead_event_loop_ends_the_bridge_so_the_process_can_restart() {
    let ctx = bridge_context(&["yard"], &[]);
    let stopping = Arc::clone(&ctx.shutdown); // down: camon is running
    let stops = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let counter = Arc::clone(&stops);
    let raise = Arc::clone(&stopping);
    let supervisor = crate::supervise::Supervisor::new(
        Arc::clone(&stopping),
        Arc::new(tokio::sync::Notify::new()),
        move || {
            counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            raise.store(true, std::sync::atomic::Ordering::Relaxed);
        },
    );

    let (_tx, rx) = tokio::sync::mpsc::channel(1);
    let (probe_tx, probe_rx) = tokio::sync::oneshot::channel();
    let bridge = supervisor.critical(
        "mqtt-bridge",
        run_bridge_with(ctx, rx, move |raw| {
            let eventloop = Eventloop::spawn(raw);
            let _ = probe_tx.send(eventloop.task.abort_handle());
            eventloop
        }),
    );

    // The poller dies the way a panic would leave it: gone, with the
    // bridge's end of the channel still open.
    probe_rx
        .await
        .expect("the bridge never built a poller")
        .abort();

    tokio::time::timeout(Duration::from_secs(10), bridge)
        .await
        .expect("the bridge ticked on for ever with nothing polling it")
        .expect("bridge task panicked");

    assert_eq!(
        supervisor.deaths(),
        vec!["mqtt-bridge (returned)".to_string()],
        "a dead poller left the bridge looking healthy"
    );
    assert_eq!(
        stops.load(std::sync::atomic::Ordering::Relaxed),
        1,
        "nothing asked for the drain"
    );
}

/// The same event during a stop is not that. The poller is *supposed* to go
/// away then — `Eventloop::stop` is how the bridge ends it — and the bridge
/// owes the drain its last transitions and the retained `offline` marker
/// before it leaves. So it keeps serving until its producers are gone, and
/// nothing is reported.
#[tokio::test]
async fn a_poller_that_ends_during_a_stop_is_not_a_death() {
    let ctx = bridge_context(&["yard"], &[]);
    ctx.shutdown
        .store(true, std::sync::atomic::Ordering::Relaxed); // a SIGTERM, just now
    let supervisor = crate::supervise::Supervisor::new(
        Arc::clone(&ctx.shutdown),
        Arc::new(tokio::sync::Notify::new()),
        || panic!("a deliberate stop asked for another drain"),
    );

    let (tx, rx) = tokio::sync::mpsc::channel(1);
    let (probe_tx, probe_rx) = tokio::sync::oneshot::channel();
    let bridge = supervisor.critical(
        "mqtt-bridge",
        run_bridge_with(ctx, rx, move |raw| {
            let eventloop = Eventloop::spawn(raw);
            let _ = probe_tx.send(eventloop.task.abort_handle());
            eventloop
        }),
    );
    probe_rx
        .await
        .expect("the bridge never built a poller")
        .abort();

    // Still receiving: an analyzer draining through phase 2 has a MotionEnd
    // to deliver, and a bridge that had left would drop it. Sent twice
    // against a channel that holds one, because a send into the buffer of a
    // receiver nobody is reading still succeeds — only the second can
    // complete, and only once the bridge has taken the first.
    let motion_end = MqttEvent::MotionEnd {
        camera_id: "yard".to_string(),
    };
    for _ in 0..2 {
        tokio::time::timeout(Duration::from_secs(5), tx.send(motion_end.clone()))
            .await
            .expect("the bridge stopped consuming when its poller went away")
            .expect("the bridge dropped the channel when its poller went away");
    }

    drop(tx);
    tokio::time::timeout(Duration::from_secs(10), bridge)
        .await
        .expect("the bridge did not stop once its producers were gone")
        .expect("bridge task panicked");
    assert!(
        supervisor.deaths().is_empty(),
        "a deliberate stop was reported as a failure: {:?}",
        supervisor.deaths()
    );
}

/// Under clean sessions the broker keeps no state under the id — what
/// stability buys is continuity in logs, ACLs and the broker's client
/// list, not session custody (see [`derive_client_id`]). It is still a
/// property worth pinning: the same instance must derive the same id for
/// as long as it is the same instance — including across the config
/// changes it lives through.
#[test]
fn one_instance_derives_the_same_client_id_every_start() {
    let path = PathBuf::from("/var/lib/camon/mqtt_entities.json");
    // Pinned, not merely self-consistent: this is the identity the broker
    // remembers between restarts, and a build that quietly computed a
    // different one would be a build whose sessions all look new.
    assert_eq!(
        derive_client_id("nvr", Some(&path)),
        "camon-c8623bcc0559",
        "the client id derivation moved; every deployment's session did too"
    );
    assert_eq!(
        derive_client_id("nvr", Some(&path)),
        derive_client_id("nvr", Some(&path))
    );

    // What the operator changes day to day is the camera list, and none of
    // it reaches the id.
    let mut ctx = bridge_context(&["yard"], &["person"]);
    ctx.entities_path = Some(path.clone());
    let mut later = bridge_context(&["yard", "gate", "front"], &[]);
    later.entities_path = Some(path);
    later.config.snapshot_interval_secs += 7;
    assert_eq!(client_id(&ctx), client_id(&later));

    // Short enough that every MQTT 3.1.1 broker must accept it, and made
    // only of characters they all take.
    let id = client_id(&ctx);
    assert!(id.len() <= 23, "{id} is longer than 23 characters");
    assert!(id.starts_with("camon-"));
    assert!(id[6..]
        .chars()
        .all(|c| c.is_ascii_hexdigit() && !c.is_uppercase()));
}

/// The failure this replaced a fixed id for: a broker disconnects the older
/// session the moment a second one claims the same id, so two camons
/// sharing one would evict each other for ever and neither would finish a
/// reconnect burst.
#[test]
fn two_camons_against_one_broker_do_not_share_a_client_id() {
    let here = PathBuf::from("/var/lib/camon/mqtt_entities.json");
    let there = PathBuf::from("/srv/camon-garage/mqtt_entities.json");

    // Two instances on one host: they cannot share a data dir, each owning
    // its hot buffer state and entity record in it.
    assert_ne!(
        derive_client_id("nvr", Some(&here)),
        derive_client_id("nvr", Some(&there))
    );
    // Two hosts publishing to one broker: same package, same data dir path,
    // different machines.
    assert_ne!(
        derive_client_id("nvr", Some(&here)),
        derive_client_id("shed", Some(&here))
    );
    // Neither dimension may be swallowed by the other.
    assert_ne!(
        derive_client_id("nvr", Some(&there)),
        derive_client_id("shed", Some(&here))
    );
    // A bridge with no entity record at all still answers, and still
    // answers for the host it runs on.
    assert_ne!(
        derive_client_id("nvr", None),
        derive_client_id("shed", None)
    );
    assert_ne!(
        derive_client_id("nvr", None),
        derive_client_id("nvr", Some(&here))
    );

    // And this machine has a name to derive from in the first place.
    assert!(hostname().is_some_and(|name| !name.is_empty()));
}

#[test]
fn the_image_budget_bounds_what_one_tick_may_queue() {
    // Pinned: the whole point of the bound is the arithmetic in its doc
    // comment, and a value edited without that arithmetic is a bound that
    // says something else.
    assert_eq!(MAX_IMAGE_BYTES_PER_TICK, 16 * 1024 * 1024);

    let budget = ImageBudget::default();
    let half = MAX_IMAGE_BYTES_PER_TICK / 2;
    assert!(budget.take(half));
    // Exactly at the bound is inside it...
    assert!(budget.take(half));
    // ...and a single byte past it is not, however small the image.
    assert!(!budget.take(1));

    // A refusal spends nothing: a 4 MiB snapshot refused must not stop the
    // next tick from taking a whole window's worth.
    budget.refill();
    assert!(budget.take(MAX_IMAGE_BYTES_PER_TICK));
    assert!(!budget.take(1));
    budget.refill();
    assert!(budget.take(MAX_IMAGE_BYTES_PER_TICK));
}

/// The bound is only a bound while something refills it, and the tick is
/// what does — one window per second, which is what the memory arithmetic
/// multiplies by.
#[tokio::test]
async fn the_tick_opens_a_new_image_window() {
    let topics = Topics::new(&MqttConfig::default());
    let ctx = bridge_context(&["yard"], &[]);
    let (client, _eventloop) = unpolled_client(capacity_for(&ctx));
    let mut link = Link {
        connected: true,
        republish_pending: false,
        ..Link::default()
    };
    assert!(link.images.take(MAX_IMAGE_BYTES_PER_TICK));
    assert!(!link.images.take(1));

    on_tick(
        &client,
        &topics,
        &mut state(),
        &ctx,
        &mut HashMap::new(),
        &mut link,
        &mut no_memory(&ctx),
        false,
    );
    assert!(link.images.take(MAX_IMAGE_BYTES_PER_TICK));
}

/// A spent budget drops the image and nothing else: the states around it
/// still go out, because they are bytes the broker cannot hold in its
/// hands and the bound exists for the ones it can.
#[tokio::test]
async fn a_sighting_crop_past_the_budget_is_dropped_not_queued() {
    let topics = Topics::new(&MqttConfig::default());
    let ctx = bridge_context(&["yard"], &["person"]);
    let sighting = || MqttEvent::Detections {
        camera_id: "yard".to_string(),
        sightings: vec![Sighting {
            class: "person".to_string(),
            frame_jpeg: Some(vec![0u8; 4096]),
        }],
    };

    // Two slots: the occupancy ON state takes one, and whether the crop
    // takes the other is the whole question.
    let (client, _eventloop) = unpolled_client(2);
    let mut link = Link {
        connected: true,
        republish_pending: false,
        ..Link::default()
    };
    assert!(link.images.take(MAX_IMAGE_BYTES_PER_TICK));
    handle_event(
        sighting(),
        &client,
        &topics,
        &mut state(),
        &ctx,
        &mut HashMap::new(),
        &mut link,
        false,
    );
    assert_eq!(publish_state(&client, "camon/filler", "x"), Published::Yes);

    // The control: with the window open the crop is queued, and that same
    // slot is gone.
    let (client, _eventloop) = unpolled_client(2);
    link.images.refill();
    handle_event(
        sighting(),
        &client,
        &topics,
        &mut state(),
        &ctx,
        &mut HashMap::new(),
        &mut link,
        false,
    );
    assert_eq!(
        publish_state(&client, "camon/filler", "x"),
        Published::QueueFull
    );
}

/// The cadence paces pictures, not attempts. A camera whose decodes all
/// fail publishes nothing at all, so counting a failure as a snapshot buys
/// the operator a full interval of silence for every failure — and then
/// another one.
#[test]
fn a_decode_that_produced_nothing_does_not_spend_the_cadence() {
    let mut s = state();
    let t0 = Instant::now();
    s.motion_start("yard");
    assert_eq!(s.due_snapshots(t0), vec!["yard".to_string()]);

    // The attempt was stamped, and the failure takes it back — but only as
    // far as the retry delay, so a camera failing for good forks one
    // ffmpeg every couple of seconds rather than every tick.
    assert!(s.note_snapshot_failed("yard", t0));
    assert!(s
        .due_snapshots(t0 + SNAPSHOT_RETRY_DELAY - Duration::from_millis(1))
        .is_empty());
    let t1 = t0 + SNAPSHOT_RETRY_DELAY;
    assert_eq!(s.due_snapshots(t1), vec!["yard".to_string()]);

    // The other branch: an attempt that produced a frame keeps the whole
    // interval, or the cadence would mean nothing.
    s.note_snapshot_decoded("yard");
    assert!(s
        .due_snapshots(t1 + INTERVAL - Duration::from_millis(1))
        .is_empty());
    assert_eq!(s.due_snapshots(t1 + INTERVAL), vec!["yard".to_string()]);
}

/// Silence is the failure nobody notices, so it is said out loud — once
/// per run of failures rather than once per retry, and again when it ends.
#[test]
fn a_failing_camera_is_reported_once_and_its_recovery_too() {
    let mut s = state();
    let t0 = Instant::now();
    s.motion_start("yard");
    assert!(s.note_snapshot_failed("yard", t0));
    assert!(!s.note_snapshot_failed("yard", t0 + INTERVAL));
    assert!(s.note_snapshot_decoded("yard"));
    assert!(!s.note_snapshot_decoded("yard"));
    // A new run of failures is news again.
    assert!(s.note_snapshot_failed("yard", t0 + INTERVAL * 2));

    // A failure reported after the run closed must not put the camera back
    // on a schedule `motion_end` just took it off: nothing is snapshotted
    // between runs.
    s.motion_end("yard");
    s.note_snapshot_failed("yard", t0 + INTERVAL * 3);
    assert!(s.due_snapshots(t0 + INTERVAL * 100).is_empty());
}

/// A decode that has ended, as the tick finds it, tagged with the run it
/// was started for.
async fn ended_snapshot(decoded: bool, run: u64) -> SnapshotTask {
    let task = SnapshotTask {
        handle: tokio::spawn(async {}),
        decoded: Arc::new(AtomicBool::new(decoded)),
        run,
    };
    while !task.handle.is_finished() {
        tokio::task::yield_now().await;
    }
    task
}

/// The outcome has to travel from the detached decode back to the cadence,
/// and the tick retiring finished tasks is the whole of that path.
#[tokio::test]
async fn the_tick_folds_a_decode_outcome_into_the_cadence() {
    let now = Instant::now();
    let mut s = state();
    s.motion_start("yard");
    s.motion_start("gate");
    s.due_snapshots(now);

    let mut tasks = HashMap::from([
        (
            "yard".to_string(),
            ended_snapshot(false, s.snapshot_run("yard")).await,
        ),
        (
            "gate".to_string(),
            ended_snapshot(true, s.snapshot_run("gate")).await,
        ),
    ]);
    retire_snapshots(&mut tasks, &mut s, now);
    assert!(tasks.is_empty(), "a finished decode was not retired");

    // The failed camera comes back around at the retry delay; the one that
    // published keeps the cadence it just spent.
    assert_eq!(
        s.due_snapshots(now + SNAPSHOT_RETRY_DELAY),
        vec!["yard".to_string()]
    );

    // A decode still running is left alone — its outcome is not in yet.
    let running = SnapshotTask {
        handle: tokio::spawn(async { tokio::time::sleep(Duration::from_secs(60)).await }),
        decoded: Arc::new(AtomicBool::new(false)),
        run: 0,
    };
    let mut tasks = HashMap::from([("front".to_string(), running)]);
    retire_snapshots(&mut tasks, &mut s, now);
    assert_eq!(tasks.len(), 1);
    tasks["front"].handle.abort();
}

/// A decode outlives the run that asked for it — fifteen seconds are
/// allowed for one, and a camera can close a run and open another well
/// inside that. The outcome belongs to the run it was started for: folded
/// into a later one it would shorten a cadence that never failed, or clear
/// a failure the new run is still having.
#[tokio::test]
async fn a_decode_that_outlived_its_run_does_not_touch_the_next_one() {
    let t0 = Instant::now();
    let mut s = state();
    s.motion_start("yard");
    let stale = s.snapshot_run("yard");
    s.due_snapshots(t0);

    // The run closes and the camera opens another before the decode ends.
    s.motion_end("yard");
    let t1 = t0 + Duration::from_millis(1);
    s.motion_start("yard");
    assert_ne!(
        s.snapshot_run("yard"),
        stale,
        "a new run is a new generation"
    );
    s.due_snapshots(t1);

    // Now the old decode reports, having produced nothing.
    let mut tasks = HashMap::from([("yard".to_string(), ended_snapshot(false, stale).await)]);
    retire_snapshots(&mut tasks, &mut s, t1);
    assert!(tasks.is_empty(), "a finished decode was not retired");

    // The new run keeps the cadence it was given...
    assert!(
        s.due_snapshots(t1 + SNAPSHOT_RETRY_DELAY).is_empty(),
        "a stale failure shortened the new run's cadence"
    );
    assert_eq!(s.due_snapshots(t1 + INTERVAL), vec!["yard".to_string()]);
    // ...and its own first failure is still news, rather than a repeat of
    // one the previous run had.
    assert!(s.note_snapshot_failed("yard", t1 + INTERVAL));

    // The same in the other direction: a stale success must not clear the
    // failure the live run is having.
    let mut tasks = HashMap::from([("yard".to_string(), ended_snapshot(true, stale).await)]);
    retire_snapshots(&mut tasks, &mut s, t1 + INTERVAL);
    assert!(
        !s.note_snapshot_failed("yard", t1 + INTERVAL),
        "a stale success cleared the live run's failure"
    );
}

/// `snapshot_interval_secs` is an operator's `u64`, and every use of it is
/// `now + interval`: adding a duration to an `Instant` panics when the
/// result is not representable. A nonsense interval reads as "effectively
/// never", which is a config to clamp and warn about, not to die on.
#[test]
fn an_absurd_snapshot_interval_is_clamped_rather_than_panicking() {
    let mut s = SensorState::new(Duration::from_secs(u64::MAX), HOLD);
    assert_eq!(s.snapshot_interval, MAX_SNAPSHOT_INTERVAL);

    let t0 = Instant::now();
    s.motion_start("yard");
    assert_eq!(s.due_snapshots(t0), vec!["yard".to_string()]);
    assert!(!s.due_snapshots(t0 + MAX_SNAPSHOT_INTERVAL).is_empty());
    // The retry path does its own arithmetic on the interval.
    s.note_snapshot_failed("yard", t0);
    assert!(!s.due_snapshots(t0 + SNAPSHOT_RETRY_DELAY).is_empty());

    // A sane interval is left exactly as configured.
    assert_eq!(SensorState::new(INTERVAL, HOLD).snapshot_interval, INTERVAL);
}

/// One GOP of MPEG-TS, straight out of ffmpeg's muxer — near enough what
/// the hot buffer holds. Muxed here rather than borrowed from the decoder's
/// fixtures, which live in a module this one cannot see. Needs an `ffmpeg`
/// binary, so only the `#[ignore]`d test below uses it.
fn recorded_segment() -> Vec<u8> {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("segment.ts");
    let status = std::process::Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "quiet",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=640x480:rate=25",
            "-t",
            "1",
            "-c:v",
            "libx264",
            "-g",
            "25",
            "-keyint_min",
            "25",
            "-sc_threshold",
            "0",
            "-f",
            "mpegts",
            path.to_str().unwrap(),
        ])
        .status()
        .expect("ffmpeg is on PATH for the ignored tests");
    assert!(status.success(), "ffmpeg could not mux a test segment");
    std::fs::read(&path).unwrap()
}

/// The budget site the memory arithmetic is actually about: the detached
/// decode, which holds the only copy of a few hundred KiB and is the thing
/// that can queue them fifteen at a time.
///
/// Two orderings matter and neither is visible from the pure-state tests.
/// The charge must happen before the publish, or the bound is decoration;
/// and the decode must be recorded as having produced a frame *before* the
/// budget is consulted, because a refused publish is a queue problem and
/// not a camera problem — treating it as a decode failure would shorten the
/// cadence of a perfectly healthy camera exactly when the queue is least
/// able to take another image, and warn the operator about the wrong fault.
///
/// Needs a real decode, so it is `#[ignore]`d like every other test that
/// forks ffmpeg.
#[tokio::test]
#[ignore]
async fn a_snapshot_charges_the_budget_before_it_publishes() {
    // The hot buffer's newest segment is not the one snapshotted — the
    // decode takes the one behind it, which is complete.
    let segment = Arc::new(recorded_segment());
    let buffer = HotBuffer::new("yard".to_string(), 60);
    {
        let mut buf = buffer.write_recover();
        for i in 0..2 {
            buf.push(crate::buffer::GopSegment {
                start_pts: i * 1_000_000_000,
                duration_ns: 1_000_000_000,
                data: Arc::clone(&segment),
                frame_count: 25,
            });
        }
    }
    let mut ctx = bridge_context(&["yard"], &[]);
    ctx.buffers = Arc::new(HashMap::from([("yard".to_string(), buffer)]));
    let topics = Topics::new(&ctx.config);

    let decode = |link: &Link, client: &AsyncClient| {
        let mut tasks = HashMap::new();
        spawn_snapshot(client, &topics, &ctx, "yard", 0, &mut tasks, link);
        tasks.remove("yard").expect("no decode was started")
    };

    // A window with room: the frame is queued, and the bytes are charged.
    let link = Link {
        connected: true,
        ..Link::default()
    };
    let (client, _eventloop) = unpolled_client(1);
    let task = decode(&link, &client);
    let decoded = Arc::clone(&task.decoded);
    task.handle.await.expect("the decode task panicked");
    assert!(
        decoded.load(Ordering::Relaxed),
        "the segment did not decode"
    );
    assert_eq!(
        publish_state(&client, "camon/filler", "x"),
        Published::QueueFull,
        "the snapshot never reached the request queue"
    );
    assert!(
        !link.images.take(MAX_IMAGE_BYTES_PER_TICK),
        "the published image was not charged against the budget"
    );

    // The same decode against a window that is already spent: the image is
    // dropped rather than queued...
    let link = Link {
        connected: true,
        ..Link::default()
    };
    assert!(link.images.take(MAX_IMAGE_BYTES_PER_TICK));
    let (client, _eventloop) = unpolled_client(1);
    let task = decode(&link, &client);
    let decoded = Arc::clone(&task.decoded);
    task.handle.await.expect("the decode task panicked");
    assert_eq!(
        publish_state(&client, "camon/filler", "x"),
        Published::Yes,
        "an image past the budget was queued anyway"
    );
    // ...and the camera is not blamed for it. This decode produced a frame;
    // that the queue could not have it is the queue's fault, and the
    // cadence must not hear about it as a failure.
    assert!(
        decoded.load(Ordering::Relaxed),
        "a refused publish was recorded as a decode that produced nothing"
    );
}

/// Two conditions, and they are not the same condition. That the queue took
/// a burst carrying the clears is one; that it took it on the session whose
/// `Disconnect` is now being proven is the other. A queued request lives
/// exactly as long as its connection: rumqttc moves what is left into its
/// pending set when the connection fails, and the next connect — camon's
/// sessions are clean — throws that set away unwritten.
#[test]
fn only_the_session_that_queued_the_clears_can_prove_them() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mqtt_entities.json");
    let topics = Topics::new(&MqttConfig::default());
    save_record(&path, &record("camon", &["yard", "gate"], &["person"])).unwrap();
    let mut ctx = bridge_context(&["yard"], &["person"]);
    ctx.entities_path = Some(path.clone());

    let mut memory = EntityMemory::load(&topics, &ctx);
    assert_eq!(memory.orphans.len(), 8);
    memory.note_burst_accepted();
    assert_eq!(load_record(&path).unwrap().pending_clears, memory.orphans);

    // The connection that took them is gone, so what it was holding is
    // gone with it. A `Disconnect` written on the *next* connection proves
    // nothing about the burst queued on the last one.
    memory.note_session_lost();
    memory.note_clears_flushed();
    assert_eq!(
        load_record(&path).unwrap().pending_clears,
        memory.orphans,
        "clears were forgotten on the strength of another session's disconnect"
    );

    // The new session re-queues them — every reconnect burst carries them —
    // and now the disconnect is evidence about the queue that holds them.
    memory.note_burst_accepted();
    memory.note_clears_flushed();
    assert!(load_record(&path).unwrap().pending_clears.is_empty());
    assert!(EntityMemory::load(&topics, &ctx).orphans.is_empty());
}

/// The same distinction where it actually bites, through the bridge: a
/// broker that drops out just before the stop leaves camon queueing its
/// `offline` marker and its `Disconnect` while down, the poller reconnects
/// and writes exactly those two, and the clears queued on the connection
/// before them were discarded by that reconnect. A bridge that took the
/// disconnect for proof would record them as done and never clear those
/// topics again — a Home Assistant entity that outlives every restart.
#[tokio::test]
async fn a_reconnect_before_the_stop_leaves_the_clears_owed() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mqtt_entities.json");
    save_record(&path, &record("camon", &["yard", "gate"], &["person"])).unwrap();
    let mut ctx = bridge_context(&["yard"], &["person"]);
    ctx.entities_path = Some(path.clone());
    let shutdown = Arc::clone(&ctx.shutdown);

    // A poller the test speaks for. The raw event loop comes back here
    // rather than being dropped: dropping it closes the request queue, and
    // every publish would then fail for a reason no live bridge has. Held
    // by the test, it is also how the test sees what the bridge has queued.
    let (link_tx, link_rx) = tokio::sync::mpsc::channel(1);
    let (mqtt_tx, mqtt_rx) = tokio::sync::mpsc::channel(1);
    let (raw_tx, raw_rx) = tokio::sync::oneshot::channel();
    let bridge = tokio::spawn(run_bridge_with(ctx, mqtt_rx, move |raw| {
        let _ = raw_tx.send(raw);
        Eventloop {
            events: link_rx,
            stop: Arc::new(tokio::sync::Notify::new()),
            task: tokio::spawn(std::future::pending()),
        }
    }));
    let mut raw = raw_rx.await.expect("the bridge never built a poller");

    // Connected: the burst goes out, clears at its head, and the record
    // says so.
    link_tx.send(LinkEvent::Connected).await.unwrap();
    let mut queued = Vec::new();
    for _ in 0..100 {
        queued = load_record(&path).unwrap().pending_clears;
        if !queued.is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(
        queued.len(),
        8,
        "the reconnect burst never queued the clears"
    );

    // The broker drops out. The channel holds one, so the second send can
    // only complete once the bridge has taken the first — the edge is
    // observed, not merely posted. `DisconnectSent` is inert in the loop.
    link_tx.send(LinkEvent::Disconnected).await.unwrap();
    link_tx.send(LinkEvent::DisconnectSent).await.unwrap();

    // Now camon stops. The marker and the disconnect are queued while the
    // link is down, and the reconnect that writes them is the session that
    // reports this.
    shutdown.store(true, Ordering::Relaxed);
    drop(mqtt_tx);

    // The `DisconnectSent` below has to be the one the *flush wait* reads,
    // not one the loop swallows on its way out — a loop that ate it would
    // leave the flush to time out, and a test whose gate is never reached
    // passes for the wrong reason. So it goes out only once the bridge's
    // own `Disconnect` request is in the queue, which happens after the
    // loop has ended and immediately before the wait begins. `clean` is
    // what rumqttc itself does with a queue it is taking over.
    let queued_disconnect = tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            raw.clean();
            if raw
                .pending
                .iter()
                .any(|request| matches!(request, rumqttc::Request::Disconnect(_)))
            {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    });
    queued_disconnect
        .await
        .expect("the bridge never asked to disconnect");

    let flushing = std::time::Instant::now();
    link_tx.send(LinkEvent::DisconnectSent).await.unwrap();
    tokio::time::timeout(Duration::from_secs(10), bridge)
        .await
        .expect("the bridge did not stop once its producers were gone")
        .expect("bridge task panicked");
    // Proof that the disconnect was taken as written rather than waited
    // out: a flush that timed out would have sat here for SHUTDOWN_FLUSH,
    // and the record below would then be unchanged for a reason that has
    // nothing to do with which session queued the clears.
    assert!(
        flushing.elapsed() < SHUTDOWN_FLUSH,
        "the flush wait never saw the disconnect, so nothing consulted the clears"
    );

    assert_eq!(
        load_record(&path).unwrap().pending_clears,
        queued,
        "a disconnect from a later session was taken as proof the clears went out"
    );
}

#[tokio::test]
async fn send_event_drops_instead_of_blocking() {
    let (tx, mut rx) = tokio::sync::mpsc::channel(1);
    assert!(send_event(
        &tx,
        MqttEvent::MotionStart {
            camera_id: "yard".to_string()
        }
    ));
    // Queue full: the producer moves on rather than stalling.
    assert!(!send_event(
        &tx,
        MqttEvent::MotionEnd {
            camera_id: "yard".to_string()
        }
    ));
    assert_eq!(
        rx.recv().await.unwrap(),
        MqttEvent::MotionStart {
            camera_id: "yard".to_string()
        }
    );

    drop(rx);
    assert!(!send_event(
        &tx,
        MqttEvent::MotionEnd {
            camera_id: "yard".to_string()
        }
    ));
}
