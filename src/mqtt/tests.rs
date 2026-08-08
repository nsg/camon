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

    assert!(s.record_sighting("yard", "person", t0));
    assert!(s.is_occupied("yard", "person"));

    assert!(s
        .expire_occupancy(t0 + HOLD - Duration::from_secs(1))
        .is_empty());
    assert!(s.is_occupied("yard", "person"));

    let expired = s.expire_occupancy(t0 + HOLD);
    assert_eq!(expired, vec![("yard".to_string(), "person".to_string())]);
    assert!(!s.is_occupied("yard", "person"));
}

#[test]
fn new_sighting_extends_the_hold() {
    let mut s = state();
    let t0 = Instant::now();
    s.record_sighting("yard", "person", t0);

    let later = t0 + HOLD - Duration::from_secs(1);
    assert!(!s.record_sighting("yard", "person", later));
    assert!(s.expire_occupancy(t0 + HOLD).is_empty());
    assert!(s.is_occupied("yard", "person"));

    assert_eq!(s.expire_occupancy(later + HOLD).len(), 1);
    assert!(!s.is_occupied("yard", "person"));
}

#[test]
fn occupancy_re_arms_after_expiring() {
    let mut s = state();
    let t0 = Instant::now();
    s.record_sighting("yard", "person", t0);
    s.expire_occupancy(t0 + HOLD);
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
    assert!(s.due_snapshots(t0).is_empty());

    assert!(s.motion_start("yard"));
    assert_eq!(s.due_snapshots(t0), vec!["yard".to_string()]);
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

fn snapshot_task(handle: tokio::task::JoinHandle<()>) -> SnapshotTask {
    SnapshotTask {
        handle,
        decoded: Arc::new(AtomicBool::new(false)),
        run: 0,
    }
}

fn no_memory(ctx: &BridgeContext) -> EntityMemory {
    EntityMemory::load(&Topics::new(&ctx.config), ctx)
}

fn announced_for(ctx: &BridgeContext) -> EntityRecord {
    EntityRecord::current(None, &Topics::new(&ctx.config), ctx)
}

fn capacity_for(ctx: &BridgeContext) -> usize {
    request_queue_capacity(ctx.camera_ids.len(), ctx.classes.len(), 0)
}

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
    let (topic, payload) = burst.last().unwrap();
    assert_eq!(topic, "camon/availability");
    assert_eq!(payload, b"online");

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

    s.motion_end("yard");
    assert_eq!(
        motion(reconnect_burst(&topics, &s, &announced_for(&ctx), &[])),
        b"OFF"
    );
}

#[test]
fn the_queue_always_has_room_for_a_whole_burst() {
    let topics = Topics::new(&MqttConfig::default());
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

        let burst = reconnect_burst(&topics, &state(), &announced_for(&ctx), &orphan_topics);
        assert_eq!(burst.len(), burst_len(cameras, classes, orphans));

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

    let (client, mut eventloop) = unpolled_client(burst.len());
    assert_eq!(publish_state(&client, "camon/filler", "x"), Published::Yes);
    assert!(!publish_burst(&client, burst.clone()));

    eventloop.clean();
    assert!(publish_burst(&client, burst));
}

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

    let mut ctx = bridge_context(&["yard"], &[]);
    ctx.entities_path = Some(path.clone());
    let memory = EntityMemory::load(&topics, &ctx);
    assert_eq!(memory.announced.classes, vec!["person".to_string()]);
    assert!(memory.orphans.is_empty());

    let burst = reconnect_burst(&topics, &state(), &memory.announced, &memory.orphans);
    assert!(burst.iter().any(
        |(topic, _)| topic == "homeassistant/binary_sensor/camon_yard_occupancy_person/config"
    ));
    assert!(burst
        .iter()
        .any(|(topic, payload)| topic == "camon/yard/occupancy/person" && payload == b"OFF"));

    let (client, _eventloop) = unpolled_client(request_queue_capacity(
        memory.announced.cameras.len(),
        memory.announced.classes.len(),
        memory.orphans.len(),
    ));
    assert!(publish_burst(&client, burst));

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
    assert!(orphans.contains(&"camon/availability".to_string()));
    assert!(orphans.iter().all(|topic| !topic.starts_with("nvr/")));

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
    save_record(&path, &previous).unwrap();
    assert_eq!(EntityMemory::load(&topics, &ctx).orphans.len(), 8);

    std::fs::remove_file(&path).unwrap();
    let memory = EntityMemory::load(&topics, &ctx);
    assert!(memory.orphans.is_empty());
    assert!(memory.on_disk.is_none());

    std::fs::write(&path, b"{ not json").unwrap();
    assert!(EntityMemory::load(&topics, &ctx).orphans.is_empty());

    let mut json = serde_json::to_value(&previous).unwrap();
    json["entities"] = serde_json::json!(["camon_gate_motion"]);
    std::fs::write(&path, serde_json::to_vec(&json).unwrap()).unwrap();
    assert!(EntityMemory::load(&topics, &ctx).orphans.is_empty());

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

    let topics = Topics::new(&MqttConfig::default());
    let mut ctx = bridge_context(&["yard"], &[]);
    ctx.entities_path = Some(path);
    let memory = EntityMemory::load(&topics, &ctx);
    assert!(memory.orphans.is_empty());
    assert_eq!(memory.announced.broker, "localhost:1883");
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

    assert_eq!(EntityMemory::load(&topics, &ctx).orphans, memory.orphans);

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
    let blocked = dir.path().join("blocked");
    std::fs::write(&blocked, b"not a directory").unwrap();
    let path = blocked.join("mqtt_entities.json");

    let topics = Topics::new(&MqttConfig::default());
    let mut ctx = bridge_context(&["yard"], &["person"]);
    ctx.entities_path = Some(path.clone());
    let mut memory = EntityMemory::load(&topics, &ctx);

    memory.note_burst_accepted();
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

    assert!(link.republish_pending);
}

#[test]
fn a_topic_that_can_never_be_published_stalls_nothing() {
    let topics = Topics::new(&MqttConfig::default());
    let ctx = bridge_context(&["ya+rd"], &[]);
    let burst = reconnect_burst(&topics, &state(), &announced_for(&ctx), &[]);
    let (client, _eventloop) = unpolled_client(capacity_for(&ctx));
    assert!(publish_burst(&client, burst));

    assert_eq!(
        publish_state(&client, "camon/ya+rd/motion", "OFF"),
        Published::ImpossibleTopic
    );
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
    assert!(!publish_burst(
        &client,
        reconnect_burst(&topics, &state(), &announced_for(&ctx), &[])
    ));
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
    assert!(tasks.is_empty());

    link.connected = true;
    spawn_snapshot(&client, &topics, &ctx, "yard", 0, &mut tasks, &link);
    assert_eq!(tasks.len(), 1);
    for task in tasks.values() {
        task.handle.abort();
    }
}

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

    assert_eq!(events.recv().await, Some(LinkEvent::Disconnected));

    stop.notify_one();
    let ended = tokio::time::timeout(Duration::from_secs(1), task).await;
    assert!(
        matches!(ended, Ok(Ok(()))),
        "the reconnect delay outlived the stop signal"
    );
    accepts.abort();
}

#[test]
fn the_bridge_stops_only_once_its_producers_have_gone() {
    let stopping = AtomicBool::new(true);
    assert!(
        !bridge_is_done(false, &stopping),
        "the bridge stopped while its analyzers were still draining"
    );
    assert!(bridge_is_done(true, &stopping));
    assert!(!bridge_is_done(true, &AtomicBool::new(false)));
}

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
    assert!(
        !state.has_motion("yard"),
        "the OFF transition was not applied"
    );
    assert!(
        tasks.is_empty(),
        "the drain's last MotionEnd forked a snapshot decode"
    );
}

#[tokio::test]
async fn the_bridge_keeps_receiving_while_the_analyzers_are_still_draining() {
    let ctx = bridge_context(&["yard"], &[]);
    ctx.shutdown.store(true, Ordering::Relaxed); // a stop signal, just now
    let (tx, rx) = tokio::sync::mpsc::channel(1);
    let bridge = tokio::spawn(run_bridge(ctx, rx));

    tokio::time::sleep(Duration::from_millis(1_500)).await;
    let motion_end = MqttEvent::MotionEnd {
        camera_id: "yard".to_string(),
    };
    tx.send(motion_end.clone())
        .await
        .expect("the bridge dropped the channel while its analyzers were still draining");

    tokio::time::timeout(Duration::from_secs(5), tx.send(motion_end))
        .await
        .expect("the bridge stopped consuming while its analyzers were still draining")
        .expect("the bridge dropped the channel while its analyzers were still draining");

    drop(tx);
    tokio::time::timeout(Duration::from_secs(10), bridge)
        .await
        .expect("the bridge did not stop once its producers were gone")
        .expect("bridge task panicked");
}

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

#[test]
fn one_instance_derives_the_same_client_id_every_start() {
    let path = PathBuf::from("/var/lib/camon/mqtt_entities.json");
    assert_eq!(
        derive_client_id("nvr", Some(&path)),
        "camon-c8623bcc0559",
        "the client id derivation moved; every deployment's session did too"
    );
    assert_eq!(
        derive_client_id("nvr", Some(&path)),
        derive_client_id("nvr", Some(&path))
    );

    let mut ctx = bridge_context(&["yard"], &["person"]);
    ctx.entities_path = Some(path.clone());
    let mut later = bridge_context(&["yard", "gate", "front"], &[]);
    later.entities_path = Some(path);
    later.config.snapshot_interval_secs += 7;
    assert_eq!(client_id(&ctx), client_id(&later));

    let id = client_id(&ctx);
    assert!(id.len() <= 23, "{id} is longer than 23 characters");
    assert!(id.starts_with("camon-"));
    assert!(id[6..]
        .chars()
        .all(|c| c.is_ascii_hexdigit() && !c.is_uppercase()));
}

#[test]
fn two_camons_against_one_broker_do_not_share_a_client_id() {
    let here = PathBuf::from("/var/lib/camon/mqtt_entities.json");
    let there = PathBuf::from("/srv/camon-garage/mqtt_entities.json");

    assert_ne!(
        derive_client_id("nvr", Some(&here)),
        derive_client_id("nvr", Some(&there))
    );
    assert_ne!(
        derive_client_id("nvr", Some(&here)),
        derive_client_id("shed", Some(&here))
    );
    assert_ne!(
        derive_client_id("nvr", Some(&there)),
        derive_client_id("shed", Some(&here))
    );
    assert_ne!(
        derive_client_id("nvr", None),
        derive_client_id("shed", None)
    );
    assert_ne!(
        derive_client_id("nvr", None),
        derive_client_id("nvr", Some(&here))
    );

    assert!(hostname().is_some_and(|name| !name.is_empty()));
}

#[test]
fn the_image_budget_bounds_what_one_tick_may_queue() {
    assert_eq!(MAX_IMAGE_BYTES_PER_TICK, 16 * 1024 * 1024);

    let budget = ImageBudget::default();
    let half = MAX_IMAGE_BYTES_PER_TICK / 2;
    assert!(budget.take(half));
    assert!(budget.take(half));
    assert!(!budget.take(1));

    budget.refill();
    assert!(budget.take(MAX_IMAGE_BYTES_PER_TICK));
    assert!(!budget.take(1));
    budget.refill();
    assert!(budget.take(MAX_IMAGE_BYTES_PER_TICK));
}

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

#[test]
fn a_decode_that_produced_nothing_does_not_spend_the_cadence() {
    let mut s = state();
    let t0 = Instant::now();
    s.motion_start("yard");
    assert_eq!(s.due_snapshots(t0), vec!["yard".to_string()]);

    assert!(s.note_snapshot_failed("yard", t0));
    assert!(s
        .due_snapshots(t0 + SNAPSHOT_RETRY_DELAY - Duration::from_millis(1))
        .is_empty());
    let t1 = t0 + SNAPSHOT_RETRY_DELAY;
    assert_eq!(s.due_snapshots(t1), vec!["yard".to_string()]);

    s.note_snapshot_decoded("yard");
    assert!(s
        .due_snapshots(t1 + INTERVAL - Duration::from_millis(1))
        .is_empty());
    assert_eq!(s.due_snapshots(t1 + INTERVAL), vec!["yard".to_string()]);
}

#[test]
fn a_failing_camera_is_reported_once_and_its_recovery_too() {
    let mut s = state();
    let t0 = Instant::now();
    s.motion_start("yard");
    assert!(s.note_snapshot_failed("yard", t0));
    assert!(!s.note_snapshot_failed("yard", t0 + INTERVAL));
    assert!(s.note_snapshot_decoded("yard"));
    assert!(!s.note_snapshot_decoded("yard"));
    assert!(s.note_snapshot_failed("yard", t0 + INTERVAL * 2));

    s.motion_end("yard");
    s.note_snapshot_failed("yard", t0 + INTERVAL * 3);
    assert!(s.due_snapshots(t0 + INTERVAL * 100).is_empty());
}

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

    assert_eq!(
        s.due_snapshots(now + SNAPSHOT_RETRY_DELAY),
        vec!["yard".to_string()]
    );

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

#[tokio::test]
async fn a_decode_that_outlived_its_run_does_not_touch_the_next_one() {
    let t0 = Instant::now();
    let mut s = state();
    s.motion_start("yard");
    let stale = s.snapshot_run("yard");
    s.due_snapshots(t0);

    s.motion_end("yard");
    let t1 = t0 + Duration::from_millis(1);
    s.motion_start("yard");
    assert_ne!(
        s.snapshot_run("yard"),
        stale,
        "a new run is a new generation"
    );
    s.due_snapshots(t1);

    let mut tasks = HashMap::from([("yard".to_string(), ended_snapshot(false, stale).await)]);
    retire_snapshots(&mut tasks, &mut s, t1);
    assert!(tasks.is_empty(), "a finished decode was not retired");

    assert!(
        s.due_snapshots(t1 + SNAPSHOT_RETRY_DELAY).is_empty(),
        "a stale failure shortened the new run's cadence"
    );
    assert_eq!(s.due_snapshots(t1 + INTERVAL), vec!["yard".to_string()]);
    assert!(s.note_snapshot_failed("yard", t1 + INTERVAL));

    let mut tasks = HashMap::from([("yard".to_string(), ended_snapshot(true, stale).await)]);
    retire_snapshots(&mut tasks, &mut s, t1 + INTERVAL);
    assert!(
        !s.note_snapshot_failed("yard", t1 + INTERVAL),
        "a stale success cleared the live run's failure"
    );
}

#[test]
fn an_absurd_snapshot_interval_is_clamped_rather_than_panicking() {
    let mut s = SensorState::new(Duration::from_secs(u64::MAX), HOLD);
    assert_eq!(s.snapshot_interval, MAX_SNAPSHOT_INTERVAL);

    let t0 = Instant::now();
    s.motion_start("yard");
    assert_eq!(s.due_snapshots(t0), vec!["yard".to_string()]);
    assert!(!s.due_snapshots(t0 + MAX_SNAPSHOT_INTERVAL).is_empty());
    s.note_snapshot_failed("yard", t0);
    assert!(!s.due_snapshots(t0 + SNAPSHOT_RETRY_DELAY).is_empty());

    assert_eq!(SensorState::new(INTERVAL, HOLD).snapshot_interval, INTERVAL);
}

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

#[tokio::test]
#[ignore]
async fn a_snapshot_charges_the_budget_before_it_publishes() {
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
    assert!(
        decoded.load(Ordering::Relaxed),
        "a refused publish was recorded as a decode that produced nothing"
    );
}

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

    memory.note_session_lost();
    memory.note_clears_flushed();
    assert_eq!(
        load_record(&path).unwrap().pending_clears,
        memory.orphans,
        "clears were forgotten on the strength of another session's disconnect"
    );

    memory.note_burst_accepted();
    memory.note_clears_flushed();
    assert!(load_record(&path).unwrap().pending_clears.is_empty());
    assert!(EntityMemory::load(&topics, &ctx).orphans.is_empty());
}

#[tokio::test]
async fn a_reconnect_before_the_stop_leaves_the_clears_owed() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mqtt_entities.json");
    save_record(&path, &record("camon", &["yard", "gate"], &["person"])).unwrap();
    let mut ctx = bridge_context(&["yard"], &["person"]);
    ctx.entities_path = Some(path.clone());
    let shutdown = Arc::clone(&ctx.shutdown);

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

    link_tx.send(LinkEvent::Disconnected).await.unwrap();
    link_tx.send(LinkEvent::DisconnectSent).await.unwrap();

    shutdown.store(true, Ordering::Relaxed);
    drop(mqtt_tx);

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
