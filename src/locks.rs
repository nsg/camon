use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

/// Extension methods for `RwLock` that recover the guard on poison instead of panicking or
/// silently skipping the work.
pub trait LockExt<T: ?Sized> {
    fn read_recover(&self) -> RwLockReadGuard<'_, T>;
    fn write_recover(&self) -> RwLockWriteGuard<'_, T>;
}

impl<T: ?Sized> LockExt<T> for RwLock<T> {
    fn read_recover(&self) -> RwLockReadGuard<'_, T> {
        self.read().unwrap_or_else(PoisonError::into_inner)
    }

    fn write_recover(&self) -> RwLockWriteGuard<'_, T> {
        self.write().unwrap_or_else(PoisonError::into_inner)
    }
}

/// The same recovery-on-poison treatment for `Mutex`, for the same reason.
pub trait MutexExt<T: ?Sized> {
    fn lock_recover(&self) -> MutexGuard<'_, T>;
}

impl<T: ?Sized> MutexExt<T> for Mutex<T> {
    fn lock_recover(&self) -> MutexGuard<'_, T> {
        self.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Serializes async work that shares a key, so that N callers racing for the same result do the
/// work once and the rest wait for it.
pub struct SingleFlight<K: Eq + Hash + Clone> {
    slots: Mutex<HashMap<K, Arc<tokio::sync::Mutex<()>>>>,
}

impl<K: Eq + Hash + Clone> Default for SingleFlight<K> {
    fn default() -> Self {
        Self {
            slots: Mutex::new(HashMap::new()),
        }
    }
}

impl<K: Eq + Hash + Clone> SingleFlight<K> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Wait until `key` is ours, then hold it until the guard is dropped.
    pub async fn acquire(&self, key: K) -> SingleFlightGuard<'_, K> {
        let slot = {
            let mut slots = self.slots.lock_recover();
            Arc::clone(slots.entry(key.clone()).or_default())
        };
        // The guard exists *before* the wait, so a caller cancelled mid-wait
        // (an HTTP client disconnecting) still runs `Drop` and prunes the
        // entry it just put in the map.
        let mut pending = SingleFlightGuard {
            flight: self,
            key,
            slot,
            guard: None,
        };
        pending.guard = Some(Arc::clone(&pending.slot).lock_owned().await);
        pending
    }

    /// How many keys are currently held or waited on — 0 once every guard for
    /// every key has been dropped.
    #[cfg(test)]
    pub fn live_keys(&self) -> usize {
        self.slots.lock_recover().len()
    }
}

/// Exclusive hold on one [`SingleFlight`] key.
pub struct SingleFlightGuard<'a, K: Eq + Hash + Clone> {
    flight: &'a SingleFlight<K>,
    key: K,
    slot: Arc<tokio::sync::Mutex<()>>,
    guard: Option<tokio::sync::OwnedMutexGuard<()>>,
}

impl<K: Eq + Hash + Clone> Drop for SingleFlightGuard<'_, K> {
    fn drop(&mut self) {
        // Hand the key on before deciding whether anyone still wants it.
        self.guard = None;
        let mut slots = self.flight.slots.lock_recover();
        // Two strong references — the map's and this guard's — mean nobody else holds or
        // waits on the key.
        if Arc::strong_count(&self.slot) == 2 {
            slots.remove(&self.key);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn recovers_poisoned_lock() {
        let lock = Arc::new(RwLock::new(0u32));

        let poisoner = {
            let lock = Arc::clone(&lock);
            std::thread::spawn(move || {
                let mut guard = lock.write_recover();
                *guard = 42;
                panic!("poison the lock while holding the write guard");
            })
        };
        assert!(poisoner.join().is_err());

        assert!(lock.read().is_err(), "lock should be poisoned");
        assert_eq!(*lock.read_recover(), 42);
        *lock.write_recover() = 7;
        assert_eq!(*lock.read_recover(), 7);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn single_flight_serializes_one_key() {
        let flight = Arc::new(SingleFlight::<&'static str>::new());
        let live = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let gate = Arc::new(tokio::sync::Semaphore::new(0));

        let tasks: Vec<_> = (0..8)
            .map(|_| {
                let (flight, live, peak, gate) = (
                    Arc::clone(&flight),
                    Arc::clone(&live),
                    Arc::clone(&peak),
                    Arc::clone(&gate),
                );
                tokio::spawn(async move {
                    let _guard = flight.acquire("same").await;
                    let now = live.fetch_add(1, Ordering::SeqCst) + 1;
                    peak.fetch_max(now, Ordering::SeqCst);
                    let permit = gate.acquire().await.unwrap();
                    permit.forget();
                    live.fetch_sub(1, Ordering::SeqCst);
                })
            })
            .collect();

        for _ in 0..64 {
            tokio::task::yield_now().await;
        }
        gate.add_permits(8);
        for t in tasks {
            t.await.unwrap();
        }

        assert_eq!(peak.load(Ordering::SeqCst), 1, "holders overlapped");
        assert_eq!(flight.live_keys(), 0, "registry retained a finished key");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn single_flight_lets_different_keys_run_together() {
        let flight = Arc::new(SingleFlight::<u32>::new());
        let barrier = Arc::new(tokio::sync::Barrier::new(2));

        let tasks: Vec<_> = (0..2)
            .map(|key| {
                let (flight, barrier) = (Arc::clone(&flight), Arc::clone(&barrier));
                tokio::spawn(async move {
                    let _guard = flight.acquire(key).await;
                    barrier.wait().await;
                })
            })
            .collect();

        for t in tasks {
            tokio::time::timeout(std::time::Duration::from_secs(5), t)
                .await
                .expect("distinct keys serialized against each other")
                .unwrap();
        }
        assert_eq!(flight.live_keys(), 0);
    }

    #[tokio::test]
    async fn single_flight_prunes_a_cancelled_waiter() {
        use std::future::Future;
        use std::task::{Context, Poll, Waker};

        let flight = SingleFlight::<&'static str>::new();
        let held = flight.acquire("key").await;

        let mut waiter = Box::pin(flight.acquire("key"));
        let mut cx = Context::from_waker(Waker::noop());
        assert!(
            matches!(waiter.as_mut().poll(&mut cx), Poll::Pending),
            "the waiter should be queued behind the holder"
        );

        drop(held);
        drop(waiter);
        assert_eq!(flight.live_keys(), 0, "a cancelled waiter orphaned its key");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn single_flight_stays_exclusive_and_empty_under_cancellation() {
        const KEYS: u32 = 16;
        const ROUNDS: u32 = 200;

        struct Occupant(Arc<AtomicUsize>);

        impl Occupant {
            fn enter(live: &Arc<AtomicUsize>, overlaps: &AtomicUsize) -> Self {
                if live.fetch_add(1, Ordering::SeqCst) != 0 {
                    overlaps.fetch_add(1, Ordering::SeqCst);
                }
                Self(Arc::clone(live))
            }
        }

        impl Drop for Occupant {
            fn drop(&mut self) {
                self.0.fetch_sub(1, Ordering::SeqCst);
            }
        }

        let flight = Arc::new(SingleFlight::<u32>::new());
        let overlaps = Arc::new(AtomicUsize::new(0));

        let loops: Vec<_> = (0..KEYS)
            .map(|key| {
                let (flight, overlaps) = (Arc::clone(&flight), Arc::clone(&overlaps));
                tokio::spawn(async move {
                    let live = Arc::new(AtomicUsize::new(0));
                    for _ in 0..ROUNDS {
                        let holder = flight.acquire(key).await;
                        let occupant = Occupant::enter(&live, &overlaps);

                        let waiter = {
                            let (flight, live, overlaps) = (
                                Arc::clone(&flight),
                                Arc::clone(&live),
                                Arc::clone(&overlaps),
                            );
                            tokio::spawn(async move {
                                let _guard = flight.acquire(key).await;
                                let _occupant = Occupant::enter(&live, &overlaps);
                                tokio::task::yield_now().await;
                            })
                        };

                        tokio::task::yield_now().await;
                        drop(occupant);
                        drop(holder);
                        waiter.abort();
                        let _ = waiter.await;
                    }
                })
            })
            .collect();

        for l in loops {
            l.await.unwrap();
        }

        assert_eq!(overlaps.load(Ordering::SeqCst), 0, "two holders of one key");
        assert_eq!(flight.live_keys(), 0, "cancellation orphaned keys");
    }

    #[tokio::test]
    async fn single_flight_releases_key_after_panic() {
        let flight = Arc::new(SingleFlight::<&'static str>::new());

        let panicker = {
            let flight = Arc::clone(&flight);
            tokio::spawn(async move {
                let _guard = flight.acquire("key").await;
                panic!("unwind while holding the key");
            })
        };
        assert!(panicker.await.is_err());

        let guard = tokio::time::timeout(std::time::Duration::from_secs(5), flight.acquire("key"))
            .await
            .expect("panicking holder left the key locked");
        drop(guard);
        assert_eq!(flight.live_keys(), 0);
    }
}
