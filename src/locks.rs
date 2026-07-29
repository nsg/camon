use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

/// Extension methods for `RwLock` that recover the guard on poison instead of
/// panicking or silently skipping the work.
///
/// Recovery is safe throughout camon: every store update is a single-step
/// push/insert of an already-constructed value, so a panic while a guard is
/// held cannot leave a multi-step invariant half-applied. Treating a poisoned
/// lock as usable therefore only risks reading data written just before an
/// unrelated panic, never a corrupt intermediate state.
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

/// Serializes async work that shares a key, so that N callers racing for the
/// same result do the work once and the rest wait for it.
///
/// One `tokio::sync::Mutex` per *live* key, not one global mutex: two callers
/// with different keys never meet, so deduplication never turns into a queue.
/// The registry itself is guarded by a plain `std::sync::Mutex` that is only
/// ever held for a map lookup — never across an await.
///
/// The map holds an entry only while someone is holding or waiting for that
/// key, and [`SingleFlightGuard`]'s `Drop` removes the last one on the way out.
/// Every way out runs it: returning, unwinding — `Drop` runs there too, and a
/// tokio mutex has no poisoning, so a panicking holder releases the key rather
/// than sealing it — and cancellation, since [`SingleFlight::acquire`] builds
/// the guard before it starts waiting. The single exception is the one every
/// RAII guard has: `mem::forget` on a guard leaks its entry and leaves that key
/// locked for the life of the process.
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
        // The guard exists *before* the wait, so that a caller who goes away
        // mid-wait — an HTTP client disconnecting is exactly this — still runs
        // `Drop` and still prunes the entry it just put in the map. Built the
        // other way round there is nothing to drop while the wait is parked,
        // and a cancelled last waiter leaves the key behind forever.
        let mut pending = SingleFlightGuard {
            flight: self,
            key,
            slot,
            guard: None,
        };
        // Owned guard so the wait does not borrow the map, which is already
        // unlocked above: a std guard held across this await would block the
        // executor thread and defeat the point.
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
        // Two strong references — the map's and this guard's — mean nobody else
        // holds or is waiting for the key, so the entry can go. Anyone arriving
        // after this point takes the map lock, finds nothing and starts a fresh
        // slot, which is correct: the work this guard covered is already done.
        //
        // A waiter parked in `acquire` counts twice, once for its own guard and
        // once for the `lock_owned` future it is suspended on; a cancelled one
        // drops that future first, so by the time its guard reaches this line
        // the count is the same 2 a finished holder leaves.
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

        // The std guards now report the lock as poisoned...
        assert!(lock.read().is_err(), "lock should be poisoned");

        // ...but the recovery helpers still hand back the last written value.
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
                    // Holds the key until the test opens the gate, so an absent
                    // guard shows up as overlap rather than as fast serial runs.
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
        // Both halves only get past the barrier if they hold their keys at the
        // same time; a global lock would deadlock here and trip the timeout.
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

    /// A waiter that goes away mid-wait — an HTTP client disconnecting, which
    /// aborts the task serving it — must take its registry entry with it.
    ///
    /// The future is polled once by hand and then dropped without a final poll,
    /// because that is what abort does and what `tokio::time::timeout` does
    /// *not*: `Timeout::poll` polls the inner future before reporting the
    /// elapsed deadline, so a timeout would hide this entirely.
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

        // The holder leaves first, so it sees the waiter's references and
        // rightly keeps the entry; the waiter is then the last one out.
        drop(held);
        drop(waiter);
        assert_eq!(flight.live_keys(), 0, "a cancelled waiter orphaned its key");
    }

    /// Cancellation against real parallelism. Each key runs its own loop of
    /// "one holder, one waiter queued behind it, waiter aborted the moment the
    /// key is handed over" — the interleaving that used to strand an entry,
    /// since the departing holder correctly sees the waiter and leaves the key
    /// for it, and the waiter then never wakes to claim or release it. Sixteen
    /// of those loops run at once on real threads, and every hold is counted so
    /// a lost mutual exclusion shows up as well as a lost key.
    #[tokio::test(flavor = "multi_thread")]
    async fn single_flight_stays_exclusive_and_empty_under_cancellation() {
        const KEYS: u32 = 16;
        const ROUNDS: u32 = 200;

        /// Counts a hold for as long as it lasts. A `Drop` impl rather than a
        /// pair of statements because an aborted holder never reaches the
        /// second one, and a missed decrement would read as an overlap.
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

        // The key is free again, not stuck behind the panicked holder.
        let guard = tokio::time::timeout(std::time::Duration::from_secs(5), flight.acquire("key"))
            .await
            .expect("panicking holder left the key locked");
        drop(guard);
        assert_eq!(flight.live_keys(), 0);
    }
}
