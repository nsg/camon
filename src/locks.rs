use std::sync::{PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

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
}
