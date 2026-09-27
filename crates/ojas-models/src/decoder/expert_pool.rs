//! Shared expert storage. An allocation is reusable only after its last owner
//! (including the current GPU batch) releases it.
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

#[derive(Debug)]
struct Ranges {
    free: BTreeMap<usize, usize>,
}
impl Ranges {
    fn new(bytes: usize) -> Self {
        Self {
            free: [(0, bytes)].into(),
        }
    }
    fn take(&mut self, bytes: usize) -> Option<usize> {
        assert!(bytes > 0);
        let (&offset, &length) = self.free.iter().find(|(_, length)| **length >= bytes)?;
        self.free.remove(&offset);
        if length > bytes {
            self.free.insert(offset + bytes, length - bytes);
        }
        Some(offset)
    }
    fn release(&mut self, mut offset: usize, mut bytes: usize) {
        if let Some((&left, &length)) = self.free.range(..offset).next_back() {
            assert!(left + length <= offset, "overlapping free range");
            if left + length == offset {
                self.free.remove(&left);
                offset = left;
                bytes += length;
            }
        }
        if let Some((&right, &length)) = self.free.range(offset..).next() {
            assert!(offset + bytes <= right, "overlapping free range");
            if offset + bytes == right {
                self.free.remove(&right);
                bytes += length;
            }
        }
        assert!(self.free.insert(offset, bytes).is_none());
    }
}

pub(crate) struct MetalExpert {
    pub buffer: metal::Buffer,
    offset: usize,
    pub len: usize,
    reservation: Option<(Arc<Mutex<Ranges>>, usize)>,
}
impl MetalExpert {
    pub fn standalone(buffer: metal::Buffer) -> Arc<Self> {
        Arc::new(Self {
            len: buffer.length() as usize,
            buffer,
            offset: 0,
            reservation: None,
        })
    }
    pub fn address(&self) -> u64 {
        self.buffer.gpu_address() + self.offset as u64
    }
    pub fn contents(&self) -> *mut u8 {
        // Construction guarantees offset + len is inside the backing buffer.
        unsafe { (self.buffer.contents() as *mut u8).add(self.offset) }
    }
}
impl Drop for MetalExpert {
    fn drop(&mut self) {
        if let Some((ranges, bytes)) = &self.reservation {
            ranges.lock().unwrap().release(self.offset, *bytes);
        }
    }
}

pub(super) struct ExpertPool {
    buffer: metal::Buffer,
    ranges: Arc<Mutex<Ranges>>,
}
impl ExpertPool {
    pub fn new(device: &metal::DeviceRef, budget: usize) -> Self {
        assert!(budget > 0 && budget as u64 <= device.max_buffer_length());
        Self {
            buffer: device.new_buffer(budget as u64, metal::MTLResourceOptions::StorageModeShared),
            ranges: Arc::new(Mutex::new(Ranges::new(budget))),
        }
    }
    pub fn allocate(&self, len: usize) -> Option<Arc<MetalExpert>> {
        let reserved = len.checked_add(255)? & !255;
        if reserved == 0 {
            return None;
        }
        let offset = self.ranges.lock().unwrap().take(reserved)?;
        Some(Arc::new(MetalExpert {
            buffer: self.buffer.clone(),
            offset,
            len,
            reservation: Some((self.ranges.clone(), reserved)),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ranges_reuse_without_overlap() {
        let mut ranges = Ranges::new(4096);
        let mut live = Vec::new();
        let mut seed = 17u64;
        for _ in 0..10000 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            if !live.is_empty() && seed & 3 == 0 {
                let (offset, size) = live.swap_remove((seed as usize >> 8) % live.len());
                ranges.release(offset, size);
            } else {
                let size = ((seed >> 16) as usize % 200) + 1;
                if let Some(offset) = ranges.take(size) {
                    assert!(offset + size <= 4096);
                    assert!(live
                        .iter()
                        .all(|&(o, n)| offset + size <= o || o + n <= offset));
                    live.push((offset, size));
                }
            }
            assert_eq!(
                ranges.free.values().sum::<usize>() + live.iter().map(|x| x.1).sum::<usize>(),
                4096
            );
        }
        for (offset, size) in live {
            ranges.release(offset, size);
        }
        assert_eq!(ranges.free, [(0, 4096)].into());
    }
    #[test]
    fn metal_expert_pin_prevents_reuse() {
        let gpu = ojas_metal::MetalGpu::new().unwrap();
        let pool = ExpertPool::new(&gpu.device, 1024);
        let cached = pool.allocate(700).unwrap(); // reserves 768 bytes
        let batch_pin = cached.clone();
        unsafe {
            std::ptr::write_bytes(cached.contents(), 0x5a, cached.len);
        }
        drop(cached); // simulate cache eviction during gather
        assert!(pool.allocate(512).is_none());
        assert_eq!(unsafe { *batch_pin.contents().add(699) }, 0x5a);
        let address = batch_pin.address();
        drop(batch_pin); // GPU has completed
        let replacement = pool.allocate(1024).unwrap();
        assert_eq!(replacement.address(), address);
        assert!(pool.allocate(1).is_none());
    }
    #[test]
    fn pooled_admission_respects_hot_entries_and_batch_pins() {
        let gpu = ojas_metal::MetalGpu::new().unwrap();
        let mut cache = super::super::ExpertCache::new(1024);
        cache.pool = Some(ExpertPool::new(&gpu.device, 1024));
        cache.insert(1, vec![0x5a; 700].into_boxed_slice());
        cache.get(1).unwrap(); // frequency two: protect this hot resident
        cache.insert(2, vec![0x99; 512].into_boxed_slice());
        assert!(cache.contains(1) && !cache.contains(2));
        let batch = cache.metal_buffer(1).unwrap();
        assert!(cache.evict_one()); // explicit eviction while GPU holds a pin
        cache.insert(3, vec![0xff; 1024].into_boxed_slice());
        assert!(!cache.contains(3)); // scratch fallback, not an overwrite
        assert_eq!(unsafe { *batch.contents().add(699) }, 0x5a);
        drop(batch);
        cache.insert(3, vec![0xff; 1024].into_boxed_slice());
        assert!(cache.contains(3));
        assert_eq!(cache.bytes(), 1024);
    }
}
