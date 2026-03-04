use allocator::{CountingAllocator, allocation_count, allocation_bytes, reset_counters};

#[global_allocator]
static ALLOC: CountingAllocator = CountingAllocator;

#[test]
fn test_counts_allocations() {
    reset_counters();
    let _v: Vec<u8> = Vec::with_capacity(1024);
    assert!(allocation_count() >= 1, "Should have counted at least one allocation");
    assert!(allocation_bytes() >= 1024, "Should have counted at least 1024 bytes");
}

#[test]
fn test_reset_clears_counters() {
    let _v: Vec<u8> = Vec::with_capacity(512);
    reset_counters();
    assert_eq!(allocation_count(), 0);
    assert_eq!(allocation_bytes(), 0);
}
