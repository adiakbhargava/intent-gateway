//! Lock-free SPSC (Single-Producer Single-Consumer) ring buffer.
//!
//! # Quick start
//!
//! ```rust
//! let (producer, consumer) = ring_buffer::new::<u32>(1024);
//! producer.try_push(42).unwrap();
//! assert_eq!(consumer.try_pop(), Some(42));
//! ```
//!
//! # Design
//!
//! `new()` returns a `(Producer<T>, Consumer<T>)` pair that can be sent to
//! separate threads. The split enforces the SPSC invariant at the type level:
//! neither half can be cloned, and the compiler prevents you from using the
//! wrong end on the wrong thread.
//!
//! All hot-path operations (`try_push`, `try_pop`, `push_slice`, `pop_slice`)
//! are allocation-free after construction. The only heap allocation is the
//! slot array created inside [`new`].

mod buffer;

pub use buffer::{new, Consumer, Producer};
