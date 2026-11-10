//! The tests of what the Leaker chooses and writes (outlives bounds, `reduce_roots()`, binders),
//! one module per file.

#[path = "../common/mod.rs"]
mod common;

mod binder_output;
mod outlives_bounds;
mod reduce_roots;
