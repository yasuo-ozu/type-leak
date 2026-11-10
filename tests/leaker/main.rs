//! The tests of the analysis (the Leaker, its scopes and the View), one module per file.

#[path = "../common/mod.rs"]
mod common;

mod bare_const;
mod bound_cycles;
mod check;
mod check_values;
mod context_id;
mod external_crates;
mod leaker;
mod outer_self;
mod requirements;
mod scopes;
mod shadowed_names;
