//! Detects whether `proc_macro::Diagnostic` (nightly) is available, to render the notes and the help
//! of an `ErrorItem` as those of the diagnostic with their own spans.

use std::env;
use std::path::PathBuf;
use std::process::{Command, Stdio};

const PROBE: &str = r#"
#![feature(proc_macro_diagnostic)]
extern crate proc_macro;
pub fn probe(span: proc_macro::Span) {
    proc_macro::Diagnostic::spanned(span, proc_macro::Level::Error, "error")
        .span_note(span, "note")
        .span_help(span, "help")
        .emit();
}
"#;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    // What the probe is compiled with.
    for var in [
        "RUSTC",
        "RUSTC_WRAPPER",
        "RUSTFLAGS",
        "CARGO_ENCODED_RUSTFLAGS",
    ] {
        println!("cargo:rerun-if-env-changed={}", var);
    }
    println!("cargo:rustc-check-cfg=cfg(type_leak_diagnostic)");
    if probe() {
        println!("cargo:rustc-cfg=type_leak_diagnostic");
    }
}

/// Returns `true` if [`PROBE`] compiles, with the wrapper and the flags that cargo runs rustc with.
fn probe() -> bool {
    let rustc = env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let probe_file = out_dir.join("probe_diagnostic.rs");
    if std::fs::write(&probe_file, PROBE).is_err() {
        return false;
    }
    // `[wrapper] rustc [flags]`, as cargo runs it: a wrapper (sccache) takes rustc as its first
    // argument, and an empty one is none.
    let mut cmd = match env::var_os("RUSTC_WRAPPER").filter(|wrapper| !wrapper.is_empty()) {
        Some(wrapper) => {
            let mut cmd = Command::new(wrapper);
            cmd.arg(rustc);
            cmd
        }
        None => Command::new(rustc),
    };
    cmd.arg("--edition=2018")
        .arg("--crate-type=lib")
        .arg("--crate-name=type_leak_probe")
        .arg("--emit=metadata")
        // The flags may deny lints (`-D missing-docs`), which the probe does not care about.
        .arg("--cap-lints=allow")
        .arg("--out-dir")
        .arg(&out_dir)
        .arg(&probe_file)
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    if let Some(target) = env::var_os("TARGET") {
        cmd.arg("--target").arg(target);
    }
    // The flags are separated by `\x1f`; cargo gives them to the build script since 1.55.
    if let Ok(flags) = env::var("CARGO_ENCODED_RUSTFLAGS") {
        cmd.args(flags.split('\x1f').filter(|flag| !flag.is_empty()));
    }
    cmd.status().map(|status| status.success()).unwrap_or(false)
}
