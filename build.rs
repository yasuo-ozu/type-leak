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
    println!("cargo:rustc-check-cfg=cfg(type_leak_diagnostic)");
    if probe() {
        println!("cargo:rustc-cfg=type_leak_diagnostic");
    }
}

/// Returns `true` if [`PROBE`] compiles.
fn probe() -> bool {
    let rustc = env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let probe_file = out_dir.join("probe_diagnostic.rs");
    if std::fs::write(&probe_file, PROBE).is_err() {
        return false;
    }
    let mut cmd = Command::new(rustc);
    cmd.arg("--edition=2018")
        .arg("--crate-type=lib")
        .arg("--crate-name=type_leak_probe")
        .arg("--emit=metadata")
        .arg("--out-dir")
        .arg(&out_dir)
        .arg(&probe_file)
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    if let Some(target) = env::var_os("TARGET") {
        cmd.arg("--target").arg(target);
    }
    cmd.status().map(|status| status.success()).unwrap_or(false)
}
