//! A property test of `type-leak` across crate boundaries.
//!
//! Each case is a generated Leaker's crate (a module of relative types, and random modules,
//! structs, enums, unions, traits and impls with `#[prop_leak]` on them, see `gen.rs`) and a
//! generated Referrer's crate that invokes the macro exported for each item. The macros are those
//! of `tests/crates/macro/prop.rs`, which analyze the item with the matching `intern_*`, emit the
//! Repeater's implementations, and, in the Referrer's crate, rewrite a copy of the item with
//! `Referrer::visitor(context, ..)`.
//!
//! The property: if the Leaker accepts an item (it may reject it, which is counted), the Referrer's
//! crate compiles, and every type position of the rewritten copy is exactly the original type.
//! The latter is checked with a witness for each position: the Leaker's crate defines
//! `pub fn w_k<generics>() -> PhantomData<OriginalType>`, and the Referrer's crate has
//! `fn chk_k<generics>() -> PhantomData<RewrittenType> { leaker::w_k::<..>() }`, with the same
//! generics (the rewritten type is the result, as a parameter of a method is, so that the bounds
//! implied by its well-formedness, such as `T: 'a` for `&'a T`, are assumed as in a real copy).
//! `Self` is what it is at the position (the struct, enum or union, the self type of an impl, or
//! the host type that implements a trait in the Referrer's crate); a position with an elided
//! lifetime has no witness, and in a copy of a module neither has a type that names what the module
//! declares, because the copy declares its own.
//!
//! All the cases are built by one `cargo build --workspace`, so that they compile in parallel.
//! The Leaker's crate must compile by itself: if it does not, it is built again without
//! `type-leak` (`prop_leak_noop`) to tell a bug of the generator from the macro's output.
//!
//! Environment variables:
//!
//! - `TYPE_LEAK_PROP_CASES` (default 24): the number of cases.
//! - `TYPE_LEAK_PROP_SEED` (default 1): the seed of the generator.
//! - `TYPE_LEAK_PROP_ITEMS` (default 8): the most items in a module.
//! - `TYPE_LEAK_PROP_FEATURES`: a comma-separated list of constructs that `type-leak` is known to
//!   fail on (see `gen::Features`). The generator avoids them by default, so that the test passes
//!   on the current library, and a new failure is a new finding. With one of `inferred_outlives`,
//!   `impl_self_outlives`, `strict_anchors` (F18), or `all`, the test fails with the reproduction
//!   of that finding (`tests/findings.rs` has each one as a library-level test; `strict_anchors`
//!   is about the items that are invoked as `strict`, a quarter of them). `unsized_params`,
//!   `bound_cycles`, `impl_assoc_self`, `method_lifetime_paths`, `module_crate_paths`,
//!   `bound_merge_syntax`, `fn_pointer_self`, `elided_fn_generics`, `impl_self_bounds` (F13),
//!   `impl_pred_leaked` (F14), `self_elision` (F15, F23), `implied_outlives` (F16),
//!   `strict_cycles` (F17, about the strict items too), `impl_self_bound_cycle` (F19),
//!   `trait_assoc_bound_cycle` (F20), `iterator_item_cycle` (F21), `supertrait_item` (F25),
//!   `impl_trait_args` (F27), `hidden_lifetimes` (F28), `projection_outlives` (F30),
//!   `item_predicate_cycle` (F33) and `self_projection_outlives` (F34) are fixed findings: they are
//!   on by default (the Leaker encodes or rewrites them, records what they need, or rejects and
//!   counts what it cannot copy, and the Referrer names the lifetime of an elided signature where
//!   `Self` adds lifetimes to it). `alias_item` (F26, a limitation) is a known failure that the
//!   generator does not make by default: with it, or `all`, the test fails with the reproduction
//!   of that finding (E0277).
//!   The other findings of `tests/findings.rs` (F22 to F24: the Leaker rejected
//!   what compiles, F29, F31, F32) are not features: a rejection is counted as success, and the
//!   test cannot show a toolchain or a lint.
//!
//! The test is ignored by default: it is slow, as it builds the generated crates. Run it with
//!
//! ```text
//! cargo test --test prop_cross_crate -- --ignored
//! TYPE_LEAK_PROP_CASES=200 TYPE_LEAK_PROP_SEED=7 cargo test --test prop_cross_crate -- --ignored
//! ```
//!
//! The sources of every case stay in `target/tmp/prop/ws-<seed>-<cases>/`, and each is a workspace
//! member that builds on its own (`cargo build` there, with `CARGO_TARGET_DIR` as the test has it).

mod gen;
mod json;

use gen::{generate, Case, Features, KINDS};
use json::Json;
use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;
use std::hash::Hasher;
use std::path::{Path, PathBuf};
use std::process::Command;

fn env_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    match std::env::var(name) {
        Ok(value) => value
            .trim()
            .parse()
            .unwrap_or_else(|_| panic!("{} is not valid: {:?}", name, value)),
        Err(_) => default,
    }
}

fn case_seed(seed: u64, index: usize) -> u64 {
    let mut hasher = DefaultHasher::new();
    hasher.write_u64(seed);
    hasher.write_u64(index as u64);
    hasher.finish()
}

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn leaker_dir(i: usize) -> String {
    format!("c{:03}_l", i)
}

fn referrer_dir(i: usize) -> String {
    format!("c{:03}_r", i)
}

fn leaker_name(i: usize) -> String {
    format!("tlp_l_{:03}", i)
}

fn referrer_name(i: usize) -> String {
    format!("tlp_r_{:03}", i)
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    /// With `type-leak`, and the Referrer's crate.
    Real,
    /// Without it, to check the generator.
    Identity,
}

/// Writes the workspace of the cases (the indexes are those of the case list).
fn write_workspace(ws: &Path, cases: &[(usize, &Case)], mode: Mode) {
    let _ = std::fs::remove_dir_all(ws);
    std::fs::create_dir_all(ws).unwrap();
    let members: Vec<String> = cases
        .iter()
        .flat_map(|(i, _)| {
            let mut dirs = vec![format!("\"{}\"", leaker_dir(*i))];
            if mode == Mode::Real {
                dirs.push(format!("\"{}\"", referrer_dir(*i)));
            }
            dirs
        })
        .collect();
    std::fs::write(
        ws.join("Cargo.toml"),
        format!(
            "[workspace]\nmembers = [{}]\nresolver = \"2\"\n",
            members.join(", ")
        ),
    )
    .unwrap();
    // The same versions as the test crates, if they are locked.
    let lock = root().join("tests/crates/macro/Cargo.lock");
    if lock.exists() {
        std::fs::copy(lock, ws.join("Cargo.lock")).unwrap();
    }
    let macro_path = root().join("tests/crates/macro");
    for (i, case) in cases {
        let dir = ws.join(leaker_dir(*i));
        std::fs::create_dir_all(dir.join("src")).unwrap();
        std::fs::write(
            dir.join("Cargo.toml"),
            format!(
                "[package]\nname = \"{}\"\nversion = \"0.0.0\"\nedition = \"2021\"\npublish = false\n\n\
                 [dependencies]\ntype-leak-test-macro = {{ path = {:?} }}\n",
                leaker_name(*i),
                macro_path.display().to_string()
            ),
        )
        .unwrap();
        let attr = match mode {
            Mode::Real => "crate::__macro::prop_leak",
            Mode::Identity => "crate::__macro::prop_leak_noop",
        };
        std::fs::write(dir.join("src/lib.rs"), case.leaker.replace(gen::ATTR, attr)).unwrap();
        if mode == Mode::Identity {
            continue;
        }
        // A binary prints what each macro reported.
        let mut main = String::from("fn main() {\n");
        for item in &case.items {
            writeln!(
                main,
                "    println!(\"{{}}\", {}::__tl_report_{}!());",
                leaker_name(*i),
                item.id
            )
            .unwrap();
        }
        main.push_str("}\n");
        std::fs::write(dir.join("src/main.rs"), main).unwrap();

        let dir = ws.join(referrer_dir(*i));
        std::fs::create_dir_all(dir.join("src")).unwrap();
        std::fs::write(
            dir.join("Cargo.toml"),
            format!(
                "[package]\nname = \"{}\"\nversion = \"0.0.0\"\nedition = \"2021\"\npublish = false\n\n\
                 [dependencies]\n{} = {{ path = \"../{}\" }}\n",
                referrer_name(*i),
                leaker_name(*i),
                leaker_dir(*i)
            ),
        )
        .unwrap();
        std::fs::write(dir.join("src/lib.rs"), &case.referrer).unwrap();
    }
}

/// A diagnostic of the build.
struct Diag {
    /// The crate.
    package: String,
    rendered: String,
    /// The `(file, line)` of its spans and of where the macros of each were invoked.
    lines: Vec<(String, u64)>,
}

struct Build {
    success: bool,
    diags: Vec<Diag>,
    executables: HashMap<String, PathBuf>,
    /// What cargo printed to stderr.
    stderr: String,
}

fn span_lines(span: &Json, out: &mut Vec<(String, u64)>) {
    if let (Some(file), Some(line)) = (
        span.get("file_name").and_then(Json::str),
        span.get("line_start").and_then(Json::num),
    ) {
        out.push((file.to_string(), line as u64));
    }
    if let Some(expansion) = span.get("expansion").and_then(|e| e.get("span")) {
        span_lines(expansion, out);
    }
}

fn cargo_build(ws: &Path, target: &Path) -> Build {
    let output = Command::new(env!("CARGO"))
        .args([
            "build",
            "--workspace",
            "--keep-going",
            "--message-format=json",
            "--manifest-path",
        ])
        .arg(ws.join("Cargo.toml"))
        .env("CARGO_TARGET_DIR", target)
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap();
    let mut build = Build {
        success: output.status.success(),
        diags: Vec::new(),
        executables: HashMap::new(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    };
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        let Some(json) = Json::parse(line) else {
            continue;
        };
        match json.get("reason").and_then(Json::str) {
            Some("compiler-message") => {
                let message = json.get("message").unwrap();
                if message.get("level").and_then(Json::str) != Some("error") {
                    continue;
                }
                let mut lines = Vec::new();
                // The primary spans first.
                let spans = message.get("spans").map_or(&[][..], Json::arr);
                for primary in [true, false] {
                    for span in spans {
                        if span.get("is_primary").and_then(Json::bool) == Some(primary) {
                            span_lines(span, &mut lines);
                        }
                    }
                }
                build.diags.push(Diag {
                    package: json
                        .path("target.name")
                        .and_then(Json::str)
                        .unwrap_or("")
                        .to_string(),
                    rendered: message
                        .get("rendered")
                        .and_then(Json::str)
                        .unwrap_or("")
                        .to_string(),
                    lines,
                });
            }
            Some("compiler-artifact") => {
                if let (Some(name), Some(exe)) = (
                    json.path("target.name").and_then(Json::str),
                    json.get("executable").and_then(Json::str),
                ) {
                    build
                        .executables
                        .insert(name.to_string(), PathBuf::from(exe));
                }
            }
            _ => (),
        }
    }
    build
}

#[derive(Debug)]
struct Failure {
    case: usize,
    item: Option<usize>,
    class: &'static str,
    text: String,
}

/// What a macro reported for an item.
#[derive(Debug, Default, Clone)]
struct Report {
    kind: String,
    accepted: bool,
    /// The reason of the rejection, or the error of the rewrite.
    detail: String,
    positions: usize,
    checked: usize,
    entries: usize,
}

fn parse_report(line: &str) -> Option<(usize, Report)> {
    let mut parts = line.splitn(4, '|');
    let id = parts.next()?.parse().ok()?;
    let kind = parts.next()?.to_string();
    let accepted = parts.next()? == "ok";
    let rest = parts.next().unwrap_or("");
    let mut report = Report {
        kind,
        accepted,
        ..Default::default()
    };
    if accepted {
        for word in rest.split(' ') {
            match word.split_once('=') {
                Some(("entries", n)) => report.entries = n.parse().unwrap_or(0),
                Some(("positions", n)) => report.positions = n.parse().unwrap_or(0),
                Some(("checked", n)) => report.checked = n.parse().unwrap_or(0),
                Some(("rewrite-error", _)) => report.detail = rest.to_string(),
                _ => (),
            }
        }
        if let Some(at) = rest.find("rewrite-error=") {
            report.detail = rest[at..].to_string();
        }
    } else {
        report.detail = rest.to_string();
    }
    Some((id, report))
}

/// The first lines of a diagnostic.
fn excerpt(text: &str, max: usize) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let mut out = lines[..lines.len().min(max)].join("\n");
    if lines.len() > max {
        write!(out, "\n... ({} more lines)", lines.len() - max).unwrap();
    }
    out
}

/// A short reason of a rejection, for the statistics.
fn reason_head(detail: &str) -> String {
    // The cycles of a bound that names a leaked type (F2) are counted together.
    if detail.contains("in a bound or where predicate") {
        return "a leaked type in a bound or where predicate that rustc cannot normalize"
            .to_string();
    }
    // An entry in the self type of an impl whose context depends on `Self` (F13) is counted
    // together.
    if detail.contains("is in the self type of the impl") {
        return "a type in the self type of an impl that a bound of the impl makes depend on `Self`"
            .to_string();
    }
    // "use absolute path: 'Vec < X >'" is counted by the start of the type.
    match detail.find('\'') {
        Some(at) => {
            let ty: String = detail[at + 1..].chars().take(24).collect();
            format!("{} '{}", detail[..at].trim(), ty)
        }
        None => detail.chars().take(60).collect(),
    }
}

#[test]
#[ignore = "slow, builds generated crates; run with `cargo test --test prop_cross_crate -- --ignored`"]
fn random_items_leak_across_crates() {
    let cases: usize = env_or("TYPE_LEAK_PROP_CASES", 24);
    let seed: u64 = env_or("TYPE_LEAK_PROP_SEED", 1);
    let items_per_module: usize = env_or("TYPE_LEAK_PROP_ITEMS", 8);
    let features = Features::parse(&std::env::var("TYPE_LEAK_PROP_FEATURES").unwrap_or_default());
    let generated: Vec<Case> = (0..cases)
        .map(|i| generate(case_seed(seed, i), i, items_per_module, features))
        .collect();

    let base = Path::new(env!("CARGO_TARGET_TMPDIR")).join("prop");
    let target = base.join("target");
    let ws = base.join(format!("ws-{}-{}", seed, cases));
    let list: Vec<(usize, &Case)> = generated.iter().enumerate().collect();
    write_workspace(&ws, &list, Mode::Real);
    let build = cargo_build(&ws, &target);

    let mut failures: Vec<Failure> = Vec::new();
    if !build.success && build.diags.is_empty() {
        panic!(
            "cargo failed without a compile error (seed {}):\n{}",
            seed,
            excerpt(&build.stderr, 60)
        );
    }

    // The errors of each crate. The Referrer's crate is built only if the Leaker's compiles.
    let mut leaker_failed: Vec<usize> = Vec::new();
    for (i, case) in generated.iter().enumerate() {
        let leaker = leaker_name(i);
        let referrer = referrer_name(i);
        let leaker_errors: Vec<&Diag> =
            build.diags.iter().filter(|d| d.package == leaker).collect();
        if !leaker_errors.is_empty() {
            leaker_failed.push(i);
            continue;
        }
        let referrer_file = format!("{}/src/lib.rs", referrer_dir(i));
        for diag in build.diags.iter().filter(|d| d.package == referrer) {
            // The item whose macro invocation the error is in.
            let line = diag
                .lines
                .iter()
                .find(|(file, _)| *file == referrer_file)
                .map(|(_, line)| *line as usize);
            let item = line.and_then(|line| {
                case.referrer_lines
                    .iter()
                    .find(|(l, _)| *l == line)
                    .map(|(_, id)| *id)
            });
            // The same error is often reported several times.
            if failures.iter().any(|f| {
                f.case == i
                    && f.item == item
                    && f.text.lines().next() == diag.rendered.lines().next()
            }) {
                continue;
            }
            failures.push(Failure {
                case: i,
                item,
                class: "the Referrer's crate does not compile",
                text: diag.rendered.clone(),
            });
        }
    }

    // A Leaker's crate that does not compile: is it the generator or the macro?
    if !leaker_failed.is_empty() {
        let id_ws = base.join(format!("ws-{}-{}-identity", seed, cases));
        let list: Vec<(usize, &Case)> = leaker_failed.iter().map(|&i| (i, &generated[i])).collect();
        write_workspace(&id_ws, &list, Mode::Identity);
        let identity = cargo_build(&id_ws, &target);
        for &i in &leaker_failed {
            let name = leaker_name(i);
            let own: Vec<&Diag> = identity
                .diags
                .iter()
                .filter(|d| d.package == name)
                .collect();
            let (class, diags) = if own.is_empty() {
                (
                    "the Leaker's crate does not compile with the output of the macro",
                    build
                        .diags
                        .iter()
                        .filter(|d| d.package == name)
                        .collect::<Vec<_>>(),
                )
            } else {
                (
                    "GENERATOR BUG: the Leaker's crate does not compile by itself",
                    own,
                )
            };
            for diag in diags {
                failures.push(Failure {
                    case: i,
                    item: None,
                    class,
                    text: diag.rendered.clone(),
                });
            }
        }
    }

    // What the macros reported.
    let mut reports: BTreeMap<(usize, usize), Report> = BTreeMap::new();
    for (i, case) in generated.iter().enumerate() {
        if leaker_failed.contains(&i) {
            continue;
        }
        let Some(exe) = build.executables.get(&leaker_name(i)) else {
            continue;
        };
        let output = Command::new(exe).output().unwrap();
        for line in String::from_utf8_lossy(&output.stdout).lines() {
            if let Some((id, report)) = parse_report(line) {
                if report.accepted && report.detail.starts_with("rewrite-error") {
                    failures.push(Failure {
                        case: i,
                        item: Some(id),
                        class: "the rewrite of an accepted item fails",
                        text: report.detail.clone(),
                    });
                }
                reports.insert((i, id), report);
            }
        }
        assert_eq!(
            reports.keys().filter(|(c, _)| *c == i).count(),
            case.items.len(),
            "a report is missing in case {}",
            i
        );
    }

    let summary = summarize(&generated, &reports, seed, cases, features);
    eprintln!("{}", summary);

    if !failures.is_empty() {
        let mut message = format!(
            "type-leak property failed (TYPE_LEAK_PROP_SEED={} TYPE_LEAK_PROP_CASES={} TYPE_LEAK_PROP_ITEMS={} TYPE_LEAK_PROP_FEATURES={:?}):\n{} failure(s).\nSources: {}\n\n{}\n",
            seed,
            cases,
            items_per_module,
            std::env::var("TYPE_LEAK_PROP_FEATURES").unwrap_or_default(),
            failures.len(),
            ws.display(),
            summary
        );
        for failure in failures.iter().take(6) {
            let case = &generated[failure.case];
            write!(
                message,
                "\n=== case {} (case seed {}), {}\n",
                failure.case,
                case_seed(seed, failure.case),
                failure.class
            )
            .unwrap();
            writeln!(
                message,
                "crates: {}/{}, {}/{}",
                ws.display(),
                leaker_dir(failure.case),
                ws.display(),
                referrer_dir(failure.case)
            )
            .unwrap();
            writeln!(message, "settings: {:?}", case.config).unwrap();
            if let Some(item) = case.items.iter().find(|i| Some(i.id) == failure.item) {
                writeln!(
                    message,
                    "item {} ({}, in {:?}, allow_self {}, strict {}):\n{}",
                    item.id,
                    item.kind,
                    item.module,
                    item.allow_self,
                    item.strict,
                    excerpt(&item.source, 60)
                )
                .unwrap();
            }
            writeln!(message, "{}", excerpt(&failure.text, 40)).unwrap();
        }
        message.push_str("\nall failures:\n");
        for failure in &failures {
            let head = failure.text.lines().next().unwrap_or("");
            writeln!(
                message,
                "  case {} item {:?} [{}]: {}",
                failure.case,
                failure.item,
                failure.class.chars().take(24).collect::<String>(),
                head.chars().take(150).collect::<String>()
            )
            .unwrap();
        }
        panic!("{}", message);
    }
}

fn summarize(
    cases: &[Case],
    reports: &BTreeMap<(usize, usize), Report>,
    seed: u64,
    n: usize,
    features: Features,
) -> String {
    let mut generated: BTreeMap<&str, usize> = BTreeMap::new();
    let mut accepted: BTreeMap<&str, usize> = BTreeMap::new();
    let mut rejected: BTreeMap<&str, usize> = BTreeMap::new();
    let mut reasons: BTreeMap<String, usize> = BTreeMap::new();
    let (mut positions, mut checked, mut entries) = (0, 0, 0);
    let mut analyzed = 0;
    for (case_index, case) in cases.iter().enumerate() {
        for item in &case.items {
            for kind in &item.contents {
                *generated.entry(kind).or_default() += 1;
            }
            if item.kind == "mod" {
                *generated.entry("mod").or_default() += 1;
            }
            if let Some(report) = reports.get(&(case_index, item.id)) {
                analyzed += 1;
                let kind = KINDS
                    .iter()
                    .find(|k| **k == report.kind)
                    .copied()
                    .unwrap_or("?");
                if report.accepted {
                    *accepted.entry(kind).or_default() += 1;
                    positions += report.positions;
                    checked += report.checked;
                    entries += report.entries;
                } else {
                    *rejected.entry(kind).or_default() += 1;
                    *reasons.entry(reason_head(&report.detail)).or_default() += 1;
                }
            }
        }
    }
    let mut s = format!(
        "type-leak property test: seed {}, {} cases, features {:?}\n",
        seed, n, features
    );
    let total: usize = generated.values().sum();
    writeln!(
        s,
        "  generated items: {} (nested items of a module included)",
        total
    )
    .unwrap();
    writeln!(
        s,
        "  {:<14} {:>9} {:>9} {:>9} {:>8}",
        "kind", "generated", "accepted", "rejected", "rate"
    )
    .unwrap();
    for kind in KINDS.iter().chain(["fn", "alias"].iter()) {
        let g = generated.get(kind).copied().unwrap_or(0);
        if g == 0 {
            continue;
        }
        let a = accepted.get(kind).copied().unwrap_or(0);
        let r = rejected.get(kind).copied().unwrap_or(0);
        let rate = if a + r > 0 {
            format!("{:.0}%", 100.0 * r as f64 / (a + r) as f64)
        } else {
            "-".into()
        };
        writeln!(s, "  {:<14} {:>9} {:>9} {:>9} {:>8}", kind, g, a, r, rate).unwrap();
    }
    writeln!(
        s,
        "  analyzed (macros applied): {}; entries: {}; type positions: {}, witnesses checked: {} ({} skipped: elided lifetimes, or names declared in a module)",
        analyzed,
        entries,
        positions,
        checked,
        positions - checked
    )
    .unwrap();
    let mut reasons: Vec<(String, usize)> = reasons.into_iter().collect();
    reasons.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
    for (reason, n) in reasons.iter().take(10) {
        writeln!(s, "  rejected {:>3}x: {}", n, reason).unwrap();
    }
    s
}
