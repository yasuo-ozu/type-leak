//! A property test of `type-leak` across crate boundaries.
//!
//! Each case is a generated Leaker's crate (a module of relative types, and random modules,
//! structs, enums, unions, traits and impls with `#[prop_leak]` on them, see `gen.rs`) and a
//! generated Referrer's crate that invokes the macro exported for each item. The macros are those
//! of `tests/crates/macro/prop.rs`, which analyze the item with the matching `intern_*` (or, for
//! some, through a scope: see below), emit the Repeater's implementations, and, in the Referrer's
//! crate, rewrite a copy of the item with `Referrer::visitor(context, ..)`.
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
//! # A run that tests nothing fails
//!
//! A rejection counts as a success of the property, so the test could pass while checking nothing.
//! It does not:
//!
//! - The watchdog. A crate that is not a case (the helper crate `tests/crates/macro`, `type-leak`,
//!   a build script) failing to build, or cargo failing without a compile error, fails the run
//!   (`build_problem()`), and so does a case that built no executable or no Referrer's library
//!   without an error saying why, a binary that fails or prints a line that is not a report, and
//!   an item with no report (`missing_artifacts()`). None of them is read as "rejected".
//! - The acceptance floor. The Leaker must accept at least a share of the items analyzed of each
//!   kind (struct, enum, union, trait, inherent impl, trait impl, module) and of each scope they
//!   are analyzed through, and the total accepted must not be 0 (`Floors`, `FLOORS`: they come
//!   from the rates measured over several seeds, and the failure says which kind is below its
//!   floor and why its items were rejected). A kind with fewer than `TYPE_LEAK_PROP_MIN_SAMPLES`
//!   items is not checked. At least `TYPE_LEAK_PROP_MIN_CHECKED` percent of the type positions of
//!   the accepted items must have been compared with their witness.
//!
//! The generator keeps the shapes that the Leaker rejects (a predicate that names a leaked type
//! that uses a generic, an `impl Trait` argument, ...) in a share of the items (`gen::Gen::cyclic`),
//! so that most items are accepted: an item is rejected with its first such shape, and a module
//! with the first of its dozen items.
//!
//! What the macro analyzes besides `intern_*`: a fifth of the structs, enums and unions are
//! analyzed nested in `enter_mod` (`View::visit_item`, its generics are local) or in
//! `enter_generics` (no `Self`), a tenth of the traits nested in `enter_mod`, another tenth with
//! `enter_trait`, and a third of the impls whose generics name no leaked type with `enter_impl`,
//! where only the self type and the arguments of the trait are visited (the shape of a macro that
//! resolves the trait by its name), and only those are compared. What it cannot express: the
//! impls in `enter_mod` (the rewritten self type is the `Self` of the entries, which the nested
//! contexts do not give; the macro cannot write a nested impl's own `Self`), and `enter_trait` for a
//! trait with a generic associated type (the scope does not know its items). A struct, an enum, a
//! union or a trait nested in `enter_mod` names its own `Self` (`nested_self` below): the macro
//! writes its entries with `Visitor::with_nested_self()`, and the check writes the Leaker's type
//! for a `Self` that no entry has.
//!
//! Environment variables:
//!
//! - `TYPE_LEAK_PROP_CASES` (default 24): the number of cases.
//! - `TYPE_LEAK_PROP_SEED` (default 1): the seed of the generator.
//! - `TYPE_LEAK_PROP_ITEMS` (default 8): the most items in a module.
//! - `TYPE_LEAK_PROP_MIN_ACCEPT`: the acceptance floors, in percent. A number sets every kind and
//!   scope, `name=number` one of them, read in order: `0` turns the floors off (the total accepted
//!   must still be positive), `40,mod=10,enter_impl=70` is 40% but 10% for modules and 70% for
//!   `enter_impl`. The names are the kinds of the table of the summary and the scopes `enter_mod`,
//!   `enter_generics`, `enter_trait` and `enter_impl`.
//! - `TYPE_LEAK_PROP_MIN_SAMPLES` (default 12): the fewest items of a kind for its floor to count.
//! - `TYPE_LEAK_PROP_MIN_CHECKED` (default 30): the least percentage of type positions compared.
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
//!   `item_predicate_cycle` (F33), `self_projection_outlives` (F34), `gat_where_clause` (F35) and
//!   `nested_self` (F36: a struct, an enum, a union or a trait analyzed nested in `enter_mod` that
//!   names its own `Self`, which the Visitor writes with `with_nested_self()`) are fixed findings,
//!   each in a share of the items (`gen::Gen::cyclic`): they are on by default (the Leaker encodes
//!   or rewrites them, records what they need, or rejects and counts what it cannot copy, and the
//!   Referrer names the lifetime of an elided signature where `Self` adds lifetimes to it). `alias_item` (F26, a limitation) is a known failure that the
//!   generator does not make by default: with it, or `all`, the test fails with the reproduction
//!   of that finding (E0277). The other findings of `tests/findings.rs` (F22 to F24: the Leaker
//!   rejected what compiles, F29, F31, F32) are not features: a rejection is counted as success, and
//!   the test cannot show a toolchain or a lint.
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

use gen::{generate, Case, Features, KINDS, SCOPES};
use json::Json;
use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, HashMap, HashSet};
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
    /// The targets that cargo built (or found fresh), by name.
    artifacts: HashSet<String>,
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
        artifacts: HashSet::new(),
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
                if let Some(name) = json.path("target.name").and_then(Json::str) {
                    build.artifacts.insert(name.to_string());
                    if let Some(exe) = json.get("executable").and_then(Json::str) {
                        build
                            .executables
                            .insert(name.to_string(), PathBuf::from(exe));
                    }
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
// ---------------------------------------------------------------------------------------------
// The watchdog: a run that tested nothing must not pass.
// ---------------------------------------------------------------------------------------------

/// The packages of the generated cases.
fn case_packages(cases: usize) -> HashSet<String> {
    (0..cases)
        .flat_map(|i| [leaker_name(i), referrer_name(i)])
        .collect()
}

/// Why a build tells nothing about the cases, or `None`. A failure of the helper crate
/// (`tests/crates/macro`), of `type-leak`, or of cargo itself is not a finding of a case, and a
/// case that no crate of it was built for would count as "rejected" or "nothing to check".
fn build_problem(build: &Build, known: &HashSet<String>) -> Option<String> {
    if !build.success && build.diags.is_empty() {
        return Some(format!(
            "cargo failed without a compile error:\n{}",
            excerpt(&build.stderr, 60)
        ));
    }
    let helper: Vec<&Diag> = build
        .diags
        .iter()
        .filter(|d| !known.contains(&d.package))
        .collect();
    if helper.is_empty() {
        return None;
    }
    let mut message = String::from(
        "a crate that is not a generated case fails to build (the helper crate \
         `tests/crates/macro`, `type-leak`, or a build script): nothing was tested, and this is \
         not a finding of the property.\n",
    );
    for diag in helper.iter().take(3) {
        write!(
            message,
            "\n--- package `{}`:\n{}\n",
            diag.package,
            excerpt(&diag.rendered, 30)
        )
        .unwrap();
    }
    Some(message)
}

/// What the build left of the case `i` when none of its crates reported an error: the Leaker's
/// executable and the Referrer's library. A missing one is reported, not counted as a rejection.
fn missing_artifacts(build: &Build, i: usize, leaker_ok: bool) -> Vec<String> {
    let mut missing = Vec::new();
    if !build.executables.contains_key(&leaker_name(i)) {
        missing.push(format!(
            "no executable was built for `{}`, so the macros reported nothing",
            leaker_name(i)
        ));
    }
    if leaker_ok && !build.artifacts.contains(&referrer_name(i)) {
        missing.push(format!(
            "`{}` was not built, and reported no error",
            referrer_name(i)
        ));
    }
    missing
}

// ---------------------------------------------------------------------------------------------
// The statistics and the acceptance floor.
// ---------------------------------------------------------------------------------------------

/// What the macros reported over all the cases.
#[derive(Default)]
struct Stats {
    /// The items generated, by kind (a module counts what it holds too).
    generated: BTreeMap<&'static str, usize>,
    accepted: BTreeMap<&'static str, usize>,
    rejected: BTreeMap<&'static str, usize>,
    /// The same by the scope an item was analyzed through (`gen::SCOPES`, by the name of the API).
    scope_accepted: BTreeMap<&'static str, usize>,
    scope_rejected: BTreeMap<&'static str, usize>,
    /// The reason of each rejection, by kind.
    reasons: BTreeMap<(&'static str, String), usize>,
    analyzed: usize,
    entries: usize,
    positions: usize,
    checked: usize,
}

fn kind_of(name: &str) -> &'static str {
    KINDS.iter().find(|k| **k == name).copied().unwrap_or("?")
}

impl Stats {
    fn collect(cases: &[Case], reports: &BTreeMap<(usize, usize), Report>) -> Stats {
        let mut stats = Stats::default();
        for (case_index, case) in cases.iter().enumerate() {
            for item in &case.items {
                for kind in &item.contents {
                    *stats.generated.entry(kind).or_default() += 1;
                }
                if item.kind == "mod" {
                    *stats.generated.entry("mod").or_default() += 1;
                }
                if let Some(report) = reports.get(&(case_index, item.id)) {
                    stats.analyzed += 1;
                    let kind = kind_of(&report.kind);
                    if let Some((_, api)) = SCOPES.iter().find(|(scope, _)| *scope == item.scope) {
                        let counts = if report.accepted {
                            &mut stats.scope_accepted
                        } else {
                            *stats
                                .reasons
                                .entry((api, reason_head(&report.detail)))
                                .or_default() += 1;
                            &mut stats.scope_rejected
                        };
                        *counts.entry(api).or_default() += 1;
                    }
                    if report.accepted {
                        *stats.accepted.entry(kind).or_default() += 1;
                        stats.positions += report.positions;
                        stats.checked += report.checked;
                        stats.entries += report.entries;
                    } else {
                        *stats.rejected.entry(kind).or_default() += 1;
                        *stats
                            .reasons
                            .entry((kind, reason_head(&report.detail)))
                            .or_default() += 1;
                    }
                }
            }
        }
        stats
    }

    /// The items accepted of a kind, or analyzed through a scope (the name of its API).
    fn accepted_of(&self, name: &str) -> usize {
        self.accepted.get(name).copied().unwrap_or(0)
            + self.scope_accepted.get(name).copied().unwrap_or(0)
    }

    fn rejected_of(&self, name: &str) -> usize {
        self.rejected.get(name).copied().unwrap_or(0)
            + self.scope_rejected.get(name).copied().unwrap_or(0)
    }

    fn total_accepted(&self) -> usize {
        self.accepted.values().sum()
    }

    /// The reasons of the rejections of a kind, most frequent first.
    fn reasons_of(&self, kind: &str) -> Vec<(&str, usize)> {
        let mut reasons: Vec<(&str, usize)> = self
            .reasons
            .iter()
            .filter(|((k, _), _)| *k == kind)
            .map(|((_, reason), n)| (reason.as_str(), *n))
            .collect();
        reasons.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
        reasons
    }
}

/// The least share of the analyzed items of each kind that the Leaker must accept, in percent.
///
/// A test that counts a rejection as a success passes when the Leaker rejects everything, or when
/// the generator makes shapes that cannot be leaked (before the floors, 93% of the modules and
/// 60% of the traits were rejected, and a macro that failed every `intern_*` still printed `ok`).
/// These floors fail the run then. Measured over the seeds 1 to 12 at 60 cases, the accepted share
/// was, per kind, lowest and mean: struct 93/95, enum 92/95, union 90/95, trait 71/74, inherent
/// impl 83/89, trait impl 71/80, module 65/74, and per scope `enter_mod` 89/91, `enter_generics`
/// 90/94, `enter_trait` 66/74, `enter_impl` 100/100. A floor is about a quarter below the lowest,
/// so that a different seed or a short run (24 cases has a third of the items) does not trip it,
/// but the old generator (module 10%, trait 39%, inherent and trait impl 41%) does.
/// `TYPE_LEAK_PROP_MIN_ACCEPT` overrides them.
const FLOORS: &[(&str, f64)] = &[
    ("struct", 75.0),
    ("enum", 75.0),
    ("union", 70.0),
    ("trait", 50.0),
    ("impl-inherent", 60.0),
    ("impl-trait", 50.0),
    ("mod", 40.0),
    // The items analyzed through a scope (`gen::SCOPES`), by the name of its API.
    ("enter_mod", 70.0),
    ("enter_generics", 70.0),
    ("enter_trait", 40.0),
    ("enter_impl", 70.0),
];

/// The fewest analyzed items of a kind for its floor to count (a short run has too few of the
/// rare kinds to say anything about their rate): `TYPE_LEAK_PROP_MIN_SAMPLES`.
const MIN_SAMPLES: usize = 12;

/// The least share of the type positions of the accepted items whose rewritten type was compared
/// with the original (the rest has an elided lifetime, `impl Trait`, or a name the module
/// declares), in percent: `TYPE_LEAK_PROP_MIN_CHECKED`. Without it the oracle could silently
/// check nothing. Measured: 48% to 53% over the seeds 1 to 12 at 60 cases (the accepted modules
/// have many positions that name what the module declares).
const MIN_CHECKED: f64 = 30.0;

#[derive(Debug, PartialEq)]
struct Floors {
    by_kind: BTreeMap<&'static str, f64>,
    min_samples: usize,
    min_checked: f64,
}

impl Floors {
    /// `spec`: the value of `TYPE_LEAK_PROP_MIN_ACCEPT`, a comma-separated list read in order of
    /// `N` (every kind) and `kind=N`: `"0"` turns the floors off, `"30,mod=5"` is 30% but 5% for
    /// a module.
    fn parse(spec: &str, min_samples: usize, min_checked: f64) -> Result<Floors, String> {
        let mut by_kind: BTreeMap<&'static str, f64> = FLOORS.iter().copied().collect();
        for token in spec.split(',').map(str::trim).filter(|t| !t.is_empty()) {
            let (kind, value) = match token.split_once('=') {
                Some((kind, value)) => (Some(kind.trim()), value.trim()),
                None => (None, token),
            };
            let value: f64 = value
                .parse()
                .map_err(|_| format!("`{}` is not a percentage", token))?;
            if !(0.0..=100.0).contains(&value) {
                return Err(format!("`{}` is not between 0 and 100", token));
            }
            match kind {
                None => {
                    for floor in by_kind.values_mut() {
                        *floor = value;
                    }
                }
                Some(kind) => match by_kind.keys().find(|k| **k == kind).copied() {
                    Some(kind) => {
                        by_kind.insert(kind, value);
                    }
                    None => {
                        return Err(format!(
                            "`{}`: no kind or scope `{}` (they are {:?})",
                            token,
                            kind,
                            by_kind.keys().collect::<Vec<_>>()
                        ))
                    }
                },
            }
        }
        Ok(Floors {
            by_kind,
            min_samples,
            min_checked,
        })
    }

    fn from_env() -> Floors {
        let spec = std::env::var("TYPE_LEAK_PROP_MIN_ACCEPT").unwrap_or_default();
        Floors::parse(
            &spec,
            env_or("TYPE_LEAK_PROP_MIN_SAMPLES", MIN_SAMPLES),
            env_or("TYPE_LEAK_PROP_MIN_CHECKED", MIN_CHECKED),
        )
        .unwrap_or_else(|e| panic!("TYPE_LEAK_PROP_MIN_ACCEPT: {}", e))
    }

    /// Why the statistics are too poor for the run to count as a test, one line each.
    fn problems(&self, stats: &Stats) -> Vec<String> {
        let mut problems = Vec::new();
        let all_off = self.by_kind.values().all(|f| *f == 0.0);
        if stats.analyzed == 0 {
            problems.push(
                "no macro was applied to any item (`analyzed` is 0): the run tested nothing"
                    .to_string(),
            );
        } else if stats.total_accepted() == 0 {
            problems.push(format!(
                "the Leaker accepted none of the {} items analyzed (a macro that fails every \
                 `intern_*` would pass otherwise: a rejection counts as a success)",
                stats.analyzed
            ));
        }
        if all_off {
            return problems;
        }
        for kind in KINDS.iter().chain(SCOPES.iter().map(|(_, api)| api)) {
            let (a, r) = (stats.accepted_of(kind), stats.rejected_of(kind));
            let floor = self.by_kind.get(kind).copied().unwrap_or(0.0);
            if a + r < self.min_samples || floor == 0.0 {
                continue;
            }
            let rate = 100.0 * a as f64 / (a + r) as f64;
            if rate < floor {
                let reasons: Vec<String> = stats
                    .reasons_of(kind)
                    .iter()
                    .take(3)
                    .map(|(reason, n)| format!("{}x {}", n, reason))
                    .collect();
                problems.push(format!(
                    "{}: {} of {} analyzed items accepted ({:.0}%), below the floor of {:.0}%; \
                     the generator makes shapes that cannot be leaked, or the Leaker rejects \
                     too much. Most frequent rejections: {}",
                    kind,
                    a,
                    a + r,
                    rate,
                    floor,
                    reasons.join("; ")
                ));
            }
        }
        if stats.total_accepted() > 0 && self.min_checked > 0.0 {
            let share = if stats.positions == 0 {
                0.0
            } else {
                100.0 * stats.checked as f64 / stats.positions as f64
            };
            if share < self.min_checked {
                problems.push(format!(
                    "only {} of {} type positions of the accepted items have a witness \
                     ({:.0}%), below {:.0}%: the oracle checks (almost) nothing",
                    stats.checked, stats.positions, share, self.min_checked
                ));
            }
        }
        problems
    }
}

fn summarize(stats: &Stats, floors: &Floors, seed: u64, n: usize, features: Features) -> String {
    let mut s = format!(
        "type-leak property test: seed {}, {} cases, features {:?}\n",
        seed, n, features
    );
    let total: usize = stats.generated.values().sum();
    writeln!(
        s,
        "  generated items: {} (nested items of a module included)",
        total
    )
    .unwrap();
    writeln!(
        s,
        "  {:<14} {:>9} {:>9} {:>9} {:>9} {:>6}",
        "kind", "generated", "accepted", "rejected", "accepted%", "floor"
    )
    .unwrap();
    for kind in KINDS
        .iter()
        .chain(["fn", "alias"].iter())
        .chain(SCOPES.iter().map(|(_, api)| api))
    {
        // An item analyzed through a scope is of a kind too: `generated` counts kinds only.
        let g = if SCOPES.iter().any(|(_, api)| api == kind) {
            stats.accepted_of(kind) + stats.rejected_of(kind)
        } else {
            stats.generated.get(kind).copied().unwrap_or(0)
        };
        if g == 0 {
            continue;
        }
        let (a, r) = (stats.accepted_of(kind), stats.rejected_of(kind));
        let rate = if a + r > 0 {
            format!("{:.0}%", 100.0 * a as f64 / (a + r) as f64)
        } else {
            "-".into()
        };
        let floor = match floors.by_kind.get(kind) {
            Some(f) if *f == 0.0 => "off".into(),
            Some(f) if a + r >= floors.min_samples => format!("{:.0}%", f),
            Some(_) => "n/a".into(),
            None => "-".into(),
        };
        writeln!(
            s,
            "  {:<14} {:>9} {:>9} {:>9} {:>9} {:>6}",
            kind, g, a, r, rate, floor
        )
        .unwrap();
    }
    writeln!(
        s,
        "  analyzed (macros applied): {}; entries: {}; type positions: {}, witnesses checked: {} ({} skipped: elided lifetimes, or names declared in a module)",
        stats.analyzed,
        stats.entries,
        stats.positions,
        stats.checked,
        stats.positions - stats.checked
    )
    .unwrap();
    for kind in KINDS.iter().chain(SCOPES.iter().map(|(_, api)| api)) {
        for (reason, n) in stats.reasons_of(kind).iter().take(3) {
            writeln!(s, "  rejected {:>4}x {:<13} {}", n, kind, reason).unwrap();
        }
    }
    s
}

// ---------------------------------------------------------------------------------------------
// The test.
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "slow, builds generated crates; run with `cargo test --test prop_cross_crate -- --ignored`"]
fn random_items_leak_across_crates() {
    let cases: usize = env_or("TYPE_LEAK_PROP_CASES", 24);
    let seed: u64 = env_or("TYPE_LEAK_PROP_SEED", 1);
    let items_per_module: usize = env_or("TYPE_LEAK_PROP_ITEMS", 8);
    let features = Features::parse(&std::env::var("TYPE_LEAK_PROP_FEATURES").unwrap_or_default());
    let floors = Floors::from_env();
    let generated: Vec<Case> = (0..cases)
        .map(|i| generate(case_seed(seed, i), i, items_per_module, features))
        .collect();

    let base = Path::new(env!("CARGO_TARGET_TMPDIR")).join("prop");
    let target = base.join("target");
    let ws = base.join(format!("ws-{}-{}", seed, cases));
    let list: Vec<(usize, &Case)> = generated.iter().enumerate().collect();
    write_workspace(&ws, &list, Mode::Real);
    let build = cargo_build(&ws, &target);

    // The watchdog: the helper crate or cargo failing is not a finding, and not a rejection.
    let known = case_packages(cases);
    if let Some(problem) = build_problem(&build, &known) {
        panic!(
            "type-leak property test: the build is broken (seed {}, {} cases):\n{}",
            seed, cases, problem
        );
    }

    let mut failures: Vec<Failure> = Vec::new();

    // The errors of each crate. The Referrer's crate is built only if the Leaker's compiles.
    let mut leaker_failed: Vec<usize> = Vec::new();
    let mut unbuilt: Vec<usize> = Vec::new();
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
        let mut referrer_failed = false;
        for diag in build.diags.iter().filter(|d| d.package == referrer) {
            referrer_failed = true;
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
        // Neither crate reported an error: both must have been built.
        if !referrer_failed {
            let missing = missing_artifacts(&build, i, true);
            if !missing.is_empty() {
                unbuilt.push(i);
                failures.push(Failure {
                    case: i,
                    item: None,
                    class: "WATCHDOG: a crate of the case was not built, and no error says why",
                    text: missing.join("\n"),
                });
            }
        }
    }

    // A Leaker's crate that does not compile: is it the generator or the macro?
    if !leaker_failed.is_empty() {
        let id_ws = base.join(format!("ws-{}-{}-identity", seed, cases));
        let list: Vec<(usize, &Case)> = leaker_failed.iter().map(|&i| (i, &generated[i])).collect();
        write_workspace(&id_ws, &list, Mode::Identity);
        let identity = cargo_build(&id_ws, &target);
        if let Some(problem) = build_problem(&identity, &known) {
            panic!(
                "type-leak property test: the build without `type-leak` is broken (seed {}, {} cases):\n{}",
                seed, cases, problem
            );
        }
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

    // What the macros reported. Every item must have a report: a missing one is a failure, not a
    // rejection.
    let mut reports: BTreeMap<(usize, usize), Report> = BTreeMap::new();
    for (i, case) in generated.iter().enumerate() {
        if leaker_failed.contains(&i) || unbuilt.contains(&i) {
            continue;
        }
        let Some(exe) = build.executables.get(&leaker_name(i)) else {
            continue;
        };
        let output = Command::new(exe).output().unwrap();
        if !output.status.success() {
            failures.push(Failure {
                case: i,
                item: None,
                class: "WATCHDOG: the binary that prints the reports fails",
                text: String::from_utf8_lossy(&output.stderr).into_owned(),
            });
            continue;
        }
        for line in String::from_utf8_lossy(&output.stdout).lines() {
            match parse_report(line) {
                Some((id, report)) => {
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
                None => failures.push(Failure {
                    case: i,
                    item: None,
                    class: "WATCHDOG: a report line cannot be read",
                    text: line.to_string(),
                }),
            }
        }
        let missing: Vec<String> = case
            .items
            .iter()
            .filter(|item| !reports.contains_key(&(i, item.id)))
            .map(|item| item.id.to_string())
            .collect();
        if !missing.is_empty() {
            failures.push(Failure {
                case: i,
                item: None,
                class: "WATCHDOG: the macros of some items reported nothing",
                text: format!("items {}", missing.join(", ")),
            });
        }
    }

    if !build.success && failures.is_empty() {
        panic!(
            "cargo failed, and no failure of a case says why (seed {}):\n{}",
            seed,
            excerpt(&build.stderr, 60)
        );
    }

    let stats = Stats::collect(&generated, &reports);
    let summary = summarize(&stats, &floors, seed, cases, features);
    eprintln!("{}", summary);
    let floor_problems = floors.problems(&stats);

    if !failures.is_empty() || !floor_problems.is_empty() {
        let mut message = format!(
            "type-leak property failed (TYPE_LEAK_PROP_SEED={} TYPE_LEAK_PROP_CASES={} TYPE_LEAK_PROP_ITEMS={} TYPE_LEAK_PROP_FEATURES={:?}):\n{} failure(s), {} acceptance problem(s).\nSources: {}\n\n{}\n",
            seed,
            cases,
            items_per_module,
            std::env::var("TYPE_LEAK_PROP_FEATURES").unwrap_or_default(),
            failures.len(),
            floor_problems.len(),
            ws.display(),
            summary
        );
        if !floor_problems.is_empty() {
            message.push_str(
                "\nACCEPTANCE: the run passes only if the Leaker accepts enough of what is \
                 generated (TYPE_LEAK_PROP_MIN_ACCEPT, TYPE_LEAK_PROP_MIN_SAMPLES and \
                 TYPE_LEAK_PROP_MIN_CHECKED change the floors; see the docs of this test):\n",
            );
            for problem in &floor_problems {
                writeln!(message, "  - {}", problem).unwrap();
            }
        }
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
        if !failures.is_empty() {
            message.push_str("\nall failures:\n");
        }
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

// ---------------------------------------------------------------------------------------------
// Tests of the checks themselves (fast; they run with `cargo test` too).
// ---------------------------------------------------------------------------------------------

#[cfg(test)]
mod checks {
    use super::*;

    fn build(success: bool, packages: &[&str]) -> Build {
        Build {
            success,
            diags: packages
                .iter()
                .map(|p| Diag {
                    package: p.to_string(),
                    rendered: "error: boom".into(),
                    lines: Vec::new(),
                })
                .collect(),
            executables: HashMap::new(),
            artifacts: HashSet::new(),
            stderr: "cargo said".into(),
        }
    }

    /// `n` reports of `kind`: `accepted` of them accepted (a case with one item each).
    fn stats_of(kinds: &[(&str, usize, usize)]) -> Stats {
        let mut stats = Stats::default();
        for (kind, accepted, rejected) in kinds {
            let kind = kind_of(kind);
            for _ in 0..*accepted {
                stats.analyzed += 1;
                *stats.accepted.entry(kind).or_default() += 1;
                stats.positions += 4;
                stats.checked += 4;
            }
            for _ in 0..*rejected {
                stats.analyzed += 1;
                *stats.rejected.entry(kind).or_default() += 1;
                *stats.reasons.entry((kind, "no".into())).or_default() += 1;
            }
        }
        stats
    }

    fn defaults() -> Floors {
        Floors::parse("", MIN_SAMPLES, MIN_CHECKED).unwrap()
    }

    #[test]
    fn a_broken_helper_crate_is_not_a_finding_and_not_a_rejection() {
        let known = case_packages(2);
        // The macro crate does not compile: no crate of any case is built, and no case reports.
        let problem = build_problem(&build(false, &["type_leak_test_macro"]), &known).unwrap();
        assert!(problem.contains("type_leak_test_macro"), "{}", problem);
        assert!(problem.contains("nothing was tested"), "{}", problem);
        // Cargo itself fails.
        let problem = build_problem(&build(false, &[]), &known).unwrap();
        assert!(problem.contains("without a compile error"), "{}", problem);
        // The errors of the cases are the findings of the test, not of the watchdog.
        assert!(build_problem(&build(false, &["tlp_l_001", "tlp_r_000"]), &known).is_none());
        assert!(build_problem(&build(true, &[]), &known).is_none());
    }

    #[test]
    fn a_case_with_nothing_built_is_reported() {
        let mut b = build(true, &[]);
        assert_eq!(missing_artifacts(&b, 3, true).len(), 2);
        b.executables.insert(leaker_name(3), PathBuf::from("/x"));
        assert_eq!(missing_artifacts(&b, 3, true).len(), 1);
        b.artifacts.insert(referrer_name(3));
        assert!(missing_artifacts(&b, 3, true).is_empty());
    }

    #[test]
    fn the_floors_fail_a_run_that_accepts_nothing() {
        // Every `intern_*` fails: all rejected.
        let all_rejected = stats_of(&[("struct", 0, 40), ("trait", 0, 40), ("mod", 0, 20)]);
        let problems = defaults().problems(&all_rejected);
        assert!(
            problems.iter().any(|p| p.contains("accepted none")),
            "{:?}",
            problems
        );
        assert!(
            problems.iter().any(|p| p.starts_with("struct:")),
            "{:?}",
            problems
        );
        // Nothing analyzed at all.
        let problems = defaults().problems(&Stats::default());
        assert!(
            problems.iter().any(|p| p.contains("tested nothing")),
            "{:?}",
            problems
        );
        // Turned off per kind, the total still has to be positive.
        let off = Floors::parse("0", MIN_SAMPLES, MIN_CHECKED).unwrap();
        assert_eq!(off.problems(&all_rejected).len(), 1);
        assert!(off.problems(&stats_of(&[("struct", 1, 40)])).is_empty());
    }

    #[test]
    fn a_kind_below_its_floor_fails_when_it_has_enough_samples() {
        let rich = stats_of(&[("struct", 90, 10), ("mod", 1, 30)]);
        let problems = defaults().problems(&rich);
        assert_eq!(problems.len(), 1, "{:?}", problems);
        assert!(problems[0].starts_with("mod:"), "{:?}", problems);
        // Too few samples of the kind: not counted.
        let poor = stats_of(&[("struct", 90, 10), ("mod", 0, MIN_SAMPLES - 1)]);
        assert!(defaults().problems(&poor).is_empty());
        // The floor is a minimum, not a bound on the rate.
        let ok = stats_of(&[("struct", 100, 0), ("mod", 4, 6)]);
        assert!(defaults().problems(&ok).is_empty());
    }

    #[test]
    fn the_floors_are_overridden_by_the_environment_format() {
        let floors = Floors::parse("30, mod=5", 7, 55.0).unwrap();
        assert_eq!(floors.by_kind["struct"], 30.0);
        assert_eq!(floors.by_kind["mod"], 5.0);
        assert_eq!((floors.min_samples, floors.min_checked), (7, 55.0));
        let floors = Floors::parse("mod=90", MIN_SAMPLES, MIN_CHECKED).unwrap();
        assert_eq!(floors.by_kind["mod"], 90.0);
        let default_of = |kind: &str| FLOORS.iter().find(|(k, _)| *k == kind).unwrap().1;
        assert_eq!(floors.by_kind["struct"], default_of("struct"));
        assert_eq!(floors.by_kind["trait"], default_of("trait"));
        // A scope's floor too.
        let floors = Floors::parse("enter_impl=1", MIN_SAMPLES, MIN_CHECKED).unwrap();
        assert_eq!(floors.by_kind["enter_impl"], 1.0);
        assert!(Floors::parse("fn=5", 1, 1.0).is_err());
        assert!(Floors::parse("150", 1, 1.0).is_err());
        assert!(Floors::parse("a", 1, 1.0).is_err());
    }

    #[test]
    fn an_oracle_that_checks_nothing_fails() {
        let mut stats = stats_of(&[("struct", 90, 10)]);
        stats.checked = 0;
        let problems = defaults().problems(&stats);
        assert!(
            problems.iter().any(|p| p.contains("witness")),
            "{:?}",
            problems
        );
        let mut stats = stats_of(&[("struct", 90, 10)]);
        stats.checked = stats.positions / 5;
        assert_eq!(defaults().problems(&stats).len(), 1);
        stats.checked = stats.positions * 9 / 10;
        assert!(defaults().problems(&stats).is_empty());
    }

    #[test]
    fn reports_are_read_back() {
        let (id, r) = parse_report("7|struct|ok|entries=2 positions=5 checked=4").unwrap();
        assert_eq!(
            (id, r.accepted, r.entries, r.positions, r.checked),
            (7, true, 2, 5, 4)
        );
        let (id, r) = parse_report("8|mod|rej|use absolute path: 'X'").unwrap();
        assert_eq!((id, r.accepted), (8, false));
        assert!(parse_report("nonsense").is_none());
    }
}
