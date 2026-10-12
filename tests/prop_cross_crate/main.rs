//! A property test of `type-leak` across crate boundaries.
//!
//! Each case is a generated Leaker's crate (a module of relative types, and random modules,
//! structs, enums, unions, traits and impls with `#[prop_leak]` on them, see `gen.rs`) and a
//! generated Referrer's crate that invokes the macro exported for each item. The macros are those
//! of `tests/crates/macro/prop.rs`, which analyze the item with the matching `intern_*` (or, for
//! some, through a scope: see below), emit the Repeater's implementations, and, in the Referrer's
//! crate, rewrite a copy of the item with `Referrer::visitor(context, ..)`.
//!
//! The property: if the Leaker accepts an item (it may reject it, which is counted, and every
//! rejection must be explained: see below), the Referrer's crate compiles, and every type position
//! of the rewritten copy is exactly the original type.
//! The latter is checked with a witness for each position: the Leaker's crate defines
//! `pub fn w_k<generics>() -> PhantomData<*mut OriginalType>`, and the Referrer's crate has
//! `fn chk_k<generics>() -> PhantomData<*mut RewrittenType> { leaker::w_k::<..>() }`, with the same
//! generics (the rewritten type is the result, as a parameter of a method is, so that the bounds
//! implied by its well-formedness, such as `T: 'a` for `&'a T`, are assumed as in a real copy).
//! `*mut` makes the types invariant: with a bare `PhantomData<T>`, which is covariant, a
//! rewritten type with a shorter lifetime in a covariant position (`&'a u8` for `&'static u8`) would
//! pass for the original, and it must be the same type, lifetimes included.
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
//! # Every rejection is explained
//!
//! A rejection counts as a success of the property, so a Leaker that rejected what it should accept
//! would pass. So every error of every rejection must be one of the reasons that the test knows
//! (`family()`: a leaked type in a bound or where predicate that rustc cannot normalize, a bare
//! trait of the prelude where the Leaker does not allow the prelude, `Self::A` where `Self` is a
//! type, the lifetime of a binder that no input keeps, a relative constant that cannot be leaked,
//! ...), and the item must have a shape that gives it: the generator tags each item with the
//! shapes it has (`gen::Item::tags`: `bound_cycle`, `self_assoc`, `binder_no_anchor`, `bare:Fn`,
//! ...). A reason outside the list fails the run ("NOT A KNOWN REASON") with the item and the
//! message, and so does a known one on an item that has no tag for it ("NO SHAPE THAT GIVES IT").
//! Either is a defect of `type-leak`, or a limitation to put in the list with its shape.
//! `TYPE_LEAK_PROP_REJECTIONS=1` prints each rejection with the tags of its item.
//!
//! # Bare and absolute spellings
//!
//! Bounds, where clauses, supertraits and trait objects name `Sized`, `?Sized`, `Send`, `Sync`,
//! `Unpin`, `Copy`, `Clone`, `Default`, `Fn`, `FnMut`, `FnOnce`, `Into`, `From`, `AsRef`, `Iterator`,
//! `PartialEq`, `Eq`, `PartialOrd` and `Ord` sometimes bare and sometimes in full
//! (`::core::marker::Sized`), whether the Leaker allows the names of the prelude or not. With it,
//! the macro analyzes the item with the traits written in full too (its twin), and the two must
//! give the same entries, generics and errors; without it, a bare trait in a bound must be
//! rejected as a relative path (`tests/crates/macro/bare.rs`). A bare `?Sized` is the exception (`?`
//! applies to the real `Sized`): the Leaker accepts it without the prelude, which
//! `gen::BARE_RELAXED_SIZED_ALWAYS_OK` says.
//!
//! # What the Referrer needs
//!
//! The macro writes the Repeater's impls of the required entries only (`Referrer::required()`) for
//! an item that the Referrer invokes as trusting, as a downstream crate does, and of all the
//! entries for one it invokes as `strict` (a quarter of the items, `gen::Item::strict`, that the
//! macro is told). A trusting Referrer that replaced an optional entry fails to compile.
//!
//! # Other checks of the macro
//!
//! - The helpers agree: `repeater_impl()` declares `impl_generics()` of the context of its entry
//!   and the parameter of the implementor, `generics()`, `iter_with_generics()`, `required()` and
//!   `must()` say the same, and `required()` comes first (`HELPERS DISAGREE`).
//! - A fifth of the cases do not allow the primitive types: `u8` is an entry then, and the copy
//!   keeps none in a type position (`PRIMITIVE KEPT`).
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
//! - `TYPE_LEAK_PROP_REJECTIONS`: if set, the rejections are printed with the tags of their items.
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
//!   `item_predicate_cycle` (F33), `self_projection_outlives` (F34), `gat_where_clause` (F35),
//!   `nested_self` (F36: a struct, an enum, a union or a trait analyzed nested in `enter_mod` that
//!   names its own `Self`, which the Visitor writes with `with_nested_self()`) and
//!   `declared_named_like_generic` (a module that declares `X`, and `crate::base::B1<X>` in
//!   `fn f<X>`) are fixed findings,
//!   each in a share of the items (`gen::Gen::cyclic`): they are on by default (the Leaker encodes
//!   or rewrites them, records what they need, or rejects and counts what it cannot copy, and the
//!   Referrer names the lifetime of an elided signature where `Self` adds lifetimes to it). `alias_item` (F26, a limitation) is a known failure that the
//!   generator does not make by default: with it, or `all`, the test fails with the reproduction
//!   of that finding (E0277). So are `bare_const_in_absolute_path` (a bare `ILEN` next to a leaked
//!   argument in a path of another crate is leaked alone as a type: E0573) and
//!   `path_named_like_generic` (`self::Rl<'a>` in a where predicate that names `Self` is not taken
//!   for a bound cycle: E0277). The other findings of `tests/findings.rs` (F22 to F24: the Leaker
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
                "    println!(\"{{}}\\u{{1e}}{{}}\", {l}::__tl_report_{id}!(), {l}::__tl_errors_{id}!());",
                l = leaker_name(*i),
                id = item.id
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

/// One error of a rejection (`NotInternableError::items()`): the message, the tokens it is about
/// and its first note.
#[derive(Debug, Default, Clone)]
struct Rejection {
    message: String,
    tokens: String,
    note: String,
}

/// What a macro reported for an item.
#[derive(Debug, Default, Clone)]
struct Report {
    kind: String,
    accepted: bool,
    /// The reason of the rejection (the messages of its errors, on one line), or the error of the
    /// rewrite.
    detail: String,
    /// The errors of the rejection.
    rejections: Vec<Rejection>,
    positions: usize,
    checked: usize,
    entries: usize,
    /// The optional entries whose Repeater impl the macro did not write (the Referrer is
    /// trusting).
    skipped: usize,
    /// The item has bare traits of the prelude and was compared with its twin.
    twin: bool,
}

fn parse_report(line: &str) -> Option<(usize, Report)> {
    // The errors of a rejection follow the report.
    let (line, errors) = line.split_once('\u{1e}').unwrap_or((line, ""));
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
                Some(("skipped", n)) => report.skipped = n.parse().unwrap_or(0),
                Some(("twin", n)) => report.twin = n == "1",
                Some(("rewrite-error", _)) => report.detail = rest.to_string(),
                _ => (),
            }
        }
        if let Some(at) = rest.find("rewrite-error=") {
            report.detail = rest[at..].to_string();
        }
    } else {
        report.rejections = errors
            .split('\u{1d}')
            .filter(|error| !error.is_empty())
            .map(|error| {
                let mut fields = error.splitn(3, '\u{1f}');
                Rejection {
                    message: fields.next().unwrap_or("").to_string(),
                    tokens: fields.next().unwrap_or("").to_string(),
                    note: fields.next().unwrap_or("").to_string(),
                }
            })
            .collect();
        report.detail = report
            .rejections
            .iter()
            .map(|r| r.message.as_str())
            .collect::<Vec<_>>()
            .join(" ; ");
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
// Every rejection is explained.
// ---------------------------------------------------------------------------------------------

/// The traits of the prelude that the generator spells bare in bounds (`gen::absolute_trait()`),
/// the names of the tags `bare:<Name>`.
const BARE_TRAITS: &[&str] = &[
    "Sized",
    "Send",
    "Sync",
    "Unpin",
    "Copy",
    "Clone",
    "Default",
    "Fn",
    "FnMut",
    "FnOnce",
    "Into",
    "From",
    "AsRef",
    "Iterator",
    "PartialEq",
    "PartialOrd",
    "Eq",
    "Ord",
];

/// A reason for which `type-leak` may reject an item: what it is, and the shapes of an item (its
/// tags, see `gen::Item::tags`) of which the item needs one to justify it.
#[derive(Debug, PartialEq)]
struct Family {
    name: &'static str,
    tags: Vec<String>,
}

fn family_of(name: &'static str, tags: &[&str]) -> Option<Family> {
    Some(Family {
        name,
        tags: tags.iter().map(|tag| tag.to_string()).collect(),
    })
}

/// The reason of a rejection among the ones that the generator knows (the allow-list), or `None`.
///
/// A rejection counts as a success of the property, so one must not hide a defect: each error of
/// it is one of these, and the generator has put its shape on the item (`family().tags`). A new
/// reason fails the run with the item and the message, to be judged: a limitation to list, or a
/// defect.
fn family(error: &Rejection) -> Option<Family> {
    let message = error.message.as_str();
    if message.contains("names `Self` in a bound or where predicate") {
        return family_of(
            "a leaked type that names `Self` in a bound or where predicate of its trait",
            &[gen::TAG_BOUND_CYCLE, gen::TAG_SELF_IN_BOUNDS],
        );
    }
    if message.contains("is in a bound or where predicate") {
        return family_of(
            "a leaked type in a bound or where predicate that rustc cannot normalize",
            &[gen::TAG_BOUND_CYCLE],
        );
    }
    if message.contains("is in the self type of the impl") {
        return family_of(
            "a type in the self type of an impl whose context depends on `Self` through a bound",
            &[gen::TAG_SELF_IN_BOUNDS, gen::TAG_BOUND_CYCLE],
        );
    }
    if message.contains("has a generic whose bound mentions `Self`") {
        return family_of(
            "a type that uses a generic whose bound mentions a `Self` that cannot be leaked",
            &[
                gen::TAG_BOUND_CYCLE,
                gen::TAG_SELF_IN_BOUNDS,
                gen::TAG_SELF_DENIED,
            ],
        );
    }
    if message.starts_with("write `<Self as ") {
        return family_of("`Self::A` where `Self` is a type", &[gen::TAG_SELF_ASSOC]);
    }
    if message.contains("leaves the lifetime") {
        return family_of(
            "a lifetime of a method that only a leaked type keeps (E0195)",
            &[gen::TAG_METHOD_LIFETIME],
        );
    }
    // A bare name (`u32`, `T`, `R0`) as an argument of a type that cannot be leaked as a whole:
    // it may be a const, so it is not leaked alone. The note says why the holder cannot be leaked,
    // a reason that is an error of the item too.
    if message.contains("may be a const, and the type that holds it cannot be leaked as a whole") {
        let tags: &[&str] = if error.note.contains("because of `'") {
            &[gen::TAG_METHOD_LIFETIME, gen::TAG_BINDER_NO_ANCHOR]
        } else {
            &[
                gen::TAG_SELF_ASSOC,
                gen::TAG_SELF_DENIED,
                gen::TAG_SELF_IN_BOUNDS,
                gen::TAG_BOUND_CYCLE,
            ]
        };
        return family_of(
            "a bare name that may be a const, in a type that cannot be leaked as a whole",
            tags,
        );
    }
    if !message.starts_with("use absolute path: '") {
        return None;
    }
    // The tokens of the path: a bound with a binder starts with it (`for<'h>Fn(..)`).
    let text: String = error.tokens.replace(char::is_whitespace, "");
    let bound = match text.strip_prefix("for<") {
        Some(rest) => rest.split_once('>').map_or(rest, |(_, rest)| rest),
        None => text.as_str(),
    };
    // A path can be several reasons at once (a bare `Fn` bound with a binder, a type with `Self`
    // and a binder): the shapes that the item has for any of them explain it.
    let mut reasons: Vec<(&'static str, Vec<String>)> = Vec::new();
    for name in BARE_TRAITS {
        for spelled in [name.to_string(), format!("?{}", name)] {
            let is_it = bound.strip_prefix(spelled.as_str()).map_or(false, |rest| {
                !rest.starts_with(|c: char| c.is_alphanumeric() || c == '_')
            });
            if is_it {
                reasons.push((
                    "a bare trait of the prelude, where the Leaker does not allow the prelude",
                    vec![format!("bare:{}", spelled)],
                ));
            }
        }
    }
    // `Rl<'h5>`: a type of the module that names a lifetime of a binder (not the `for<'h5>` of one).
    if text.match_indices("<'h").any(|(at, _)| {
        !text[..at].ends_with("for") && text[at + 3..].starts_with(|c: char| c.is_ascii_digit())
    }) {
        reasons.push((
            "a type that names the lifetime of a binder that no input keeps",
            vec![gen::TAG_BINDER_NO_ANCHOR.to_string()],
        ));
    }
    // A path that names a lifetime of a method (`it cannot be leaked as a whole because of 'x`), and
    // that cannot be split either (a part of it must be leaked on its own, as a primitive type
    // that is not allowed, or a trait object takes the lifetime of the reference around it).
    if error
        .note
        .starts_with("it cannot be leaked as a whole because of `'")
    {
        reasons.push((
            "a path that names a lifetime of a method or a binder and cannot be split",
            vec![
                gen::TAG_METHOD_LIFETIME.to_string(),
                gen::TAG_BINDER_NO_ANCHOR.to_string(),
            ],
        ));
    }
    // `Self::A0`, `Tr<'a, T>::A0`, `Tr<'a>::G<'c>`: an associated type of the trait.
    let assoc = text
        .match_indices("::A")
        .any(|(at, _)| text[at + 3..].starts_with(|c: char| c.is_ascii_digit()))
        || text.contains("::G<");
    if assoc {
        reasons.push((
            "`Self::A` where `Self` is a type",
            vec![gen::TAG_SELF_ASSOC.to_string()],
        ));
    }
    // A type that names `Self`, where `allow_self` is false.
    if text.contains("Self") {
        reasons.push((
            "a type that names `Self`, which the item does not allow",
            vec![
                gen::TAG_SELF_ASSOC.to_string(),
                gen::TAG_SELF_DENIED.to_string(),
            ],
        ));
    }
    // `RLEN`, `crate::base::BLEN`: a constant, a value, which is not leaked: in the default of a
    // parameter, or in the length of an array of a type that is split into parts because it
    // names the `Self` that the item does not allow.
    if text.ends_with("LEN") {
        reasons.push((
            "a relative constant that is not leaked as a whole",
            vec![
                gen::TAG_CONST_DEFAULT.to_string(),
                gen::TAG_SELF_DENIED.to_string(),
            ],
        ));
    }
    let name = reasons.first()?.0;
    let mut tags: Vec<String> = reasons.into_iter().flat_map(|(_, tags)| tags).collect();
    tags.sort();
    tags.dedup();
    Some(Family { name, tags })
}

/// What is wrong with the way a rejection is explained, one line for each error: it is not one of
/// the reasons that are known, or the item has no shape that gives it.
fn unexplained(tags: &std::collections::BTreeSet<String>, report: &Report) -> Vec<String> {
    if report.rejections.is_empty() {
        return vec!["the report of the rejection has no error".to_string()];
    }
    let mut problems = Vec::new();
    for error in &report.rejections {
        match family(error) {
            None => problems.push(format!(
                "NOT A KNOWN REASON: {} [tokens: {}] [note: {}]",
                error.message, error.tokens, error.note
            )),
            Some(family) => {
                if !family.tags.iter().any(|tag| tags.contains(tag)) {
                    problems.push(format!(
                        "NO SHAPE THAT GIVES IT: {} [tokens: {}]: {}, and the item has none of the tags {:?} (its tags are {:?})",
                        error.message, error.tokens, family.name, family.tags, tags
                    ));
                }
            }
        }
    }
    problems
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
    /// The Repeater impls of optional entries that were not written, for the items invoked as
    /// trusting.
    skipped: usize,
    /// The items with a bare trait of the prelude that were compared with their twin (the
    /// prelude is allowed), and the rejections for a bare trait (it is not).
    twins: usize,
    bare_rejected: usize,
    /// The items analyzed where the Leaker does not allow the primitive types.
    primitive_off: usize,
    /// The rejections by their reason (`family()`).
    families: BTreeMap<&'static str, usize>,
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
                    if !case.config.primitive {
                        stats.primitive_off += 1;
                    }
                    stats.skipped += report.skipped;
                    stats.twins += usize::from(report.twin);
                    for error in &report.rejections {
                        if let Some(family) = family(error) {
                            *stats.families.entry(family.name).or_default() += 1;
                            if family.name.starts_with("a bare trait") {
                                stats.bare_rejected += 1;
                            }
                        }
                    }
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
    writeln!(
        s,
        "  bare traits of the prelude: {} items compared with their twin, {} errors of rejections; optional entries without an impl (trusting items): {}; items without allow_primitive: {}",
        stats.twins, stats.bare_rejected, stats.skipped, stats.primitive_off
    )
    .unwrap();
    for (name, n) in &stats.families {
        writeln!(s, "  explained rejection errors: {:>4}x {}", n, name).unwrap();
    }
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

    // Every rejection has a reason that is known, and an item that has its shape.
    for ((i, id), report) in &reports {
        if report.accepted {
            continue;
        }
        let item = generated[*i]
            .items
            .iter()
            .find(|item| item.id == *id)
            .unwrap();
        for problem in unexplained(&item.tags, report) {
            failures.push(Failure {
                case: *i,
                item: Some(*id),
                class: "UNEXPLAINED REJECTION: type-leak rejects an item for a reason the generator does not explain",
                text: problem,
            });
        }
    }
    if std::env::var_os("TYPE_LEAK_PROP_REJECTIONS").is_some() {
        for ((i, id), report) in &reports {
            if report.accepted {
                continue;
            }
            let item = generated[*i]
                .items
                .iter()
                .find(|item| item.id == *id)
                .unwrap();
            eprintln!(
                "REJECTED case {} item {} ({}, scope {:?}, prelude {}, tags {:?}):",
                i, id, item.kind, item.scope, generated[*i].config.prelude, item.tags
            );
            for r in &report.rejections {
                eprintln!("    {} || {} || {}", r.message, r.tokens, r.note);
            }
        }
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

    fn rejection(message: &str, tokens: &str) -> Rejection {
        Rejection {
            message: message.to_string(),
            tokens: tokens.to_string(),
            note: String::new(),
        }
    }

    fn tags(names: &[&str]) -> std::collections::BTreeSet<String> {
        names.iter().map(|name| name.to_string()).collect()
    }

    #[test]
    fn rejections_are_classified_into_the_known_reasons() {
        let tags_of =
            |message: &str, tokens: &str| family(&rejection(message, tokens)).map(|f| f.tags);
        // A bare trait of the prelude: its name is the tag, with the binder and the `?` left out.
        assert_eq!(
            tags_of("use absolute path: 'Clone'", "Clone"),
            Some(vec!["bare:Clone".to_string()])
        );
        assert_eq!(
            tags_of("use absolute path: '? Sized'", "? Sized"),
            Some(vec!["bare:?Sized".to_string()])
        );
        assert_eq!(
            tags_of(
                "use absolute path: 'for < 'h3 > Fn(& 'h3 u8) -> & 'h3 u8'",
                "for < 'h3 > Fn(& 'h3 u8) -> & 'h3 u8"
            ),
            Some(vec!["bare:Fn".to_string()])
        );
        // A path can be several reasons at once: the shapes of any of them explain it.
        let tags = tags_of(
            "use absolute path: 'Into < Self :: A0 >'",
            "Into < Self :: A0 >",
        )
        .unwrap();
        assert!(tags.contains(&"bare:Into".to_string()), "{:?}", tags);
        assert!(
            tags.contains(&gen::TAG_SELF_ASSOC.to_string()),
            "{:?}",
            tags
        );
        // The binder of a bound is not a type that names one.
        let tags = tags_of(
            "use absolute path: 'for < 'h1 > Fn(& 'h1 u8) -> & 'h1 u8'",
            "for < 'h1 > Fn(& 'h1 u8) -> & 'h1 u8",
        )
        .unwrap();
        assert!(
            !tags.contains(&gen::TAG_BINDER_NO_ANCHOR.to_string()),
            "{:?}",
            tags
        );
        // The name of another trait is not the name of the prelude's (`Fn` of `FnMut`, `Fnord`).
        assert_eq!(
            tags_of("use absolute path: 'Fnord'", "Fnord"),
            None,
            "a relative path that is not a trait of the prelude is not explained"
        );
        assert_eq!(
            tags_of("use absolute path: 'FnMut(u8)'", "FnMut(u8)"),
            Some(vec!["bare:FnMut".to_string()])
        );
        // The other reasons.
        assert_eq!(
            tags_of("use absolute path: 'Rl < 'h5 >'", "Rl < 'h5 >"),
            Some(vec![gen::TAG_BINDER_NO_ANCHOR.to_string()])
        );
        assert_eq!(
            tags_of(
                "use absolute path: 'Tr1_1 < 'a > :: A0'",
                "Tr1_1 < 'a > :: A0"
            ),
            Some(vec![gen::TAG_SELF_ASSOC.to_string()])
        );
        assert_eq!(
            tags_of(
                "write `<Self as ::k::Tr>::A0` instead of `Self::A0` in `Wrap<Self::A0>`: the copy cannot tell",
                "Wrap<Self::A0>"
            ),
            Some(vec![gen::TAG_SELF_ASSOC.to_string()])
        );
        assert_eq!(
            tags_of(
                "`Wrap < T >` is in a bound or where predicate, and a bound of its generics names a leaked type",
                "Wrap < T >"
            ),
            Some(vec![gen::TAG_BOUND_CYCLE.to_string()])
        );
        assert_eq!(
            tags_of(
                "use absolute path: 'crate :: base :: BLEN'",
                "crate :: base :: BLEN"
            )
            .map(|tags| tags.contains(&gen::TAG_CONST_DEFAULT.to_string())),
            Some(true)
        );
        // A reason nobody listed.
        assert_eq!(tags_of("`Self` has no meaning here", "Self"), None);
        assert_eq!(tags_of("use absolute path: 'Strange'", "Strange"), None);
    }

    #[test]
    fn a_rejection_needs_a_reason_that_is_known_and_a_shape_that_gives_it() {
        let report = |errors: Vec<Rejection>| Report {
            kind: "trait".into(),
            rejections: errors,
            ..Default::default()
        };
        let bare = report(vec![rejection("use absolute path: 'Default'", "Default")]);
        // The item has the shape: explained.
        assert!(unexplained(&tags(&["bare:Default"]), &bare).is_empty());
        // The item has none (it spells another trait bare, or none), so something else rejected it.
        let problems = unexplained(&tags(&["bare:Clone", "bound_cycle"]), &bare);
        assert_eq!(problems.len(), 1, "{:?}", problems);
        assert!(
            problems[0].starts_with("NO SHAPE THAT GIVES IT"),
            "{:?}",
            problems
        );
        // A bare `?Sized` is explained by its own tag, which the generator does not put on when
        // `gen::BARE_RELAXED_SIZED_ALWAYS_OK` is set.
        let sized = report(vec![rejection("use absolute path: '? Sized'", "? Sized")]);
        assert!(unexplained(&tags(&["bare:Sized"]), &sized).len() == 1);
        assert!(unexplained(&tags(&["bare:?Sized"]), &sized).is_empty());
        // A reason that is not in the list.
        let unknown = report(vec![rejection("`Self` has no meaning here", "Self")]);
        let problems = unexplained(&tags(&["bound_cycle"]), &unknown);
        assert!(
            problems[0].starts_with("NOT A KNOWN REASON"),
            "{:?}",
            problems
        );
        // Every error counts: one that is explained does not explain the other.
        let two = report(vec![
            rejection("use absolute path: 'Default'", "Default"),
            rejection("`Self` has no meaning here", "Self"),
        ]);
        assert_eq!(unexplained(&tags(&["bare:Default"]), &two).len(), 1);
        // A rejection with no error (the report was not read) is not explained either.
        assert_eq!(unexplained(&tags(&[]), &report(Vec::new())).len(), 1);
    }

    #[test]
    fn the_errors_of_a_rejection_are_read_back_from_the_report() {
        let line = "8|trait|rej|use absolute path: 'Clone' use absolute path: 'Default'\u{1e}\
                    use absolute path: 'Clone'\u{1f}Clone\u{1f}\u{1d}\
                    use absolute path: 'Default'\u{1f}Default\u{1f}the note";
        let (id, report) = parse_report(line).unwrap();
        assert_eq!((id, report.accepted), (8, false));
        assert_eq!(report.rejections.len(), 2);
        assert_eq!(report.rejections[1].tokens, "Default");
        assert_eq!(report.rejections[1].note, "the note");
        // An accepted item has no errors, and says what the macro wrote.
        let (_, report) =
            parse_report("3|struct|ok|entries=2 positions=5 checked=4 skipped=1 twin=1\u{1e}")
                .unwrap();
        assert!(report.accepted && report.rejections.is_empty());
        assert_eq!((report.skipped, report.twin), (1, true));
    }

    #[test]
    fn the_generated_cases_say_what_the_macro_needs() {
        let mut bare_sized = 0;
        let mut strict = 0;
        for seed in 0..12 {
            let case = generate(seed, 0, 6, Features::parse(""));
            for item in &case.items {
                let attr = case
                    .leaker
                    .lines()
                    .find(|line| line.contains(&format!("(id = {},", item.id)))
                    .unwrap_or_else(|| panic!("no attribute for item {}", item.id));
                assert!(
                    attr.contains(&format!("strict = {},", item.strict)),
                    "{} {}",
                    attr,
                    item.strict
                );
                assert!(attr.contains(&format!("primitive = {},", case.config.primitive)));
                assert!(attr.contains(&format!(
                    "relaxed_sized = {})",
                    gen::BARE_RELAXED_SIZED_ALWAYS_OK
                )));
                strict += usize::from(item.strict);
                // A bare trait without the prelude gets its tag: a bare `?Sized` unless it is accepted.
                if item.tags.contains("bare:?Sized") {
                    assert!(!case.config.prelude && !gen::BARE_RELAXED_SIZED_ALWAYS_OK);
                    bare_sized += 1;
                }
                if case.config.prelude {
                    assert!(
                        !item.tags.iter().any(|tag| tag.starts_with("bare:")),
                        "{:?}",
                        item.tags
                    );
                }
            }
        }
        assert!(strict > 0, "no item is strict in 12 cases");
        let _ = bare_sized;
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
