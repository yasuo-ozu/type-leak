//! A seeded generator of random but valid Rust: the Leaker's crate and the Referrer's crate.
//!
//! The Leaker's crate has a module `base` of types and traits (`B0`, `B1<T>`, `BL<'a, T>`, `BC<T,
//! N>`, `BB<B>`, `Bl<'a>`, `BT0`, `BT1<T>`, `BT2`, `BLEN`), then
//!
//! - modules `m0..` whose items are analyzed one by one (`intern_struct`, `intern_enum`,
//!   `intern_union`, `intern_trait`, `intern_impl`), some with a nested module, and
//! - modules `mm0..` analyzed as a whole (`intern_mod`), some with a nested module, and with free
//!   functions and type aliases besides.
//!
//! Each module declares the same types as `base` under other names (`R0`, `R1<T>`, ...), imports
//! some of those of `base` with `use` (`I1`, from `super::base` or `crate::base`, or `J1`, from the
//! Leaker's own crate by its name) and names them by path (`super::base::B1`, `crate::base::B1`,
//! `::tlp_l_000::base::B1`, `self::R1`), so the types are relative in several ways.
//!
//! Types: primitives, `String`, tuples, arrays (with literal, const generic and constant lengths),
//! `Option`, `Vec`, `Box`, `Rc`, `Arc`, `Cell`, `Result`, maps and `PhantomData` (by absolute path
//! or, if the Leaker allows the prelude, bare), references (with `'static` or named lifetimes),
//! raw pointers, function pointers (also `for<'h>`, `unsafe` and `extern`), `dyn` objects (of
//! `Debug`, `Any`, `Fn*`, `Iterator`, and the traits of the module, with bounds), the types of the
//! module with lifetime, type and const arguments, generic parameters (with `?Sized` and `T::Item`
//! if they are iterators), `Self` where it is valid, and `Self::A` in traits.
//!
//! Items: structs (named, tuple and unit), enums, unions; traits (generics with bounds, `T: 'a`
//! and where clauses, supertraits, associated types and consts, methods with every receiver,
//! generic methods, `where Self: Sized`); inherent impls (of the structs and enums, and of the
//! types of the module) and impls of the traits of the module.
//!
//! The grammar is built to be valid: every type is generated with what the generics in scope
//! promise (sizedness, bounds, outlives), so the Leaker's crate compiles by itself. If it does not,
//! the harness builds it again without `type-leak` and reports a bug of the generator.
//!
//! What `type-leak` is known to fail on is avoided unless a feature asks for it ([`Features`]):
//! each is a finding of the test, with a reduced reproduction in the report of the test. The
//! generator also avoids the constructs that cannot be copied to another crate whatever
//! `type-leak` does, such as an inherent impl on a type of another module in the copy of a module.

use std::collections::BTreeSet;

// ---------------------------------------------------------------------------------------------
// Random numbers.
// ---------------------------------------------------------------------------------------------

/// splitmix64
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut rng = Rng(seed ^ 0x9e37_79b9_7f4a_7c15);
        rng.next();
        rng
    }

    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// A number in `0..n`.
    pub fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    /// `true` with the probability `percent`%.
    pub fn chance(&mut self, percent: u32) -> bool {
        self.below(100) < percent as usize
    }

    pub fn range(&mut self, lo: usize, hi: usize) -> usize {
        lo + self.below(hi - lo + 1)
    }

    pub fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        &xs[self.below(xs.len())]
    }
}

// ---------------------------------------------------------------------------------------------
// Configuration and results.
// ---------------------------------------------------------------------------------------------

/// The kinds the statistics count.
pub const KINDS: &[&str] = &[
    "struct",
    "enum",
    "union",
    "trait",
    "impl-inherent",
    "impl-trait",
    "mod",
];

/// Constructs that `type-leak` is known to fail on (see the findings in the report of the test):
/// the generator avoids them unless they are asked for, so the test passes where nothing new is
/// found.
#[derive(Clone, Copy, Default, Debug)]
pub struct Features {
    /// Type parameters with `?Sized`: the Repeater encodes the generics of an entry in a tuple.
    pub unsized_params: bool,
    /// A bound or where predicate that names a leaked type using a generic (`T: Into<Wrap<T>>`,
    /// `T: Iterator<Item = Wrap<'a>>`): the copy of the bound is cyclic, or the type is not found.
    pub bound_cycles: bool,
    /// A field of a struct whose outlives bounds (`T: 'a`) only rustc infers (RFC 2093).
    pub inferred_outlives: bool,
    /// `Self::A`, an associated type of the trait, in an impl of the trait.
    pub impl_assoc_self: bool,
    /// `dyn Fn(&Rel<T>) -> &u8`, with the elided lifetime of the input given to the output, where
    /// the leaked type uses generics: the path that replaces it names their lifetimes, so the
    /// elision is ambiguous (E0106).
    pub elided_fn_generics: bool,
    /// A bound `T: Fn(A) -> &'static dyn X` next to other bounds: type-leak moves the where
    /// predicates to the parameter and prints `T: Fn(A) -> &'static dyn X + Y`, which is ambiguous.
    pub bound_merge_syntax: bool,
    /// `Self: 'y` in a method of an impl or a struct: the copy lacks `T: 'y` of `Self = S<T>` (E0309).
    pub impl_self_outlives: bool,
    /// An impl for a function pointer (`impl Tr for fn()`) with `where Self: 'y`: type-leak prints
    /// the predicate `fn(): 'y`, which rustc reads as the start of a return type.
    pub fn_pointer_self: bool,
    /// A leaked path (`Wrap<'x>`) that names a lifetime of a method which another argument does
    /// not keep, with the lifetime in the return type: E0195 in the impl of the Referrer.
    pub method_lifetime_paths: bool,
    /// A path relative to the crate (`crate::base::B1<R0>`) in a type of a module analyzed as a
    /// whole that mentions an item of the module: the Leaker makes it an entry, but the copy of the
    /// module hides it, because the module declares `R0` itself, so the path stays as written.
    pub module_crate_paths: bool,
}

impl Features {
    pub fn parse(list: &str) -> Self {
        let mut features = Features::default();
        for name in list.split(',').map(str::trim).filter(|n| !n.is_empty()) {
            match name {
                "all" => {
                    features.unsized_params = true;
                    features.bound_cycles = true;
                    features.inferred_outlives = true;
                    features.impl_assoc_self = true;
                    features.bound_merge_syntax = true;
                    features.elided_fn_generics = true;
                    features.impl_self_outlives = true;
                    features.fn_pointer_self = true;
                    features.method_lifetime_paths = true;
                    features.module_crate_paths = true;
                }
                "inferred_outlives" => features.inferred_outlives = true,
                "impl_assoc_self" => features.impl_assoc_self = true,
                "bound_merge_syntax" => features.bound_merge_syntax = true,
                "elided_fn_generics" => features.elided_fn_generics = true,
                "impl_self_outlives" => features.impl_self_outlives = true,
                "fn_pointer_self" => features.fn_pointer_self = true,
                "method_lifetime_paths" => features.method_lifetime_paths = true,
                "module_crate_paths" => features.module_crate_paths = true,
                "unsized_params" => features.unsized_params = true,
                "bound_cycles" => features.bound_cycles = true,
                other => panic!("unknown feature `{}`", other),
            }
        }
        features
    }
}

/// Settings of a generated crate.
#[derive(Clone, Debug)]
pub struct Config {
    /// The name of the Leaker's crate.
    pub krate: String,
    /// The Leaker allows the names of the standard prelude, and the generator uses them.
    pub prelude: bool,
    /// The Leaker allows `::std`.
    pub allow_std: bool,
    /// The Leaker allows its own crate.
    pub allow_crate: bool,
}

impl Config {
    fn allow_arg(&self) -> String {
        let mut allow = Vec::new();
        if self.allow_std {
            allow.push("::std".to_string());
        }
        if self.allow_crate {
            allow.push(format!("::{}", self.krate));
        }
        allow.join(",")
    }
}

/// An item with a macro on it, which the Referrer invokes.
#[derive(Clone)]
pub struct Item {
    pub id: usize,
    pub kind: &'static str,
    /// The module that contains it, from the root.
    pub module: Vec<String>,
    pub strict: bool,
    pub allow_self: bool,
    /// The kinds of the items in it (for a module), and the item itself otherwise.
    pub contents: Vec<&'static str>,
    /// Its source, with the attribute.
    pub source: String,
}

/// A generated case.
pub struct Case {
    pub config: Config,
    /// The Leaker's `lib.rs`, with `@ATTR@` where the macro is: `@ATTR@` is replaced.
    pub leaker: String,
    pub referrer: String,
    pub items: Vec<Item>,
    /// The line of the Referrer's `lib.rs` where each item is invoked.
    pub referrer_lines: Vec<(usize, usize)>,
}

pub const ATTR: &str = "@ATTR@";

// ---------------------------------------------------------------------------------------------
// The model: bounds, types, the generics in scope.
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
struct Bounds {
    clone: bool,
    debug: bool,
    default: bool,
}

impl Bounds {
    fn is_none(self) -> bool {
        !self.clone && !self.debug && !self.default
    }

    fn contains(self, other: Bounds) -> bool {
        (!other.clone || self.clone)
            && (!other.debug || self.debug)
            && (!other.default || self.default)
    }

    fn only_clone() -> Self {
        Bounds {
            clone: true,
            ..Default::default()
        }
    }

    fn only_debug() -> Self {
        Bounds {
            debug: true,
            ..Default::default()
        }
    }

    fn only_default() -> Self {
        Bounds {
            default: true,
            ..Default::default()
        }
    }
}

#[derive(Clone, Debug)]
struct TyParam {
    name: String,
    sized: bool,
    bounds: Bounds,
    /// The text of the bounds as written (`::core::clone::Clone`, `'a`, ...).
    bound_texts: Vec<String>,
    /// The lifetimes it outlives, as declared.
    outlives: Vec<String>,
    /// It is an `Iterator`: `T::Item` is a type.
    iter: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
enum SelfKind {
    /// `Self` has no meaning, or must not be named.
    #[default]
    No,
    Sized,
    /// A trait method: `Self` may be unsized, but the method can add `where Self: Sized`.
    MaybeUnsized,
    /// A trait: `Self` may be unsized and nothing can require otherwise.
    Unsized,
}

#[derive(Clone, Default, Debug)]
struct Scope {
    lifetimes: Vec<String>,
    /// `(a, b)` is `a: b`.
    outlives: Vec<(String, String)>,
    tys: Vec<TyParam>,
    consts: Vec<(String, &'static str)>,
    self_kind: SelfKind,
    self_outlives: Vec<String>,
    /// `Self::A` of a trait, with the bounds of `A`.
    assoc: Vec<(String, Bounds)>,
    /// `Self` is the struct, enum or union that is declared: it must not contain itself.
    adt_self: bool,
    /// The outlives bounds of the type parameters in a field are inferred by rustc.
    infer: bool,
    /// The lifetimes of a method: only references may use them.
    ref_only: Vec<String>,
    /// `(parameter, "iter" | "fn")`: a parameter has one `Iterator` and one `Fn` bound at most,
    /// or the bounds conflict.
    bounded: Vec<(String, &'static str)>,
}

/// What a type to generate must be.
#[derive(Clone, Copy)]
struct Con {
    sized: bool,
    bounds: Bounds,
    /// No generics, no lifetime other than `'static`, no `Self`.
    closed: bool,
    /// The type is not behind a pointer: `Self` would contain itself.
    direct: bool,
    /// No type parameter and no `Self`, and no lifetime but `'static`.
    no_params: bool,
    /// No lifetime but `'static`.
    no_lifetimes: bool,
}

impl Con {
    fn sized() -> Self {
        Con {
            sized: true,
            bounds: Bounds::default(),
            closed: false,
            direct: true,
            no_params: false,
            no_lifetimes: false,
        }
    }

    fn unsized_ok() -> Self {
        Con {
            sized: false,
            ..Self::sized()
        }
    }

    fn indirect(self) -> Self {
        Con {
            direct: false,
            ..self
        }
    }
}

/// A generated type with what it mentions.
#[derive(Clone, Debug, Default)]
struct Ty {
    text: String,
    tys: BTreeSet<String>,
    lts: BTreeSet<String>,
    consts: BTreeSet<String>,
    uses_self: bool,
    /// It needs `Self: Sized`.
    self_sized: bool,
    /// Only forms for which rustc infers the outlives bounds of a field.
    infer_ok: bool,
}

impl Ty {
    fn new(text: impl Into<String>) -> Self {
        Ty {
            text: text.into(),
            infer_ok: true,
            ..Default::default()
        }
    }

    fn with(text: String, parts: &[&Ty]) -> Self {
        let mut ty = Ty::new(text);
        for part in parts {
            ty.tys.extend(part.tys.iter().cloned());
            ty.lts.extend(part.lts.iter().cloned());
            ty.consts.extend(part.consts.iter().cloned());
            ty.uses_self |= part.uses_self;
            ty.self_sized |= part.self_sized;
            ty.infer_ok &= part.infer_ok;
        }
        ty
    }

    fn no_infer(mut self) -> Self {
        self.infer_ok = false;
        self
    }
}

fn outlives(sc: &Scope, ty: &Ty, lt: &str, infer: bool) -> bool {
    for p in &ty.tys {
        let declared = sc
            .tys
            .iter()
            .find(|t| &t.name == p)
            .map_or(false, |t| t.outlives.iter().any(|l| l == lt));
        if !declared && !(infer && ty.infer_ok && lt != "'static") {
            return false;
        }
    }
    for l in &ty.lts {
        if l != lt && l != "'static" && !sc.outlives.iter().any(|(a, b)| a == l && b == lt) {
            return false;
        }
    }
    !ty.uses_self || sc.self_outlives.iter().any(|l| l == lt)
}

// ---------------------------------------------------------------------------------------------
// The module environment: the names of types available in a module.
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Shape {
    Unit,
    T1,
    T2,
    /// `Name<'a, T: ?Sized + 'a>`
    Lt,
    /// `Name<'a>`
    LOnly,
    /// `Name<T: ?Sized, const N: usize>`
    TC,
    /// `Name<const B: bool>`
    CB,
}

const SHAPES: &[(&str, Shape)] = &[
    ("0", Shape::Unit),
    ("1", Shape::T1),
    ("2", Shape::T2),
    ("L", Shape::Lt),
    ("C", Shape::TC),
    ("B", Shape::CB),
    ("l", Shape::LOnly),
];

#[derive(Clone, Debug)]
struct Head {
    name: String,
    shape: Shape,
}

/// A trait for `dyn`: no generics, one type argument, or an associated type `Out`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum TShape {
    T0,
    T1,
    Out,
}

#[derive(Clone, Debug)]
struct THead {
    name: String,
    shape: TShape,
}

#[derive(Clone, Debug, Default)]
struct ModEnv {
    heads: Vec<Head>,
    theads: Vec<THead>,
    /// Paths of constants of type `usize`.
    consts: Vec<String>,
}

/// The types every module declares (`prefix` is `R`) and the `base` module has (`B`).
fn types_template(p: &str) -> String {
    let mut s = String::new();
    let phantom = "::core::marker::PhantomData";
    s += &format!("#[derive(Debug, Clone, Copy, Default, PartialEq)] pub struct {p}0;\n");
    s += &format!("pub struct {p}1<T: ?::core::marker::Sized>({phantom}<*const T>);\n");
    s += &format!("pub struct {p}2<T: ?::core::marker::Sized, U: ?::core::marker::Sized>({phantom}<(*const T, *const U)>);\n");
    s += &format!("pub struct {p}L<'a, T: ?::core::marker::Sized + 'a>({phantom}<&'a T>);\n");
    s += &format!(
        "pub struct {p}C<T: ?::core::marker::Sized, const N: usize>({phantom}<*const T>);\n"
    );
    s += &format!("pub struct {p}B<const B: bool>;\n");
    s += &format!("pub struct {p}l<'a>({phantom}<&'a ()>);\n");
    for (name, generics, args) in [
        ("1", "T: ?::core::marker::Sized", "T"),
        (
            "2",
            "T: ?::core::marker::Sized, U: ?::core::marker::Sized",
            "T, U",
        ),
        ("L", "'a, T: ?::core::marker::Sized + 'a", "'a, T"),
        ("C", "T: ?::core::marker::Sized, const N: usize", "T, N"),
        ("B", "const B: bool", "B"),
        ("l", "'a", "'a"),
    ] {
        let ty = format!("{p}{name}<{args}>");
        s += &format!(
            "impl<{generics}> ::core::clone::Clone for {ty} {{ fn clone(&self) -> {ty} {{ loop {{}} }} }}\n\
             impl<{generics}> ::core::marker::Copy for {ty} {{}}\n\
             impl<{generics}> ::core::fmt::Debug for {ty} {{ fn fmt(&self, _: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {{ loop {{}} }} }}\n\
             impl<{generics}> ::core::default::Default for {ty} {{ fn default() -> {ty} {{ loop {{}} }} }}\n"
        );
    }
    s += &format!("pub trait {p}T0 {{}}\n");
    s += &format!("pub trait {p}T1<T: ?::core::marker::Sized> {{}}\n");
    s += &format!("pub trait {p}T2 {{ type Out; }}\n");
    s += &format!("pub const {p}LEN: usize = 2;\n");
    s
}

// ---------------------------------------------------------------------------------------------
// Generics.
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum Level {
    Item,
    Method,
}

/// The generics of an item or a method.
#[derive(Clone, Debug, Default)]
struct Skeleton {
    lifetimes: Vec<String>,
    outlives: Vec<(String, String)>,
    tys: Vec<TyParam>,
    consts: Vec<(String, &'static str)>,
    preds: Vec<String>,
    bounded: Vec<(String, &'static str)>,
}

impl Skeleton {
    /// `<'a, 'b: 'a, T: ?Sized + Clone + 'a, const N: usize>`
    fn params(&self) -> String {
        let mut parts = Vec::new();
        for l in &self.lifetimes {
            let bounds: Vec<&str> = self
                .outlives
                .iter()
                .filter(|(a, _)| a == l)
                .map(|(_, b)| b.as_str())
                .collect();
            if bounds.is_empty() {
                parts.push(l.clone());
            } else {
                parts.push(format!("{}: {}", l, bounds.join(" + ")));
            }
        }
        for t in &self.tys {
            if t.bound_texts.is_empty() {
                parts.push(t.name.clone());
            } else {
                parts.push(format!("{}: {}", t.name, t.bound_texts.join(" + ")));
            }
        }
        for (n, ty) in &self.consts {
            parts.push(format!("const {}: {}", n, ty));
        }
        if parts.is_empty() {
            String::new()
        } else {
            format!("<{}>", parts.join(", "))
        }
    }

    /// `<'a, T, N>`
    fn args(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        parts.extend(self.lifetimes.iter().cloned());
        parts.extend(self.tys.iter().map(|t| t.name.clone()));
        parts.extend(self.consts.iter().map(|(n, _)| n.clone()));
        if parts.is_empty() {
            String::new()
        } else {
            format!("<{}>", parts.join(", "))
        }
    }

    fn where_clause(&self) -> String {
        if self.preds.is_empty() {
            String::new()
        } else {
            format!(" where {}", self.preds.join(", "))
        }
    }

    /// The scope with these generics added.
    fn scope(&self, outer: &Scope) -> Scope {
        let mut sc = outer.clone();
        sc.lifetimes.extend(self.lifetimes.iter().cloned());
        sc.outlives.extend(self.outlives.iter().cloned());
        sc.tys.extend(self.tys.iter().cloned());
        sc.consts.extend(self.consts.iter().cloned());
        sc.bounded.extend(self.bounded.iter().cloned());
        sc
    }
}

// ---------------------------------------------------------------------------------------------
// The generator.
// ---------------------------------------------------------------------------------------------

/// A trait declared in a module, which an impl can implement.
#[derive(Clone)]
struct TraitInfo {
    name: String,
    sk: Skeleton,
    supers: Bounds,
    assoc: Vec<(String, Bounds)>,
    /// `const C0: ::core::marker::PhantomData<..>` or `const K0: usize`, with its value in an impl.
    consts: Vec<(String, String)>,
    /// The signatures of the methods, with their where clauses.
    methods: Vec<String>,
    implemented: bool,
    /// A method or a constant names `Self::A`.
    uses_assoc_self: bool,
    /// A method has `where Self: 'x`, or a receiver `&'x self` and names `Self`.
    uses_self_outlives: bool,
}

pub struct Gen {
    pub rng: Rng,
    cfg: Config,
    features: Features,
    env: ModEnv,
    no_self: bool,
    /// A module analyzed as a whole is being generated: its copy declares its own items.
    module_mode: bool,
    /// Only absolute paths are written (no relative name, nor one of the prelude).
    abs_only: bool,
    /// No `T::Item`: it does not constrain `T` in the self type of an impl.
    no_assoc_paths: bool,
    /// The probability of `Self` where it is valid, in percent.
    self_prob: u32,
    next_id: usize,
    /// A counter for the names of lifetimes bound by `for<..>`.
    next_binder: usize,
    /// The id of the item being generated, which its names carry.
    cur_id: usize,
    /// A counter for the names of the declarations in an item or a module.
    next_name: usize,
    traits: Vec<TraitInfo>,
    /// Structs, enums and unions declared in the module, with the generics they declare.
    adts: Vec<(String, Skeleton)>,
    items: Vec<Item>,
    /// The kinds of the items in the module analyzed as a whole.
    contents: Vec<&'static str>,
}

const PRIMS: &[&str] = &[
    "u8", "u16", "u32", "i32", "i64", "usize", "bool", "char", "f64", "()",
];

impl Gen {
    pub fn new(seed: u64, krate: String, features: Features) -> Self {
        let mut rng = Rng::new(seed);
        let cfg = Config {
            krate,
            prelude: rng.chance(50),
            allow_std: rng.chance(35),
            allow_crate: rng.chance(30),
        };
        Gen {
            rng,
            cfg,
            features,
            env: ModEnv::default(),
            no_self: false,
            module_mode: false,
            abs_only: false,
            no_assoc_paths: false,
            self_prob: 30,
            next_id: 0,
            next_binder: 0,
            cur_id: 0,
            next_name: 0,
            traits: Vec::new(),
            adts: Vec::new(),
            items: Vec::new(),
            contents: Vec::new(),
        }
    }

    fn binder(&mut self) -> String {
        self.next_binder += 1;
        format!("'h{}", self.next_binder)
    }

    fn name(&mut self, prefix: &str) -> String {
        self.next_name += 1;
        format!("{}{}_{}", prefix, self.cur_id, self.next_name)
    }

    // -----------------------------------------------------------------------------------------
    // Names.
    // -----------------------------------------------------------------------------------------

    fn bare(&mut self) -> bool {
        let p = if self.cfg.prelude { 35 } else { 4 };
        let chosen = self.rng.chance(p);
        // The names of the prelude are not leaked if the Leaker allows them.
        chosen && (!self.abs_only || self.cfg.prelude)
    }

    fn clone_path(&mut self) -> String {
        if self.bare() {
            "Clone".into()
        } else {
            "::core::clone::Clone".into()
        }
    }

    fn default_path(&mut self) -> String {
        if self.bare() {
            "Default".into()
        } else {
            "::core::default::Default".into()
        }
    }

    fn bound_texts(&mut self, b: Bounds) -> Vec<String> {
        let mut texts = Vec::new();
        if b.clone {
            texts.push(self.clone_path());
        }
        if b.debug {
            texts.push("::core::fmt::Debug".to_string());
        }
        if b.default {
            texts.push(self.default_path());
        }
        texts
    }

    // -----------------------------------------------------------------------------------------
    // Types.
    // -----------------------------------------------------------------------------------------

    fn leaf(&mut self, sc: &Scope, c: Con) -> Ty {
        for _ in 0..8 {
            let pick = self.rng.below(100);
            let ty = match pick {
                0..=29 => Some(Ty::new(*self.rng.pick(PRIMS))),
                30..=34 => {
                    let bare = self.bare();
                    if !bare && self.abs_only && !self.cfg.allow_std {
                        None
                    } else {
                        Some(Ty::new(if bare {
                            "String"
                        } else {
                            "::std::string::String"
                        }))
                    }
                }
                35..=54 => self.head_unit(),
                55..=79 => self.param(sc, c),
                80..=99 => self.self_ty(sc, c),
                _ => None,
            };
            if let Some(ty) = ty {
                return ty;
            }
        }
        Ty::new("u8")
    }

    fn head_unit(&mut self) -> Option<Ty> {
        let units: Vec<String> = self
            .env
            .heads
            .iter()
            .filter(|h| h.shape == Shape::Unit)
            .map(|h| h.name.clone())
            .collect();
        if units.is_empty() {
            return None;
        }
        Some(Ty::new(self.rng.pick(&units).clone()))
    }

    fn param(&mut self, sc: &Scope, c: Con) -> Option<Ty> {
        if c.closed || c.no_params || sc.tys.is_empty() {
            return None;
        }
        let candidates: Vec<&TyParam> = sc
            .tys
            .iter()
            .filter(|p| (p.sized || !c.sized) && p.bounds.contains(c.bounds))
            .collect();
        if candidates.is_empty() {
            return None;
        }
        let p = (*self.rng.pick(&candidates)).clone();
        // `T::Item` and `<T as Iterator>::Item` are associated type paths.
        if p.iter && c.bounds.is_none() && !self.no_assoc_paths && self.rng.chance(40) {
            let text = if self.rng.chance(50) {
                format!("{}::Item", p.name)
            } else if self.bare() {
                format!("<{} as Iterator>::Item", p.name)
            } else {
                format!("<{} as ::core::iter::Iterator>::Item", p.name)
            };
            let mut ty = Ty::new(text).no_infer();
            ty.tys.insert(p.name);
            return Some(ty);
        }
        let mut ty = Ty::new(p.name.clone());
        ty.tys.insert(p.name);
        Some(ty)
    }

    fn self_ty(&mut self, sc: &Scope, c: Con) -> Option<Ty> {
        if self.no_self || c.closed || c.no_params || !c.bounds.is_none() {
            return None;
        }
        if sc.assoc.is_empty() && sc.self_kind == SelfKind::No {
            return None;
        }
        if !self.rng.chance(self.self_prob) {
            return None;
        }
        // `Self::A`
        if !sc.assoc.is_empty() && self.rng.chance(40) {
            let (name, _) = self.rng.pick(&sc.assoc).clone();
            let mut ty = Ty::new(format!("Self::{}", name)).no_infer();
            ty.uses_self = true;
            return Some(ty);
        }
        let mut ty = Ty::new("Self");
        ty.uses_self = true;
        match sc.self_kind {
            SelfKind::No => return None,
            SelfKind::Sized => {
                if sc.adt_self && c.direct {
                    return None;
                }
            }
            SelfKind::MaybeUnsized => ty.self_sized = c.sized,
            SelfKind::Unsized => {
                if c.sized {
                    return None;
                }
            }
        }
        Some(ty)
    }

    fn ty(&mut self, sc: &Scope, c: Con, depth: u32) -> Ty {
        if depth >= 3 || self.rng.chance(30) {
            return self.leaf(sc, c);
        }
        for _ in 0..6 {
            let pick = self.rng.below(100);
            let ty = match pick {
                0..=10 => self.tuple(sc, c, depth),
                11..=17 => self.array(sc, c, depth),
                18..=34 => self.std_wrap(sc, c, depth),
                35..=40 => self.std_two(sc, c, depth),
                41..=54 => self.head_app(sc, c, depth),
                55..=64 => self.reference(sc, c, depth),
                65..=68 => self.pointer(sc, c, depth),
                69..=74 => self.fn_ptr(sc, c, depth),
                75..=84 => self.dyn_obj(sc, c, depth),
                85..=89 if !c.sized => self.slice_or_str(sc, c, depth),
                _ => None,
            };
            if let Some(ty) = ty {
                return ty;
            }
        }
        self.leaf(sc, c)
    }

    fn tuple(&mut self, sc: &Scope, c: Con, depth: u32) -> Option<Ty> {
        let n = self.rng.range(2, 3);
        let elem = Con { sized: true, ..c };
        let parts: Vec<Ty> = (0..n).map(|_| self.ty(sc, elem, depth + 1)).collect();
        let text = format!(
            "({})",
            parts
                .iter()
                .map(|t| t.text.clone())
                .collect::<Vec<_>>()
                .join(", ")
        );
        Some(Ty::with(text, &parts.iter().collect::<Vec<_>>()))
    }

    fn array(&mut self, sc: &Scope, c: Con, depth: u32) -> Option<Ty> {
        let elem = Con { sized: true, ..c };
        let inner = self.ty(sc, elem, depth + 1);
        let len = self.array_len(sc, c.bounds.default || c.no_params || c.closed);
        let text = format!("[{}; {}]", inner.text, len);
        let mut ty = Ty::with(text, &[&inner]);
        if sc.consts.iter().any(|(n, _)| *n == len) {
            ty.consts.insert(len);
        }
        Some(ty)
    }

    fn array_len(&mut self, sc: &Scope, literal_only: bool) -> String {
        let usizes: Vec<&String> = sc
            .consts
            .iter()
            .filter(|(_, ty)| *ty == "usize")
            .map(|(n, _)| n)
            .collect();
        if !literal_only && !usizes.is_empty() && self.rng.chance(45) {
            return (*self.rng.pick(&usizes)).clone();
        }
        if !self.env.consts.is_empty() && self.rng.chance(20) {
            return self.rng.pick(&self.env.consts).clone();
        }
        self.rng.pick(&["2", "3", "0", "1"]).to_string()
    }

    fn std_wrap(&mut self, sc: &Scope, c: Con, depth: u32) -> Option<Ty> {
        // (names, argument may be unsized, behind a pointer, ok with the bounds)
        let which = self.rng.below(7);
        let (name, unsized_ok, indirect): (String, bool, bool) = match which {
            0 => (
                if self.abs_only {
                    if self.cfg.prelude && self.rng.chance(30) {
                        "Option".to_string()
                    } else {
                        "::core::option::Option".to_string()
                    }
                } else {
                    self.rng
                        .pick(&["::core::option::Option", "::std::option::Option", "Option"])
                        .to_string()
                },
                false,
                false,
            ),
            1 => (
                if self.bare() {
                    "Vec".into()
                } else {
                    "::std::vec::Vec".into()
                },
                false,
                true,
            ),
            2 => (
                if self.bare() {
                    "Box".into()
                } else {
                    "::std::boxed::Box".into()
                },
                true,
                true,
            ),
            3 => ("::std::rc::Rc".into(), true, true),
            4 => ("::std::sync::Arc".into(), true, true),
            5 => {
                if !c.bounds.clone && !c.bounds.debug {
                    ("::core::cell::Cell".into(), false, false)
                } else {
                    ("::std::vec::Vec".into(), false, true)
                }
            }
            _ => ("::core::marker::PhantomData".into(), true, true),
        };
        if name == "Option" && !self.cfg.prelude && !self.rng.chance(15) {
            return None;
        }
        // A path of `::std` is an optional entry unless the Leaker allows the crate.
        if self.abs_only && name.starts_with("::std") && !self.cfg.allow_std {
            return None;
        }
        let allow_unsized = unsized_ok && c.bounds.is_none();
        let inner_con = Con {
            sized: !allow_unsized,
            direct: c.direct && !indirect,
            ..c
        };
        let inner_con = if name.ends_with("PhantomData") {
            Con {
                bounds: Bounds::default(),
                ..inner_con
            }
        } else {
            inner_con
        };
        let inner = self.ty(sc, inner_con, depth + 1);
        Some(Ty::with(format!("{}<{}>", name, inner.text), &[&inner]))
    }

    fn std_two(&mut self, sc: &Scope, c: Con, depth: u32) -> Option<Ty> {
        let (name, ok) = match self.rng.below(3) {
            0 => (
                if self.bare() {
                    "Result"
                } else {
                    "::core::result::Result"
                },
                !c.bounds.default,
            ),
            1 => ("::std::collections::BTreeMap", true),
            _ => ("::std::collections::HashMap", true),
        };
        if !ok || (name == "Result" && !self.cfg.prelude && !self.rng.chance(20)) {
            return None;
        }
        if self.abs_only && name.starts_with("::std") && !self.cfg.allow_std {
            return None;
        }
        let inner = Con { sized: true, ..c };
        let direct = name.ends_with("Result");
        let inner = Con {
            direct: c.direct && direct,
            ..inner
        };
        let a = self.ty(sc, inner, depth + 1);
        let b = self.ty(sc, inner, depth + 1);
        Some(Ty::with(
            format!("{}<{}, {}>", name, a.text, b.text),
            &[&a, &b],
        ))
    }

    /// A lifetime to use in `sc`, or `None` if the type must not use one.
    ///
    /// `for_ref`: the lifetime of a reference. The lifetimes of a method are used by references
    /// only: a leaked path or trait object that names one gives a method that does not match the
    /// trait in the Referrer's crate (E0195, `method_lifetime_paths`).
    fn lifetime(&mut self, sc: &Scope, c: Con, for_ref: bool) -> String {
        let pool: Vec<&String> = sc
            .lifetimes
            .iter()
            .filter(|l| for_ref || self.features.method_lifetime_paths || !sc.ref_only.contains(*l))
            .collect();
        if c.closed || c.no_params || c.no_lifetimes || pool.is_empty() || self.rng.chance(15) {
            "'static".to_string()
        } else {
            (*self.rng.pick(&pool)).clone()
        }
    }

    fn head_app(&mut self, sc: &Scope, c: Con, depth: u32) -> Option<Ty> {
        if self.env.heads.is_empty() {
            return None;
        }
        let head = self.rng.pick(&self.env.heads.clone()).clone();
        let inner = Con {
            sized: false,
            bounds: Bounds::default(),
            direct: false,
            ..c
        };
        match head.shape {
            Shape::Unit => Some(Ty::new(head.name)),
            Shape::T1 => {
                let a = self.ty(sc, inner, depth + 1);
                Some(Ty::with(format!("{}<{}>", head.name, a.text), &[&a]))
            }
            Shape::T2 => {
                let a = self.ty(sc, inner, depth + 1);
                let b = self.ty(sc, inner, depth + 1);
                Some(Ty::with(
                    format!("{}<{}, {}>", head.name, a.text, b.text),
                    &[&a, &b],
                ))
            }
            Shape::Lt => {
                let lt = self.lifetime(sc, c, false);
                let a = self.ty(sc, inner, depth + 1);
                // The bound `T: 'a` of the definition must be written, as the README says.
                if !outlives(sc, &a, &lt, false) {
                    return None;
                }
                let mut ty =
                    Ty::with(format!("{}<{}, {}>", head.name, lt, a.text), &[&a]).no_infer();
                if lt != "'static" {
                    ty.lts.insert(lt);
                }
                Some(ty)
            }
            Shape::LOnly => {
                let lt = self.lifetime(sc, c, false);
                let mut ty = Ty::new(format!("{}<{}>", head.name, lt));
                if lt != "'static" {
                    ty.lts.insert(lt);
                }
                Some(ty)
            }
            Shape::TC => {
                let a = self.ty(sc, inner, depth + 1);
                let len = self.array_len(sc, c.no_params || c.closed);
                let arg = braced_path(&len, &sc.consts);
                let mut ty = Ty::with(format!("{}<{}, {}>", head.name, a.text, arg), &[&a]);
                if sc.consts.iter().any(|(n, _)| *n == len) {
                    ty.consts.insert(len);
                }
                Some(ty)
            }
            Shape::CB => {
                let bools: Vec<&String> = sc
                    .consts
                    .iter()
                    .filter(|(_, ty)| *ty == "bool")
                    .map(|(n, _)| n)
                    .collect();
                let value = if !bools.is_empty() && !c.no_params && !c.closed && self.rng.chance(60)
                {
                    (*self.rng.pick(&bools)).clone()
                } else {
                    self.rng.pick(&["true", "false"]).to_string()
                };
                let mut ty = Ty::new(format!("{}<{}>", head.name, value));
                if sc.consts.iter().any(|(n, _)| *n == value) {
                    ty.consts.insert(value);
                }
                Some(ty)
            }
        }
    }

    fn reference(&mut self, sc: &Scope, c: Con, depth: u32) -> Option<Ty> {
        if !c.bounds.is_none() {
            return None;
        }
        let lt = self.lifetime(sc, c, true);
        let inner_con = Con {
            sized: false,
            direct: false,
            ..c
        };
        let a = self.ty(sc, inner_con, depth + 1);
        if !outlives(sc, &a, &lt, sc.infer) {
            return None;
        }
        let m = if self.rng.chance(25) { "mut " } else { "" };
        let mut ty = Ty::with(format!("&{} {}{}", lt, m, paren(&a.text)), &[&a]);
        if lt != "'static" {
            ty.lts.insert(lt);
        }
        Some(ty)
    }

    fn pointer(&mut self, sc: &Scope, c: Con, depth: u32) -> Option<Ty> {
        if !c.bounds.is_none() {
            return None;
        }
        let inner_con = Con {
            sized: false,
            direct: false,
            ..c
        };
        let a = self.ty(sc, inner_con, depth + 1);
        let m = if self.rng.chance(50) { "const" } else { "mut" };
        Some(Ty::with(format!("*{} {}", m, paren(&a.text)), &[&a]))
    }

    fn fn_ptr(&mut self, sc: &Scope, c: Con, depth: u32) -> Option<Ty> {
        if !c.bounds.is_none() {
            return None;
        }
        let arg = Con {
            sized: true,
            direct: false,
            ..c
        };
        let prefix = *self
            .rng
            .pick(&["", "", "unsafe ", "extern \"C\" ", "unsafe extern \"C\" "]);
        let n = self.rng.range(0, 2);
        let mut parts: Vec<Ty> = Vec::new();
        let (text, hr) = if self.rng.chance(40) {
            // `for<'h> fn(&'h A, B) -> &'h R`: an input keeps `'h`.
            let a = self.ty(sc, Con::unsized_ok().with_scope(c), depth + 1);
            let h = self.binder();
            let mut args = vec![format!("&{} {}", h, paren(&a.text))];
            parts.push(a);
            // A type of the module with the lifetime of the binder: `Rl<'h>`.
            if self.rng.chance(40) {
                let heads: Vec<String> = self
                    .env
                    .heads
                    .iter()
                    .filter(|head| head.shape == Shape::LOnly)
                    .map(|head| head.name.clone())
                    .collect();
                if !heads.is_empty() {
                    args.push(format!("{}<{}>", self.rng.pick(&heads), h));
                }
            }
            for _ in 0..n {
                let t = self.ty(sc, arg, depth + 1);
                args.push(t.text.clone());
                parts.push(t);
            }
            let ret = if self.rng.chance(60) {
                let r = self.ty(sc, Con::unsized_ok().with_scope(c), depth + 1);
                let text = format!(" -> &{} {}", h, paren(&r.text));
                parts.push(r);
                text
            } else if self.rng.chance(50) {
                let r = self.ty(sc, arg, depth + 1);
                let text = format!(" -> {}", r.text);
                parts.push(r);
                text
            } else {
                String::new()
            };
            (
                format!("for<{}> {}fn({}){}", h, prefix, args.join(", "), ret),
                true,
            )
        } else {
            let mut args = Vec::new();
            for _ in 0..n {
                let t = self.ty(sc, arg, depth + 1);
                args.push(t.text.clone());
                parts.push(t);
            }
            let ret = if self.rng.chance(50) {
                let r = self.ty(sc, arg, depth + 1);
                let text = format!(" -> {}", r.text);
                parts.push(r);
                text
            } else {
                String::new()
            };
            (format!("{}fn({}){}", prefix, args.join(", "), ret), false)
        };
        let _ = hr;
        Some(Ty::with(text, &parts.iter().collect::<Vec<_>>()).no_infer())
    }

    fn slice_or_str(&mut self, sc: &Scope, c: Con, depth: u32) -> Option<Ty> {
        if self.rng.chance(30) {
            return Some(Ty::new("str"));
        }
        let elem = Con {
            sized: true,
            direct: false,
            ..c
        };
        let a = self.ty(sc, elem, depth + 1);
        Some(Ty::with(format!("[{}]", a.text), &[&a]))
    }

    /// `dyn Trait`, with arguments that do not depend on the generics (`open` allows them).
    fn dyn_bare(&mut self, sc: &Scope, depth: u32, open: bool) -> Ty {
        let closed = Con {
            sized: true,
            bounds: Bounds::default(),
            closed: !open,
            direct: false,
            no_params: false,
            no_lifetimes: false,
        };
        let mut parts: Vec<Ty> = Vec::new();
        // `elided_fn_generics` asks for the elided form, with generics in the input.
        let pick = if self.features.elided_fn_generics && open && self.rng.chance(60) {
            8
        } else {
            self.rng.below(9)
        };
        let text = match pick {
            0 => "dyn ::core::fmt::Debug".to_string(),
            1 => "dyn ::core::any::Any".to_string(),
            2 | 3 if !self.env.theads.is_empty() => {
                let ts: Vec<THead> = self.env.theads.clone();
                let t = self.rng.pick(&ts).clone();
                match t.shape {
                    TShape::T0 => format!("dyn {}", t.name),
                    TShape::T1 => {
                        let a = self.ty(sc, closed, depth + 1);
                        let text = format!("dyn {}<{}>", t.name, a.text);
                        parts.push(a);
                        text
                    }
                    TShape::Out => {
                        let a = self.ty(sc, closed, depth + 1);
                        let text = format!("dyn {}<Out = {}>", t.name, a.text);
                        parts.push(a);
                        text
                    }
                }
            }
            2 | 3 => "dyn ::core::fmt::Debug".to_string(),
            4 | 5 => {
                let f = *self.rng.pick(&["Fn", "FnMut", "FnOnce"]);
                let path = if self.cfg.prelude && self.bare() {
                    f.to_string()
                } else {
                    format!("::core::ops::{}", f)
                };
                let a = self.ty(sc, closed, depth + 1);
                let r = self.ty(sc, closed, depth + 1);
                let text = format!("dyn {}({}) -> {}", path, a.text, ret_paren(&r.text));
                parts.push(a);
                parts.push(r);
                text
            }
            6 => {
                let path = if self.bare() {
                    "Iterator"
                } else {
                    "::core::iter::Iterator"
                };
                let a = self.ty(sc, closed, depth + 1);
                let text = format!("dyn {}<Item = {}>", path, a.text);
                parts.push(a);
                text
            }
            7 => {
                // A higher-ranked bound, which the input anchors.
                let h = self.binder();
                let a = self.ty(sc, closed, depth + 1);
                let r = self.ty(sc, closed, depth + 1);
                let text = format!(
                    "dyn for<{h}> ::core::ops::Fn(&{h} {}) -> &{h} {}",
                    paren(&a.text),
                    ret_paren(&paren(&r.text))
                );
                parts.push(a);
                parts.push(r);
                text
            }
            _ => {
                // Elided lifetimes in the sugar of `Fn`: one in the input, so it is the output's.
                let a = self.ty(sc, closed, depth + 1);
                let r = self.ty(sc, closed, depth + 1);
                if a.text.contains(['\'', '&']) {
                    return self.dyn_bare(sc, depth, open);
                }
                // The generics of a leaked type are lifetimes and types in the path that replaces
                // it, which stop the input from being the only lifetime to elide
                // (`elided_fn_generics`).
                if open && !self.features.elided_fn_generics {
                    return self.dyn_bare(sc, depth, false);
                }
                let text = format!(
                    "dyn ::core::ops::Fn(&{}) -> &{}",
                    paren(&a.text),
                    ret_paren(&paren(&r.text))
                );
                parts.push(a);
                parts.push(r);
                text
            }
        };
        let text = if self.rng.chance(12) && !text.contains("Any") {
            format!("{} + ::core::marker::Send", text)
        } else {
            text
        };
        Ty::with(text, &parts.iter().collect::<Vec<_>>()).no_infer()
    }

    fn dyn_obj(&mut self, sc: &Scope, c: Con, depth: u32) -> Option<Ty> {
        if !c.bounds.is_none() {
            return None;
        }
        let open = !c.closed && !c.no_params && !c.no_lifetimes && self.rng.chance(30);
        let obj = self.dyn_bare(sc, depth, open);
        // The bound of the object: `'static` by default for a box, the lifetime of a reference.
        let lt_bound = if open || self.rng.chance(35) {
            Some(self.lifetime(sc, c, false))
        } else {
            None
        };
        if let Some(lt) = &lt_bound {
            // `Any: 'static`, so only `dyn Any + 'static` is well formed (E0478).
            if !outlives(sc, &obj, lt, false) || (obj.text.contains("Any") && lt != "'static") {
                return None;
            }
        } else if open && (!obj.tys.is_empty() || !obj.lts.is_empty() || obj.uses_self) {
            // Without a bound the object must be `'static`.
            return None;
        }
        let with_bound = |text: &str, lt: &str| format!("{} + {}", text, lt);
        let mut ty;
        // `Box` and `Rc` are paths of `::std`.
        let std_ok = !self.abs_only || self.cfg.allow_std;
        match if std_ok { self.rng.below(4) } else { 2 } {
            0 | 1 => {
                let text = match &lt_bound {
                    Some(lt) => with_bound(&obj.text, lt),
                    None => obj.text.clone(),
                };
                ty = Ty::with(format!("::std::boxed::Box<{}>", text), &[&obj]);
            }
            2 => {
                // `&'a dyn Tr`: the bound of the object is `'a`.
                let lt = match &lt_bound {
                    Some(lt) => lt.clone(),
                    None => self.lifetime(sc, c, true),
                };
                if !outlives(sc, &obj, &lt, false) {
                    return None;
                }
                let text = match &lt_bound {
                    Some(b) => format!("&{} ({})", lt, with_bound(&obj.text, b)),
                    None => format!("&{} {}", lt, paren(&obj.text)),
                };
                ty = Ty::with(text, &[&obj]);
                if lt != "'static" {
                    ty.lts.insert(lt);
                }
            }
            _ => {
                let text = match &lt_bound {
                    Some(lt) => with_bound(&obj.text, lt),
                    None => obj.text.clone(),
                };
                ty = Ty::with(format!("::std::rc::Rc<{}>", text), &[&obj]);
            }
        }
        if let Some(lt) = lt_bound {
            if lt != "'static" {
                ty.lts.insert(lt);
            }
        }
        Some(ty.no_infer())
    }

    // -----------------------------------------------------------------------------------------
    // Generics.
    // -----------------------------------------------------------------------------------------

    fn skeleton(&mut self, level: Level, outer: &Scope, want_bounds: Bounds) -> Skeleton {
        let (lts, tys, consts): (&[&str], &[&str], &[&str]) = match level {
            Level::Item => (&["'a", "'b", "'c"], &["T", "U", "V"], &["N", "K"]),
            Level::Method => (&["'x", "'y"], &["X", "Y"], &["M"]),
        };
        let mut sk = Skeleton::default();
        let n_lt = match self.rng.below(100) {
            0..=44 => 0,
            45..=79 => 1,
            _ => 2,
        }
        .min(lts.len());
        let n_ty = match self.rng.below(100) {
            0..=24 => 0,
            25..=64 => 1,
            65..=92 => 2,
            _ => 3,
        }
        .min(tys.len());
        let n_const = if self.rng.chance(30) { 1 } else { 0 };
        for l in lts.iter().take(n_lt) {
            sk.lifetimes.push(l.to_string());
        }
        if sk.lifetimes.len() >= 2 && self.rng.chance(30) {
            sk.outlives
                .push((sk.lifetimes[1].clone(), sk.lifetimes[0].clone()));
        }
        if level == Level::Method
            && !sk.lifetimes.is_empty()
            && !outer.lifetimes.is_empty()
            && self.rng.chance(15)
        {
            let outer_lt = self.rng.pick(&outer.lifetimes).clone();
            sk.outlives.push((sk.lifetimes[0].clone(), outer_lt));
        }
        let all_lts: Vec<String> = outer
            .lifetimes
            .iter()
            .chain(sk.lifetimes.iter())
            .cloned()
            .collect();
        for name in tys.iter().take(n_ty) {
            let sized = !self.features.unsized_params || self.rng.chance(82);
            let mut bounds = Bounds {
                clone: self.rng.chance(20) && sized,
                debug: self.rng.chance(20),
                default: self.rng.chance(15) && sized,
            };
            if !sized {
                bounds.clone = false;
                bounds.default = false;
            }
            if level == Level::Item && !want_bounds.is_none() {
                bounds.clone |= want_bounds.clone && sized;
                bounds.debug |= want_bounds.debug;
                bounds.default |= want_bounds.default && sized;
            }
            let mut bound_texts = Vec::new();
            if !sized {
                bound_texts.push("?::core::marker::Sized".to_string());
            }
            bound_texts.extend(self.bound_texts(bounds));
            let mut outlives_list = Vec::new();
            for l in &all_lts {
                if self.rng.chance(45) {
                    outlives_list.push(l.clone());
                    bound_texts.push(l.clone());
                }
            }
            if self.rng.chance(5) {
                outlives_list.push("'static".to_string());
                bound_texts.push("'static".to_string());
            }
            let iter = sized && self.rng.chance(10);
            if iter {
                bound_texts.push(if self.bare() {
                    "Iterator".to_string()
                } else {
                    "::core::iter::Iterator".to_string()
                });
            }
            sk.tys.push(TyParam {
                name: name.to_string(),
                sized,
                bounds,
                bound_texts,
                outlives: outlives_list,
                iter,
            });
        }
        for name in consts.iter().take(n_const) {
            let ty = if self.rng.chance(70) { "usize" } else { "bool" };
            sk.consts.push((name.to_string(), ty));
        }
        sk
    }

    /// Where predicates on the generics of `sc`.
    fn preds(&mut self, sc: &Scope, sk: &mut Skeleton, max: usize) {
        // `Self` is not known to be `Sized` while the predicates are checked.
        let mut sc = sc.clone();
        if sc.self_kind == SelfKind::MaybeUnsized {
            sc.self_kind = SelfKind::Unsized;
        }
        let sc = &sc;
        // A bound that names a leaked type that uses a generic is cyclic once rewritten, if the
        // generic has a bound itself, and a where predicate that names a lifetime is part of the
        // context of the types that use the lifetime (`bound_cycles`).
        let sized = Con {
            no_params: !self.features.bound_cycles,
            ..Con::sized()
        };
        let unsized_ok = Con {
            no_params: !self.features.bound_cycles,
            ..Con::unsized_ok()
        };
        let mut bounded = sc.bounded.clone();
        // The Repeater of a type with a bound that names an item of the module would be
        // for the Leaker's item, and the copy of the module has its own.
        let saved_env = self.env.clone();
        // Rewritten, a bound that names a leaked type is a projection, which rustc has to
        // normalize in the environment of the item while the other types of the item need the
        // original bounds to be normalized themselves: the types of a bound are absolute
        // (`bound_cycles`).
        let saved_abs = self.abs_only;
        if !self.features.bound_cycles {
            self.abs_only = true;
            let crate_ok = self.cfg.allow_crate;
            self.env
                .heads
                .retain(|h| crate_ok && h.name.starts_with("::"));
            self.env
                .theads
                .retain(|h| crate_ok && h.name.starts_with("::"));
            self.env.consts.retain(|c| crate_ok && c.starts_with("::"));
        }
        if self.module_mode {
            let declared = |name: &str| {
                name.starts_with('R') || name.starts_with("self::R") || name.starts_with("super::R")
            };
            self.env.heads.retain(|h| !declared(&h.name));
            self.env.theads.retain(|h| !declared(&h.name));
            self.env.consts.retain(|c| !declared(c));
        }
        let n = self.rng.below(max + 1);
        for _ in 0..n {
            let sized_params: Vec<TyParam> = sc.tys.iter().filter(|p| p.sized).cloned().collect();
            let pred = match self.rng.below(8) {
                0 if !sc.tys.is_empty() => {
                    let p = self.rng.pick(&sc.tys).clone();
                    let b = if p.sized {
                        *self.rng.pick(&[
                            Bounds::only_clone(),
                            Bounds::only_debug(),
                            Bounds::only_default(),
                        ])
                    } else {
                        Bounds::only_debug()
                    };
                    format!("{}: {}", p.name, self.bound_texts(b).join(" + "))
                }
                1 if !sized_params.is_empty() => {
                    let p = self.rng.pick(&sized_params).clone();
                    let a = self.ty(sc, sized, 1);
                    format!("{}: ::core::convert::Into<{}>", p.name, a.text)
                }
                2 if !sc.tys.is_empty() => {
                    let p = self.rng.pick(&sc.tys).clone();
                    let a = self.ty(sc, unsized_ok, 1);
                    format!("{}: ::core::convert::AsRef<{}>", p.name, a.text)
                }
                3 if !sized_params.is_empty() => {
                    let p = self.rng.pick(&sized_params).clone();
                    if bounded.contains(&(p.name.clone(), "iter")) {
                        continue;
                    }
                    bounded.push((p.name.clone(), "iter"));
                    let a = self.ty(sc, sized, 1);
                    format!("{}: ::core::iter::Iterator<Item = {}>", p.name, a.text)
                }
                4 if !sized_params.is_empty() => {
                    let p = self.rng.pick(&sized_params).clone();
                    if bounded.contains(&(p.name.clone(), "fn")) {
                        continue;
                    }
                    bounded.push((p.name.clone(), "fn"));
                    let a = self.ty(sc, sized, 1);
                    let r = self.ty(sc, sized, 1);
                    let r = if self.features.bound_merge_syntax {
                        r.text
                    } else {
                        ret_paren(&r.text)
                    };
                    format!("{}: ::core::ops::Fn({}) -> {}", p.name, a.text, r)
                }
                5 if !sized_params.is_empty() => {
                    let h = self.binder();
                    let p = self.rng.pick(&sized_params).clone();
                    if bounded.contains(&(p.name.clone(), "fn")) {
                        continue;
                    }
                    bounded.push((p.name.clone(), "fn"));
                    let a = self.ty(sc, sized, 1);
                    let r = self.ty(sc, sized, 1);
                    let r = paren(&r.text);
                    let r = if self.features.bound_merge_syntax {
                        r
                    } else {
                        ret_paren(&r)
                    };
                    format!(
                        "{}: for<{h}> ::core::ops::Fn(&{h} {}) -> &{h} {}",
                        p.name,
                        paren(&a.text),
                        r
                    )
                }
                6 => {
                    // A type with bounds it has: `Vec<T>: Debug`.
                    let b = *self.rng.pick(&[
                        Bounds::only_clone(),
                        Bounds::only_debug(),
                        Bounds::only_default(),
                    ]);
                    let a = self.ty(
                        sc,
                        Con {
                            bounds: b,
                            no_lifetimes: !self.features.bound_cycles,
                            ..Con::sized()
                        },
                        1,
                    );
                    format!("{}: {}", a.text, self.bound_texts(b).join(" + "))
                }
                _ => continue,
            };
            sk.preds.push(pred);
        }
        sk.bounded = bounded;
        self.env = saved_env;
        self.abs_only = saved_abs;
    }

    // -----------------------------------------------------------------------------------------
    // Methods.
    // -----------------------------------------------------------------------------------------

    /// A parameter or the result of a method. The lifetimes of the method are written in a
    /// reference at the top of the type only, which the copy keeps: a lifetime that is in a leaked
    /// type only is early-bound in the copy, and late-bound in the trait (E0195, see
    /// `method_lifetime_paths`).
    fn method_ty(&mut self, sc: &Scope) -> Ty {
        if self.features.method_lifetime_paths || sc.ref_only.is_empty() {
            return self.ty(sc, Con::sized(), 0);
        }
        let mut inner = sc.clone();
        inner.lifetimes.retain(|l| !sc.ref_only.contains(l));
        if self.rng.chance(30) {
            let lt = self.rng.pick(&sc.ref_only).clone();
            let a = self.ty(&inner, Con::unsized_ok().indirect(), 1);
            if outlives(sc, &a, &lt, false) {
                let mut ty = Ty::with(format!("&{} {}", lt, paren(&a.text)), &[&a]);
                ty.lts.insert(lt);
                return ty;
            }
        }
        self.ty(&inner, Con::sized(), 0)
    }

    /// The signature of a method with its where clause. `in_trait` is a method of a trait.
    fn method_sig(&mut self, outer: &Scope, name: &str, in_trait: bool) -> String {
        let sk = self.skeleton(Level::Method, outer, Bounds::default());
        let mut sc = sk.scope(outer);
        sc.ref_only.extend(sk.lifetimes.iter().cloned());
        // The receiver.
        let has_self = outer.self_kind != SelfKind::No;
        let typed_self_ok = !self.no_self && has_self && !outer.adt_self;
        let by_value_ok = !self.no_self && has_self;
        let mut self_sized = false;
        let receiver = match self.rng.below(100) {
            0..=9 => None,
            _ if !has_self => None,
            10..=39 => Some("&self".to_string()),
            40..=54 => Some("&mut self".to_string()),
            55..=64 if !sk.lifetimes.is_empty() => {
                let l = sk.lifetimes[0].clone();
                sc.self_outlives.push(l.clone());
                Some(format!("&{} self", l))
            }
            65..=76 if by_value_ok => {
                self_sized = true;
                Some("self".to_string())
            }
            77..=88 if typed_self_ok => Some("self: ::std::boxed::Box<Self>".to_string()),
            89..=94 if typed_self_ok => Some("self: ::std::rc::Rc<Self>".to_string()),
            _ => Some("&self".to_string()),
        };
        let mut pre_preds: Vec<String> = Vec::new();
        if !sk.lifetimes.is_empty()
            && receiver.is_some()
            && by_value_ok
            && (in_trait || self.features.impl_self_outlives)
            && self.rng.chance(12)
        {
            let l = sk.lifetimes[sk.lifetimes.len() - 1].clone();
            pre_preds.push(format!("Self: {}", l));
            sc.self_outlives.push(l);
        }
        let mut params = Vec::new();
        if let Some(r) = &receiver {
            params.push(r.clone());
        }
        let n = match self.rng.below(100) {
            0..=24 => 0,
            25..=59 => 1,
            60..=84 => 2,
            _ => 3,
        };
        for i in 0..n {
            let t = self.method_ty(&sc);
            self_sized |= t.self_sized;
            params.push(format!("a{}: {}", i, t.text));
        }
        let mut ret = String::new();
        if self.rng.chance(60) {
            let t = self.method_ty(&sc);
            self_sized |= t.self_sized;
            ret = format!(" -> {}", t.text);
        }
        let mut sk = sk;
        sk.preds = pre_preds;
        self.preds(&sc, &mut sk, 2);
        if self_sized && outer.self_kind == SelfKind::MaybeUnsized {
            sk.preds.push("Self: ::core::marker::Sized".to_string());
        }
        format!(
            "fn {}{}({}){}{}",
            name,
            sk.params(),
            params.join(", "),
            ret,
            sk.where_clause()
        )
    }
}

impl Con {
    /// Keeps the restrictions of `c` on the generics.
    fn with_scope(self, c: Con) -> Con {
        Con {
            closed: c.closed,
            no_params: c.no_params,
            no_lifetimes: c.no_lifetimes,
            direct: false,
            ..self
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Items.
// ---------------------------------------------------------------------------------------------

/// The generics of `sk` that the type `ty` uses.
fn prune(sk: &Skeleton, ty: &Ty) -> Skeleton {
    let mut out = Skeleton {
        lifetimes: sk
            .lifetimes
            .iter()
            .filter(|l| ty.lts.contains(*l))
            .cloned()
            .collect(),
        ..Default::default()
    };
    out.outlives = sk
        .outlives
        .iter()
        .filter(|(a, b)| out.lifetimes.contains(a) && out.lifetimes.contains(b))
        .cloned()
        .collect();
    for t in sk.tys.iter().filter(|t| ty.tys.contains(&t.name)) {
        let mut t = t.clone();
        let kept: Vec<String> = t
            .outlives
            .iter()
            .filter(|l| *l == "'static" || out.lifetimes.contains(l))
            .cloned()
            .collect();
        let dropped: Vec<String> = t
            .outlives
            .iter()
            .filter(|l| !kept.contains(l))
            .cloned()
            .collect();
        t.bound_texts.retain(|b| !dropped.contains(b));
        t.outlives = kept;
        out.tys.push(t);
    }
    out.consts = sk
        .consts
        .iter()
        .filter(|(n, _)| ty.consts.contains(n))
        .cloned()
        .collect();
    out
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum AdtKind {
    Struct,
    Enum,
    Union,
}

impl Gen {
    fn new_id(&mut self) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        self.cur_id = id;
        self.next_name = 0;
        id
    }

    fn attr(&self, id: usize, allow_self: bool) -> String {
        format!(
            "#[{}(id = {}, allow_self = {}, prelude = {}, allow = \"{}\")]\n",
            ATTR,
            id,
            allow_self,
            self.cfg.prelude,
            self.cfg.allow_arg()
        )
    }

    /// The names of the types available in the module `path`, and the `use`s that import some.
    fn build_env(&mut self, path: &[String]) -> (ModEnv, String) {
        let up = "super::".repeat(path.len());
        let krate = self.cfg.krate.clone();
        let mut env = ModEnv::default();
        let mut uses = String::new();
        let mut names: Vec<(String, String, Option<Shape>, Option<TShape>)> = Vec::new();
        for (suffix, shape) in SHAPES {
            names.push((suffix.to_string(), suffix.to_string(), Some(*shape), None));
        }
        for (suffix, shape) in [("T0", TShape::T0), ("T1", TShape::T1), ("T2", TShape::Out)] {
            names.push((suffix.to_string(), suffix.to_string(), None, Some(shape)));
        }
        for (suffix, _, shape, tshape) in &names {
            let mut paths: Vec<String> = Vec::new();
            // Declared in the module.
            paths.push(format!("R{}", suffix));
            if self.rng.chance(30) {
                paths.push(format!("self::R{}", suffix));
            }
            if path.len() > 1 && self.rng.chance(30) {
                paths.push(format!("super::R{}", suffix));
            }
            // Declared in `base`, by path or imported.
            paths.push(format!("{}base::B{}", up, suffix));
            paths.push(format!("crate::base::B{}", suffix));
            paths.push(format!("::{}::base::B{}", krate, suffix));
            if self.rng.chance(45) {
                let from = if self.rng.chance(70) {
                    format!("{}base", up)
                } else {
                    "crate::base".to_string()
                };
                uses += &format!("use {}::B{} as I{};\n", from, suffix, suffix);
                paths.push(format!("I{}", suffix));
            }
            if self.rng.chance(15) {
                uses += &format!("use ::{}::base::B{} as J{};\n", krate, suffix, suffix);
                paths.push(format!("J{}", suffix));
            }
            let skip_crate = self.module_mode && !self.features.module_crate_paths;
            for name in paths {
                if skip_crate && name.starts_with("crate::") {
                    continue;
                }
                if let Some(shape) = shape {
                    env.heads.push(Head {
                        name,
                        shape: *shape,
                    });
                } else if let Some(shape) = tshape {
                    env.theads.push(THead {
                        name,
                        shape: *shape,
                    });
                }
            }
        }
        env.consts.push("RLEN".to_string());
        env.consts.push("self::RLEN".to_string());
        // In a module analyzed as a whole, a constant of another module in a type that names an
        // item of the module cannot be leaked (`module_crate_paths`).
        let relative_consts = !self.module_mode || self.features.module_crate_paths;
        if relative_consts {
            env.consts.push(format!("{}base::BLEN", up));
            env.consts.push("crate::base::BLEN".to_string());
        }
        env.consts.push(format!("::{}::base::BLEN", krate));
        if self.rng.chance(30) {
            uses += &format!("use {}base::BLEN as ILEN;\n", up);
            if relative_consts {
                env.consts.push("ILEN".to_string());
            }
        }
        (env, uses)
    }

    fn adt_field_count(&mut self) -> usize {
        match self.rng.below(100) {
            0..=14 => 0,
            15..=49 => 1,
            50..=79 => 2,
            _ => 3,
        }
    }

    fn gen_adt(&mut self, kind: AdtKind, ann: Option<(usize, bool)>) -> String {
        let (prefix, kw) = match kind {
            AdtKind::Struct => ("S", "struct"),
            AdtKind::Enum => ("E", "enum"),
            AdtKind::Union => ("U", "union"),
        };
        let name = self.name(prefix);
        let allow_self = ann.map_or(true, |(_, a)| a);
        let saved_prob = self.self_prob;
        if !allow_self {
            self.self_prob = 6;
        }
        let base = Scope {
            self_kind: SelfKind::Sized,
            adt_self: true,
            infer: self.features.inferred_outlives && self.rng.chance(35),
            ..Default::default()
        };
        let mut sk = self.skeleton(Level::Item, &Scope::default(), Bounds::default());
        let sc = sk.scope(&base);

        let mut used_tys = BTreeSet::new();
        let mut used_lts = BTreeSet::new();
        let mut field = |g: &mut Gen, wrap: bool| -> String {
            let t = g.ty(&sc, Con::sized(), 0);
            used_tys.extend(t.tys.iter().cloned());
            used_lts.extend(t.lts.iter().cloned());
            if wrap {
                format!("::core::mem::ManuallyDrop<{}>", t.text)
            } else {
                t.text
            }
        };
        // The fields of each variant (one for a struct or a union).
        let n_variants = if kind == AdtKind::Enum {
            self.rng.range(1, 3)
        } else {
            1
        };
        let mut variants: Vec<Vec<String>> = Vec::new();
        for _ in 0..n_variants {
            let n = if kind == AdtKind::Union {
                self.rng.range(1, 3)
            } else {
                self.adt_field_count()
            };
            let fields: Vec<String> = (0..n)
                .map(|_| field(self, kind == AdtKind::Union))
                .collect();
            variants.push(fields);
        }
        // Every type and lifetime parameter must be used.
        let mut phantoms: Vec<String> = Vec::new();
        for l in &sk.lifetimes {
            if !used_lts.contains(l) {
                phantoms.push(format!("::core::marker::PhantomData<&{} ()>", l));
            }
        }
        for t in &sk.tys {
            if !used_tys.contains(&t.name) {
                phantoms.push(format!("::core::marker::PhantomData<*const {}>", t.name));
            }
        }
        self.preds(&sc, &mut sk, 2);
        self.self_prob = saved_prob;

        let attr = ann.map_or(String::new(), |(id, a)| self.attr(id, a));
        let params = sk.params();
        let wc = sk.where_clause();
        let text = match kind {
            AdtKind::Union => {
                let mut fields = variants.remove(0);
                for p in phantoms {
                    fields.push(format!("::core::mem::ManuallyDrop<{}>", p));
                }
                let body: Vec<String> = fields
                    .iter()
                    .enumerate()
                    .map(|(i, t)| format!("pub f{}: {}", i, t))
                    .collect();
                format!(
                    "pub union {}{}{} {{ {} }}",
                    name,
                    params,
                    wc,
                    body.join(", ")
                )
            }
            AdtKind::Struct => {
                let mut fields = variants.remove(0);
                fields.extend(phantoms);
                let style = self.rng.below(100);
                if fields.is_empty() {
                    format!("pub struct {}{}{};", name, params, wc)
                } else if style < 55 {
                    let body: Vec<String> = fields
                        .iter()
                        .enumerate()
                        .map(|(i, t)| format!("pub f{}: {}", i, t))
                        .collect();
                    format!(
                        "pub struct {}{}{} {{ {} }}",
                        name,
                        params,
                        wc,
                        body.join(", ")
                    )
                } else {
                    let body: Vec<String> = fields.iter().map(|t| format!("pub {}", t)).collect();
                    format!("pub struct {}{}({}){};", name, params, body.join(", "), wc)
                }
            }
            AdtKind::Enum => {
                let mut texts: Vec<String> = Vec::new();
                for (i, fields) in variants.iter().enumerate() {
                    let text = if fields.is_empty() {
                        format!("V{}", i)
                    } else if self.rng.chance(50) {
                        format!("V{}({})", i, fields.join(", "))
                    } else {
                        let body: Vec<String> = fields
                            .iter()
                            .enumerate()
                            .map(|(j, t)| format!("f{}: {}", j, t))
                            .collect();
                        format!("V{} {{ {} }}", i, body.join(", "))
                    };
                    texts.push(text);
                }
                if !phantoms.is_empty() {
                    texts.push(format!("Phantom({})", phantoms.join(", ")));
                }
                format!(
                    "pub enum {}{}{} {{ {} }}",
                    name,
                    params,
                    wc,
                    texts.join(", ")
                )
            }
        };
        let _ = kw;
        self.adts.push((name, sk));
        format!("{}{}\n", attr, text)
    }

    fn gen_trait(&mut self, ann: Option<(usize, bool)>) -> String {
        let name = self.name("Tr");
        let allow_self = ann.map_or(true, |(_, a)| a);
        let saved_prob = self.self_prob;
        if !allow_self {
            self.self_prob = 6;
        }
        let mut sk = self.skeleton(Level::Item, &Scope::default(), Bounds::default());
        // Supertraits.
        let supers = Bounds {
            clone: self.rng.chance(15),
            debug: self.rng.chance(25),
            default: self.rng.chance(10),
        };
        let mut super_texts = self.bound_texts(supers);
        let explicit_sized = self.rng.chance(8);
        if explicit_sized {
            super_texts.push("::core::marker::Sized".to_string());
        }
        let self_sized = supers.clone || supers.default || explicit_sized;
        // Associated types.
        let mut assoc: Vec<(String, Bounds)> = Vec::new();
        let mut assoc_texts: Vec<String> = Vec::new();
        for i in 0..self.rng.range(0, 2) {
            let b = Bounds {
                clone: self.rng.chance(30),
                debug: self.rng.chance(30),
                default: self.rng.chance(20),
            };
            let mut texts = self.bound_texts(b);
            if self.rng.chance(20) {
                texts.push("::core::convert::Into<u32>".to_string());
            }
            let n = format!("A{}", i);
            assoc_texts.push(if texts.is_empty() {
                format!("type {};", n)
            } else {
                format!("type {}: {};", n, texts.join(" + "))
            });
            assoc.push((n, b));
        }
        let base = Scope {
            self_kind: if self_sized {
                SelfKind::Sized
            } else {
                SelfKind::Unsized
            },
            assoc: assoc.clone(),
            ..Default::default()
        };
        let sc = sk.scope(&base);
        // A where clause of the trait that names `Self::A` is cyclic once the trait is implemented.
        let mut pred_scope = sc.clone();
        pred_scope.assoc.clear();
        self.preds(&pred_scope, &mut sk, 1);
        let sc = sk.scope(&base);
        // Constants.
        let mut consts: Vec<(String, String)> = Vec::new();
        let mut const_texts: Vec<String> = Vec::new();
        for i in 0..self.rng.range(0, 2) {
            if self.rng.chance(70) {
                let t = self.ty(&sc, Con::unsized_ok().indirect(), 1);
                let decl = format!("const C{}: ::core::marker::PhantomData<{}>", i, t.text);
                consts.push((decl.clone(), "::core::marker::PhantomData".to_string()));
                const_texts.push(format!("{};", decl));
            } else {
                let (ty, value) = *self
                    .rng
                    .pick(&[("usize", "0"), ("bool", "false"), ("u8", "1")]);
                let decl = format!("const K{}: {}", i, ty);
                consts.push((decl.clone(), value.to_string()));
                const_texts.push(format!("{};", decl));
            }
        }
        // Methods.
        let method_scope = Scope {
            self_kind: if self_sized {
                SelfKind::Sized
            } else {
                SelfKind::MaybeUnsized
            },
            ..sc.clone()
        };
        let mut methods: Vec<String> = Vec::new();
        for _ in 0..self.rng.range(0, 4) {
            let m = self.name("m");
            methods.push(self.method_sig(&method_scope, &m, true));
        }
        self.self_prob = saved_prob;

        let attr = ann.map_or(String::new(), |(id, a)| self.attr(id, a));
        let supers_text = if super_texts.is_empty() {
            String::new()
        } else {
            format!(": {}", super_texts.join(" + "))
        };
        let body: Vec<String> = assoc_texts
            .iter()
            .cloned()
            .chain(const_texts.iter().cloned())
            .chain(methods.iter().map(|m| format!("{};", m)))
            .collect();
        let text = format!(
            "{}pub trait {}{}{}{} {{\n{}\n}}\n",
            attr,
            name,
            sk.params(),
            supers_text,
            sk.where_clause(),
            body.join("\n")
        );
        self.traits.push(TraitInfo {
            name,
            sk,
            supers,
            assoc,
            consts,
            uses_assoc_self: body.iter().any(|item| item.contains("Self::")),
            uses_self_outlives: body.iter().any(|item| {
                item.contains("Self: '")
                    || ((item.contains("(&'x self") || item.contains("(&'y self"))
                        && item.contains("Self"))
            }),
            methods,
            implemented: false,
        });
        text
    }

    /// An impl of a trait of this module that is not implemented yet.
    fn gen_impl_trait(
        &mut self,
        ann: Option<(usize, bool)>,
        path: &[String],
        whole: bool,
    ) -> Option<String> {
        let candidates: Vec<usize> = (0..self.traits.len())
            .filter(|&i| !self.traits[i].implemented)
            .filter(|&i| !self.traits[i].uses_assoc_self || self.features.impl_assoc_self)
            .filter(|&i| {
                !self.traits[i].uses_self_outlives
                    || self.features.impl_self_outlives
                    || self.features.fn_pointer_self
            })
            .collect();
        if candidates.is_empty() {
            return None;
        }
        let idx = *self.rng.pick(&candidates);
        self.traits[idx].implemented = true;
        let tr = self.traits[idx].clone();
        let allow_self = ann.map_or(true, |(_, a)| a);
        let saved_prob = self.self_prob;
        if !allow_self {
            self.self_prob = 6;
        }
        // The impl declares the generics of the trait, and maybe one more.
        let mut sk = tr.sk.clone();
        let mut extra: Option<String> = None;
        if self.rng.chance(30) {
            let mut bounds = tr.supers;
            bounds.clone |= self.rng.chance(20);
            let mut texts = self.bound_texts(bounds);
            // Besides `Sized`, which a type parameter has by default.
            texts.retain(|t| !t.is_empty());
            sk.tys.push(TyParam {
                name: "E".to_string(),
                sized: true,
                bounds,
                bound_texts: texts,
                outlives: Vec::new(),
                iter: false,
            });
            extra = Some("E".to_string());
        }
        let sc = sk.scope(&Scope {
            self_kind: SelfKind::Sized,
            assoc: tr.assoc.clone(),
            ..Default::default()
        });
        // The self type: it cannot name `Self`.
        let saved_no_self = std::mem::replace(&mut self.no_self, true);
        let mut self_ty = self.ty(
            &sc,
            Con {
                bounds: tr.supers,
                ..Con::sized()
            },
            0,
        );
        // Not a blanket impl, which would overlap with the impl for the host in the Referrer.
        // `T::Item` is as general.
        let top_projection = |text: &str| {
            sc.tys.iter().any(|p| {
                text == p.name
                    || text.starts_with(&format!("{}::", p.name))
                    || text.starts_with(&format!("<{} as ", p.name))
            }) && !text.contains(['(', ',', '<', '['])
                || text.starts_with('<')
        };
        if top_projection(&self_ty.text) {
            self_ty = Ty::new("u8");
        }
        if !self.features.fn_pointer_self && self_ty.text.contains("fn(") {
            self_ty = Ty::new("u8");
        }
        // `fn_pointer_self` asks for a function pointer to implement the trait.
        if self.features.fn_pointer_self && !tr.supers.default && extra.is_none() {
            self_ty = Ty::new("&'static unsafe fn(i32)");
        }
        self.no_self = saved_no_self;
        let self_text = match &extra {
            Some(e) => format!("({}, {})", self_ty.text, e),
            None => self_ty.text.clone(),
        };
        self.self_prob = saved_prob;

        let tr_path = if whole {
            format!("{}{}", tr.name, tr.sk.args())
        } else {
            let mut p = format!("::{}", self.cfg.krate);
            for m in path {
                p += &format!("::{}", m);
            }
            format!("{}::{}{}", p, tr.name, tr.sk.args())
        };
        let mut body: Vec<String> = Vec::new();
        for (n, _) in &tr.assoc {
            body.push(format!("type {} = u8;", n));
        }
        for (decl, value) in &tr.consts {
            body.push(format!("{} = {};", decl, value));
        }
        for m in &tr.methods {
            body.push(format!("{} {{ loop {{}} }}", m));
        }
        let attr = ann.map_or(String::new(), |(id, a)| self.attr(id, a));
        Some(format!(
            "{}impl{} {} for {}{} {{\n{}\n}}\n",
            attr,
            sk.params(),
            tr_path,
            self_text,
            sk.where_clause(),
            body.join("\n")
        ))
    }

    fn gen_impl_inherent(&mut self, ann: Option<(usize, bool)>) -> String {
        let allow_self = ann.map_or(true, |(_, a)| a);
        let saved_prob = self.self_prob;
        if !allow_self {
            self.self_prob = 6;
        }
        // A copy of a module cannot have an inherent impl on a type of another module, which
        // the Referrer replaces by a projection.
        let own_only = self.no_self;
        let (sk, self_text) = if !self.adts.is_empty() && self.rng.chance(55) {
            // An ADT of this module, with the generics it declares.
            let (name, sk) = self.rng.pick(&self.adts.clone()).clone();
            let text = format!("{}{}", name, sk.args());
            (sk, text)
        } else {
            // A type of the module or of `base`, with generics all used. In the copy of a module, a
            // parameter that only a leaked type uses is not constrained by the self type (E0207).
            let sk = if own_only {
                Skeleton::default()
            } else {
                self.skeleton(Level::Item, &Scope::default(), Bounds::default())
            };
            let sc = sk.scope(&Scope {
                self_kind: SelfKind::Sized,
                ..Default::default()
            });
            let saved_no_self = std::mem::replace(&mut self.no_self, true);
            self.no_assoc_paths = true;
            let mut ty = None;
            let saved_heads = self.env.heads.clone();
            if own_only {
                self.env.heads.retain(|h| h.name.starts_with('R'));
            }
            for _ in 0..20 {
                if let Some(t) = self.head_app(&sc, Con::sized(), 0) {
                    ty = Some(t);
                    break;
                }
            }
            self.env.heads = saved_heads;
            self.no_assoc_paths = false;
            self.no_self = saved_no_self;
            let ty = ty.unwrap_or_else(|| Ty::new("R0"));
            (prune(&sk, &ty), ty.text)
        };
        let sc = sk.scope(&Scope {
            self_kind: SelfKind::Sized,
            ..Default::default()
        });
        let mut body: Vec<String> = Vec::new();
        for _ in 0..self.rng.range(1, 3) {
            let m = self.name("m");
            let mut sig = self.method_sig(&sc, &m, false);
            // The copy of a check has no receiver, so `Self: 'x` would not be implied.
            if !self.features.impl_self_outlives && sig.contains("Self") {
                sig = sig
                    .replace("(&'x self", "(&self")
                    .replace("(&'y self", "(&self");
            }
            body.push(format!("pub {} {{ loop {{}} }}", sig));
        }
        if self.rng.chance(35) {
            let t = self.ty(&sc, Con::unsized_ok().indirect(), 1);
            let n = self.name("C");
            body.push(format!(
                "pub const {}: ::core::marker::PhantomData<{}> = ::core::marker::PhantomData;",
                n, t.text
            ));
        }
        self.self_prob = saved_prob;
        let attr = ann.map_or(String::new(), |(id, a)| self.attr(id, a));
        format!(
            "{}impl{} {}{} {{\n{}\n}}\n",
            attr,
            sk.params(),
            self_text,
            sk.where_clause(),
            body.join("\n")
        )
    }

    /// A free function of a module analyzed as a whole.
    fn gen_fn(&mut self) -> String {
        let name = self.name("f");
        let sig = self.method_sig(&Scope::default(), &name, false);
        format!("pub {} {{ loop {{}} }}\n", sig)
    }

    /// A type alias of a module analyzed as a whole.
    fn gen_alias(&mut self) -> String {
        let name = self.name("Al");
        let sk = self.skeleton(Level::Item, &Scope::default(), Bounds::default());
        let sc = sk.scope(&Scope::default());
        let ty = self.ty(&sc, Con::sized(), 0);
        let sk = prune(&sk, &ty);
        format!("pub type {}{} = {};\n", name, sk.params(), ty.text)
    }

    /// One item of the module `path`. `whole`: the module is analyzed as a whole.
    fn gen_one(&mut self, path: &[String], whole: bool) -> String {
        let (ann, allow_self) = if whole {
            (None, true)
        } else {
            let id = self.new_id();
            let allow_self = self.rng.chance(70);
            (Some((id, allow_self)), allow_self)
        };
        let ann_id = ann.map(|(id, _)| id);
        let pick = self.rng.below(100);
        let (kind, text): (&'static str, String) = match pick {
            0..=21 => ("struct", self.gen_adt(AdtKind::Struct, ann)),
            22..=36 => ("enum", self.gen_adt(AdtKind::Enum, ann)),
            37..=43 => ("union", self.gen_adt(AdtKind::Union, ann)),
            44..=63 => ("trait", self.gen_trait(ann)),
            64..=75 => ("impl-inherent", self.gen_impl_inherent(ann)),
            76..=93 => match self.gen_impl_trait(ann, path, whole) {
                Some(text) => ("impl-trait", text),
                None => ("trait", self.gen_trait(ann)),
            },
            94..=96 if whole => {
                let text = self.gen_fn();
                self.contents.push("fn");
                return text;
            }
            97..=99 if whole => {
                let text = self.gen_alias();
                self.contents.push("alias");
                return text;
            }
            _ => ("struct", self.gen_adt(AdtKind::Struct, ann)),
        };
        if let Some(id) = ann_id {
            let strict = self.rng.chance(25);
            self.items.push(Item {
                id,
                kind,
                module: path.to_vec(),
                strict,
                allow_self,
                contents: vec![kind],
                source: text.clone(),
            });
        } else {
            self.contents.push(kind);
        }
        text
    }

    /// The body of the module `path`: the `use`s, the types it declares, items and modules.
    fn gen_module_body(
        &mut self,
        path: &[String],
        whole: bool,
        items: usize,
        nested: usize,
    ) -> String {
        let (env, uses) = self.build_env(path);
        let saved_env = std::mem::replace(&mut self.env, env);
        let saved_traits = std::mem::take(&mut self.traits);
        let saved_adts = std::mem::take(&mut self.adts);
        let mut text = String::new();
        text += &uses;
        text += &types_template("R");
        for _ in 0..items {
            text += &self.gen_one(path, whole);
        }
        for i in 0..nested {
            let name = format!("n{}", i);
            let mut sub = path.to_vec();
            sub.push(name.clone());
            let n = self.rng.range(2, items.max(3));
            let body = self.gen_module_body(&sub, whole, n, 0);
            text += &format!("pub mod {} {{\n{}\n}}\n", name, body);
        }
        self.env = saved_env;
        self.traits = saved_traits;
        self.adts = saved_adts;
        text
    }
}

/// Generates the case `index` of the seed.
pub fn generate(seed: u64, index: usize, items_per_module: usize, features: Features) -> Case {
    let krate = format!("tlp_l_{:03}", index);
    let mut g = Gen::new(seed, krate.clone(), features);
    let mut lib = String::new();
    lib += "#![allow(warnings)]\n";
    lib += &format!("extern crate self as {};\n", krate);
    lib += "#[doc(hidden)]\npub use type_leak_test_macro as __macro;\n";
    lib += "type_leak_test_macro::define_repeater!(pub Repeater);\n";
    lib += &format!("pub mod base {{\n{}\n}}\n", types_template("B"));

    // Modules whose items are analyzed one by one.
    let n_item_modules = g.rng.range(2, 3);
    for i in 0..n_item_modules {
        let name = format!("m{}", i);
        let n = g.rng.range(4, items_per_module);
        let nested = usize::from(g.rng.chance(50));
        let body = g.gen_module_body(std::slice::from_ref(&name), false, n, nested);
        lib += &format!("pub mod {} {{\n{}\n}}\n", name, body);
    }
    // Modules analyzed as a whole.
    let n_whole = g.rng.range(1, 2);
    for i in 0..n_whole {
        let name = format!("mm{}", i);
        let id = g.new_id();
        g.no_self = true;
        g.module_mode = true;
        g.contents.clear();
        let n = g.rng.range(4, items_per_module);
        let nested = usize::from(g.rng.chance(60));
        let body = g.gen_module_body(std::slice::from_ref(&name), true, n, nested);
        g.no_self = false;
        g.module_mode = false;
        let attr = g.attr(id, true);
        let module = format!("{}pub mod {} {{\n{}\n}}\n", attr, name, body);
        lib += &module;
        let strict = g.rng.chance(25);
        let contents = std::mem::take(&mut g.contents);
        g.items.push(Item {
            id,
            kind: "mod",
            module: Vec::new(),
            strict,
            allow_self: true,
            contents,
            source: module,
        });
    }

    // The Referrer's crate invokes the macro of each item.
    let mut referrer = String::from("#![allow(warnings)]\n");
    let mut referrer_lines = Vec::new();
    let mut items = g.items.clone();
    items.sort_by_key(|item| item.id);
    for item in &items {
        let mut path = format!("::{}", krate);
        for m in &item.module {
            path += &format!("::{}", m);
        }
        let line = referrer.lines().count() + 1;
        referrer += &format!(
            "{}::__tl_prop_{}!(path = {}{});\n",
            krate,
            item.id,
            path,
            if item.strict { ", strict" } else { "" }
        );
        referrer_lines.push((line, item.id));
    }
    Case {
        config: g.cfg.clone(),
        leaker: lib,
        referrer,
        items,
        referrer_lines,
    }
}

/// Parenthesizes a trait object with several bounds behind a reference or a pointer.
fn paren(text: &str) -> String {
    if text.starts_with("dyn ") && text.contains(" + ") {
        format!("({})", text)
    } else {
        text.to_string()
    }
}

/// A constant argument of a path: a path of several segments needs braces.
fn braced_path(len: &str, params: &[(String, &'static str)]) -> String {
    let is_param = params.iter().any(|(n, _)| n == len);
    let is_literal = len.chars().all(|c| c.is_ascii_digit());
    if is_param || is_literal {
        len.to_string()
    } else {
        format!("{{ {} }}", len)
    }
}

/// A return type of the sugar of `Fn` that ends with a trait object, which a bound after it would
/// join.
fn ret_paren(text: &str) -> String {
    if text.contains("dyn ") {
        format!("({})", text)
    } else {
        text.to_string()
    }
}
