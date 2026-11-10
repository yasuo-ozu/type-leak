//! The proc macros of `tests/prop_cross_crate`: they leak random items of a generated crate (the
//! Leaker's) and rewrite copies of them in another generated crate (the Referrer's), where every
//! rewritten type is checked to be exactly the original one.
//!
//! In the Leaker's crate, `#[prop_leak(id = 17, allow_self = true, ..)]` is put on a struct, an
//! enum, a union, a trait, an impl or a module (analyzed by the matching `intern_*`). It emits
//!
//! - the item itself,
//! - the Repeater's implementation of each entry, next to the item (inside the module the entry is
//!   keyed by, for a module),
//! - a witness `pub fn __tl_w_17_<j><generics>() -> PhantomData<*mut OriginalType>` for each type
//!   position `j` of the item (a field, a parameter, a return type, ...; the `*mut` makes the types
//!   invariant, so that a check compares them for equality and not as subtypes), and for the header
//!   of an impl of a trait (the self type, with the trait reference as a where predicate),
//! - the same signatures as the methods of a trait `__TlRev_17` (see [`reverse_impls()`]), and
//! - the exported `macro_rules! __tl_prop_17`, which carries the serialized `Referrer`, the
//!   `ContextId` and the item, and `__tl_report_17`, a report of the analysis.
//!
//! In the Referrer's crate, `leaker::__tl_prop_17!(path = ::leaker::m)` (`path` is the module that
//! contains the item) expands `__prop_apply!`, which rewrites a copy of the item with
//! `Referrer::visitor(context, ..)` and emits
//!
//! - the copy, for a struct, an enum, a union and a module,
//! - an implementation of the trait for a host type, with the rewritten signatures, for a trait,
//! - `fn __tl_chk_17_<j><generics>() -> PhantomData<*mut RewrittenType> { path::__tl_w_17_<j>::<..>() }`
//!   for each type position, which compiles only if the rewritten type is exactly the original and
//!   the generics around it have at least the bounds of the original's,
//! - an impl of `__TlRev_17` for a type of the Referrer's crate with the copy's signatures as
//!   methods, which compiles only if the copy's bounds are no stronger than the original's,
//! - for the positions of a module that name what the module declares, which the witness cannot
//!   be called with: three functions in the copy's module that compare the copy with the original
//!   written for this crate, both ways ([`Reroot`]).
//!
//! An impl is not emitted (the orphan rules), so only these checks compare it: its positions, and
//! its header (`header_position()`), unless rustc cannot normalize that (`header_cycles()`).
//!
//! The `Self` of a position is the type itself (a struct, an enum or a union), the self type (an
//! impl) or the host type (a trait). A module cannot name `Self`, so its items do not.
//!
//! The copy replaces every entry through the Repeater, so the Leaker's crate has the impls of all.
//!
//! With `#[prop_leak_accept]` (`TYPE_LEAK_PROP_ACCEPT`, see `tests/prop_cross_crate/main.rs`), an item
//! that the Leaker rejects is emitted as an accepted item with the recovered Referrer
//! (`LeakerResult::into_referrer()`: the report says `acc`, and has the errors) if
//! `LeakerResult::allow(cause)` for the cause of every error drops every error. Its copy is made in
//! the Referrer's crate of the accepted items only (`accept` in the invocation, [`apply()`]), where
//! the parts that it keeps as written resolve.
//!
//! With `stages = true` in the attribute (`TYPE_LEAK_PROP_FEATURES=allow_stages`) the macro also
//! makes two copies of the item for each [`Rule`], which the Referrer's crates of `allow_stages`
//! build ([`rule_stage()`], [`stage_copy()`]): after the analysis without the rule, whose result is
//! given the rule (`Rules::prelude()`, ...) and then `allow(cause)` for the causes of
//! the errors that remain, and after the analysis of a config that has the rule. When no error
//! remains after the rule, the copy is also made in a crate with none of the Leaker's names (the
//! invocation says `clean`). Their Repeater's impls have an `ID` of their own
//! ([`Args::repeater_id()`]).
//!
//! With `#[prop_leak_result]` the Leaker does not decide: an item with errors is emitted with its
//! `LeakerResult` (the Referrer, then the errors, as tokens). The Referrer's invocation says what
//! it allows (`allow(std_crates, trait Path, type Ty)`: `Rules::std_crates()`,
//! `allow(cause)`), [`apply()`] reports every error that remains with
//! `ErrorItem::emit()`, and copies the item with the recovered Referrer either way.

use proc_macro2::{Punct, Spacing, Span, TokenStream};
use std::collections::HashSet;
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;
use syn::visit::Visit;
use syn::visit_mut::VisitMut;
use syn::*;
use template_quote::{quote, ToTokens};
use type_leak::*;

/// Arguments of `#[prop_leak(..)]`.
#[derive(Clone)]
pub(crate) struct Args {
    /// The `ID` of the Repeater's `KEY`, the index of the item in the Leaker's crate.
    id: u64,
    allow_self: bool,
    /// Allow the names of the standard prelude (`Vec`, `Option`, `Clone`, ...).
    prelude: bool,
    /// Allowed crates (`::std`).
    allow: Vec<Path>,
    /// How the item is analyzed: `""` by the `intern_*` method of its kind, or by a scope (see
    /// [`analyze()`]): `"mod"`, `"generics"`, `"trait"` or `"impl"`.
    scope: String,
    /// The Leaker allows the primitive types (`Rules::primitive()`).
    primitive: bool,
    /// A trait (analyzed as the item itself, or of a scope `mod`): the Referrer's crate declares
    /// the rewritten trait too (`<Name>Copy`), not only an impl of the Leaker's trait for the host.
    /// Its parameter defaults are rewritten for a declaration, which cannot name the defaulted
    /// parameter itself (E0128): see [`rewrite_defaults()`].
    redeclare: bool,
    /// The Leaker reads the values of the item (`check_values`): the default of an
    /// associated const, the value of a const or static, the body of a function. A type in a type
    /// position of a value is an entry, and the copy replaces it. The witnesses compare the types
    /// of the signatures only; the copy of a value compiles in the Referrer's crate only if every
    /// type in it that the crate cannot name is replaced, and the tests compare what it computes.
    values: bool,
    /// The values of the item are checked as its signatures are (`TYPE_LEAK_PROP_FEATURES=values`): the
    /// types in the type positions of a body, a constant and a default are positions, with a witness
    /// ([`ValueTypes`]), and the host of a trait has the default bodies and values of the copy, not
    /// `unimplemented!()` and a value of the type of the constant.
    bodies: bool,
    /// The Leaker accepts an entry that names the own `Self` of a nested item together with a bound
    /// that does ([`Options::nested_self_entries()`]).
    nested_self_entries: bool,
    /// `marker = "::path::Marker"`: the impls of the entries that name no `Self`, nor do their
    /// generics, are for this type (`ImplSelf::Marker`) and the Referrer writes it as their owner,
    /// where they are otherwise for a primitive type that the id and the index choose
    /// (`ImplSelf::Bucket`). If some entry needs `Self`, the call for a marker fails: then every
    /// impl is for a parameter (`ImplSelf::Param`), which accepts the marker as the owner. The type
    /// is the Leaker's, written by hand, by an absolute path that both crates resolve.
    marker: Option<Type>,
    /// The Referrer's visitor is not given `Visitor::with_nested_self()`: a macro that has no entry
    /// to write for the own `Self` of a nested item does not need it, and the parts of the item
    /// must be left as written all the same.
    no_nested_self: bool,
    /// `TYPE_LEAK_PROP_FEATURES=allow_stages`: for each [`Rule`] the macro also tells how many errors
    /// remain with the rule given to the config and to the result, and makes a copy after each
    /// ([`rule_stage()`]).
    stages: bool,
    /// Not an argument of the attribute: set by [`leak()`] when the item has errors and is accepted
    /// with the recovered Referrer (`prop_leak_accept`, `TYPE_LEAK_PROP_ACCEPT`). The Referrer copies
    /// such an item in the crate of its own that `tests/prop_cross_crate` makes for them (`accept`,
    /// [`apply()`]), and in no other.
    accepted: bool,
    /// Not an argument of the attribute: `0`, or the number ([`Rule::number()`]) of the arguments of
    /// the copy that [`stage_copy()`] makes (`TYPE_LEAK_PROP_FEATURES=allow_stages`). Its Repeater's
    /// impls have another `ID` ([`Args::repeater_id()`]), so that they do not clash with those of the
    /// ordinary analysis and with each other.
    stage: u8,
    /// How the item is analyzed within its `scope` (`Via`): by other calls of the same API. The copy
    /// and the checks are those of the scope.
    via: String,
}

impl Parse for Args {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut args = Args {
            id: 0,
            allow_self: true,
            prelude: false,
            allow: Vec::new(),
            scope: String::new(),
            primitive: true,
            redeclare: false,
            marker: None,
            values: false,
            bodies: false,
            nested_self_entries: false,
            no_nested_self: false,
            stages: false,
            accepted: false,
            stage: 0,
            via: String::new(),
        };
        for kv in Punctuated::<MetaNameValue, Token![,]>::parse_terminated(input)? {
            let key = kv
                .path
                .get_ident()
                .map(|i| i.to_string())
                .unwrap_or_default();
            let lit = match &kv.value {
                Expr::Lit(ExprLit { lit, .. }) => lit.clone(),
                other => return Err(Error::new(other.span(), "expected a literal")),
            };
            match (key.as_str(), lit) {
                ("id", Lit::Int(i)) => args.id = i.base10_parse()?,
                ("allow_self", Lit::Bool(b)) => args.allow_self = b.value,
                ("prelude", Lit::Bool(b)) => args.prelude = b.value,
                ("scope", Lit::Str(s)) => args.scope = s.value(),
                ("primitive", Lit::Bool(b)) => args.primitive = b.value,
                ("redeclare", Lit::Bool(b)) => args.redeclare = b.value,
                ("marker", Lit::Str(s)) => args.marker = Some(s.parse()?),
                ("values", Lit::Bool(b)) => args.values = b.value,
                ("bodies", Lit::Bool(b)) => args.bodies = b.value,
                ("nested_self_entries", Lit::Bool(b)) => args.nested_self_entries = b.value,
                ("no_nested_self", Lit::Bool(b)) => args.no_nested_self = b.value,
                ("stages", Lit::Bool(b)) => args.stages = b.value,
                ("accepted", Lit::Bool(b)) => args.accepted = b.value,
                ("stage", Lit::Int(i)) => args.stage = i.base10_parse()?,
                ("via", Lit::Str(s)) => args.via = s.value(),
                ("allow", Lit::Str(s)) => {
                    for path in s.value().split(',').filter(|p| !p.trim().is_empty()) {
                        args.allow.push(parse_str(path.trim())?);
                    }
                }
                _ => return Err(Error::new(kv.path.span(), "unknown argument")),
            }
        }
        Ok(args)
    }
}

impl Args {
    fn tokens(&self) -> TokenStream {
        let (id, allow_self, prelude) = (self.id, self.allow_self, self.prelude);
        let primitive = self.primitive;
        let (redeclare, values) = (self.redeclare, self.values);
        let bodies = self.bodies;
        let (nested_self_entries, accepted) = (self.nested_self_entries, self.accepted);
        let no_nested_self = self.no_nested_self;
        let (stages, stage) = (self.stages, self.stage);
        let via = &self.via;
        let marker = self.marker.as_ref().map(|marker| {
            let marker = quote!(#marker).to_string().replace(' ', "");
            quote!(marker = #marker,)
        });
        let scope = &self.scope;
        let allow = self
            .allow
            .iter()
            .map(|p| quote!(#p).to_string().replace(' ', ""))
            .collect::<Vec<_>>()
            .join(",");
        quote!(id = #id, allow_self = #allow_self, prelude = #prelude, scope = #scope, allow = #allow, #marker primitive = #primitive, redeclare = #redeclare, values = #values, bodies = #bodies, nested_self_entries = #nested_self_entries, no_nested_self = #no_nested_self, stages = #stages, accepted = #accepted, stage = #stage, via = #via)
    }

    /// The item is nested in `enter_mod` and has a `Self` of its own: the copy is wrapped in a
    /// file and the visitor is given `with_nested_self()`. Not with `via = "nested"`, which opens
    /// the scope of the type inside the module's: the copy is that of a type analyzed by itself.
    fn nested(&self) -> bool {
        self.scope == "mod" && self.via != "nested"
    }

    /// The `ID` of the Repeater's impls of the entries: the item's, and for the copy of
    /// `allow_stages` the item's with a number of the rule and the side in the bits that no item
    /// has.
    fn repeater_id(&self) -> u64 {
        self.id | (u64::from(self.stage) << 40)
    }
}

/// Which analysis a copy of `TYPE_LEAK_PROP_FEATURES=allow_stages` is made after: the impls of its
/// entries and those of the ordinary analysis of the same item are written next to each other in
/// the Leaker's crate, each with an `ID` of their own.
#[derive(Clone, Copy, PartialEq)]
enum Side {
    /// The analysis without the rule, whose result is given the rule (`LeakerResult::allow()`).
    Result,
    /// The analysis of a config that has the rule (`Rules`). Its copy is what the
    /// copy of `Result` is compared with: if the one that fails is only `Result`'s, the rule after
    /// the analysis is what makes it fail.
    Config,
}

impl Side {
    fn index(self) -> u8 {
        match self {
            Side::Result => 0,
            Side::Config => 1,
        }
    }

    fn word(self) -> &'static str {
        match self {
            Side::Result => "res",
            Side::Config => "cfg",
        }
    }
}

/// An `allow*()` rule that `TYPE_LEAK_PROP_FEATURES=allow_stages` gives to the config and to the
/// result of the analysis without it.
///
/// - The path rules (`Prelude`, `Crate`, `Path`, `StdCrates`) judge an analysis made without them,
///   so the result can leave other errors and entries than the config (`tests/allow_stages.rs`).
/// - The cause and place rules (`Values`, `Default`) give the same errors and Referrer at both
///   stages: the macro tells whether they do ([`rule_stage()`]).
#[derive(Clone, Copy, PartialEq)]
enum Rule {
    /// `Rules::prelude()`: for every item, with the analysis not allowing the prelude.
    Prelude,
    /// `Rules::crate_paths()`.
    Crate,
    /// `Rules::or_if(super::base)`: the module `base` as the modules at depth 1 (`m0`, `mm0`) name it.
    /// The paths of the deeper ones (`super::super::base`) are other paths.
    Path,
    /// `Rules::std_crates()`.
    StdCrates,
    /// `Rules::values()`: the items that the Leaker reads the values of.
    Values,
    /// `Rules::defaults()`.
    Default,
}

impl Rule {
    const ALL: [Rule; 6] = [
        Rule::Prelude,
        Rule::Crate,
        Rule::Path,
        Rule::StdCrates,
        Rule::Values,
        Rule::Default,
    ];

    fn index(self) -> u8 {
        Rule::ALL.iter().position(|rule| *rule == self).unwrap() as u8
    }

    fn name(self) -> &'static str {
        match self {
            Rule::Prelude => "prelude",
            Rule::Crate => "crate",
            Rule::Path => "path",
            Rule::StdCrates => "std",
            Rule::Values => "values",
            Rule::Default => "default",
        }
    }

    /// The number in the `ID` of the Repeater's impls of a copy ([`Args::repeater_id()`]).
    fn number(self, side: Side) -> u8 {
        1 + self.index() * 2 + side.index()
    }

    /// The names of the macro of the Referrer's crates of the copies of the rule on a side, and of
    /// its report: `__tl_st_<rule>_<res|cfg>_<id>!` (`gen::Case::stages`).
    fn names(self, side: Side, id: u64) -> (Ident, Ident) {
        let word = format!("{}_{}", self.name(), side.word());
        (
            ident(&format!("__tl_st_{}_{}", word, id)),
            ident(&format!("__tl_st_{}_report_{}", word, id)),
        )
    }

    fn stages_name(self, id: u64) -> Ident {
        ident(&format!("__tl_stages_{}_{}", self.name(), id))
    }

    fn on_result(self, result: LeakerResult) -> LeakerResult {
        match self {
            Rule::Prelude => result.allow(Rules::new().prelude()),
            Rule::Crate => result.allow(Rules::new().crate_paths()),
            Rule::Path => result.allow(crate::starts_with(vec![base_path()])),
            Rule::StdCrates => result.allow(Rules::new().std_crates()),
            Rule::Values => result.allow(Rules::new().values()),
            Rule::Default => result.allow(Rules::new().defaults()),
        }
    }
}

/// The path that [`Rule::Path`] allows: the module `base`, as the modules at depth 1 name it.
fn base_path() -> Path {
    parse_quote!(super::base)
}

/// `std`, `core` or `alloc` as a relative path (`Rules::std_crates()`).
fn is_std_crate(path: &Path) -> bool {
    path.leading_colon.is_none()
        && path.segments.len() == 1
        && ["std", "core", "alloc"]
            .iter()
            .any(|name| path.is_ident(name))
}

fn kind_name(item: &Item) -> &'static str {
    match item {
        Item::Struct(_) => "struct",
        Item::Enum(_) => "enum",
        Item::Union(_) => "union",
        Item::Trait(_) => "trait",
        Item::Impl(i) if i.trait_.is_some() => "impl-trait",
        Item::Impl(_) => "impl-inherent",
        Item::Mod(_) => "mod",
        _ => "other",
    }
}

fn ident(name: &str) -> Ident {
    Ident::new(name, Span::call_site())
}

/// Analyzes `item` like a macro of a downstream crate would.
///
/// By the `intern_*` method of its kind, or, with `scope`, through the scopes and the `View` that
/// a macro with other needs uses:
///
/// - `mod`: a struct, an enum, a union or a trait, as a nested item of `enter_mod(..)`
///   (`View::visit_item()`), the module declaring its name. Its context is the module's, its
///   generics are local, and it has a `Self` of its own, whose entries the copy's visitor writes
///   with `Visitor::with_nested_self()` (`Projector::project_nested()`).
/// - `generics`: a struct, an enum or a union with no `Self`, in `enter_generics(..)`.
/// - `trait`: a trait without a generic associated type, in `enter_trait(..)`.
/// - `impl`: an impl in `enter_impl(.., None, ..)`, where only the self type and the arguments of
///   the trait are visited (the shape of a macro that resolves the trait by its name, so it gives
///   none). The generics and the items of the impl are not analyzed, and not compared.
fn analyze(item: &Item, args: &Args) -> (LeakerResult, ContextId, TokenStream) {
    analyze_with(item, args, None)
}

/// [`analyze()`] with `rule` given to the config as well: `allow_stages` compares it with the same
/// rule given to the result.
fn analyze_with(
    item: &Item,
    args: &Args,
    rule: Option<Rule>,
) -> (LeakerResult, ContextId, TokenStream) {
    let mut config = Rules::new().or_if(crate::starts_with(args.allow.clone()));
    if args.primitive {
        config = config.primitive();
    }
    if args.prelude || rule == Some(Rule::Prelude) {
        config = config.prelude();
    }
    if rule == Some(Rule::Crate) {
        config = config.crate_paths();
    }
    if rule == Some(Rule::StdCrates) {
        config = config.std_crates();
    }
    if rule == Some(Rule::Values) {
        config = config.values();
    }
    if rule == Some(Rule::Default) {
        config = config.defaults();
    }
    let path_rule = crate::starts_with(vec![base_path()]);
    let config = config.or_if(move |cause: &Cause, place: Place| {
        rule == Some(Rule::Path) && path_rule(cause, place)
    });
    analyze_config(item, args, config)
}

/// [`analyze()`] with the whole `config`.
fn analyze_config<A: Allow>(
    item: &Item,
    args: &Args,
    config: A,
) -> (LeakerResult, ContextId, TokenStream) {
    let mut options = Options::new();
    if args.values {
        options = options.check_values();
    }
    if args.nested_self_entries {
        options = options.nested_self_entries();
    }
    let mut leaker = Leaker::from_config(config, options);
    let allow_self = args.allow_self;
    let scope = args.scope.as_str();
    CHECKED.with(|checked| checked.borrow_mut().clear());
    let context = match (args.via.as_str(), scope, item) {
        // `enter_type` and the walk: what `intern_struct()` does.
        ("type", "", Item::Struct(_) | Item::Enum(_) | Item::Union(_)) => {
            let (ident, generics) = adt_head(item);
            leaker
                .enter_type(ident, generics, allow_self, |view| walk_adt(view, item))
                .0
        }
        // The same, with the generics, the where clause and the types given to the view one by
        // one, in a scope of the type, of `enter_generics` (no `Self`) or of the trait.
        ("parts", "", Item::Struct(_) | Item::Enum(_) | Item::Union(_)) => {
            let (ident, generics) = adt_head(item);
            leaker
                .enter_type(ident, generics, allow_self, |view| adt_parts(view, item))
                .0
        }
        ("parts", "generics", Item::Struct(_) | Item::Enum(_) | Item::Union(_)) => {
            let (_, generics) = adt_head(item);
            leaker
                .enter_generics(generics, |view| adt_parts(view, item))
                .0
        }
        ("parts", "trait", Item::Trait(i)) => {
            leaker
                .enter_trait(&i.ident, &i.generics, allow_self, |view| {
                    visit_generics_parts(view, &i.generics);
                    for bound in &i.supertraits {
                        view.visit_type_param_bound(bound);
                    }
                    for trait_item in &i.items {
                        Visit::visit_trait_item(view, trait_item);
                    }
                })
                .0
        }
        // `View::check()` of the types first, then the walk: the check must change nothing
        // (`via = "unchecked"` is the same without the calls of `check()`, which `leak()` compares).
        (via @ ("check" | "unchecked"), "", Item::Struct(_) | Item::Enum(_) | Item::Union(_)) => {
            let (ident, generics) = adt_head(item);
            leaker
                .enter_type(ident, generics, allow_self, |view| {
                    check_positions(view, item, args, via == "check");
                    walk_adt(view, item)
                })
                .0
        }
        (
            via @ ("check" | "unchecked"),
            "generics",
            Item::Struct(_) | Item::Enum(_) | Item::Union(_),
        ) => {
            let (_, generics) = adt_head(item);
            leaker
                .enter_generics(generics, |view| {
                    check_positions(view, item, args, via == "check");
                    walk_adt(view, item)
                })
                .0
        }
        (via @ ("check" | "unchecked"), "trait", Item::Trait(i)) => {
            leaker
                .enter_trait(&i.ident, &i.generics, allow_self, |view| {
                    check_positions(view, item, args, via == "check");
                    syn::visit::visit_item_trait(view, i)
                })
                .0
        }
        // The scope of the type opened in the view of `enter_mod`, which declares its name. The copy
        // is that of the type analyzed by itself, at the context of the nested scope.
        ("nested", "mod", Item::Struct(_) | Item::Enum(_) | Item::Union(_)) => {
            let (ident, generics) = adt_head(item);
            leaker
                .enter_mod(format!("prop{}", args.id), [ident.clone()], |view| {
                    view.enter_type(ident, generics, allow_self, |view| walk_adt(view, item))
                        .0
                })
                .1
        }
        // The path of the trait is given to the scope (it is absolute, and checked there).
        ("some", "impl", Item::Impl(i)) if i.trait_.is_some() => {
            let path = i.trait_.as_ref().map(|(_, path, _)| path);
            leaker
                .enter_impl(&i.generics, &i.self_ty, path, allow_self, |view| {
                    view.visit_type(&i.self_ty);
                    if let Some((_, path, _)) = &i.trait_ {
                        for seg in &path.segments {
                            view.visit_path_arguments(&seg.arguments);
                        }
                    }
                })
                .0
        }
        // `enter_impl` and the walk of syn: what `intern_impl()` does if no bound names `Self`.
        ("visit", "", Item::Impl(i)) => {
            let path = i.trait_.as_ref().map(|(_, path, _)| path);
            leaker
                .enter_impl(&i.generics, &i.self_ty, path, allow_self, |view| {
                    syn::visit::visit_item_impl(view, i)
                })
                .0
        }
        ("", _, _) => analyze_scope(&mut leaker, item, args),
        (via, scope, _) => panic!("test macro: `via = {:?}` with the scope {:?}", via, scope),
    };
    let assertions = leaker.assertions();
    (leaker.result(), context, assertions)
}

/// The analysis of `item` through its scope by the calls that name them: `intern_*`, `enter_mod`,
/// `enter_generics`, `enter_trait` and `enter_impl`.
fn analyze_scope<A: Allow>(leaker: &mut Leaker<A>, item: &Item, args: &Args) -> ContextId {
    let allow_self = args.allow_self;
    match (args.scope.as_str(), item) {
        ("mod", Item::Struct(_) | Item::Enum(_) | Item::Union(_) | Item::Trait(_)) => {
            let name = match item {
                Item::Struct(i) => i.ident.clone(),
                Item::Enum(i) => i.ident.clone(),
                Item::Union(i) => i.ident.clone(),
                Item::Trait(i) => i.ident.clone(),
                _ => unreachable!(),
            };
            leaker
                .enter_mod(format!("prop{}", args.id), [name], |view| {
                    view.visit_item(item)
                })
                .0
        }
        ("generics", Item::Struct(i)) => {
            leaker
                .enter_generics(&i.generics, |view| syn::visit::visit_item_struct(view, i))
                .0
        }
        ("generics", Item::Enum(i)) => {
            leaker
                .enter_generics(&i.generics, |view| syn::visit::visit_item_enum(view, i))
                .0
        }
        ("generics", Item::Union(i)) => {
            leaker
                .enter_generics(&i.generics, |view| syn::visit::visit_item_union(view, i))
                .0
        }
        ("trait", Item::Trait(i)) => {
            leaker
                .enter_trait(&i.ident, &i.generics, allow_self, |view| {
                    syn::visit::visit_item_trait(view, i)
                })
                .0
        }
        ("impl", Item::Impl(i)) => {
            leaker
                .enter_impl(&i.generics, &i.self_ty, None, allow_self, |view| {
                    view.visit_type(&i.self_ty);
                    if let Some((_, path, _)) = &i.trait_ {
                        for seg in &path.segments {
                            view.visit_path_arguments(&seg.arguments);
                        }
                    }
                })
                .0
        }
        ("", Item::Struct(i)) => leaker.intern_struct(i, allow_self),
        ("", Item::Enum(i)) => leaker.intern_enum(i, allow_self),
        ("", Item::Union(i)) => leaker.intern_union(i, allow_self),
        ("", Item::Trait(i)) => leaker.intern_trait(i, allow_self),
        ("", Item::Impl(i)) => leaker.intern_impl(i, allow_self),
        ("", Item::Mod(i)) => leaker.intern_mod(i),
        _ => panic!("unsupported item"),
    }
}

/// The name and the generics of a struct, an enum or a union.
fn adt_head(item: &Item) -> (&Ident, &Generics) {
    match item {
        Item::Struct(i) => (&i.ident, &i.generics),
        Item::Enum(i) => (&i.ident, &i.generics),
        Item::Union(i) => (&i.ident, &i.generics),
        _ => unreachable!("not a struct, an enum or a union"),
    }
}

/// The walk of syn over a struct, an enum or a union, in a scope of the type.
fn walk_adt<A: Allow>(view: &mut View<'_, '_, A>, item: &Item) {
    match item {
        Item::Struct(i) => syn::visit::visit_item_struct(view, i),
        Item::Enum(i) => syn::visit::visit_item_enum(view, i),
        Item::Union(i) => syn::visit::visit_item_union(view, i),
        _ => unreachable!("not a struct, an enum or a union"),
    }
}

/// The generics of the scope, given to the view by their parts: the bounds of a type parameter by
/// `View::visit_type_param_bound()` and the where clause by `View::visit_where_clause()`. A
/// parameter with a default is analyzed whole (the default has a scope of its own), and so is a
/// const parameter.
fn visit_generics_parts<A: Allow>(view: &mut View<'_, '_, A>, generics: &Generics) {
    for param in &generics.params {
        match param {
            GenericParam::Type(t) if t.default.is_none() => {
                for bound in &t.bounds {
                    view.visit_type_param_bound(bound);
                }
            }
            GenericParam::Type(t) => Visit::visit_type_param(view, t),
            GenericParam::Const(c) => Visit::visit_const_param(view, c),
            GenericParam::Lifetime(_) => (),
        }
    }
    if let Some(where_clause) = &generics.where_clause {
        view.visit_where_clause(where_clause);
    }
}

/// A struct, an enum or a union given to the view by its parts: the generics (`visit_generics_parts()`)
/// and each type of a field with `View::visit_type()`.
fn adt_parts<A: Allow>(view: &mut View<'_, '_, A>, item: &Item) {
    let (_, generics) = adt_head(item);
    visit_generics_parts(view, generics);
    match item {
        Item::Struct(i) => {
            for field in &i.fields {
                view.visit_type(&field.ty);
            }
        }
        Item::Enum(i) => {
            for variant in &i.variants {
                Visit::visit_variant(view, variant);
            }
        }
        Item::Union(i) => {
            for field in &i.fields.named {
                view.visit_type(&field.ty);
            }
        }
        _ => unreachable!("not a struct, an enum or a union"),
    }
}

/// What `View::check()` said of a type position of the item (`CheckResult`), to compare with the copy
/// ([`check_against_copy()`]).
#[derive(Clone, Debug)]
struct Checked {
    /// The index of the position (`collect_positions()`).
    j: usize,
    must: usize,
    impossible: usize,
    is_own: bool,
}

thread_local! {
    /// The results of the `View::check()` calls of the last analysis (`via = "check"`).
    static CHECKED: std::cell::RefCell<Vec<Checked>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// `View::check()` on each type position of the item, before its walk. The positions of a method
/// are checked in a scope nested in the view for the generics of the method. The defaults of the
/// parameters are not checked: they have a scope of their own.
fn check_positions<A: Allow>(view: &mut View<'_, '_, A>, item: &Item, args: &Args, check: bool) {
    let skip = match item {
        Item::Struct(i) => default_positions(&i.generics).len(),
        Item::Enum(i) => default_positions(&i.generics).len(),
        Item::Union(i) => default_positions(&i.generics).len(),
        Item::Trait(i) => default_positions(&i.generics).len(),
        _ => 0,
    };
    for (j, pos) in collect_positions(item, None, args)
        .iter()
        .enumerate()
        .skip(skip)
    {
        // The scope of a method is entered with and without the call: it takes a context id.
        let result = match pos.layers.len() {
            1 => check.then(|| view.check(&pos.ty)),
            2 => {
                view.enter_generics(&pos.layers[1], |view| check.then(|| view.check(&pos.ty)))
                    .1
            }
            _ => continue,
        };
        let result = match result {
            Some(result) => result,
            None => continue,
        };
        CHECKED.with(|checked| {
            checked.borrow_mut().push(Checked {
                j,
                must: result.must.len(),
                impossible: result.impossible.len(),
                is_own: result.is_own,
            })
        });
    }
}

/// The analysis of `item` as [`crate::bare::check()`] compares them: the Referrer if no error remains,
/// else the errors.
fn analysis(item: &Item, args: &Args) -> crate::bare::Analysis {
    let (result, context, assertions) = analyze(item, args);
    result
        .finish()
        .map(|referrer| (referrer, context, assertions))
}

pub(crate) fn identity_args(generics: &Generics) -> Vec<GenericArgument> {
    generics
        .params
        .iter()
        .map(|param| match param {
            GenericParam::Lifetime(l) => GenericArgument::Lifetime(l.lifetime.clone()),
            GenericParam::Type(t) => {
                let ident = &t.ident;
                GenericArgument::Type(parse_quote!(#ident))
            }
            GenericParam::Const(c) => {
                let ident = &c.ident;
                GenericArgument::Type(parse_quote!(#ident))
            }
        })
        .collect()
}

/// `Name<'a, T, N>`
fn type_with_params(name: &Ident, generics: &Generics) -> Type {
    let args = identity_args(generics);
    parse_quote!(#name<#(#args),*>)
}

// ---------------------------------------------------------------------------------------------
// The Leaker's side: Repeater implementations.
// ---------------------------------------------------------------------------------------------

/// The impls with the path of the module each is written in.
type Impls = Vec<(Vec<Ident>, TokenStream)>;

/// What `repeater_impls()` wrote the impls of an item for. The entries that need `Self` are for what
/// `Self` stands for whatever it is, and under `Bucket` only they are for the parameter.
#[derive(Clone, Copy)]
enum Used<'a> {
    /// `ImplSelf::Bucket`: the entries that do not need `Self` are for a primitive type.
    Bucket,
    /// `ImplSelf::Marker`: the item has a marker, and no entry needs `Self`.
    Marker(&'a Type),
    /// `ImplSelf::Param` for every entry: the item has a marker, but an entry needs `Self`, and
    /// then the call for a marker fails.
    Param,
}

/// The Repeater's implementation of each entry with the path of the module it is keyed by,
/// relative to the analyzed module (`[]` for it and for the entries of no module), as a downstream
/// crate writes them.
fn repeater_impls(
    result: &LeakerResult,
    referrer: &Referrer,
    id: u64,
    marker: Option<&Type>,
) -> Result<Impls> {
    let repeater: Path = parse_quote!(crate::Repeater);
    // For `Self` bounded by the trait, the implementation is for `__S` (the implementor).
    let self_param = ident("__S");
    check_entry_helpers(result, referrer)?;
    let check_marker: Type = parse_quote!(__Marker);
    // The call for a marker fails exactly when some entry needs `Self`.
    crate::check_bucket_self(
        referrer,
        id,
        result.repeater_impls(
            &repeater,
            id,
            ImplSelf::Marker(&check_marker),
            slot_const(&repeater),
        ),
    )?;
    let some_need_self = referrer
        .iter()
        .any(|(ix, _, _)| referrer.bucket_self(ix, id).is_none());
    let (used, targets): (Used, Vec<_>) = match marker {
        // Every entry is for the marker, or, if one of them needs `Self`, for the parameter (the
        // blanket impls accept the marker as `Self`, so the copy still compiles).
        Some(marker) if !some_need_self => (
            Used::Marker(marker),
            result
                .repeater_impls(
                    &repeater,
                    id,
                    ImplSelf::Marker(marker),
                    slot_const(&repeater),
                )
                .map_err(Error::from)?
                .collect(),
        ),
        Some(_) => (
            Used::Param,
            result
                .repeater_impls(
                    &repeater,
                    id,
                    ImplSelf::Param(&self_param),
                    slot_const(&repeater),
                )
                .map_err(Error::from)?
                .collect(),
        ),
        None => (
            Used::Bucket,
            result
                .repeater_impls(
                    &repeater,
                    id,
                    ImplSelf::Bucket { param: &self_param },
                    slot_const(&repeater),
                )
                .map_err(Error::from)?
                .collect(),
        ),
    };
    let mut impls = Vec::new();
    for r in targets {
        let ix = r.entry;
        check_impl_pieces(referrer, ix, id, &r, &self_param, used)?;
        let module = crate::module_path(&r.module)
            .map(|p| p.to_vec())
            .unwrap_or_default();
        // `ToTokens` writes the `KEY`: the trait path of the written impl is the one that the
        // copy's side writes.
        let tokens = r.to_token_stream();
        let trait_path = repeater_trait_path(&repeater, &r.args, id, ix);
        let written: ItemImpl = parse2(tokens.clone())?;
        match &written.trait_ {
            Some((_, path, _)) if *path == trait_path => (),
            other => {
                let path = other.as_ref().map(|(_, path, _)| path);
                return Err(helpers_disagree(format!(
                    "the path of the impl of the entry {} is `{}` and it should be `{}`",
                    ix,
                    quote!(#path),
                    quote!(#trait_path)
                )));
            }
        }
        impls.push((module, tokens));
    }
    Ok(impls)
}

fn helpers_disagree(what: String) -> Error {
    Error::new(
        Span::call_site(),
        format!(
            "HELPERS DISAGREE (the helpers of the Referrer are inconsistent): {}",
            what
        ),
    )
}

/// The generics as text: the parameters and the where clause.
fn generics_text(generics: &Generics) -> String {
    let (params, where_clause) = (&generics.params, &generics.where_clause);
    quote!(#params ; #where_clause).to_string()
}

/// The entries and their helpers agree: `is_empty()` and `iter()`, and the module of the impl of an
/// entry (`RepeaterImpl::module`) is `Itself` of the module `m` exactly when the context of the
/// entry is `In(m)`, and `None` otherwise.
fn check_entry_helpers(result: &LeakerResult, referrer: &Referrer) -> Result<()> {
    let n = referrer.iter().count();
    if referrer.is_empty() != (n == 0) {
        return Err(helpers_disagree(
            "`is_empty()` and `iter()` differ".to_string(),
        ));
    }
    let param = ident("__S");
    let repeater: Path = parse_quote!(crate::Repeater);
    let impls = result
        .repeater_impls(&repeater, 0, ImplSelf::Param(&param), slot_const(&repeater))
        .map_err(Error::from)?;
    for ((ix, _, context), r) in referrer.iter().zip(impls) {
        if r.entry != ix {
            return Err(helpers_disagree(format!(
                "the impl for the entry {} says it is for the entry {}",
                ix, r.entry
            )));
        }
        let agree = match (&context.module, &r.module) {
            (Some(ModuleScope::In(module)), Some(ModuleScope::Itself { id, .. })) => id == module,
            (Some(ModuleScope::In(_)), _) => false,
            (_, module) => module.is_none(),
        };
        if !agree {
            return Err(helpers_disagree(format!(
                "the module of the impl of the entry {} ({:?}) is not the context of the module \
                 that the context of the entry is in ({:?})",
                ix, r.module, context.module
            )));
        }
    }
    Ok(())
}

/// Normalizes the text of generics that the impl of an entry declares, or that `declared_generics()`
/// returns, so that the two can be compared: the implementor `__S` replaces `Self` (and `<__S>::A`
/// is `Self::A`), and the parentheses that rustc needs around a first qualified path are dropped.
fn impl_text(text: &str, self_param: &Ident) -> String {
    text.replace(char::is_whitespace, "")
        .replace(&self_param.to_string(), "Self")
        .replace("<Self>::", "Self::")
        .replace(['(', ')'], "")
}

/// Takes the parameter before the first const parameter (or the last one) off the generics: where
/// `LeakerResult::repeater_impls()` declares the implementor. (They are collected again, so that no
/// comma is left after the parameter before it.)
fn implementor_param(generics: &mut Generics) -> Option<GenericParam> {
    let mut params: Vec<GenericParam> = generics.params.iter().cloned().collect();
    let at = params
        .iter()
        .position(|param| matches!(param, GenericParam::Const(_)))
        .unwrap_or(params.len());
    let taken = at.checked_sub(1).map(|at| params.remove(at));
    generics.params = params.into_iter().collect();
    taken
}

/// The generics of a context without the lifetime bounds that the trait arguments of a Repeater
/// impl encode: a bound by a lifetime parameter of the generics (`T: 'a`, `'b: 'a`).
fn declared_generics(generics: &Generics) -> Generics {
    let lifetimes: Vec<Lifetime> = generics
        .params
        .iter()
        .filter_map(|param| match param {
            GenericParam::Lifetime(param) => Some(param.lifetime.clone()),
            _ => None,
        })
        .collect();
    let mut declared = generics.clone();
    for param in declared.params.iter_mut() {
        match param {
            GenericParam::Lifetime(param) => {
                param.bounds = param
                    .bounds
                    .iter()
                    .filter(|bound| !lifetimes.contains(bound))
                    .cloned()
                    .collect();
                if param.bounds.is_empty() {
                    param.colon_token = None;
                }
            }
            GenericParam::Type(param) => {
                param.bounds = param
                    .bounds
                    .iter()
                    .filter(|bound| {
                        !matches!(bound, TypeParamBound::Lifetime(lifetime)
                            if lifetimes.contains(lifetime))
                    })
                    .cloned()
                    .collect();
                if param.bounds.is_empty() {
                    param.colon_token = None;
                }
            }
            GenericParam::Const(_) => (),
        }
    }
    declared
}

/// The generics of the impl of an entry are those of its context without the lifetime bounds that
/// its trait arguments encode (`declared_generics()`), with the parameter of the implementor added
/// (`__S: ?Sized`, and the trait for a `Self` that a trait bounds) unless the impl is for a type;
/// its module is the context's.
///
/// An entry that `Referrer::bucket_self()` gives a type for (and only one that does not name `Self`,
/// in its type or in the generics of its context) is for that type, and declares no parameter,
/// unless `used` is `Used::Param`: then it is for the parameter, like the others.
fn check_impl_pieces(
    referrer: &Referrer,
    ix: EntryId,
    id: u64,
    r: &RepeaterImpl,
    self_param: &Ident,
    used: Used<'_>,
) -> Result<()> {
    let context = crate::entry_context(referrer, ix);
    let declared = declared_generics(&context.generics);
    let mut got = r.generics.clone();
    let bucket = referrer.bucket_self(ix, id);
    let bucketable = context.self_ty.is_none() && !names_self(&context.generics);
    if bucket.is_some() != bucketable {
        return Err(helpers_disagree(format!(
            "`bucket_self()` of the entry {} is `{}`, and its context {} `Self`",
            ix,
            bucket.as_ref().map_or(String::new(), |ty| quote!(#ty).to_string()),
            if bucketable { "does not name" } else { "names" }
        )));
    }
    match (&bucket, context.self_ty.as_ref()) {
        (Some(ty), _) if !matches!(used, Used::Param) => {
            // The marker of the item, if it has one (`Args::marker`), else the bucket.
            let expected = match used {
                Used::Marker(marker) => marker,
                _ => ty,
            };
            if r.self_ty != *expected {
                let self_ty = &r.self_ty;
                return Err(helpers_disagree(format!(
                    "the impl of the entry {} is for `{}` and it should be for `{}`",
                    ix,
                    quote!(#self_ty),
                    quote!(#expected)
                )));
            }
            if matches!(used, Used::Bucket)
                && !matches!(ty, Type::Path(TypePath { qself: None, path })
                if path.leading_colon.is_some()
                    && path.segments.len() == 3
                    && path.segments[0].ident == "core"
                    && path.segments[1].ident == "primitive")
            {
                return Err(helpers_disagree(format!(
                    "`bucket_self()` of the entry {} is not a primitive type: `{}`",
                    ix,
                    quote!(#ty)
                )));
            }
        }
        (None, Some(SelfRequirement::Type(_))) => (),
        _ => match implementor_param(&mut got) {
            Some(GenericParam::Type(t)) if t.ident == *self_param => {
                // `?Sized`, and the trait of a `Self` that is bound by it. Any type that the
                // bounds hold for is a `Sized` one (a struct, an enum or a union), so it has no
                // bound if a bound of the generics needs it `Sized`, and `?Sized` if not.
                let expected = match context.self_ty.as_ref() {
                    Some(SelfRequirement::Bound(_)) => 2..=2,
                    Some(SelfRequirement::Any) => 0..=1,
                    _ => 1..=1,
                };
                if !expected.contains(&t.bounds.len()) {
                    return Err(helpers_disagree(format!(
                        "the parameter of the implementor of the entry {} has the bounds `{}`",
                        ix,
                        quote!(#t)
                    )));
                }
            }
            other => {
                return Err(helpers_disagree(format!(
                    "the impl of the entry {} does not declare the implementor as `{}`: {}",
                    ix,
                    self_param,
                    quote!(#other)
                )))
            }
        },
    }
    if impl_text(&generics_text(&got), self_param)
        != impl_text(&generics_text(&declared), self_param)
    {
        return Err(helpers_disagree(format!(
            "the impl of the entry {} declares `{}` and the generics of its context declare `{}`",
            ix,
            generics_text(&got),
            generics_text(&declared)
        )));
    }
    // The module of the impl is the module's own context, `Itself`, of the one the context is in.
    let itself = match &r.module {
        Some(ModuleScope::Itself { id, .. }) => Some(*id),
        _ => None,
    };
    if itself != crate::in_module(context) || r.module.is_some() != itself.is_some() {
        return Err(helpers_disagree(format!(
            "the impl of the entry {} has another module than `Context::module`",
            ix
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Type positions.
// ---------------------------------------------------------------------------------------------

/// What `Self` is at a type position, for the witness.
#[derive(Clone)]
pub(crate) struct SelfSub {
    pub(crate) ty: Type,
    /// The trait `Self::A` is an associated type of.
    pub(crate) trait_ref: Option<Path>,
    /// A trait: the witness has the implementor as a type parameter.
    pub(crate) bound: bool,
}

/// A type that the item copies: a field, a parameter, a return type, an associated type or const.
#[derive(Clone)]
pub(crate) struct Position {
    /// The modules from the analyzed one, starting with its own name.
    pub(crate) mods: Vec<Ident>,
    /// The generics around, outermost first.
    pub(crate) layers: Vec<Generics>,
    pub(crate) self_sub: Option<SelfSub>,
    pub(crate) ty: Type,
}

/// `host`: in the Referrer's crate, the type that implements the trait, and the path of the module
/// of the trait in the Leaker's crate.
type Host = (Type, Path);

/// `scope`: how the item is analyzed (`Args::scope`). With `impl`, only the fragments the scope
/// visits are positions: the self type and the arguments of the trait.
fn collect_positions(item: &Item, host: Option<&Host>, args: &Args) -> Vec<Position> {
    let mut out = Vec::new();
    collect_item(
        item,
        &mut Vec::new(),
        &mut out,
        true,
        host,
        args.scope == "impl",
        args.bodies,
    );
    out
}

/// Whether the tokens name `Self`.
fn tokens_name_self(tokens: TokenStream) -> bool {
    tokens.into_iter().any(|tree| match tree {
        proc_macro2::TokenTree::Ident(ident) => ident == "Self",
        proc_macro2::TokenTree::Group(group) => tokens_name_self(group.stream()),
        _ => false,
    })
}

/// The bounds of `bounds` that do not name `Self`.
fn bounds_without_self(
    bounds: &Punctuated<TypeParamBound, Token![+]>,
) -> Punctuated<TypeParamBound, Token![+]> {
    bounds
        .iter()
        .filter(|bound| !tokens_name_self(quote!(#bound)))
        .cloned()
        .collect()
}

/// The defaults of the type parameters of `generics`, each with the generics before it: a default
/// is analyzed with the parameters before it, and the where predicates that name only those (the
/// context of its entries has them, so the Repeater's impl needs them). The bounds that name `Self`
/// are left out: a default has no `Self`, and for a struct, an enum or a union nested in a module
/// the Leaker leaves them out of the context of an entry in a default (the copy cannot write `Self`
/// there), where the others stay.
fn default_positions(generics: &Generics) -> Vec<(Generics, Type)> {
    fn name(param: &GenericParam) -> String {
        match param {
            GenericParam::Type(t) => t.ident.to_string(),
            GenericParam::Const(c) => c.ident.to_string(),
            GenericParam::Lifetime(l) => l.lifetime.to_string(),
        }
    }
    let mut out = Vec::new();
    for (i, param) in generics.params.iter().enumerate() {
        if let GenericParam::Type(TypeParam {
            default: Some(default),
            ..
        }) = param
        {
            // A default has no `Self`: the predicates that name it are not in its context either.
            let later: HashSet<String> = generics
                .params
                .iter()
                .skip(i)
                .map(name)
                .collect();
            let mentions_later = |predicate: &WherePredicate| {
                let text = quote!(#predicate).to_string();
                later.iter().any(|name| {
                    text.split(|c: char| !(c.is_alphanumeric() || c == '_'))
                        .any(|word| word == name)
                })
            };
            let predicates: Punctuated<WherePredicate, Token![,]> = generics
                .where_clause
                .iter()
                .flat_map(|wc| wc.predicates.iter())
                .filter(|p| !mentions_later(p))
                .filter_map(|p| match p {
                    WherePredicate::Type(p) => {
                        let bounded = &p.bounded_ty;
                        let bounds = bounds_without_self(&p.bounds);
                        (!tokens_name_self(quote!(#bounded)) && !bounds.is_empty()).then(|| {
                            WherePredicate::Type(PredicateType {
                                bounds,
                                ..p.clone()
                            })
                        })
                    }
                    other => Some(other.clone()),
                })
                .collect();
            let params = generics.params.iter().take(i).cloned().map(|mut param| {
                if let GenericParam::Type(t) = &mut param {
                    t.bounds = bounds_without_self(&t.bounds);
                }
                param
            });
            let before = Generics {
                lt_token: Some(Default::default()),
                params: params.collect(),
                gt_token: Some(Default::default()),
                where_clause: (!predicates.is_empty()).then(|| WhereClause {
                    where_token: Default::default(),
                    predicates,
                }),
            };
            out.push((before, default.clone()));
        }
    }
    out
}

fn collect_item(
    item: &Item,
    mods: &mut Vec<Ident>,
    out: &mut Vec<Position>,
    with_self: bool,
    host: Option<&Host>,
    fragments_only: bool,
    bodies: bool,
) {
    let mut push = |layers: &[&Generics], sub: &Option<SelfSub>, ty: &Type| {
        out.push(Position {
            mods: mods.clone(),
            layers: layers.iter().map(|g| (*g).clone()).collect(),
            self_sub: sub.clone(),
            ty: ty.clone(),
        })
    };
    let fields_of =
        |fields: &Fields| -> Vec<Type> { fields.iter().map(|f| f.ty.clone()).collect() };
    // The `Self` of a struct, an enum or a union: the type itself, or, for the copy of a nested one,
    // the Leaker's type, which the copy's `Self` stays different from (`host_arg()`).
    let adt_ty = |ident: &Ident, generics: &Generics| -> Type {
        match host {
            Some((ty, _)) => ty.clone(),
            None => type_with_params(ident, generics),
        }
    };
    match item {
        Item::Struct(i) => {
            let sub = with_self.then(|| SelfSub {
                ty: adt_ty(&i.ident, &i.generics),
                trait_ref: None,
                bound: false,
            });
            for (before, ty) in default_positions(&i.generics) {
                push(&[&before], &None, &ty);
            }
            for ty in fields_of(&i.fields) {
                push(&[&i.generics], &sub, &ty);
            }
        }
        Item::Enum(i) => {
            let sub = with_self.then(|| SelfSub {
                ty: adt_ty(&i.ident, &i.generics),
                trait_ref: None,
                bound: false,
            });
            for (before, ty) in default_positions(&i.generics) {
                push(&[&before], &None, &ty);
            }
            for variant in &i.variants {
                for ty in fields_of(&variant.fields) {
                    push(&[&i.generics], &sub, &ty);
                }
            }
        }
        Item::Union(i) => {
            let sub = with_self.then(|| SelfSub {
                ty: adt_ty(&i.ident, &i.generics),
                trait_ref: None,
                bound: false,
            });
            for (before, ty) in default_positions(&i.generics) {
                push(&[&before], &None, &ty);
            }
            for field in &i.fields.named {
                push(&[&i.generics], &sub, &field.ty);
            }
        }
        Item::Trait(i) => {
            let sub = with_self.then(|| {
                let (name, args) = (&i.ident, identity_args(&i.generics));
                match host {
                    Some((host_ty, leaker_path)) => SelfSub {
                        ty: host_ty.clone(),
                        trait_ref: Some(parse_quote!(#leaker_path::#name<#(#args),*>)),
                        bound: true,
                    },
                    None => SelfSub {
                        ty: parse_quote!(__Self),
                        trait_ref: Some(parse_quote!(#name<#(#args),*>)),
                        bound: true,
                    },
                }
            });
            for (before, ty) in default_positions(&i.generics) {
                // A bound that names `Self` (`T: AsRef<Self>`, `where T: AsRef<Self>`) is for the
                // implementor, as in a method: the position is generic over it, with the trait's
                // generics (the bound on the implementor names the defaulted parameter too).
                match (names_self(&i.generics), &sub) {
                    (true, Some(_)) => push(&[&i.generics], &sub, &ty),
                    _ => push(&[&before], &None, &ty),
                }
            }
            for trait_item in &i.items {
                match trait_item {
                    TraitItem::Fn(f) => {
                        let layers = [&i.generics, &f.sig.generics];
                        for ty in sig_types(&f.sig) {
                            push(&layers, &sub, &ty);
                        }
                        if let Some(block) = &f.default {
                            for ty in value_types(bodies, |v| v.visit_block(block)) {
                                push(&layers, &sub, &ty);
                            }
                        }
                    }
                    TraitItem::Const(c) => {
                        let layers = [&i.generics, &c.generics];
                        push(&layers, &sub, &c.ty);
                        if let Some((_, value)) = &c.default {
                            for ty in value_types(bodies, |v| v.visit_expr(value)) {
                                push(&layers, &sub, &ty);
                            }
                        }
                    }
                    _ => (),
                }
            }
        }
        Item::Impl(i) => {
            let sub = with_self.then(|| SelfSub {
                ty: (*i.self_ty).clone(),
                trait_ref: i.trait_.as_ref().map(|(_, path, _)| path.clone()),
                bound: false,
            });
            push(&[&i.generics], &sub, &i.self_ty);
            if let Some((_, path, _)) = &i.trait_ {
                for seg in &path.segments {
                    if let PathArguments::AngleBracketed(args) = &seg.arguments {
                        for arg in &args.args {
                            if let GenericArgument::Type(ty) = arg {
                                // A const parameter is parsed as a type.
                                let is_const = matches!(ty, Type::Path(TypePath { qself: None, path })
                                    if i.generics.params.iter().any(|param| matches!(param,
                                        GenericParam::Const(c) if path.is_ident(&c.ident))));
                                if !is_const {
                                    push(&[&i.generics], &sub, ty);
                                }
                            }
                        }
                    }
                }
            }
            if fragments_only {
                return;
            }
            for impl_item in &i.items {
                match impl_item {
                    ImplItem::Fn(f) => {
                        let layers = [&i.generics, &f.sig.generics];
                        for ty in sig_types(&f.sig) {
                            push(&layers, &sub, &ty);
                        }
                        for ty in value_types(bodies, |v| v.visit_block(&f.block)) {
                            push(&layers, &sub, &ty);
                        }
                    }
                    ImplItem::Const(c) => {
                        let layers = [&i.generics, &c.generics];
                        push(&layers, &sub, &c.ty);
                        for ty in value_types(bodies, |v| v.visit_expr(&c.expr)) {
                            push(&layers, &sub, &ty);
                        }
                    }
                    ImplItem::Type(t) => push(&[&i.generics, &t.generics], &sub, &t.ty),
                    _ => (),
                }
            }
        }
        Item::Fn(f) => {
            for ty in sig_types(&f.sig) {
                push(&[&f.sig.generics], &None, &ty);
            }
            for ty in value_types(bodies, |v| v.visit_block(&f.block)) {
                push(&[&f.sig.generics], &None, &ty);
            }
        }
        Item::Type(t) => push(&[&t.generics], &None, &t.ty),
        Item::Const(c) => {
            push(&[], &None, &c.ty);
            for ty in value_types(bodies, |v| v.visit_expr(&c.expr)) {
                push(&[], &None, &ty);
            }
        }
        Item::Static(st) => {
            for ty in value_types(bodies, |v| v.visit_expr(&st.expr)) {
                push(&[], &None, &ty);
            }
        }
        Item::Mod(m) => {
            if let Some((_, items)) = &m.content {
                mods.push(m.ident.clone());
                for item in items {
                    collect_item(item, mods, out, false, host, fragments_only, bodies);
                }
                mods.pop();
            }
        }
        _ => (),
    }
}

/// Whether `generics` name `Self`.
fn names_self(generics: &Generics) -> bool {
    fn has_self(tokens: TokenStream) -> bool {
        tokens.into_iter().any(|tree| match tree {
            proc_macro2::TokenTree::Ident(ident) => ident == "Self",
            proc_macro2::TokenTree::Group(group) => has_self(group.stream()),
            _ => false,
        })
    }
    let where_clause = &generics.where_clause;
    has_self(quote!(#generics #where_clause))
}

/// The types in the type positions of a value (`TYPE_LEAK_PROP_FEATURES=values`): the annotation of a
/// `let`, a cast, the arguments of a path (a turbofish), a qualified path. Each is one position, and
/// the types inside it are not, so the original and the copy (whose types are replaced by projections)
/// have as many.
struct ValueTypes(Vec<Type>);

impl<'ast> Visit<'ast> for ValueTypes {
    fn visit_type(&mut self, ty: &'ast Type) {
        self.0.push(ty.clone());
    }
}

fn value_types(bodies: bool, visit: impl FnOnce(&mut ValueTypes)) -> Vec<Type> {
    let mut types = ValueTypes(Vec::new());
    if bodies {
        visit(&mut types);
    }
    types.0
}

/// The types of a signature besides the receiver: the parameters, then the return type.
fn sig_types(sig: &Signature) -> Vec<Type> {
    let mut types: Vec<Type> = sig
        .inputs
        .iter()
        .filter_map(|input| match input {
            FnArg::Typed(pat) => Some((*pat.ty).clone()),
            FnArg::Receiver(_) => None,
        })
        .collect();
    if let ReturnType::Type(_, ty) = &sig.output {
        types.push((**ty).clone());
    }
    types
}

/// Merges the generics of the layers into the generics of one function: lifetimes first.
fn merge_generics(layers: &[Generics]) -> Generics {
    let mut params: Punctuated<GenericParam, Token![,]> = Punctuated::new();
    for layer in layers {
        for param in &layer.params {
            if matches!(param, GenericParam::Lifetime(_)) {
                params.push(param.clone());
            }
        }
    }
    // A function cannot have a default for a parameter.
    let without_default = |param: &GenericParam| -> GenericParam {
        let mut param = param.clone();
        match &mut param {
            GenericParam::Type(t) => {
                t.eq_token = None;
                t.default = None;
            }
            GenericParam::Const(c) => {
                c.eq_token = None;
                c.default = None;
            }
            GenericParam::Lifetime(_) => (),
        }
        param
    };
    let mut predicates: Punctuated<WherePredicate, Token![,]> = Punctuated::new();
    for layer in layers {
        for param in &layer.params {
            if !matches!(param, GenericParam::Lifetime(_)) {
                params.push(without_default(param));
            }
        }
        if let Some(wc) = &layer.where_clause {
            predicates.extend(wc.predicates.iter().cloned());
        }
    }
    Generics {
        lt_token: Some(Default::default()),
        params,
        gt_token: Some(Default::default()),
        where_clause: if predicates.is_empty() {
            None
        } else {
            Some(WhereClause {
                where_token: Default::default(),
                predicates,
            })
        },
    }
}

/// Wraps `ty` in parentheses.
fn parenthesized(ty: &mut Type) {
    let inner = std::mem::replace(ty, Type::Verbatim(TokenStream::new()));
    *ty = Type::Paren(TypeParen {
        paren_token: Default::default(),
        elem: Box::new(inner),
    });
}

/// Whether a type is printed so that rustc cannot read it as the bounded type of a where predicate:
/// a function pointer without an output, reached through a reference or a pointer, takes the `:`
/// for the start of its output (`&fn(A): 'y`: "return types are denoted using `->`").
fn colon_is_taken_as_output(ty: &Type) -> bool {
    let mut under_pointer = false;
    let mut ty = ty;
    loop {
        match ty {
            Type::Reference(TypeReference { elem, .. }) | Type::Ptr(TypePtr { elem, .. }) => {
                under_pointer = true;
                ty = elem;
            }
            Type::BareFn(TypeBareFn {
                output: ReturnType::Type(_, output),
                ..
            }) => ty = output,
            Type::BareFn(_) => return under_pointer,
            _ => return false,
        }
    }
}

/// Replaces `Self` by what it is at the position of the witness. The type is substituted as
/// tokens, so it is parenthesized where its precedence matters: as the bounded type of a where
/// predicate (`(&fn(A)): 'y`) and as the pointee of a reference or a pointer (`&(dyn A + B)`).
pub(crate) struct SelfSubst<'a>(pub(crate) &'a SelfSub);

impl VisitMut for SelfSubst<'_> {
    fn visit_predicate_type_mut(&mut self, i: &mut PredicateType) {
        syn::visit_mut::visit_predicate_type_mut(self, i);
        if colon_is_taken_as_output(&i.bounded_ty) {
            parenthesized(&mut i.bounded_ty);
        }
    }

    fn visit_type_reference_mut(&mut self, i: &mut TypeReference) {
        syn::visit_mut::visit_type_reference_mut(self, i);
        self.parenthesize_pointee(&mut i.elem);
    }

    fn visit_type_ptr_mut(&mut self, i: &mut TypePtr) {
        syn::visit_mut::visit_type_ptr_mut(self, i);
        self.parenthesize_pointee(&mut i.elem);
    }

    fn visit_type_mut(&mut self, i: &mut Type) {
        if let Type::Path(TypePath { qself: None, path }) = i {
            if path.leading_colon.is_none()
                && path
                    .segments
                    .first()
                    .map_or(false, |seg| seg.ident == "Self")
            {
                if path.segments.len() == 1 {
                    *i = self.0.ty.clone();
                    return;
                }
                // `Self::A` is an associated type of the trait: `<Ty as Trait>::A`.
                if let Some(trait_ref) = &self.0.trait_ref {
                    let mut qpath = trait_ref.clone();
                    let position = qpath.segments.len();
                    qpath.segments.extend(path.segments.iter().skip(1).cloned());
                    *i = Type::Path(TypePath {
                        qself: Some(QSelf {
                            lt_token: Default::default(),
                            ty: Box::new(self.0.ty.clone()),
                            position,
                            as_token: Some(Default::default()),
                            gt_token: Default::default(),
                        }),
                        path: qpath,
                    });
                }
            }
        }
        syn::visit_mut::visit_type_mut(self, i)
    }
}

impl SelfSubst<'_> {
    fn parenthesize_pointee(&self, elem: &mut Type) {
        if matches!(elem, Type::TraitObject(object) if object.bounds.len() > 1) {
            parenthesized(elem);
        }
    }
}

fn witness_name(id: u64, j: usize) -> Ident {
    ident(&format!("__tl_w_{}_{}", id, j))
}

/// How the implementor of a trait is named in the function that a position becomes.
#[derive(Clone, Copy, PartialEq)]
enum Implementor {
    /// A type parameter of the function, `__Self: ?Sized + Trait<..>` (the witness).
    Param,
    /// The type parameter `__Self` of the reverse trait: a predicate `__Self: Trait<..>` of the
    /// method (the trait of the Leaker's crate, [`reverse_trait()`]).
    Where,
    /// The host, which `Self` is replaced by (the method of the Referrer's impl of the reverse trait).
    Fixed,
}

/// The generics and the type of the function that a position becomes (the witness, the check, a
/// method of the reverse trait): `Self` is what it is at the position, and each lifetime is in the
/// type (`pinned()`).
fn lowered(pos: &Position, implementor: Implementor) -> (Generics, Type) {
    let mut ty = pos.ty.clone();
    let mut generics = merge_generics(&pos.layers);
    if let Some(sub) = &pos.self_sub {
        SelfSubst(sub).visit_type_mut(&mut ty);
        SelfSubst(sub).visit_generics_mut(&mut generics);
        if sub.bound {
            let trait_ref = sub.trait_ref.as_ref().unwrap();
            match implementor {
                Implementor::Param => generics
                    .params
                    .push(parse_quote!(__Self: ?::core::marker::Sized + #trait_ref)),
                Implementor::Where => generics
                    .make_where_clause()
                    .predicates
                    .push(parse_quote!(__Self: #trait_ref)),
                Implementor::Fixed => (),
            }
        }
    }
    let ty = pinned(&generics, ty);
    (generics, ty)
}

/// `<vis> fn __tl_w_<id>_<j><generics>() -> PhantomData<*mut ty> where ..` with `tail` after it: the
/// body of the witness, of the default in the reverse trait and of its impl.
fn method(
    id: u64,
    j: usize,
    pos: &Position,
    implementor: Implementor,
    vis: TokenStream,
) -> TokenStream {
    let (generics, ty) = lowered(pos, implementor);
    let name = witness_name(id, j);
    let params = &generics.params;
    let where_clause = &generics.where_clause;
    let invariant = invariant_in(ty);
    quote! {
        #[allow(non_snake_case, dead_code)]
        #vis fn #name<#params>() -> ::core::marker::PhantomData<#invariant> #where_clause {
            ::core::marker::PhantomData
        }
    }
}

/// `pub fn __tl_w_<id>_<j><generics>() -> PhantomData<ty>`, with the generics around the position.
pub(crate) fn witness_fn(id: u64, j: usize, pos: &Position) -> TokenStream {
    method(id, j, pos, Implementor::Param, quote!(pub))
}

/// The name of the trait that has the witness of each position as a method (with its body as the
/// default), in the module of the witnesses: the Referrer's crate implements it for a type of its
/// own with the methods of the copy, and rustc compares each with the original's (see
/// [`reverse_impls()`]). The trait of a trait is generic over the implementor, `__Self: ?Sized`.
fn reverse_trait(id: u64) -> Ident {
    ident(&format!("__TlRev_{}", id))
}

/// `*mut ty`: invariant in `ty`. `PhantomData<ty>` is covariant, so a Referrer's type with a
/// shorter lifetime (`&'a u8` for `&'static u8`, `dyn X + 'a` for `+ 'static`) in a covariant
/// position would pass for the original type. With `*mut`, the witness and the check compare the
/// types for equality, lifetimes included: each fails if the other's type is not the same.
pub(crate) fn invariant_in(mut ty: Type) -> Type {
    // `*mut dyn A + B` is ambiguous.
    if matches!(&ty, Type::TraitObject(object) if object.bounds.len() > 1) {
        parenthesized(&mut ty);
    }
    parse_quote!(*mut #ty)
}

/// The primitive types, as a path of one segment.
const PRIMITIVES: &[&str] = &[
    "u8", "u16", "u32", "u64", "u128", "usize", "i8", "i16", "i32", "i64", "i128", "isize", "f32",
    "f64", "bool", "char", "str",
];

/// The names of the standard prelude that the generated items use.
const PRELUDE: &[&str] = &[
    "Box",
    "String",
    "Vec",
    "Option",
    "Result",
    "Copy",
    "Send",
    "Sized",
    "Sync",
    "Unpin",
    "Drop",
    "Fn",
    "FnMut",
    "FnOnce",
    "Clone",
    "PartialEq",
    "PartialOrd",
    "Eq",
    "Ord",
    "AsRef",
    "AsMut",
    "Into",
    "From",
    "Default",
    "Iterator",
    "IntoIterator",
    "Extend",
    "DoubleEndedIterator",
    "ExactSizeIterator",
    "Some",
    "None",
    "Ok",
    "Err",
    "ToOwned",
    "ToString",
    "drop",
];

/// Finds the paths that are relative: they mean something else in another crate.
struct Portable<'a> {
    generics: &'a HashSet<String>,
    prelude: bool,
    ok: bool,
}

impl Portable<'_> {
    fn check_path(&mut self, path: &Path) {
        if path.leading_colon.is_some() {
            return;
        }
        let first = match path.segments.first() {
            Some(seg) => seg.ident.to_string(),
            None => return,
        };
        let known = self.generics.contains(&first)
            || PRIMITIVES.contains(&first.as_str())
            || (self.prelude && PRELUDE.contains(&first.as_str()));
        if !known {
            self.ok = false;
        }
    }
}

impl<'ast> Visit<'ast> for Portable<'_> {
    fn visit_type_path(&mut self, i: &'ast TypePath) {
        if i.qself.is_none() {
            self.check_path(&i.path);
        }
        syn::visit::visit_type_path(self, i)
    }
    fn visit_trait_bound(&mut self, i: &'ast TraitBound) {
        self.check_path(&i.path);
        syn::visit::visit_trait_bound(self, i)
    }
    fn visit_expr_path(&mut self, i: &'ast ExprPath) {
        if i.qself.is_none() {
            self.check_path(&i.path);
        }
    }
    fn visit_type_macro(&mut self, _: &'ast TypeMacro) {
        self.ok = false;
    }
}

/// Finds an elided lifetime that the witness cannot name: one outside the sugar of `Fn` and a
/// function pointer, where it is the binder's.
struct Elided(bool);

impl<'ast> Visit<'ast> for Elided {
    fn visit_type_reference(&mut self, i: &'ast TypeReference) {
        if i.lifetime.is_none() {
            self.0 = true;
        }
        syn::visit::visit_type_reference(self, i)
    }
    fn visit_lifetime(&mut self, i: &'ast Lifetime) {
        if i.ident == "_" {
            self.0 = true;
        }
    }
    fn visit_type_bare_fn(&mut self, _: &'ast TypeBareFn) {}
    fn visit_parenthesized_generic_arguments(&mut self, _: &'ast ParenthesizedGenericArguments) {}
}

fn has_elided(ty: &Type) -> bool {
    let mut elided = Elided(false);
    elided.visit_type(ty);
    elided.0
}

/// Whether a type or a path names a lifetime, or a reference (an elided lifetime).
struct NamesLifetime(bool);

impl<'ast> Visit<'ast> for NamesLifetime {
    fn visit_lifetime(&mut self, _: &'ast Lifetime) {
        self.0 = true;
    }
    fn visit_type_reference(&mut self, _: &'ast TypeReference) {
        self.0 = true;
    }
}

/// Finds `Self` and `Self::A` in the inputs of a function pointer or of the sugar of `Fn(..)`
/// whose output has an elided lifetime, which elision takes from the inputs.
#[derive(Default)]
struct SelfInElidedInputs {
    /// How many signatures with such an output the visit is in the inputs of.
    depth: usize,
    /// `Self` is in the inputs.
    bare: bool,
    /// `Self::A` is in the inputs.
    assoc: bool,
}

impl SelfInElidedInputs {
    fn signature<'ast>(
        &mut self,
        inputs: impl IntoIterator<Item = &'ast Type>,
        output: &'ast ReturnType,
    ) {
        let mut elided = Elided(false);
        elided.visit_return_type(output);
        let counts = usize::from(elided.0);
        self.depth += counts;
        for input in inputs {
            self.visit_type(input);
        }
        self.depth -= counts;
        self.visit_return_type(output);
    }
}

impl<'ast> Visit<'ast> for SelfInElidedInputs {
    fn visit_type_bare_fn(&mut self, i: &'ast TypeBareFn) {
        self.signature(i.inputs.iter().map(|arg| &arg.ty), &i.output)
    }
    fn visit_parenthesized_generic_arguments(&mut self, i: &'ast ParenthesizedGenericArguments) {
        self.signature(&i.inputs, &i.output)
    }
    fn visit_type_path(&mut self, i: &'ast TypePath) {
        if self.depth > 0 && i.qself.is_none() && i.path.leading_colon.is_none() {
            if let Some(first) = i.path.segments.first().filter(|seg| seg.ident == "Self") {
                if i.path.segments.len() == 1 {
                    self.bare |= first.arguments.is_none();
                } else {
                    self.assoc = true;
                }
            }
        }
        syn::visit::visit_type_path(self, i)
    }
}

/// Whether the witness and the check of a position cannot write its `Self` out. They replace `Self`
/// by the type it is at the position (`SelfSubst`), and `Self::A` by `<Ty as Trait>::A`. Where that
/// has lifetimes (the type itself `S<'a>`, the self type of an impl, the lifetime parameters of
/// the trait), a function pointer or `Fn(..)` with `Self` or `Self::A` in its inputs and an elided
/// output has more input lifetimes than the original: rustc cannot tell which one the output has
/// (E0106, F15). The position is not compared, like one with an elided lifetime (`has_elided()`).
/// The library does not rewrite `Self` this way: the Repeater's impl for the implementors writes
/// `Self::A` as `<__S>::A`, and the Leaker rejects an item whose copy would, or leaks the whole
/// signature (`Visitor::with_self()`, F15).
fn self_breaks_elision(pos: &Position) -> bool {
    let sub = match &pos.self_sub {
        Some(sub) => sub,
        None => return false,
    };
    // The witness and the check have the generics around the position too.
    let mut found = SelfInElidedInputs::default();
    found.visit_type(&pos.ty);
    for layer in &pos.layers {
        found.visit_generics(layer);
    }
    let ty_lifetimes = || {
        let mut names = NamesLifetime(false);
        names.visit_type(&sub.ty);
        names.0
    };
    let trait_lifetimes = || {
        let mut names = NamesLifetime(false);
        if let Some(trait_ref) = &sub.trait_ref {
            names.visit_path(trait_ref);
        }
        names.0
    };
    (found.bare && ty_lifetimes()) || (found.assoc && (ty_lifetimes() || trait_lifetimes()))
}

/// Finds an argument-position `impl Trait`: its type is not a type that a witness can return.
struct HasImplTrait(bool);

impl<'ast> Visit<'ast> for HasImplTrait {
    fn visit_type_impl_trait(&mut self, _: &'ast TypeImplTrait) {
        self.0 = true;
    }
}

fn has_impl_trait(ty: &Type) -> bool {
    let mut found = HasImplTrait(false);
    found.visit_type(ty);
    found.0
}

/// Whether a position has no witness: it has an elided lifetime or an `impl Trait`, or `Self`
/// cannot be written out in it.
fn has_no_witness(pos: &Position) -> bool {
    has_elided(&pos.ty) || has_impl_trait(&pos.ty) || self_breaks_elision(pos)
}

// ---------------------------------------------------------------------------------------------
// The Referrer's side: rewriting a copy.
// ---------------------------------------------------------------------------------------------

/// Builds `<Self as Repeater<..>>::Type` for an entry.
struct Projector<'a> {
    referrer: &'a Referrer,
    repeater: Path,
    id: u64,
    root_params: Punctuated<GenericParam, Token![,]>,
    root_args: Vec<GenericArgument>,
    /// `Self` of an entry whose context bounds `Self` by the trait: the host type.
    bound_self: Option<Type>,
    /// `Self` of an entry whose context says what `Self` is: the type, as the Referrer names it.
    type_self: Option<Type>,
    /// The type that the Referrer writes as the owner of the entries which do not need `Self`
    /// (`Args::marker`).
    marker: Option<Type>,
    /// The entries that the copy replaced.
    used: std::cell::RefCell<std::collections::BTreeSet<EntryId>>,
    /// `root_args` are the arguments of an instantiation (`Instance`): what an entry names `Self`
    /// as is written with them too.
    at_instance: bool,
}

impl Projector<'_> {
    fn project(&self, ix: EntryId, context: &Context) -> Result<Type> {
        let self_ty: Type = match context.self_ty.as_ref() {
            Some(SelfRequirement::Bound(_)) => self
                .bound_self
                .clone()
                .ok_or_else(|| Error::new(Span::call_site(), "test macro: no host for Self"))?,
            Some(SelfRequirement::Type(_)) => self
                .type_self
                .clone()
                .ok_or_else(|| Error::new(Span::call_site(), "test macro: no type for Self"))?,
            // The default of a parameter of a trait analyzed as the item itself, or nested in the
            // analyzed syntax, whose generics have a bound that names `Self` (`trait Tr<T:
            // AsRef<Self>, U = Rel<T>>`): the Leaker analyzed it without `Self`, and the entry is
            // for any type that the bound holds for. The implementor is the host, which has it
            // (`T: AsRef<Host>`), not `()`.
            _ => match (self.referrer.bucket_self(ix, self.id), &self.marker) {
                // The impl is for the marker of the item (`Args::marker`), or for this primitive
                // type (`ImplSelf::Bucket`).
                (Some(_), Some(marker)) => marker.clone(),
                (Some(bucket), None) => bucket,
                (None, _)
                    if self.bound_self.is_some()
                        && names_self(&context.generics) =>
                {
                    self.bound_self.clone().unwrap()
                }
                (None, _) => parse_quote!(()),
            },
        };
        self.project_with(ix, context, self_ty)
    }

    /// The path through the Repeater of an entry that depends on the own `Self` of a nested item
    /// (`Visitor::with_nested_self()`). `base` is the module of the Leaker's crate that the
    /// Visitor's context is in. The Repeater's impl is for the Leaker's own type, which the copy
    /// cannot name as `Self`: a struct, an enum or a union is `base::<modules>::Name<args>`, and a
    /// trait is the host. An impl would need its self type written out in the Leaker's crate,
    /// which the generator does not make.
    ///
    /// An entry whose type does not name `Self`, and whose bounds do (`SelfRequirement::Any`), has
    /// an impl for any type that the bounds hold for: the copy passes its own type, `Self`.
    fn project_nested(
        &self,
        nested: &NestedSelf,
        base: &Path,
        ix: EntryId,
        context: &Context,
    ) -> Result<Type> {
        if let Some(SelfRequirement::Any) = context.self_ty {
            return self.project_with(ix, context, parse_quote!(Self));
        }
        let self_ty: Type = match &nested.requirement {
            SelfRequirement::Bound(_) => self
                .bound_self
                .clone()
                .ok_or_else(|| Error::new(Span::call_site(), "test macro: no host for Self"))?,
            SelfRequirement::Type(Type::Path(TypePath { qself: None, path }))
                if nested.ident.is_some() =>
            {
                let mut full = base.clone();
                full.segments
                    .extend(nested.modules.iter().cloned().map(PathSegment::from));
                full.segments.extend(path.segments.iter().cloned());
                Type::Path(TypePath {
                    qself: None,
                    path: full,
                })
            }
            _ => {
                return Err(Error::new(
                    Span::call_site(),
                    "test macro: no type for the Self of a nested impl",
                ))
            }
        };
        self.project_with(ix, context, self_ty)
    }

    fn project_with(&self, ix: EntryId, context: &Context, self_ty: Type) -> Result<Type> {
        self.used.borrow_mut().insert(ix);
        let site = UseSite::substituting(&self.root_params, &self.root_args);
        // The type of `Self` is written with the generics of the item (`Name<'a, T>`).
        let self_ty = if self.at_instance {
            site.substitute(&self_ty, None)
        } else {
            self_ty
        };
        let encoded =
            crate::repeater_args(context, &site, Some(&self_ty), slot_const(&self.repeater))
                .map_err(Error::from)?;
        Ok(repeater_projection(&self.repeater, &self_ty, &encoded, self.id, ix))
    }
}

struct Rewritten {
    item: Item,
    /// The name of the host type of a trait.
    host: Option<Ident>,
    /// The host is `dyn <host>`, an unsized type (the implementor of the trait is not `Sized`).
    dyn_host: bool,
    /// The entries that the copy replaced through the Repeater.
    used: std::collections::BTreeSet<EntryId>,
}

/// The host type: a unit struct `Host`, or `dyn Host + 'static` if the implementor must be unsized.
fn host_type(host: &Ident, dyn_host: bool) -> Type {
    if dyn_host {
        parse_quote!(dyn #host + 'static)
    } else {
        parse_quote!(#host)
    }
}

/// The path of the Repeater, from the path of a module of the Leaker's crate.
fn repeater_path(leaker_path: &Path) -> Path {
    let mut repeater = leaker_path.clone();
    repeater.segments = repeater.segments.iter().take(1).cloned().collect();
    repeater.segments.push(ident("Repeater").into());
    repeater
}

/// Rewrites a copy of `item`, which is in the module `leaker_path` of the Leaker's crate.
fn rewrite(
    referrer: &Referrer,
    context: ContextId,
    item: &Item,
    args: &Args,
    leaker_path: &Path,
    dyn_host: bool,
) -> Result<Rewritten> {
    rewrite_with(
        referrer,
        context,
        item,
        args,
        leaker_path,
        dyn_host,
        None,
        None,
    )
}

/// [`rewrite()`] for a use of the item at an instantiation of its generics (`inst`: the entries are
/// written with its arguments, the host of a trait is its own), or inside a module or a block that
/// declares names (`wrap`: the copy is the item, as the visitor leaves it there).
#[allow(clippy::too_many_arguments)]
fn rewrite_with(
    referrer: &Referrer,
    context: ContextId,
    item: &Item,
    args: &Args,
    leaker_path: &Path,
    dyn_host: bool,
    inst: Option<&Instance>,
    wrap: Option<&Wrap>,
) -> Result<Rewritten> {
    let repeater = repeater_path(leaker_path);
    let generics = match item {
        Item::Struct(i) => Some(&i.generics),
        Item::Enum(i) => Some(&i.generics),
        Item::Union(i) => Some(&i.generics),
        Item::Trait(i) => Some(&i.generics),
        Item::Impl(i) => Some(&i.generics),
        _ => None,
    };
    let mut projector = Projector {
        referrer,
        repeater,
        id: args.repeater_id(),
        root_params: generics.map(|g| g.params.clone()).unwrap_or_default(),
        root_args: match inst {
            Some(inst) => inst.completed.clone(),
            None => generics.map(identity_args).unwrap_or_default(),
        },
        bound_self: None,
        type_self: None,
        marker: args.marker.clone(),
        used: Default::default(),
        at_instance: inst.is_some(),
    };
    let host = matches!(item, Item::Trait(_)).then(|| match inst {
        Some(inst) => inst.host(),
        None => ident(&format!("Host{}", args.id)),
    });
    let mut self_ty = None;
    match item {
        Item::Struct(_) | Item::Enum(_) | Item::Union(_) => {
            let (name, generics) = match item {
                Item::Struct(i) => (&i.ident, &i.generics),
                Item::Enum(i) => (&i.ident, &i.generics),
                Item::Union(i) => (&i.ident, &i.generics),
                _ => unreachable!(),
            };
            let args = match inst {
                Some(inst) => inst.completed.clone(),
                None => identity_args(generics),
            };
            let ty: Type = parse_quote!(#leaker_path::#name<#(#args),*>);
            projector.type_self = Some(ty.clone());
            self_ty = Some(ty);
        }
        Item::Trait(_) => {
            // `Self` is the host in the copy, which is an impl for it.
            projector.bound_self = Some(host_type(host.as_ref().unwrap(), dyn_host));
        }
        Item::Impl(i) => {
            // The self type is written in the Leaker's crate: rewrite it like any type.
            let error = std::cell::RefCell::new(None);
            let rewritten = referrer.expand(context, (*i.self_ty).clone(), |_leaked, ix, entry| {
                projector.project(ix, entry).unwrap_or_else(|e| {
                    *error.borrow_mut() = Some(e);
                    parse_quote!(())
                })
            });
            if let Some(e) = error.into_inner() {
                return Err(e);
            }
            projector.type_self = Some(rewritten);
            // Inside a module or a block, only the entries that the visitor replaces count.
            if wrap.is_some() {
                projector.used.borrow_mut().clear();
            }
        }
        _ => (),
    }
    let error = std::cell::RefCell::new(None);
    let mut rewritten = item.clone();
    {
        // The module of the Leaker's crate that the context is in: the module of the item, or the
        // analyzed module itself.
        let mut base = leaker_path.clone();
        if let Item::Mod(m) = item {
            base.segments.push(m.ident.clone().into());
        }
        let visitor = referrer.visitor(context, |_leaked, ix, entry| {
            projector.project(ix, entry).unwrap_or_else(|e| {
                *error.borrow_mut() = Some(e);
                parse_quote!(())
            })
        });
        // The item of a scope `mod` is a nested item, which has a `Self` of its own.
        let visitor = match (self_ty, args.nested()) {
            (Some(self_ty), false) => visitor.with_self(self_ty),
            _ => visitor,
        };
        // A struct, an enum, a union or a trait nested in the module (the item of a scope `mod`)
        // has a `Self` of its own, whose entries the visitor finds.
        let mut visitor: Box<dyn VisitMut + '_> = if args.no_nested_self {
            Box::new(visitor)
        } else {
            Box::new(visitor.with_nested_self(|nested, _leaked, ix, entry| {
                projector
                    .project_nested(nested, &base, ix, entry)
                    .unwrap_or_else(|e| {
                        *error.borrow_mut() = Some(e);
                        parse_quote!(())
                    })
            }))
        };
        match &mut rewritten {
            // The item inside a module or a block that declares names (`shadow_check()`).
            other if wrap.is_some() => {
                let wrap = wrap.unwrap();
                let mut file = File {
                    shebang: None,
                    attrs: Vec::new(),
                    items: vec![wrap.around(std::mem::replace(
                        other,
                        Item::Verbatim(TokenStream::new()),
                    ))],
                };
                visitor.visit_file_mut(&mut file);
                *other = wrap.inside(file.items.remove(0));
            }
            other if args.nested() => {
                let mut file = File {
                    shebang: None,
                    attrs: Vec::new(),
                    items: vec![std::mem::replace(other, Item::Verbatim(TokenStream::new()))],
                };
                visitor.visit_file_mut(&mut file);
                *other = file.items.remove(0);
            }
            Item::Mod(m) => {
                if let Some((_, items)) = &mut m.content {
                    // The items of a module are declared together, as those of a file.
                    let mut file = File {
                        shebang: None,
                        attrs: Vec::new(),
                        items: std::mem::take(items),
                    };
                    visitor.visit_file_mut(&mut file);
                    *items = file.items;
                }
            }
            other => visitor.visit_item_mut(other),
        }
    }
    // An item nested in a module or a block has a `Self` of its own: an impl cannot write it.
    if let (Some(e), None) = (error.into_inner(), wrap) {
        return Err(e);
    }
    if let (true, Item::Trait(original), Item::Trait(copy)) = (args.redeclare, item, &mut rewritten)
    {
        if !matches!(args.scope.as_str(), "" | "mod") {
            return Err(Error::new(
                Span::call_site(),
                "test macro: `redeclare` is for a trait analyzed as the item itself and for the traits of a scope `mod`",
            ));
        }
        let mut base = leaker_path.clone();
        if let Item::Mod(m) = item {
            base.segments.push(m.ident.clone().into());
        }
        rewrite_defaults(referrer, context, original, copy, &projector, &base)?;
    }
    Ok(Rewritten {
        item: rewritten,
        host,
        dyn_host,
        used: projector.used.into_inner(),
    })
}

// ---------------------------------------------------------------------------------------------
// Use sites with arguments, and names that hide entries.
// ---------------------------------------------------------------------------------------------

/// The choices of the macro, from the seed that the invocation carries (splitmix64).
struct Draw(u64);

impl Draw {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn chance(&mut self, percent: u64) -> bool {
        self.next() % 100 < percent
    }
}

/// What the macro counted of the copy (`inst`, `shadow`): a line on its stderr, which cargo passes on
/// and `tests/prop_cross_crate` reads: `TLCOUNT <what> <id> <n>`. (A lint of the output does not
/// reach it: the lints are not reported for the output of a macro of another crate.)
fn counted(what: &str, id: u64, n: usize) {
    eprintln!("TLCOUNT {} {} {}", what, id, n);
}

/// The name of a generic parameter: `'a`, `T` or `N`.
fn param_name(param: &GenericParam) -> String {
    match param {
        GenericParam::Lifetime(l) => l.lifetime.to_string(),
        GenericParam::Type(t) => t.ident.to_string(),
        GenericParam::Const(c) => c.ident.to_string(),
    }
}

/// How many times the tokens name `name` as an identifier.
fn count_name(tokens: TokenStream, name: &str) -> usize {
    tokens
        .into_iter()
        .map(|tree| match tree {
            proc_macro2::TokenTree::Ident(ident) => usize::from(ident == name),
            proc_macro2::TokenTree::Group(group) => count_name(group.stream(), name),
            _ => 0,
        })
        .sum()
}

/// Maps the lifetimes of an instantiation (`'a` to `'static` or `'q0`) wherever it visits.
struct MapLifetimes<'a>(&'a [(Lifetime, Lifetime)]);

impl VisitMut for MapLifetimes<'_> {
    fn visit_lifetime_mut(&mut self, lifetime: &mut Lifetime) {
        if let Some((_, to)) = self.0.iter().find(|(from, _)| from == lifetime) {
            *lifetime = to.clone();
        }
    }
}

/// The marker traits and comparisons that every type of [`concrete_type()`] has: a parameter whose
/// bounds are these and lifetimes can be given any of them.
const FRIENDLY: &[&str] = &[
    "Sized",
    "Clone",
    "Copy",
    "Debug",
    "Default",
    "Send",
    "Sync",
    "Unpin",
    "PartialEq",
    "Eq",
    "PartialOrd",
    "Ord",
];

/// Adds the names of the bounds to `traits`, or says that one is not one of [`FRIENDLY`]
/// (`?Sized` is written `?Sized`).
fn friendly_bounds<'a>(
    bounds: impl IntoIterator<Item = &'a TypeParamBound>,
    traits: &mut std::collections::BTreeSet<String>,
) -> bool {
    for bound in bounds {
        match bound {
            TypeParamBound::Lifetime(_) => (),
            TypeParamBound::Trait(tb) if tb.lifetimes.is_none() && tb.paren_token.is_none() => {
                let seg = match tb.path.segments.last() {
                    Some(seg) if seg.arguments.is_none() => seg,
                    _ => return false,
                };
                let name = seg.ident.to_string();
                match tb.modifier {
                    TraitBoundModifier::Maybe(_) if name == "Sized" => {
                        traits.insert("?Sized".to_string());
                    }
                    TraitBoundModifier::None if FRIENDLY.contains(&name.as_str()) => {
                        traits.insert(name);
                    }
                    _ => return false,
                }
            }
            _ => return false,
        }
    }
    true
}

/// Looks for what the generics of an item need of a parameter `name` (`friendly_type()`,
/// `friendly_const()`): any bound or predicate of the item, at any level, that names it.
struct Needs<'a> {
    name: &'a Ident,
    /// The traits of the bounds on the parameter alone, if they are all of [`FRIENDLY`].
    traits: std::collections::BTreeSet<String>,
    /// A bound or a predicate names the parameter in another way.
    other: bool,
    /// A bound on the parameter alone is not one of [`FRIENDLY`].
    unfit: bool,
}

impl Needs<'_> {
    fn names(&self, node: impl ToTokens) -> bool {
        count_name(quote!(#node), &self.name.to_string()) > 0
    }
}

impl<'ast> Visit<'ast> for Needs<'_> {
    fn visit_predicate_type(&mut self, p: &'ast PredicateType) {
        let alone = matches!(&p.bounded_ty, Type::Path(TypePath { qself: None, path })
            if path.is_ident(self.name));
        if alone {
            if p.lifetimes.is_some() || !friendly_bounds(&p.bounds, &mut self.traits) {
                self.unfit = true;
            }
        } else if self.names(p) {
            self.other = true;
        }
    }

    fn visit_type_param(&mut self, t: &'ast TypeParam) {
        if t.ident == *self.name {
            if !friendly_bounds(&t.bounds, &mut self.traits) {
                self.unfit = true;
            }
        } else if self.names(&t.bounds) {
            self.other = true;
        }
    }

    fn visit_item_trait(&mut self, i: &'ast ItemTrait) {
        if self.names(&i.supertraits) {
            self.other = true;
        }
        syn::visit::visit_item_trait(self, i)
    }

    fn visit_trait_item_type(&mut self, i: &'ast TraitItemType) {
        if self.names(&i.bounds) {
            self.other = true;
        }
        syn::visit::visit_trait_item_type(self, i)
    }
}

/// Whether a bound or a predicate of the item names the parameter, other than its own declaration
/// (`T: Clone` in the generics): `U: Into<T>`, `where T: Clone` of a method.
fn bounded_elsewhere(item: &Item, param: &GenericParam) -> bool {
    let name = match param {
        GenericParam::Type(t) => &t.ident,
        GenericParam::Const(c) => &c.ident,
        GenericParam::Lifetime(_) => return false,
    };
    struct Elsewhere<'a>(&'a Ident, bool);
    impl<'ast> Visit<'ast> for Elsewhere<'_> {
        fn visit_predicate_type(&mut self, p: &'ast PredicateType) {
            self.1 |= count_name(quote!(#p), &self.0.to_string()) > 0;
        }
        fn visit_type_param(&mut self, t: &'ast TypeParam) {
            if t.ident != *self.0 {
                let bounds = &t.bounds;
                self.1 |= count_name(quote!(#bounds), &self.0.to_string()) > 0;
            }
        }
        fn visit_item_trait(&mut self, i: &'ast ItemTrait) {
            let supertraits = &i.supertraits;
            self.1 |= count_name(quote!(#supertraits), &self.0.to_string()) > 0;
            syn::visit::visit_item_trait(self, i)
        }
    }
    let mut found = Elsewhere(name, false);
    found.visit_item(item);
    found.1
}

/// The traits that the type parameter is bounded by, if it can be given a type of
/// [`concrete_type()`]: they are all of [`FRIENDLY`] (or lifetimes), and its name is in no other
/// bound or predicate of the item, in the generics or in a method (`U: Into<T>` and `Vec<T>:
/// Clone` need what they name, `T: Fn(A)` more than a type has).
fn friendly_type(item: &Item, t: &TypeParam) -> Option<std::collections::BTreeSet<String>> {
    let mut needs = Needs {
        name: &t.ident,
        traits: Default::default(),
        other: false,
        unfit: false,
    };
    needs.visit_item(item);
    (!needs.other && !needs.unfit).then_some(needs.traits)
}

/// A const parameter whose name is in no bound and no predicate of the item can be given any value.
fn friendly_const(item: &Item, c: &ConstParam) -> bool {
    let mut needs = Needs {
        name: &c.ident,
        traits: Default::default(),
        other: false,
        unfit: false,
    };
    needs.visit_item(item);
    !needs.other && !needs.unfit
}

/// A type that has the `traits` (the marker traits and comparisons of [`FRIENDLY`]): not a type
/// parameter, other than the generics (`Vec<u8>`, `(u8, u16)`), and unsized if `?Sized` allows it.
fn concrete_type(traits: &std::collections::BTreeSet<String>, draw: &mut Draw) -> Type {
    let mut choices: Vec<Type> = vec![
        parse_quote!(u8),
        parse_quote!(()),
        parse_quote!((u8, u16)),
        parse_quote!([u8; 2]),
        parse_quote!(&'static str),
        parse_quote!(bool),
        parse_quote!(char),
    ];
    if !traits.contains("Copy") {
        choices.push(parse_quote!(::std::vec::Vec<u8>));
        choices.push(parse_quote!(::std::string::String));
        choices.push(parse_quote!(::core::option::Option<u8>));
    }
    let unsized_ok = traits.contains("?Sized")
        && traits.iter().all(|t| {
            matches!(
                t.as_str(),
                "?Sized" | "Debug" | "PartialEq" | "Eq" | "PartialOrd" | "Ord" | "Send" | "Sync" | "Unpin"
            )
        });
    if unsized_ok {
        choices.push(parse_quote!(str));
        choices.push(parse_quote!([u8]));
    }
    choices.swap_remove(draw.below(choices.len()))
}

/// A type that holds a lifetime and has the marker traits and comparisons of [`FRIENDLY`], for a
/// parameter that [`friendly_type()`] allows: written into a function pointer or `Fn(..)` whose
/// output has an elided lifetime, it adds one to the inputs, which the copy must still name (the
/// `for<'e>` binder of `UseSite::substitute()`).
fn lifetime_type(draw: &mut Draw) -> Type {
    let mut choices: Vec<Type> = vec![
        parse_quote!(&'static str),
        parse_quote!((&'static str, u8)),
        parse_quote!(::core::option::Option<&'static str>),
        parse_quote!([&'static str; 2]),
        parse_quote!(&'static [u8]),
    ];
    choices.swap_remove(draw.below(choices.len()))
}

/// The lifetimes of the inputs of a signature as elision counts them: each elided one is its own,
/// the named ones count once. A function pointer or `Fn(..)` inside has its own, and so has a bound
/// with a `for<..>` binder for the lifetimes it declares.
#[derive(Default)]
struct SignatureLifetimes {
    elided: usize,
    named: Vec<Lifetime>,
    declared: Vec<Lifetime>,
}

impl<'ast> Visit<'ast> for SignatureLifetimes {
    fn visit_type_reference(&mut self, i: &'ast TypeReference) {
        if i.lifetime.is_none() {
            self.elided += 1;
        }
        syn::visit::visit_type_reference(self, i)
    }

    fn visit_lifetime(&mut self, i: &'ast Lifetime) {
        if i.ident == "_" {
            self.elided += 1;
        } else if !self.declared.contains(i) && !self.named.contains(i) {
            self.named.push(i.clone());
        }
    }

    fn visit_trait_bound(&mut self, i: &'ast TraitBound) {
        let outer = self.declared.len();
        if let Some(binder) = &i.lifetimes {
            for param in &binder.lifetimes {
                if let GenericParam::Lifetime(l) = param {
                    self.declared.push(l.lifetime.clone());
                }
            }
        }
        syn::visit::visit_trait_bound(self, i);
        self.declared.truncate(outer);
    }

    fn visit_type_bare_fn(&mut self, _: &'ast TypeBareFn) {}

    fn visit_parenthesized_generic_arguments(&mut self, _: &'ast ParenthesizedGenericArguments) {}
}

/// The names of the parameters among `names` that appear in the inputs of each signature of an item
/// (a function pointer, or the sugar of `Fn(..)`) whose output has an elided lifetime that the one
/// lifetime of its inputs gives: the signatures that a use site must keep valid by naming it
/// (`UseSite::substitute()`).
struct ElidedSignatures {
    names: Vec<String>,
    found: Vec<ElidedSignature>,
    /// The parameters named in the inputs of a signature whose output elides a lifetime that a path
    /// hides (`Fn(Chars, *const T) -> &u8`): the copy cannot name it, so an argument with a lifetime
    /// breaks it (F69).
    hidden: Vec<String>,
}

/// A signature that relies on elision and names type parameters in its inputs: which ones, and the
/// tokens of each input.
struct ElidedSignature {
    params: Vec<String>,
    inputs: Vec<String>,
}

impl ElidedSignatures {
    fn signature<'t>(
        &mut self,
        inputs: impl Iterator<Item = &'t Type> + Clone,
        output: &ReturnType,
    ) {
        let output = match output {
            ReturnType::Type(_, output) => output,
            ReturnType::Default => return,
        };
        let mut out = SignatureLifetimes::default();
        out.visit_type(output);
        if out.elided == 0 {
            return;
        }
        let mut lifetimes = SignatureLifetimes::default();
        for input in inputs.clone() {
            lifetimes.visit_type(input);
        }
        let shown = lifetimes.elided + lifetimes.named.len();
        if shown > 1 {
            return;
        }
        let named: Vec<String> = self
            .names
            .iter()
            .filter(|name| inputs.clone().any(|input| count_name(quote!(#input), name) > 0))
            .cloned()
            .collect();
        if shown == 0 {
            // `Self` is the item at the use site, which holds every parameter: an argument with a
            // lifetime adds one to the inputs that a path hides.
            if inputs
                .clone()
                .any(|input| count_name(quote!(#input), "Self") > 0)
            {
                self.hidden.extend(self.names.iter().cloned());
            } else {
                self.hidden.extend(named);
            }
        } else if !named.is_empty() {
            self.found.push(ElidedSignature {
                params: named,
                inputs: inputs.map(|input| quote!(#input).to_string()).collect(),
            });
        }
    }
}

impl<'ast> Visit<'ast> for ElidedSignatures {
    fn visit_type_bare_fn(&mut self, i: &'ast TypeBareFn) {
        self.signature(i.inputs.iter().map(|arg| &arg.ty), &i.output);
        syn::visit::visit_type_bare_fn(self, i)
    }

    fn visit_parenthesized_generic_arguments(&mut self, i: &'ast ParenthesizedGenericArguments) {
        self.signature(i.inputs.iter(), &i.output);
        syn::visit::visit_parenthesized_generic_arguments(self, i)
    }
}

/// The signatures of `item` whose output elides a lifetime, with the type parameters of `generics`
/// they name in their inputs.
fn signatures_of(item: &Item, generics: &Generics) -> ElidedSignatures {
    let names: Vec<String> = generics
        .params
        .iter()
        .filter_map(|param| match param {
            GenericParam::Type(t) => Some(t.ident.to_string()),
            _ => None,
        })
        .collect();
    let mut found = ElidedSignatures {
        names,
        found: Vec::new(),
        hidden: Vec::new(),
    };
    found.visit_item(item);
    found
}

/// For each signature of `item` that relies on elision, the type parameters of `generics` it names
/// in its inputs.
fn elided_signatures(item: &Item, generics: &Generics) -> Vec<ElidedSignature> {
    signatures_of(item, generics).found
}

/// The type parameters of `generics` that a signature of `item` names in its inputs, whose output
/// elides a lifetime that a path hides: no type with a lifetime is given to them.
fn hidden_inputs(item: &Item, generics: &Generics) -> std::collections::HashSet<String> {
    signatures_of(item, generics).hidden.into_iter().collect()
}

/// The type parameters of `generics` that a signature of `item` which relies on elision names in its
/// inputs.
fn elided_inputs(item: &Item, generics: &Generics) -> std::collections::HashSet<String> {
    elided_signatures(item, generics)
        .into_iter()
        .flat_map(|signature| signature.params)
        .collect()
}

/// The pairs of a type parameter of `generics` and a signature of `item` that relies on elision and
/// names it in its inputs, where an argument with a lifetime breaks the copy unless it names the
/// elided lifetime (F69): a parameter that can be given any type ([`friendly_type()`]), with no path
/// hiding a lifetime of that signature ([`hidden_inputs()`]). With `replaced` (the tokens of each
/// entry that the copy replaced), only the signatures that have one in their inputs: the path
/// through the Repeater names the lifetime too (F70). A name for each pair.
fn elided_candidates(item: &Item, generics: &Generics, replaced: Option<&[String]>) -> Vec<String> {
    let hidden = hidden_inputs(item, generics);
    let friendly = |name: &String| {
        generics.params.iter().any(|param| match param {
            GenericParam::Type(t) => t.ident == *name && friendly_type(item, t).is_some(),
            _ => false,
        })
    };
    elided_signatures(item, generics)
        .into_iter()
        .filter(|signature| {
            replaced.map_or(true, |replaced| {
                signature
                    .inputs
                    .iter()
                    .any(|input| replaced.iter().any(|entry| input.contains(entry.as_str())))
            })
        })
        .flat_map(|signature| signature.params)
        .filter(|name| !hidden.contains(name) && friendly(name))
        .collect()
}

/// A value for a const parameter of the type `ty`, as an argument.
fn concrete_const(ty: &Type, draw: &mut Draw) -> GenericArgument {
    let text = match ty {
        Type::Path(TypePath { path, .. }) => path
            .segments
            .last()
            .map(|seg| seg.ident.to_string())
            .unwrap_or_default(),
        _ => String::new(),
    };
    let choices: &[&str] = match text.as_str() {
        "usize" => &["3", "0", "{ 2 + 1 }"],
        "bool" => &["true", "false"],
        "char" => &["'q'"],
        _ => &["7", "1"],
    };
    let value: Expr = parse_str(choices[draw.below(choices.len())]).unwrap();
    GenericArgument::Const(value)
}

/// A use of the item with arguments for its generics: `impl Tr<'static, u8, 3> for Host`, a field
/// of a `S<(u8, u16)>`, a method of `impl<T> S<T>` with a `T` of its own. The arguments are
///
/// - for a lifetime: `'static`, or a lifetime that the use site declares (`'q0`), all of them one
///   or the other;
/// - for a type parameter that only has marker bounds and appears in no other bound and predicate
///   ([`friendly_type()`]): a type of [`concrete_type()`], or one that holds a lifetime
///   ([`lifetime_type()`]) if a function pointer or `Fn(..)` whose output has an elided lifetime
///   names it in its inputs (always with `lifetime_args`, and with an entry in the inputs, [`elided_candidates()`]); for a
///   const parameter that appears
///   in no bound, a value; and otherwise the parameter itself, which the use site declares as the
///   item does;
/// - left out for the defaults at the end of the list (if no predicate or bound names them):
///   [`Referrer::complete_args()`] gives them.
///
/// The copy is written with the arguments ([`encode_generics_args_with_params()`],
/// [`UseSite::substitute()`]) and compared with the original at them: the witness of each
/// position is called with them.
pub(crate) struct Instance {
    id: u64,
    /// The generics of the item.
    params: Punctuated<GenericParam, Token![,]>,
    /// The arguments given, as the use site writes them (without the defaults it leaves out).
    given: Vec<GenericArgument>,
    /// All the arguments, the defaults completed.
    completed: Vec<GenericArgument>,
    /// What the use site declares: the parameters that it keeps, the lifetimes that it names, and
    /// the predicates that remain.
    declared: Generics,
    /// The lifetimes of the item and what they became.
    lifetimes: Vec<(Lifetime, Lifetime)>,
    /// The type and const parameters given a type or a value: they are not declared.
    concrete: std::collections::HashSet<String>,
}

impl Instance {
    /// The host of a trait at the instantiation: another one than the host of the generic copy,
    /// which implements the trait for any arguments.
    fn host(&self) -> Ident {
        ident(&format!("Host{}I", self.id))
    }

    fn site(&self) -> UseSite {
        UseSite::substituting(&self.params, &self.completed)
    }

    /// How many times a parameter that was given a type with a lifetime is named in the inputs of a
    /// signature of the item that relies on elision (a function pointer, `Fn(..)`): one for each
    /// pair. The copy at the instantiation must name the elided lifetime there. How many of them
    /// are in a signature that has an entry in its inputs that the copy replaced (`replaced`: the
    /// tokens of each): the path through the Repeater names the lifetime too. And how many pairs of
    /// each kind the generator could have made ([`elided_candidates()`]), those whose parameter
    /// the instantiation gives an argument: `[all, with_entry, all_candidates, entry_candidates]`.
    fn lifetime_args_in_elided_signatures(&self, item: &Item, replaced: &[String]) -> [usize; 4] {
        let generics = Generics {
            params: self.params.clone(),
            ..Default::default()
        };
        let holds_lifetime = |name: &String| {
            matches!(self.arg_of(name), Some(arg) if self.concrete.contains(name)
                && quote!(#arg).to_string().contains('\''))
        };
        let given = |name: &String| {
            self.params
                .iter()
                .position(|param| param_name(param) == *name)
                .map_or(false, |at| at < self.given.len())
        };
        let count = |pairs: &[String]| {
            (
                pairs.iter().filter(|name| holds_lifetime(name)).count(),
                pairs.iter().filter(|name| given(name)).count(),
            )
        };
        let (all, all_candidates) = count(&elided_candidates(item, &generics, None));
        let (with_entry, entry_pairs) = count(&elided_candidates(item, &generics, Some(replaced)));
        [all, with_entry, all_candidates, entry_pairs]
    }

    /// Sets what the use site declares from the generics of the copy (the entries in their bounds are
    /// replaced): the parameters that it keeps, the lifetimes that it names instead of the item's,
    /// and the predicates but those on a parameter that is given a type or a value.
    fn declare(&mut self, copy: &Generics) {
        let mut declared = Generics {
            lt_token: Some(Default::default()),
            gt_token: Some(Default::default()),
            ..Default::default()
        };
        for param in copy.params.iter().take(self.given.len()) {
            let mut param = param.clone();
            match &mut param {
                GenericParam::Lifetime(l) => {
                    let to = self
                        .lifetimes
                        .iter()
                        .find(|(from, _)| *from == l.lifetime)
                        .map(|(_, to)| to.clone())
                        .unwrap();
                    if to.to_string() == "'static" {
                        continue;
                    }
                    l.lifetime = to;
                    for bound in l.bounds.iter_mut() {
                        MapLifetimes(&self.lifetimes).visit_lifetime_mut(bound);
                    }
                }
                GenericParam::Type(t) => {
                    if self.concrete.contains(&t.ident.to_string()) {
                        continue;
                    }
                    t.eq_token = None;
                    t.default = None;
                    for bound in t.bounds.iter_mut() {
                        MapLifetimes(&self.lifetimes).visit_type_param_bound_mut(bound);
                    }
                }
                GenericParam::Const(c) => {
                    if self.concrete.contains(&c.ident.to_string()) {
                        continue;
                    }
                    c.eq_token = None;
                    c.default = None;
                }
            }
            declared.params.push(param);
        }
        if let Some(where_clause) = &copy.where_clause {
            let mut kept: Punctuated<WherePredicate, Token![,]> = Punctuated::new();
            for predicate in &where_clause.predicates {
                let on_concrete = matches!(predicate, WherePredicate::Type(p)
                    if matches!(&p.bounded_ty, Type::Path(TypePath { qself: None, path })
                        if path.get_ident().map_or(false, |i| self.concrete.contains(&i.to_string()))));
                if on_concrete {
                    continue;
                }
                let mut predicate = predicate.clone();
                MapLifetimes(&self.lifetimes).visit_where_predicate_mut(&mut predicate);
                kept.push(predicate);
            }
            if !kept.is_empty() {
                declared.where_clause = Some(WhereClause {
                    where_token: Default::default(),
                    predicates: kept,
                });
            }
        }
        self.declared = declared;
    }

    /// The argument that the instantiation gives the parameter of the item with this name.
    fn arg_of(&self, name: &str) -> Option<&GenericArgument> {
        let at = self.params.iter().position(|p| param_name(p) == name)?;
        self.completed.get(at)
    }
}

/// What `inst(N)` asks of the instantiation: the `seed` that draws it, and `lifetime_args`, which
/// gives every type parameter that can be given any type a type that holds a lifetime.
#[derive(Clone, Copy)]
struct InstAsk {
    seed: u64,
    lifetime_args: bool,
}

/// Plans the instantiation of `generics` that `ask` draws (see [`Instance`]), or `None` if it has
/// no generic. `type_fn` writes a type that a default names through the Repeater, for
/// [`Referrer::complete_args()`].
fn plan_instance(
    referrer: &Referrer,
    context: ContextId,
    id: u64,
    item: &Item,
    generics: &Generics,
    ask: InstAsk,
    replaced: &[String],
    type_fn: impl FnMut(Type, EntryId, &Context, &[GenericArgument]) -> Type,
) -> Result<Option<Instance>> {
    let params: Vec<&GenericParam> = generics.params.iter().collect();
    let n = params.len();
    if n == 0 {
        return Ok(None);
    }
    let mut draw = Draw(ask.seed);
    let lifetime_args = ask.lifetime_args;
    let has_default = |param: &GenericParam| {
        matches!(param, GenericParam::Type(TypeParam { default: Some(_), .. }))
            || matches!(param, GenericParam::Const(ConstParam { default: Some(_), .. }))
    };
    // The defaults at the end of the list are left out, if nothing else names them.
    let first_default = (0..n).find(|&i| params[i..].iter().all(|p| has_default(p)));
    let given_n = match first_default {
        // What rustc takes for a default is not a type that a bound or a predicate of the item,
        // outside the declaration of the parameter, would hold for.
        Some(first)
            if draw.chance(60) && !params[first..].iter().any(|p| bounded_elsewhere(item, p)) =>
        {
            first
        }
        _ => n,
    };
    let all_static = draw.chance(50);
    // The parameters that a signature relying on elision names: they are given a type that holds a
    // lifetime in a share of the instantiations, which that signature must still compile with.
    let elided = elided_inputs(item, generics);
    let hidden = hidden_inputs(item, generics);
    // Those that also have an entry in the inputs always are ([`elided_candidates()`]).
    let candidates: std::collections::HashSet<String> =
        elided_candidates(item, generics, Some(replaced))
            .into_iter()
            .collect();
    let mut lifetimes: Vec<(Lifetime, Lifetime)> = Vec::new();
    let mut given: Vec<GenericArgument> = Vec::new();
    let mut concrete = std::collections::HashSet::new();
    for (i, param) in params.iter().take(given_n).enumerate() {
        match param {
            GenericParam::Lifetime(l) => {
                let to: Lifetime = if all_static {
                    parse_quote!('static)
                } else {
                    Lifetime::new(&format!("'q{}", i), Span::call_site())
                };
                lifetimes.push((l.lifetime.clone(), to.clone()));
                given.push(GenericArgument::Lifetime(to));
            }
            GenericParam::Type(t) => match friendly_type(item, t) {
                Some(traits)
                    if lifetime_args
                        || draw.chance(75)
                        || candidates.contains(&t.ident.to_string()) =>
                {
                    concrete.insert(t.ident.to_string());
                    let ty = if lifetime_args
                        || (elided.contains(&t.ident.to_string()) && draw.chance(80))
                        || candidates.contains(&t.ident.to_string())
                    {
                        lifetime_type(&mut draw)
                    } else {
                        concrete_type(&traits, &mut draw)
                    };
                    // A lifetime that a path hides cannot be named, so the copy breaks if the argument
                    // adds one to that signature.
                    let ty = if hidden.contains(&t.ident.to_string())
                        && quote!(#ty).to_string().contains('\'')
                    {
                        parse_quote!(u8)
                    } else {
                        ty
                    };
                    given.push(GenericArgument::Type(ty));
                }
                _ => {
                    let ident = &t.ident;
                    given.push(GenericArgument::Type(parse_quote!(#ident)));
                }
            },
            GenericParam::Const(c) => {
                if friendly_const(item, c) {
                    concrete.insert(c.ident.to_string());
                    given.push(concrete_const(&c.ty, &mut draw));
                } else {
                    // A const parameter that a bound names is passed by its own name.
                    let ident = &c.ident;
                    given.push(GenericArgument::Type(parse_quote!(#ident)));
                }
            }
        }
    }
    let mut completed = referrer
        .complete_args(context, &generics.params, given.iter(), type_fn)
        .map_err(Error::from)?;
    // A const parameter that is its own argument is written bare (`{ N }` is not allowed in the
    // arguments of a path where `N` is a generic).
    for (param, arg) in generics.params.iter().zip(completed.iter_mut()) {
        if let GenericParam::Const(c) = param {
            if names_param(arg, &c.ident) {
                let ident = &c.ident;
                *arg = GenericArgument::Type(parse_quote!(#ident));
            }
        }
    }
    Ok(Some(Instance {
        id,
        params: generics.params.clone(),
        given,
        completed,
        declared: Generics::default(),
        lifetimes,
        concrete,
    }))
}

/// Substitutes the instantiation in the syntax of a copy (a signature, an associated type): the
/// types by [`UseSite::substitute()`] and the lifetimes that appear outside a type.
struct SiteSubst<'a> {
    site: &'a UseSite,
    /// What `Self` is written as (`None`: as it is).
    self_ty: Option<&'a Type>,
    lifetimes: &'a [(Lifetime, Lifetime)],
    /// The const parameters of the item and their arguments: a bare name in the arguments of a path
    /// is parsed as a type.
    consts: Vec<(Ident, GenericArgument)>,
}

impl<'a> SiteSubst<'a> {
    fn new(site: &'a UseSite, self_ty: Option<&'a Type>, inst: &'a Instance) -> Self {
        let consts = inst
            .params
            .iter()
            .zip(&inst.completed)
            .filter_map(|(param, arg)| match param {
                GenericParam::Const(c) => Some((c.ident.clone(), arg.clone())),
                _ => None,
            })
            .collect();
        SiteSubst {
            site,
            self_ty,
            lifetimes: &inst.lifetimes,
            consts,
        }
    }
}

impl VisitMut for SiteSubst<'_> {
    fn visit_type_mut(&mut self, ty: &mut Type) {
        *ty = self.site.substitute(ty, self.self_ty);
    }

    fn visit_generic_argument_mut(&mut self, arg: &mut GenericArgument) {
        if let GenericArgument::Type(Type::Path(TypePath { qself: None, path })) = arg {
            if let Some(ident) = path.get_ident() {
                if let Some((_, to)) = self.consts.iter().find(|(name, _)| name == ident) {
                    *arg = to.clone();
                    return;
                }
            }
        }
        syn::visit_mut::visit_generic_argument_mut(self, arg)
    }

    fn visit_lifetime_mut(&mut self, lifetime: &mut Lifetime) {
        MapLifetimes(self.lifetimes).visit_lifetime_mut(lifetime)
    }

    // The bodies are not copied at an instantiation.
    fn visit_block_mut(&mut self, _: &mut Block) {}
}

/// The trait of the copy at the instantiation: the types of the signatures and the associated items
/// written with the arguments.
fn substituted_trait(copy: &ItemTrait, inst: &Instance) -> ItemTrait {
    let site = inst.site();
    let mut copy = copy.clone();
    let mut subst = SiteSubst::new(&site, None, inst);
    for trait_item in copy.items.iter_mut() {
        subst.visit_trait_item_mut(trait_item);
    }
    copy
}

/// Whether the argument is the parameter itself (`T`, or `N` as a type or as a const `{ N }`).
fn names_param(arg: &GenericArgument, name: &Ident) -> bool {
    let text = quote!(#arg).to_string().replace([' ', '{', '}'], "");
    name == text.as_str()
}

/// The function that compares a position of the copy at the instantiation with the original at the
/// same arguments: `fn __tl_ins_<id>_<j><declared>() -> PhantomData<*mut RewrittenAtArguments> {
/// witness::<arguments>() }`. The witness is the original's, called with the arguments of the
/// instantiation for the generics of the item.
fn instance_check_fn(
    id: u64,
    j: usize,
    pos: &Position,
    inst: &Instance,
    target: &Path,
    host: Option<&Type>,
) -> Option<TokenStream> {
    let (generics, ty) = lowered(pos, Implementor::Fixed);
    let ty = inst.site().substitute(&ty, None);
    let mut params: Punctuated<GenericParam, Token![,]> = Punctuated::new();
    let mut targs: Vec<TokenStream> = Vec::new();
    // A parameter of a method that is named like one of the item would hide it.
    let own: Vec<String> = pos
        .layers
        .iter()
        .skip(1)
        .flat_map(|layer| layer.params.iter().map(param_name))
        .collect();
    for param in generics.params.iter() {
        let name = param_name(param);
        let is_root = inst.params.iter().any(|p| param_name(p) == name);
        if is_root && own.contains(&name) {
            return None;
        }
        let arg = if is_root { inst.arg_of(&name) } else { None };
        let mut param = param.clone();
        match &mut param {
            GenericParam::Lifetime(l) => {
                if let Some(GenericArgument::Lifetime(to)) = arg {
                    if to.to_string() == "'static" {
                        continue;
                    }
                    l.lifetime = to.clone();
                }
                for bound in l.bounds.iter_mut() {
                    MapLifetimes(&inst.lifetimes).visit_lifetime_mut(bound);
                }
            }
            GenericParam::Type(t) => {
                // The argument that is the parameter itself keeps it; any other replaces it.
                let kept = match arg {
                    Some(arg) => {
                        targs.push(quote!(#arg));
                        names_param(arg, &t.ident)
                    }
                    None => {
                        let ident = &t.ident;
                        targs.push(quote!(#ident));
                        true
                    }
                };
                if !kept {
                    continue;
                }
            }
            GenericParam::Const(c) => {
                let kept = match arg {
                    Some(arg) => {
                        targs.push(quote!(#arg));
                        names_param(arg, &c.ident)
                    }
                    None => {
                        let ident = &c.ident;
                        targs.push(quote!(#ident));
                        true
                    }
                };
                if !kept {
                    continue;
                }
            }
        }
        params.push(param);
    }
    // The bounds name the types and the lifetimes of the item: written with the arguments.
    let site = inst.site();
    let mut subst = SiteSubst::new(&site, None, inst);
    for param in params.iter_mut() {
        subst.visit_generic_param_mut(param);
    }
    let mut where_clause = generics.where_clause.clone();
    if let Some(wc) = &mut where_clause {
        let mut kept: Punctuated<WherePredicate, Token![,]> = Punctuated::new();
        for predicate in wc.predicates.iter() {
            let on_concrete = matches!(predicate, WherePredicate::Type(p)
                if matches!(&p.bounded_ty, Type::Path(TypePath { qself: None, path })
                    if path.get_ident().map_or(false, |i| inst.concrete.contains(&i.to_string()))));
            if on_concrete {
                continue;
            }
            let mut predicate = predicate.clone();
            subst.visit_where_predicate_mut(&mut predicate);
            kept.push(predicate);
        }
        wc.predicates = kept;
    }
    if pos.self_sub.as_ref().map_or(false, |sub| sub.bound) {
        let host = host.expect("a trait has a host");
        targs.push(quote!(#host));
    }
    let name = ident(&format!("__tl_ins_{}_{}", id, j));
    let witness = witness_name(id, j);
    let ty = invariant_in(ty);
    let mut path = target.clone();
    path.segments
        .extend(pos.mods.iter().cloned().map(PathSegment::from));
    Some(quote! {
        #[allow(non_snake_case, dead_code)]
        fn #name<#params>() -> ::core::marker::PhantomData<#ty> #where_clause {
            #path::#witness::<#(#targs),*>()
        }
    })
}

/// The check that the arguments that `Referrer::complete_args()` gave for the defaults that the use
/// site left out are the defaults that rustc takes: for a struct, an enum or a union, `S<given>` is
/// `S<completed>` (a function takes one and returns the other); for a trait, the host that
/// implements `Tr<given>` implements `Tr<completed>`.
fn defaults_check(
    inst: &Instance,
    item: &Item,
    target: &Path,
    host: Option<&Type>,
) -> TokenStream {
    if inst.given.len() == inst.params.len() {
        return TokenStream::new();
    }
    let (given, completed) = (&inst.given, &inst.completed);
    let name = ident(&format!("__tl_def_{}", inst.id));
    // The generics that the use site declares, in a function: `Self` is the type or the host.
    let with_self = |self_ty: Type| -> (Punctuated<GenericParam, Token![,]>, Option<WhereClause>) {
        let site = inst.site();
        let mut subst = SiteSubst::new(&site, Some(&self_ty), inst);
        let mut params = inst.declared.params.clone();
        for param in params.iter_mut() {
            subst.visit_generic_param_mut(param);
        }
        let mut where_clause = inst.declared.where_clause.clone();
        if let Some(where_clause) = &mut where_clause {
            subst.visit_where_clause_mut(where_clause);
        }
        (params, where_clause)
    };
    match item {
        Item::Struct(ItemStruct { ident: n, .. })
        | Item::Enum(ItemEnum { ident: n, .. })
        | Item::Union(ItemUnion { ident: n, .. }) => {
            let self_ty: Type = parse_quote!(#target::#n<#(#completed),*>);
            let (params, where_clause) = with_self(self_ty);
            quote! {
                #[allow(non_snake_case, dead_code)]
                fn #name<#params>(x: #target::#n<#(#given),*>) -> #target::#n<#(#completed),*> #where_clause {
                    x
                }
            }
        }
        Item::Trait(ItemTrait { ident: n, .. }) => {
            let host = host.expect("a trait has a host");
            let (params, where_clause) = with_self(host.clone());
            // `need` has the bounds of the trait for any `__X` (`T: AsRef<__X>`), and those for the host
            // that the arguments name (`<Host as Repeater<..>>::Type` for a default); its caller
            // has the bounds for the host.
            let (mut need_params, mut need_where) = with_self(parse_quote!(__X));
            let (host_params, host_where) = (params.clone(), where_clause.clone());
            for param in need_params.iter_mut() {
                if let GenericParam::Type(t) = param {
                    let more = host_params.iter().find_map(|p| match p {
                        GenericParam::Type(o) if o.ident == t.ident => Some(o.bounds.clone()),
                        _ => None,
                    });
                    for bound in more.into_iter().flatten() {
                        if !t.bounds.iter().any(|b| quote!(#b).to_string() == quote!(#bound).to_string()) {
                            t.bounds.push(bound);
                        }
                    }
                }
            }
            if let Some(host_where) = &host_where {
                let need_where = need_where.get_or_insert_with(|| WhereClause {
                    where_token: Default::default(),
                    predicates: Punctuated::new(),
                });
                for predicate in &host_where.predicates {
                    let text = quote!(#predicate).to_string();
                    if !need_where.predicates.iter().any(|p| quote!(#p).to_string() == text) {
                        need_where.predicates.push(predicate.clone());
                    }
                }
            }
            let args: Vec<TokenStream> = params
                .iter()
                .filter_map(|param| match param {
                    GenericParam::Type(t) => {
                        let ident = &t.ident;
                        Some(quote!(#ident))
                    }
                    GenericParam::Const(c) => {
                        let ident = &c.ident;
                        Some(quote!(#ident))
                    }
                    GenericParam::Lifetime(_) => None,
                })
                .collect();
            let need = ident(&format!("__tl_need_{}", inst.id));
            let mut need_params = need_params;
            need_params.push(parse_quote!(__X: ?::core::marker::Sized + #target::#n<#(#completed),*>));
            quote! {
                #[allow(non_snake_case, dead_code)]
                fn #need<#need_params>() #need_where {}
                #[allow(non_snake_case, dead_code)]
                fn #name<#params>() #where_clause {
                    #need::<#(#args,)* #host>();
                }
            }
        }
        _ => TokenStream::new(),
    }
}

/// Checks `UseSite::generic_args()` on the arguments of the generics of the item for the
/// generics of the context of every entry: the identity arguments give the generics' own names,
/// and a list with one argument too many, or one of the other kind in the first place, or the last
/// one missing, is refused (the last one missing only where the generics name that parameter).
fn check_refusals(referrer: &Referrer, params: &Punctuated<GenericParam, Token![,]>) -> Result<()> {
    let (first, last) = match (params.first(), params.last()) {
        (Some(first), Some(last)) => (first, last),
        _ => return Ok(()),
    };
    let generics = Generics {
        params: params.clone(),
        ..Default::default()
    };
    let valid = identity_args(&generics);
    let fail = |what: String| Error::new(Span::call_site(), format!("GENERIC_ARGS: {}", what));
    let mut extra = valid.clone();
    extra.push(GenericArgument::Lifetime(parse_quote!('static)));
    let mut wrong = valid.clone();
    wrong[0] = match first {
        GenericParam::Lifetime(_) => GenericArgument::Type(parse_quote!(u8)),
        _ => GenericArgument::Lifetime(parse_quote!('static)),
    };
    let mut short = valid.clone();
    short.pop();
    let entries = referrer
        .iter()
        .map(|(ix, _, context)| (format!("the context of the entry {}", ix), context));
    for (context_name, context) in entries {
        let generics = &context.generics;
        let own = UseSite::substituting(params, &valid).generic_args(generics);
        if own.is_err() {
            return Err(fail(format!(
                "refused the arguments that are the generics' own names in {}",
                context_name
            )));
        }
        for (what, args) in [("one too many", &extra), ("one of another kind", &wrong)] {
            match UseSite::substituting(params, args).generic_args(generics) {
                Err(EncodeError::UnexpectedArgument(_)) => (),
                other => {
                    return Err(fail(format!(
                        "{} in {} is not an `UnexpectedArgument`: {:?}",
                        what,
                        context_name,
                        other.map(|a| a.len())
                    )))
                }
            }
        }
        let needs = generics
            .params
            .iter()
            .any(|param| param_name(param) == param_name(last));
        match (
            needs,
            UseSite::substituting(params, &short).generic_args(generics),
        ) {
            (true, Err(EncodeError::MissingArgument(_))) | (false, Ok(_)) => (),
            (_, other) => {
                return Err(fail(format!(
                    "the last argument missing in {} (its generics {} `{}`) gives {:?}",
                    context_name,
                    if needs { "name" } else { "do not name" },
                    param_name(last),
                    other.map(|a| a.len())
                )))
            }
        }
    }
    Ok(())
}

/// The names of the entries' types that a module or a block can declare again: the first segment of
/// the relative paths in them, which are no generic of the context of the entry.
fn shadowable(referrer: &Referrer, ix: EntryId) -> std::collections::BTreeSet<String> {
    struct Heads(std::collections::BTreeSet<String>);
    impl<'ast> Visit<'ast> for Heads {
        fn visit_path(&mut self, path: &'ast Path) {
            if path.leading_colon.is_none() {
                if let Some(seg) = path.segments.first() {
                    self.0.insert(seg.ident.to_string());
                }
            }
            syn::visit::visit_path(self, path)
        }
    }
    let (ty, context) = (crate::entry_type(referrer, ix), crate::entry_context(referrer, ix));
    let mut heads = Heads(Default::default());
    heads.visit_type(ty);
    for param in &context.generics.params {
        heads.0.remove(&param_name(param));
    }
    for name in ["Self", "self", "super", "crate"] {
        heads.0.remove(name);
    }
    heads.0
}

/// A module (`mod __tl_shadow { .. }`) or a block (`fn __tl_shadow() { .. }`) that declares `names`
/// around an item, as the copy of an item could be placed in one.
pub(crate) struct Wrap {
    block: bool,
    names: Vec<Ident>,
}

impl Wrap {
    fn around(&self, item: Item) -> Item {
        let stubs = self
            .names
            .iter()
            .map(|name| -> Item { parse_quote!(pub struct #name;) });
        let wrapper = ident("__tl_shadow");
        if self.block {
            let mut function: ItemFn = parse_quote!(fn #wrapper() {});
            function.block.stmts = stubs.chain(std::iter::once(item)).map(Stmt::Item).collect();
            Item::Fn(function)
        } else {
            let mut module: ItemMod = parse_quote!(mod #wrapper {});
            module.content = Some((Default::default(), stubs.chain(std::iter::once(item)).collect()));
            Item::Mod(module)
        }
    }

    /// The item that `around()` put in.
    fn inside(&self, wrapper: Item) -> Item {
        match wrapper {
            Item::Fn(function) => match function.block.stmts.into_iter().last() {
                Some(Stmt::Item(item)) => item,
                _ => unreachable!("the wrapper holds the item"),
            },
            Item::Mod(module) => module
                .content
                .and_then(|(_, items)| items.into_iter().last())
                .expect("the wrapper holds the item"),
            _ => unreachable!("the wrapper is a module or a function"),
        }
    }
}

/// The entries that a module or a block that declares a name hides, which the visitor must leave as
/// written inside it (`Visitor::shadow_items()`): the item is copied inside a module or a block
/// that declares some of the names its entries start with, and the entries that the visitor
/// replaces there are those it replaces in a module or a block that declares nothing, but for the
/// ones that name a declared name as the first segment of a relative path. Returns how many
/// entries were hidden.
fn shadow_check(
    referrer: &Referrer,
    context: ContextId,
    item: &Item,
    args: &Args,
    leaker_path: &Path,
    dyn_host: bool,
    seed: u64,
) -> Result<usize> {
    // An item of a scope `mod` has entries keyed by a module: a module does not hide them.
    if args.nested() || args.via == "nested" || args.redeclare || matches!(item, Item::Mod(_)) {
        return Ok(0);
    }
    let mut draw = Draw(seed);
    let shadowable: Vec<std::collections::BTreeSet<String>> = referrer
        .iter()
        .map(|(ix, _, _)| shadowable(referrer, ix))
        .collect();
    let candidates: std::collections::BTreeSet<&String> = shadowable.iter().flatten().collect();
    if candidates.is_empty() {
        return Ok(0);
    }
    let mut names: Vec<Ident> = candidates
        .iter()
        .filter(|_| draw.chance(60))
        .map(|name| ident(name))
        .collect();
    if names.is_empty() {
        names.push(ident(candidates.iter().next().unwrap()));
    }
    let block = draw.chance(50);
    let plain_wrap = Wrap {
        block,
        names: Vec::new(),
    };
    let shadowing = Wrap { block, names };
    let control = rewrite_with(
        referrer,
        context,
        item,
        args,
        leaker_path,
        dyn_host,
        None,
        Some(&plain_wrap),
    )?;
    let shadowed = rewrite_with(
        referrer,
        context,
        item,
        args,
        leaker_path,
        dyn_host,
        None,
        Some(&shadowing),
    )?;
    let declared: std::collections::HashSet<String> =
        shadowing.names.iter().map(|name| name.to_string()).collect();
    let hidden = |ix: &EntryId| shadowable[ix.index()].iter().any(|name| declared.contains(name));
    let expected: std::collections::BTreeSet<EntryId> =
        control.used.iter().copied().filter(|ix| !hidden(ix)).collect();
    let hides = control.used.iter().filter(|ix| hidden(ix)).count();
    if shadowed.used != expected {
        let entry = |ix: &EntryId| {
            let ty = crate::entry_type(referrer, *ix);
            quote!(#ty).to_string()
        };
        let list = |set: Vec<&EntryId>| set.into_iter().map(entry).collect::<Vec<_>>().join(", ");
        return Err(Error::new(
            Span::call_site(),
            format!(
                "SHADOWED ENTRIES (in {} that declares {:?}): the visitor replaced {} that a declaration of its first segment hides, and left {} that nothing hides",
                if block { "a block" } else { "a module" },
                declared,
                list(shadowed.used.iter().filter(|ix| !expected.contains(ix)).collect()),
                list(expected.iter().filter(|ix| !shadowed.used.contains(ix)).collect()),
            ),
        ));
    }
    Ok(hides)
}

/// What an invocation of the item with `inst(N)` adds: the checks at an instantiation of its
/// generics (see [`Instance`]) and of `UseSite::generic_args()`. Returns the tokens, the
/// number of positions checked at the instantiation, and how many times the instantiation passed an
/// argument that holds a lifetime to a signature that relies on elision, how many of those have
/// a replaced entry in their inputs, and how many pairs of each kind it could have made
/// ([`Instance::lifetime_args_in_elided_signatures()`]).
fn instance_part(
    referrer: &Referrer,
    context: ContextId,
    item: &Item,
    args: &Args,
    path: &Path,
    dyn_host: bool,
    ask: InstAsk,
) -> Result<(TokenStream, usize, [usize; 4])> {
    let generics = match item {
        Item::Struct(i) => &i.generics,
        Item::Enum(i) => &i.generics,
        Item::Union(i) => &i.generics,
        Item::Trait(i) => &i.generics,
        Item::Impl(i) => &i.generics,
        _ => return Ok((TokenStream::new(), 0, [0; 4])),
    };
    // The copy of a trait that is declared again passes `()` for its defaults.
    if generics.params.is_empty() || args.redeclare {
        return Ok((TokenStream::new(), 0, [0; 4]));
    }
    check_refusals(referrer, &generics.params)?;
    let host_name = ident(&format!("Host{}I", args.id));
    let error = std::cell::RefCell::new(None);
    let projector_with = |prefix: Vec<GenericArgument>| Projector {
        referrer,
        repeater: repeater_path(path),
        id: args.repeater_id(),
        root_params: generics.params.clone(),
        root_args: prefix,
        bound_self: matches!(item, Item::Trait(_)).then(|| host_type(&host_name, dyn_host)),
        type_self: None,
        marker: args.marker.clone(),
        used: Default::default(),
        at_instance: true,
    };
    let replaced: Vec<String> = rewrite(referrer, context, item, args, path, dyn_host)?
        .used
        .iter()
        .filter_map(|ix| referrer.iter().nth(ix.index()))
        .map(|(_, entry, _)| quote!(#entry).to_string())
        .collect();
    let planned = plan_instance(
        referrer,
        context,
        args.id,
        item,
        generics,
        ask,
        &replaced,
        |_leaked, ix, entry, prefix| {
            // A default kept as written is not substituted again: write it at the site.
            projector_with(prefix.to_vec())
                .project(ix, entry)
                .unwrap_or_else(|e| {
                    *error.borrow_mut() = Some(e);
                    parse_quote!(())
                })
        },
    )?;
    if let Some(e) = error.into_inner() {
        return Err(e);
    }
    let inst = match planned {
        Some(inst) => inst,
        None => return Ok((TokenStream::new(), 0, [0; 4])),
    };
    let mut inst = inst;
    let rewritten = rewrite_with(
        referrer,
        context,
        item,
        args,
        path,
        dyn_host,
        Some(&inst),
        None,
    )?;
    match &rewritten.item {
        Item::Struct(i) => inst.declare(&i.generics),
        Item::Enum(i) => inst.declare(&i.generics),
        Item::Union(i) => inst.declare(&i.generics),
        Item::Trait(i) => inst.declare(&i.generics),
        Item::Impl(i) => inst.declare(&i.generics),
        _ => (),
    }
    let host = host_arg(&rewritten, args, path);
    let host_ty = host.as_ref().map(|(ty, _)| ty);
    let mut out = TokenStream::new();
    let mut positions = 0;
    for check in comparable(&rewritten.item, item, args, host.as_ref(), path) {
        if check.expected.is_some() {
            continue;
        }
        if let Some(function) = instance_check_fn(args.id, check.j, &check.pos, &inst, path, host_ty) {
            out.extend(function);
            positions += 1;
        }
    }
    if let (Item::Trait(copy), Item::Trait(original), Some(host)) =
        (&rewritten.item, item, rewritten.host.as_ref())
    {
        let copy = substituted_trait(copy, &inst);
        out.extend(host_impl(&copy, original, host, dyn_host, path, false, Some(&inst)));
    }
    out.extend(defaults_check(&inst, item, path, host_ty));
    let counts = inst.lifetime_args_in_elided_signatures(item, &replaced);
    Ok((out, positions, counts))
}


/// Rewrites the defaults of the type parameters of a trait for a declaration (`Args::redeclare`).
///
/// The Repeater's arguments of an entry that names the `Self` of the trait are the generics of
/// `Self: Tr<T, U>` too, the defaulted parameter among them. A default that passes `U` names the
/// parameter that it is the default of (E0128), so the copy passes `()` for the defaulted
/// parameters, and the host implements the Leaker's trait for any type of them. The result is the
/// same type: the entry does not depend on them.
fn rewrite_defaults(
    referrer: &Referrer,
    context: ContextId,
    original: &ItemTrait,
    copy: &mut ItemTrait,
    projector: &Projector<'_>,
    base: &Path,
) -> Result<()> {
    let first = original.generics.params.iter().position(
        |param| matches!(param, GenericParam::Type(TypeParam { default: Some(_), .. })),
    );
    let first = match first {
        Some(first) => first,
        None => return Ok(()),
    };
    let mut root_args = projector.root_args.clone();
    for (arg, param) in root_args
        .iter_mut()
        .zip(&original.generics.params)
        .skip(first)
    {
        if matches!(param, GenericParam::Type(_)) {
            *arg = GenericArgument::Type(parse_quote!(()));
        }
    }
    let placeholder = Projector {
        referrer,
        repeater: projector.repeater.clone(),
        id: projector.id,
        root_params: projector.root_params.clone(),
        root_args,
        bound_self: projector.bound_self.clone(),
        type_self: projector.type_self.clone(),
        marker: projector.marker.clone(),
        used: Default::default(),
        at_instance: projector.at_instance,
    };
    // The generics only: the same scope, and so the same entries, with no item to rewrite.
    let mut shell = original.clone();
    shell.items.clear();
    shell.supertraits.clear();
    let error = std::cell::RefCell::new(None);
    let mut file = File {
        shebang: None,
        attrs: Vec::new(),
        items: vec![Item::Trait(shell)],
    };
    referrer
        .visitor(context, |_leaked, ix, entry| {
            placeholder.project(ix, entry).unwrap_or_else(|e| {
                *error.borrow_mut() = Some(e);
                parse_quote!(())
            })
        })
        .with_nested_self(|nested, _leaked, ix, entry| {
            placeholder
                .project_nested(nested, base, ix, entry)
                .unwrap_or_else(|e| {
                    *error.borrow_mut() = Some(e);
                    parse_quote!(())
                })
        })
        .visit_file_mut(&mut file);
    if let Some(e) = error.into_inner() {
        return Err(e);
    }
    projector.used.borrow_mut().extend(placeholder.used.into_inner());
    let shell = match file.items.remove(0) {
        Item::Trait(shell) => shell,
        _ => unreachable!(),
    };
    for (param, rewritten) in copy.generics.params.iter_mut().zip(shell.generics.params) {
        if let (GenericParam::Type(param), GenericParam::Type(rewritten)) = (param, rewritten) {
            param.default = rewritten.default;
        }
    }
    Ok(())
}

/// Drops the imports of names the Leaker leaks: a copy of a module in another crate cannot resolve
/// them, and every use of them is rewritten. Imports of absolute paths and of the module's own
/// items stay.
///
/// With `constants` the import of `ILEN` stays: the paths of values and defaults that a rule allows
/// name it (`clean` copies of `Rules::values()` and `Rules::defaults()`). With `imports` all of
/// them stay: the config that has the rule wrote every relative path of a value or a default as it
/// is, whatever it names.
fn drop_relative_uses(items: &mut Vec<Item>, constants: bool, imports: bool) {
    fn renames_ilen(tree: &UseTree) -> bool {
        match tree {
            UseTree::Path(p) => renames_ilen(&p.tree),
            UseTree::Rename(r) => r.rename == "ILEN",
            _ => false,
        }
    }
    items.retain(|item| match item {
        Item::Use(u) => {
            u.leading_colon.is_some()
                || matches!(&u.tree, UseTree::Path(p) if p.ident == "self")
                || imports
                || (constants && renames_ilen(&u.tree))
        }
        _ => true,
    });
    for item in items.iter_mut() {
        if let Item::Mod(m) = item {
            if let Some((_, items)) = &mut m.content {
                drop_relative_uses(items, constants, imports);
            }
        }
    }
}

/// The value of an associated const of the host, whose type is `ty`.
fn const_value(ty: &Type) -> Expr {
    if let Type::Path(TypePath { qself: None, path }) = ty {
        match path.segments.last() {
            Some(seg) if seg.ident == "PhantomData" => {
                return parse_quote!(::core::marker::PhantomData)
            }
            Some(seg) if seg.ident == "bool" => return parse_quote!(false),
            _ => (),
        }
    }
    parse_quote!(0)
}

/// The host of a trait. For a struct, an enum or a union nested in the module (scope `mod`) it is
/// the Leaker's type: the visitor leaves a `Self` that is not in an entry (`Vec<Self>` as a field,
/// or in an absolute type, `::p::W<Self>`, which no Referrer replaces: F38), which is the copy's
/// own type, and the check writes the Leaker's for it, as the witness does. (So the check cannot
/// tell the copy's `Self` from the Leaker's; the fixtures of `tests/cross_crate.rs` do.) The copy of a struct that is not nested has its `Self` replaced
/// (`Visitor::with_self()`).
fn host_arg(rewritten: &Rewritten, args: &Args, leaker_path: &Path) -> Option<Host> {
    if let Some(host) = &rewritten.host {
        return Some((host_type(host, rewritten.dyn_host), leaker_path.clone()));
    }
    if !args.nested() {
        return None;
    }
    let (ident, generics) = match &rewritten.item {
        Item::Struct(i) => (&i.ident, &i.generics),
        Item::Enum(i) => (&i.ident, &i.generics),
        Item::Union(i) => (&i.ident, &i.generics),
        _ => return None,
    };
    let params = identity_args(generics);
    let ty: Type = parse_quote!(#leaker_path::#ident<#(#params),*>);
    Some((ty, leaker_path.clone()))
}

/// An implementation of the Leaker's trait for the host, with the signatures of the copy.
///
/// `original` is the trait as the Leaker has it: the value of a constant depends on its type, which
/// the copy may have replaced.
fn host_impl(
    item: &ItemTrait,
    original: &ItemTrait,
    host: &Ident,
    dyn_host: bool,
    leaker_path: &Path,
    bodies: bool,
    at: Option<&Instance>,
) -> TokenStream {
    // At an instantiation the impl declares what the use site declares, and names the arguments the
    // use site gives (the defaults of the trait are left to rustc).
    let (impl_generics, ty_generics, where_clause) = match at {
        Some(inst) => {
            let (impl_generics, _, where_clause) = inst.declared.split_for_impl();
            let given = &inst.given;
            let args = (!given.is_empty()).then(|| quote!(<#(#given),*>));
            (quote!(#impl_generics), quote!(#args), quote!(#where_clause))
        }
        None => {
            let (impl_generics, ty_generics, where_clause) = item.generics.split_for_impl();
            (
                quote!(#impl_generics),
                quote!(#ty_generics),
                quote!(#where_clause),
            )
        }
    };
    let name = &item.ident;
    let mut items = TokenStream::new();
    for (trait_item, original) in item.items.iter().zip(&original.items) {
        match (trait_item, original) {
            // With `bodies` the host has the default bodies and values of the copy (`Args::bodies`):
            // the rewritten types in them are checked as the ones of the signatures are.
            (TraitItem::Fn(f), _) => {
                let sig = &f.sig;
                match (&f.default, bodies) {
                    (Some(block), true) => items.extend(quote!(#sig #block)),
                    _ => items.extend(quote!(#sig { ::core::unimplemented!() })),
                }
            }
            (TraitItem::Const(c), TraitItem::Const(original)) => {
                let (name, ty) = (&c.ident, &c.ty);
                match (&c.default, bodies) {
                    (Some((_, value)), true) => items.extend(quote!(const #name: #ty = #value;)),
                    _ => {
                        let value = const_value(&original.ty);
                        items.extend(quote!(const #name: #ty = #value;));
                    }
                }
            }
            (TraitItem::Type(t), _) => {
                // A generic associated type keeps its parameters and its where clause.
                let (name, generics) = (&t.ident, &t.generics);
                let where_clause = &generics.where_clause;
                items.extend(quote!(type #name #generics = u8 #where_clause;));
            }
            _ => (),
        }
    }
    let host_ty = host_type(host, dyn_host);
    // An unsized host is a trait object of a trait of its own: a local type, so that the impl of
    // the Leaker's trait for it is allowed in the Referrer's crate.
    let declaration = if dyn_host {
        quote!(pub trait #host {})
    } else {
        quote! {
            #[derive(Debug, Clone, Default)]
            pub struct #host;
        }
    };
    quote! {
        #declaration
        impl #impl_generics #leaker_path::#name #ty_generics for #host_ty #where_clause {
            #items
        }
    }
}

/// Without `Rules::primitive()`, `u8` is a relative path like any other (a module may
/// declare its own `u8`): the type that has it is an entry, and the copy replaces it. A copy that
/// keeps a primitive in a type position copies a name that means something else in the Referrer's
/// crate, whatever it is equal to there.
fn check_primitives_replaced(
    copy: &Item,
    original: &Item,
    args: &Args,
    host: Option<&Host>,
) -> Result<()> {
    if args.primitive {
        return Ok(());
    }
    const PRIMITIVES: &[&str] = &[
        "u8", "u16", "u32", "u64", "u128", "usize", "i8", "i16", "i32", "i64", "i128", "isize",
        "f32", "f64", "bool", "char", "str",
    ];
    /// A primitive type written as a path of one segment (not `str` of `::core::str::Chars`).
    struct Primitive(Option<String>);
    impl<'ast> Visit<'ast> for Primitive {
        fn visit_type_path(&mut self, i: &'ast TypePath) {
            if i.qself.is_none() && i.path.leading_colon.is_none() && i.path.segments.len() == 1 {
                let ident = i.path.segments[0].ident.to_string();
                if PRIMITIVES.contains(&ident.as_str()) {
                    self.0 = Some(ident);
                }
            }
            syn::visit::visit_type_path(self, i)
        }
    }
    let original = collect_positions(original, None, args);
    let copied = collect_positions(copy, host, args);
    for (orig, pos) in original.iter().zip(&copied) {
        if has_no_witness(orig) {
            continue;
        }
        let ty = &pos.ty;
        let text = quote!(#ty).to_string();
        let mut found = Primitive(None);
        found.visit_type(ty);
        if let Some(name) = found.0 {
            return Err(Error::new(
                Span::call_site(),
                format!(
                    "PRIMITIVE KEPT: the copy keeps `{}` in the type `{}`, which the Leaker does not allow (primitive = false) and so leaks",
                    name, text
                ),
            ));
        }
    }
    Ok(())
}

/// A position of the copy that is compared with the original.
pub(crate) struct Check {
    /// The index of the position (`collect_positions()`), or after the last one for the header of an
    /// impl.
    j: usize,
    /// The position in the copy.
    pos: Position,
    /// `None`: the original is the Leaker's witness. `Some`: the original written for the Referrer's
    /// crate ([`Reroot`]), a position of a module that names what the module declares, where the
    /// witness (in the Leaker's module) cannot be called with the copy's own items.
    expected: Option<Position>,
    /// The header of an impl ([`header_position()`]), not a type position of the item.
    header: bool,
    /// The reverse trait compares it ([`reverse_impls()`]). Not a position of an impl with a
    /// projection of `Self` (`Self::A`): the impl's `Self` is a concrete type, so `<Self as
    /// Tr<'a, 'c>>::G<'c>` is normalized away in the signature that the trait would compare, and
    /// with it the outlives bounds that the trait's declaration gives the original (`'c: 'a`)
    /// and that the Repeater's arguments carry as `&'a &'c ()`: the copy would seem to demand
    /// more than the original. The check against the witness and the header still compare it.
    reverse: bool,
}

/// The header of an impl of a trait as a position: the self type, with the generics of the impl and
/// the trait reference as a where predicate (`SelfTy: Trait<..>`). The copy cannot be declared as an
/// impl (the orphan rules), but both crates can state the header: it holds in the Leaker's crate by
/// the impl itself, so a function there assumes it, and the copy's must follow from it, and the
/// other way round. The trait path and the lifetimes and constants of its arguments are compared
/// that way, besides the types, and so are the generics of the impl where no type position can be.
fn header_position(item: &Item, args: &Args) -> Option<Position> {
    let i = match item {
        Item::Impl(i) if args.scope.is_empty() => i,
        _ => return None,
    };
    let path = match &i.trait_ {
        Some((None, path, _)) => path,
        _ => return None,
    };
    let mut generics = i.generics.clone();
    generics
        .make_where_clause()
        .predicates
        .push(WherePredicate::Type(PredicateType {
            lifetimes: None,
            bounded_ty: (*i.self_ty).clone(),
            colon_token: Default::default(),
            bounds: std::iter::once(TypeParamBound::Trait(TraitBound {
                paren_token: None,
                modifier: TraitBoundModifier::None,
                lifetimes: None,
                path: path.clone(),
            }))
            .collect(),
        }));
    let mut elided = Elided(false);
    elided.visit_path(path);
    if elided.0 {
        return None;
    }
    Some(Position {
        mods: Vec::new(),
        layers: vec![generics],
        self_sub: Some(SelfSub {
            ty: (*i.self_ty).clone(),
            trait_ref: Some(path.clone()),
            bound: false,
        }),
        ty: (*i.self_ty).clone(),
    })
}

/// The entries of the copy that rustc must normalize to use a bound, and the names around: not
/// those in an associated type (`Item = <entry>`, `-> <entry>`), which it takes as they are.
#[derive(Default)]
struct Entries {
    /// Every name seen.
    names: HashSet<String>,
    any: bool,
    /// The names in the arguments of the entries that have some.
    args: HashSet<String>,
}

impl<'ast> Visit<'ast> for Entries {
    fn visit_ident(&mut self, i: &'ast Ident) {
        self.names.insert(i.to_string());
    }
    fn visit_type(&mut self, i: &'ast Type) {
        if let (true, Type::Path(TypePath { qself: Some(qself), path })) = (is_entry(i), i) {
            self.any = true;
            if let PathArguments::AngleBracketed(a) = &path.segments[qself.position - 1].arguments {
                if let Some(GenericArgument::Type(Type::Tuple(t))) = a.args.first() {
                    let mut inner = Entries::default();
                    inner.visit_type_tuple(t);
                    self.args.extend(inner.names);
                }
            }
        }
        syn::visit::visit_type(self, i)
    }
    fn visit_assoc_type(&mut self, _: &'ast AssocType) {}
    fn visit_parenthesized_generic_arguments(&mut self, i: &'ast ParenthesizedGenericArguments) {
        i.inputs.iter().for_each(|input| self.visit_type(input));
    }
}

/// Whether the header of the copy cannot be stated as a where predicate (`SelfTy: Trait`): it names
/// an entry with arguments, whose Repeater impl has the bounds of the generics it is given, and the
/// function states a bound on one of those through another entry (`T: AsRef<<X as
/// Repeater<..>>::Type>`). rustc normalizes the predicate with the bounds not yet normalized and
/// fails to prove the impl's (E0277, though the copy of the impl compiles: its self type is not in
/// its environment). The header is left out then, and the reverse trait's method of it.
fn header_cycles(header: &Position) -> bool {
    let mut generics = header.layers[0].clone();
    let predicate = generics.make_where_clause().predicates.pop();
    let mut in_header = Entries::default();
    in_header.visit_type(&header.ty);
    predicate.iter().for_each(|p| in_header.visit_where_predicate(p.value()));
    let cycles = |visit: &dyn Fn(&mut Entries)| {
        let mut unit = Entries::default();
        visit(&mut unit);
        unit.any && unit.names.intersection(&in_header.args).next().is_some()
    };
    generics.type_params().any(|p| cycles(&|e| e.visit_type_param(p)))
        || generics
            .where_clause
            .iter()
            .flat_map(|w| &w.predicates)
            .any(|p| cycles(&|e| e.visit_where_predicate(p)))
}

/// The header of the copy of an impl whose original has a witness, and whether it is left out
/// ([`header_cycles()`]).
fn header_of_copy(copy: &Item, item: &Item, args: &Args) -> Option<(Position, bool)> {
    match (header_position(copy, args), header_position(item, args)) {
        (Some(copied), Some(original)) if !has_no_witness(&original) => {
            let cycles = header_cycles(&copied);
            Some((copied, cycles))
        }
        _ => None,
    }
}

/// The names of the generics of the layers of a position.
fn generic_names(pos: &Position) -> HashSet<String> {
    pos.layers
        .iter()
        .flat_map(|g| g.params.iter())
        .map(|param| match param {
            GenericParam::Lifetime(l) => l.lifetime.to_string(),
            GenericParam::Type(t) => t.ident.to_string(),
            GenericParam::Const(c) => c.ident.to_string(),
        })
        .collect()
}

/// Whether the type or the generics of a position have a projection of `Self` (`Self::A`,
/// `<Self as Tr>::A`).
fn projects_self(pos: &Position) -> bool {
    struct Projects(bool);
    impl<'ast> Visit<'ast> for Projects {
        fn visit_type_path(&mut self, i: &'ast TypePath) {
            let is_self = |ty: &Type| {
                matches!(ty, Type::Path(TypePath { qself: None, path }) if path.is_ident("Self"))
            };
            match &i.qself {
                Some(qself) => self.0 |= is_self(&qself.ty),
                None => {
                    let first = i.path.segments.first();
                    self.0 |= i.path.leading_colon.is_none()
                        && i.path.segments.len() > 1
                        && first.map_or(false, |seg| seg.ident == "Self");
                }
            }
            syn::visit::visit_type_path(self, i)
        }
    }
    let mut projects = Projects(false);
    projects.visit_type(&pos.ty);
    for layer in &pos.layers {
        projects.visit_generics(layer);
    }
    projects.0
}

/// Whether every relative path of a position of a module is a generic, a primitive type or a name
/// of the prelude: no name that the module declares, nor one that it imports.
fn portable(pos: &Position, args: &Args) -> bool {
    let generics = generic_names(pos);
    let mut portable = Portable {
        generics: &generics,
        prelude: args.prelude,
        ok: true,
    };
    portable.visit_type(&pos.ty);
    for layer in &pos.layers {
        portable.visit_generics(layer);
    }
    portable.ok
}

/// The type positions of the copy that can be checked, with their index, and the header of an impl.
///
/// `item` is the original, `copy` the rewritten item and `leaker_path` the module of the Leaker's
/// crate that has the item. A position with an elided lifetime has no witness. In a copy of a
/// module, the names it declares mean its own items, which the witness (in the Leaker's module)
/// does not: such a position is compared with the original written for the Referrer's crate
/// ([`Reroot`]), and is left out only if that cannot be written.
fn comparable(
    copy: &Item,
    item: &Item,
    args: &Args,
    host: Option<&Host>,
    leaker_path: &Path,
) -> Vec<Check> {
    let in_module = matches!(copy, Item::Mod(_));
    let original = collect_positions(item, None, args);
    let copied = collect_positions(copy, host, args);
    assert_eq!(original.len(), copied.len(), "test macro: positions differ");
    let header = match header_of_copy(copy, item, args) {
        Some((copied, false)) => Some(Check {
            j: original.len(),
            pos: copied,
            expected: None,
            header: true,
            reverse: true,
        }),
        _ => None,
    };
    let reroot = if in_module {
        Reroot::new(item, leaker_path, args)
    } else {
        None
    };
    let mut out = Vec::new();
    for (j, (pos, orig)) in copied.into_iter().zip(original).enumerate() {
        if has_no_witness(&orig) {
            continue;
        }
        let expected = match &reroot {
            Some(reroot) if !portable(&pos, args) => match reroot.position(&orig, &pos) {
                Some(expected) => Some(expected),
                None => continue,
            },
            _ => None,
        };
        let impl_self = matches!(item, Item::Impl(_)) && projects_self(&orig);
        out.push(Check {
            j,
            pos,
            expected,
            header: false,
            reverse: !impl_self,
        });
    }
    out.extend(header);
    out
}

/// The paths of the original in a module, written for the Referrer's crate.
///
/// A relative path of a type of a module means an item there (the copy declares it too), an
/// import, a generic or a name that every crate has. The witness is written in the Leaker's module,
/// so it names the Leaker's items by the names the module declares; the copy names its own. The
/// expected type has the names that the module declares as they are, where the copy declares them
/// (the check is written in the copy's module), and every other path from the root of the Leaker's
/// crate: `crate::base::B0` is `::tlp_l_000::base::B0`, `super::super::base::B0` too, an import
/// `I0` is the path it stands for. Nothing is looked up in the copy, so the Visitor's idea of what
/// the module declares is checked, not assumed.
struct Reroot<'a> {
    /// The segments of the path of the module that has the analyzed one (the crate, then modules).
    outer: Vec<Ident>,
    /// The analyzed module.
    module: &'a ItemMod,
    prelude: bool,
}

/// The state of the rewriting of one position.
struct Rerooting<'a, 'b> {
    reroot: &'a Reroot<'b>,
    /// The items of each module from the analyzed one down to the module of the position.
    levels: Vec<&'b [Item]>,
    generics: HashSet<String>,
    /// The modules from the analyzed one to the module of the position.
    mods: Vec<Ident>,
    /// The names that the module declares are the Leaker's items (inside an entry), not the copy's.
    leaker: bool,
    /// The last path was resolved to a name that the module declares.
    declared: bool,
    /// An import followed: the depth, against cycles.
    depth: usize,
    failed: bool,
}

/// The names declared by the items (not by `use`).
fn declares(items: &[Item], name: &str) -> bool {
    items.iter().any(|item| match item {
        Item::Struct(ItemStruct { ident, .. })
        | Item::Enum(ItemEnum { ident, .. })
        | Item::Union(ItemUnion { ident, .. })
        | Item::Trait(ItemTrait { ident, .. })
        | Item::TraitAlias(ItemTraitAlias { ident, .. })
        | Item::Type(ItemType { ident, .. })
        | Item::Const(ItemConst { ident, .. })
        | Item::Static(ItemStatic { ident, .. })
        | Item::Mod(ItemMod { ident, .. })
        | Item::Fn(ItemFn {
            sig: Signature { ident, .. },
            ..
        })
        | Item::Macro(ItemMacro {
            ident: Some(ident), ..
        }) => ident == name,
        _ => false,
    })
}

/// The path that `use` items bring `name` in as, if they do.
fn imported(items: &[Item], name: &str) -> Option<Path> {
    fn find(
        tree: &UseTree,
        prefix: &mut Vec<PathSegment>,
        colon: bool,
        name: &str,
    ) -> Option<Path> {
        let path = |prefix: &[PathSegment]| Path {
            leading_colon: colon.then(Default::default),
            segments: prefix.iter().cloned().collect(),
        };
        match tree {
            UseTree::Path(UsePath { ident, tree, .. }) => {
                prefix.push(ident.clone().into());
                let found = find(tree, prefix, colon, name);
                prefix.pop();
                found
            }
            UseTree::Name(UseName { ident }) => {
                if ident == "self" {
                    let last = prefix.last()?;
                    return (last.ident == name).then(|| path(prefix));
                }
                (ident == name).then(|| {
                    let mut full = prefix.clone();
                    full.push(ident.clone().into());
                    path(&full)
                })
            }
            UseTree::Rename(UseRename { ident, rename, .. }) => (rename == name).then(|| {
                let mut full = prefix.clone();
                if ident != "self" {
                    full.push(ident.clone().into());
                }
                path(&full)
            }),
            UseTree::Group(UseGroup { items, .. }) => items
                .iter()
                .find_map(|tree| find(tree, prefix, colon, name)),
            UseTree::Glob(_) => None,
        }
    }
    items.iter().find_map(|item| match item {
        Item::Use(u) => find(&u.tree, &mut Vec::new(), u.leading_colon.is_some(), name),
        _ => None,
    })
}

impl<'a> Reroot<'a> {
    fn new(item: &'a Item, leaker_path: &Path, args: &Args) -> Option<Self> {
        match item {
            Item::Mod(module) => Some(Reroot {
                outer: leaker_path.segments.iter().map(|s| s.ident.clone()).collect(),
                module,
                // The copy of an item accepted by `allow()` keeps the names that conflict.
                prelude: args.prelude || args.accepted,
            }),
            _ => None,
        }
    }

    /// The original position written for the Referrer's crate, walking the copy's position with it
    /// ([`Rerooting::ty()`]): its type and generics, or `None` if a path in them cannot be resolved,
    /// or the copy has another shape.
    fn position(&self, orig: &Position, copy: &Position) -> Option<Position> {
        let mut levels: Vec<&'a [Item]> = vec![self.module.content.as_ref()?.1.as_slice()];
        for name in orig.mods.iter().skip(1) {
            let items = *levels.last()?;
            let nested = items.iter().find_map(|item| match item {
                Item::Mod(m) if m.ident == *name => {
                    m.content.as_ref().map(|(_, items)| items.as_slice())
                }
                _ => None,
            })?;
            levels.push(nested);
        }
        let mut state = Rerooting {
            reroot: self,
            levels,
            generics: generic_names(orig),
            mods: orig.mods.clone(),
            leaker: false,
            declared: false,
            depth: 0,
            failed: false,
        };
        if orig.layers.len() != copy.layers.len() {
            return None;
        }
        let ty = state.ty(&orig.ty, &copy.ty)?;
        let layers = orig
            .layers
            .iter()
            .zip(&copy.layers)
            .map(|(orig, copy)| state.generics(orig, copy))
            .collect::<Option<Vec<_>>>()?;
        (!state.failed).then(|| Position {
            mods: orig.mods.clone(),
            layers,
            self_sub: None,
            ty,
        })
    }
}

impl Rerooting<'_, '_> {
    /// The path `rest` from the root of the Leaker's crate, below the first `keep` segments of the
    /// path of the module that has the analyzed one.
    fn absolute<'p>(
        &self,
        keep: usize,
        rest: impl Iterator<Item = &'p PathSegment>,
    ) -> Path {
        let mut path = Path {
            leading_colon: Some(Default::default()),
            segments: self.reroot.outer[..keep]
                .iter()
                .cloned()
                .map(PathSegment::from)
                .collect(),
        };
        path.segments.extend(rest.cloned());
        path
    }

    /// The path `p` written in the module `level` (0 is the analyzed module), for the module of the
    /// position.
    fn path(&mut self, level: usize, p: &Path) -> Option<Path> {
        if p.leading_colon.is_some() {
            return Some(p.clone());
        }
        let current = self.levels.len() - 1;
        match p.segments.first()?.ident.to_string().as_str() {
            "crate" => Some(self.absolute(1, p.segments.iter().skip(1))),
            "self" => self.named(level, current, p, 1),
            "super" => {
                let up = p.segments.iter().take_while(|s| s.ident == "super").count();
                if up <= level {
                    self.named(level - up, current, p, up)
                } else {
                    // Out of the analyzed module: `up - level - 1` modules above the one that has it.
                    let above = up - level - 1;
                    let kept = self.reroot.outer.len().checked_sub(above)?;
                    (kept >= 1).then(|| self.absolute(kept, p.segments.iter().skip(up)))
                }
            }
            _ => self.named(level, current, p, 0),
        }
    }

    /// The segment `skip` of `p` is a name in the module `level`: what it is, written for the module
    /// of the position `current`.
    fn named(&mut self, level: usize, current: usize, p: &Path, skip: usize) -> Option<Path> {
        let segment = p.segments.get(skip)?;
        let name = segment.ident.to_string();
        if skip == 0 && self.generics.contains(&name) {
            return Some(p.clone());
        }
        let items = self.levels[level];
        if declares(items, &name) {
            self.declared = true;
            if self.leaker {
                return Some(self.leaker_path(level, p.segments.iter().skip(skip).cloned()));
            }
            return Some(p.clone());
        }
        if let Some(target) = imported(items, &name) {
            if self.depth >= 8 {
                return None;
            }
            self.depth += 1;
            let resolved = self.path(level, &target);
            self.depth -= 1;
            let mut resolved = resolved?;
            // A path relative to another module means something else here.
            if resolved.leading_colon.is_none() && level != current {
                return None;
            }
            if let Some(last) = resolved.segments.last_mut() {
                last.arguments = segment.arguments.clone();
            }
            resolved
                .segments
                .extend(p.segments.iter().skip(skip + 1).cloned());
            return Some(resolved);
        }
        let everywhere = PRIMITIVES.contains(&name.as_str())
            || ["std", "core", "alloc"].contains(&name.as_str())
            // Inside an entry a name of the prelude means the prelude's, as the Leaker's crate has
            // it; elsewhere the copy keeps it only if the Leaker allows it.
            || ((self.reroot.prelude || self.leaker) && PRELUDE.contains(&name.as_str()));
        (skip == 0 && everywhere).then(|| p.clone())
    }

    /// Writes the path of a type, an expression or a trait bound for the Referrer's crate.
    fn rewrite(&mut self, qself: &mut Option<QSelf>, path: &mut Path) {
        match qself {
            None => match self.path(self.levels.len() - 1, path) {
                Some(rewritten) => *path = rewritten,
                None => self.failed = true,
            },
            Some(qself) if qself.position > 0 => {
                let trait_path = Path {
                    leading_colon: path.leading_colon,
                    segments: path.segments.iter().take(qself.position).cloned().collect(),
                };
                match self.path(self.levels.len() - 1, &trait_path) {
                    Some(rewritten) => {
                        qself.position = rewritten.segments.len();
                        let rest: Vec<PathSegment> =
                            path.segments.iter().skip(trait_path.segments.len()).cloned().collect();
                        *path = rewritten;
                        path.segments.extend(rest);
                    }
                    None => self.failed = true,
                }
            }
            Some(_) => (),
        }
    }
}

/// Whether the copy has an entry here: a projection through the Repeater.
fn is_entry(ty: &Type) -> bool {
    matches!(ty, Type::Path(TypePath { qself: Some(qself), path })
        if qself.position > 0
            && path.segments.len() > qself.position
            && path.segments[qself.position - 1].ident == "Repeater"
            && path.segments.last().map_or(false, |seg| seg.ident == "Type"))
}

/// The walk of the original and the copy together, which writes what the copy must be equal to.
///
/// The names that the module declares are the copy's own items, except inside an entry: the
/// Repeater's impl of an entry is written in the Leaker's module, so the type it gives names the
/// Leaker's items (`Rel<R0>`, `crate::base::B1<R0>` with `R0` declared: the copy of the module
/// refers to the Leaker's `R0` there, as the docs of `Leaker::enter_mod()` say). The copy says
/// where its entries are; a bare name that the module declares is none, so it is the copy's.
impl Rerooting<'_, '_> {
    /// The path of the module `level` of the position in the Leaker's crate, then `rest`.
    fn leaker_path(&self, level: usize, rest: impl Iterator<Item = PathSegment>) -> Path {
        let mut path = self.absolute(self.reroot.outer.len(), std::iter::empty::<&PathSegment>());
        path.segments
            .extend(self.mods[..=level].iter().cloned().map(PathSegment::from));
        path.segments.extend(rest);
        path
    }

    /// The original type written for the Referrer's crate, with the names that the module declares
    /// as the Leaker's items (an entry), or as the copy's.
    fn whole(&mut self, orig: &Type, leaker: bool) -> Option<Type> {
        let (saved, mut ty) = (self.leaker, orig.clone());
        self.leaker = leaker;
        self.visit_type_mut(&mut ty);
        self.leaker = saved;
        (!self.failed).then_some(ty)
    }

    /// Whether the head of the path is a name that the module declares (a generic is none).
    fn declared_head(&mut self, path: &Path) -> bool {
        let current = self.levels.len() - 1;
        let saved = std::mem::replace(&mut self.declared, false);
        let _ = self.path(current, path);
        std::mem::replace(&mut self.declared, saved)
    }

    /// Whether a constant in the type (a path in an expression, or a bare name written like a
    /// constant) is a path that is not the module's: an import, or from the crate.
    fn foreign_const(&mut self, ty: &Type) -> bool {
        struct Foreign<'r, 'a, 'b> {
            state: &'r mut Rerooting<'a, 'b>,
            found: bool,
        }
        impl Foreign<'_, '_, '_> {
            fn check(&mut self, path: &Path) {
                let current = self.state.levels.len() - 1;
                let saved = std::mem::replace(&mut self.state.declared, false);
                let resolved = self.state.path(current, path);
                self.state.declared = saved;
                self.found |= resolved.map_or(true, |resolved| resolved != *path);
            }
        }
        impl<'ast> Visit<'ast> for Foreign<'_, '_, '_> {
            fn visit_expr_path(&mut self, i: &'ast ExprPath) {
                if i.qself.is_none() {
                    self.check(&i.path);
                }
            }
            fn visit_generic_argument(&mut self, i: &'ast GenericArgument) {
                if let GenericArgument::Type(Type::Path(TypePath { qself: None, path })) = i {
                    if path
                        .get_ident()
                        .map_or(false, |ident| written_as_const(&ident.to_string()))
                    {
                        return self.check(path);
                    }
                }
                syn::visit::visit_generic_argument(self, i)
            }
        }
        let mut foreign = Foreign {
            state: self,
            found: false,
        };
        foreign.visit_type(ty);
        foreign.found
    }

    fn ty(&mut self, orig: &Type, copy: &Type) -> Option<Type> {
        if is_entry(copy) {
            // A type that the module declares is not an entry, unless a constant in it is a path
            // that is not the module's (`RC<bool, ILEN>`): that cannot be leaked alone, so the
            // whole type is, and its `RC` is the Leaker's.
            let declared = match orig {
                Type::Path(TypePath { qself: None, path }) => {
                    self.declared_head(path) && !self.foreign_const(orig)
                }
                _ => false,
            };
            return self.whole(orig, !declared);
        }
        Some(match (orig, copy) {
            (Type::Path(o), Type::Path(c)) if o.qself.is_none() && c.qself.is_none() => {
                Type::Path(TypePath {
                    qself: None,
                    path: self.path_pair(&o.path, &c.path)?,
                })
            }
            (Type::Path(o), Type::Path(c)) if o.qself.is_some() && c.qself.is_some() => {
                let (oq, cq) = (o.qself.as_ref()?, c.qself.as_ref()?);
                if oq.position != cq.position {
                    return None;
                }
                let mut qself = oq.clone();
                qself.ty = Box::new(self.ty(&oq.ty, &cq.ty)?);
                let path = self.path_pair(&o.path, &c.path)?;
                // The trait path may be longer now: where the associated item starts.
                let tail = o.path.segments.len() - oq.position;
                qself.position = path.segments.len() - tail;
                Type::Path(TypePath {
                    qself: Some(qself),
                    path,
                })
            }
            (Type::Reference(o), Type::Reference(c)) => {
                let mut ty = o.clone();
                ty.elem = Box::new(self.ty(&o.elem, &c.elem)?);
                Type::Reference(ty)
            }
            (Type::Ptr(o), Type::Ptr(c)) => {
                let mut ty = o.clone();
                ty.elem = Box::new(self.ty(&o.elem, &c.elem)?);
                Type::Ptr(ty)
            }
            (Type::Slice(o), Type::Slice(c)) => {
                let mut ty = o.clone();
                ty.elem = Box::new(self.ty(&o.elem, &c.elem)?);
                Type::Slice(ty)
            }
            (Type::Array(o), Type::Array(c)) => {
                let mut ty = o.clone();
                ty.elem = Box::new(self.ty(&o.elem, &c.elem)?);
                self.visit_expr_mut(&mut ty.len);
                Type::Array(ty)
            }
            (Type::Paren(o), Type::Paren(c)) => {
                let mut ty = o.clone();
                ty.elem = Box::new(self.ty(&o.elem, &c.elem)?);
                Type::Paren(ty)
            }
            (Type::Group(o), Type::Group(c)) => {
                let mut ty = o.clone();
                ty.elem = Box::new(self.ty(&o.elem, &c.elem)?);
                Type::Group(ty)
            }
            (Type::Tuple(o), Type::Tuple(c)) if o.elems.len() == c.elems.len() => {
                let mut ty = o.clone();
                for (elem, copy) in ty.elems.iter_mut().zip(&c.elems) {
                    *elem = self.ty(elem, copy)?;
                }
                Type::Tuple(ty)
            }
            (Type::BareFn(o), Type::BareFn(c)) if o.inputs.len() == c.inputs.len() => {
                let mut ty = o.clone();
                for (input, copy) in ty.inputs.iter_mut().zip(&c.inputs) {
                    input.ty = self.ty(&input.ty, &copy.ty)?;
                }
                if let (ReturnType::Type(_, o), ReturnType::Type(_, c)) = (&o.output, &c.output) {
                    if let ReturnType::Type(_, out) = &mut ty.output {
                        **out = self.ty(o, c)?;
                    }
                }
                Type::BareFn(ty)
            }
            (Type::TraitObject(o), Type::TraitObject(c)) if o.bounds.len() == c.bounds.len() => {
                let mut ty = o.clone();
                self.bounds(&mut ty.bounds, &c.bounds)?;
                Type::TraitObject(ty)
            }
            (Type::ImplTrait(o), Type::ImplTrait(c)) if o.bounds.len() == c.bounds.len() => {
                let mut ty = o.clone();
                self.bounds(&mut ty.bounds, &c.bounds)?;
                Type::ImplTrait(ty)
            }
            (Type::Never(_), Type::Never(_)) | (Type::Infer(_), Type::Infer(_)) => orig.clone(),
            // A shape the copy changed: the original as it is, which the check then compares.
            _ => return self.whole(orig, false),
        })
    }

    /// The path of the original written for the Referrer's crate, with the arguments of each
    /// segment taken pairwise with the copy's.
    fn path_pair(&mut self, orig: &Path, copy: &Path) -> Option<Path> {
        if orig.segments.len() != copy.segments.len() {
            return None;
        }
        let mut path = orig.clone();
        for (segment, copy) in path.segments.iter_mut().zip(&copy.segments) {
            segment.arguments = self.arguments(&segment.arguments, &copy.arguments)?;
        }
        self.path(self.levels.len() - 1, &path)
    }

    fn arguments(&mut self, orig: &PathArguments, copy: &PathArguments) -> Option<PathArguments> {
        Some(match (orig, copy) {
            (PathArguments::None, PathArguments::None) => PathArguments::None,
            (PathArguments::AngleBracketed(o), PathArguments::AngleBracketed(c))
                if o.args.len() == c.args.len() =>
            {
                let mut args = o.clone();
                for (arg, copy) in args.args.iter_mut().zip(&c.args) {
                    match (&*arg, copy) {
                        (GenericArgument::Type(o), GenericArgument::Type(c)) => {
                            let ty = self.ty(o, c)?;
                            // A constant written bare (`LEN3`) that is a longer path now.
                            *arg = self.argument(o, ty);
                        }
                        (GenericArgument::AssocType(o), GenericArgument::AssocType(c)) => {
                            let mut assoc = o.clone();
                            assoc.ty = self.ty(&o.ty, &c.ty)?;
                            *arg = GenericArgument::AssocType(assoc);
                        }
                        (GenericArgument::Lifetime(_), _) => (),
                        (other, _) => {
                            let mut other = other.clone();
                            self.visit_generic_argument_mut(&mut other);
                            *arg = other;
                        }
                    }
                }
                PathArguments::AngleBracketed(args)
            }
            (PathArguments::Parenthesized(o), PathArguments::Parenthesized(c))
                if o.inputs.len() == c.inputs.len() =>
            {
                let mut args = o.clone();
                for (input, copy) in args.inputs.iter_mut().zip(&c.inputs) {
                    *input = self.ty(input, copy)?;
                }
                if let (ReturnType::Type(_, o), ReturnType::Type(_, c)) = (&o.output, &c.output) {
                    if let ReturnType::Type(_, out) = &mut args.output {
                        **out = self.ty(o, c)?;
                    }
                }
                PathArguments::Parenthesized(args)
            }
            _ => return None,
        })
    }

    /// The generic argument for the type `ty` that stands for `orig`: braces if a bare name became a
    /// path of several segments (see [`written_as_const()`]).
    fn argument(&self, orig: &Type, ty: Type) -> GenericArgument {
        if let (Type::Path(TypePath { qself: None, path: o }), Type::Path(TypePath { path, .. })) =
            (orig, &ty)
        {
            if o.leading_colon.is_none()
                && o.segments.len() == 1
                && o.segments[0].arguments.is_none()
                && path.segments.len() > 1
                && written_as_const(&o.segments[0].ident.to_string())
            {
                let path = path.clone();
                return GenericArgument::Const(parse_quote!({ #path }));
            }
        }
        GenericArgument::Type(ty)
    }

    fn bounds(
        &mut self,
        orig: &mut Punctuated<TypeParamBound, Token![+]>,
        copy: &Punctuated<TypeParamBound, Token![+]>,
    ) -> Option<()> {
        if orig.len() != copy.len() {
            return None;
        }
        for (bound, copy) in orig.iter_mut().zip(copy) {
            if let (TypeParamBound::Trait(o), TypeParamBound::Trait(c)) = (&mut *bound, copy) {
                // `?Sized` is the language's, whatever the module declares.
                if !matches!(o.modifier, TraitBoundModifier::Maybe(_)) {
                    o.path = self.path_pair(&o.path.clone(), &c.path)?;
                }
            }
        }
        Some(())
    }

    fn generics(&mut self, orig: &Generics, copy: &Generics) -> Option<Generics> {
        if orig.params.len() != copy.params.len() {
            return None;
        }
        let mut generics = orig.clone();
        for (param, copy) in generics.params.iter_mut().zip(&copy.params) {
            match (param, copy) {
                (GenericParam::Type(o), GenericParam::Type(c)) => {
                    o.default = None;
                    self.bounds(&mut o.bounds, &c.bounds)?;
                }
                (GenericParam::Const(o), GenericParam::Const(_)) => {
                    o.default = None;
                }
                (GenericParam::Lifetime(_), GenericParam::Lifetime(_)) => (),
                _ => return None,
            }
        }
        match (&mut generics.where_clause, copy.where_clause.as_ref()) {
            (None, None) => (),
            (Some(clause), Some(copy)) if clause.predicates.len() == copy.predicates.len() => {
                for (predicate, copy) in clause.predicates.iter_mut().zip(&copy.predicates) {
                    match (predicate, copy) {
                        (WherePredicate::Type(p), WherePredicate::Type(c)) => {
                            p.bounded_ty = self.ty(&p.bounded_ty, &c.bounded_ty)?;
                            self.bounds(&mut p.bounds, &c.bounds)?;
                        }
                        (WherePredicate::Lifetime(_), WherePredicate::Lifetime(_)) => (),
                        _ => return None,
                    }
                }
            }
            _ => return None,
        }
        Some(generics)
    }
}

/// Whether a name is written like a constant (`BLEN`, `LEN3`): upper case, with at least three
/// letters. The argument of a path is a type or a constant, and only a constant that is a single
/// identifier can be written without braces.
fn written_as_const(name: &str) -> bool {
    name.chars()
        .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
        && name.chars().filter(|c| c.is_ascii_uppercase()).count() >= 3
}

impl VisitMut for Rerooting<'_, '_> {
    fn visit_generic_argument_mut(&mut self, i: &mut GenericArgument) {
        // A bare name that is an import of a constant is a longer path now: it needs braces.
        let bare = match i {
            GenericArgument::Type(Type::Path(TypePath { qself: None, path }))
                if path.leading_colon.is_none()
                    && path.segments.len() == 1
                    && path.segments[0].arguments.is_none() =>
            {
                Some(path.segments[0].ident.to_string())
            }
            _ => None,
        };
        syn::visit_mut::visit_generic_argument_mut(self, i);
        if let (Some(name), GenericArgument::Type(Type::Path(TypePath { path, .. }))) =
            (bare, &*i)
        {
            if path.segments.len() > 1 && written_as_const(&name) {
                let path = path.clone();
                *i = GenericArgument::Const(parse_quote!({ #path }));
            }
        }
    }

    fn visit_type_path_mut(&mut self, i: &mut TypePath) {
        self.rewrite(&mut i.qself, &mut i.path);
        syn::visit_mut::visit_type_path_mut(self, i)
    }

    fn visit_expr_path_mut(&mut self, i: &mut ExprPath) {
        self.rewrite(&mut i.qself, &mut i.path);
        syn::visit_mut::visit_expr_path_mut(self, i)
    }

    fn visit_trait_bound_mut(&mut self, i: &mut TraitBound) {
        // `?Sized` is the language's, whatever the module declares.
        if !matches!(i.modifier, TraitBoundModifier::Maybe(_)) {
            self.rewrite(&mut None, &mut i.path);
        }
        syn::visit_mut::visit_trait_bound_mut(self, i)
    }

    fn visit_type_macro_mut(&mut self, _: &mut TypeMacro) {
        self.failed = true;
    }

    fn visit_expr_macro_mut(&mut self, _: &mut ExprMacro) {
        self.failed = true;
    }
}

pub(crate) fn check_fn(
    id: u64,
    j: usize,
    pos: &Position,
    target: &Path,
    host: Option<&Type>,
) -> TokenStream {
    let (generics, ty) = lowered(pos, Implementor::Fixed);
    let mut targs = type_args(&generics);
    if pos.self_sub.as_ref().map_or(false, |sub| sub.bound) {
        let host = host.expect("a trait has a host");
        targs.push(quote!(#host));
    }
    let name = ident(&format!("__tl_chk_{}_{}", id, j));
    let witness = witness_name(id, j);
    let (params, where_clause) = (&generics.params, &generics.where_clause);
    let mut path = target.clone();
    path.segments
        .extend(pos.mods.iter().cloned().map(PathSegment::from));
    // The type is the result: like the parameter of a method, it gives the implied bounds (`T: 'a`
    // for `&'a T`) that the original relies on. It is under a `*mut` as the witness has it
    // (`invariant_in()`), so the two types must be equal, not only one a subtype of the other.
    let ty = invariant_in(ty);
    quote! {
        #[allow(non_snake_case, dead_code)]
        fn #name<#params>() -> ::core::marker::PhantomData<#ty> #where_clause {
            #path::#witness::<#(#targs),*>()
        }
    }
}

/// The type and const arguments that a call passes on to a function with the generics (the
/// lifetimes are inferred).
fn type_args(generics: &Generics) -> Vec<TokenStream> {
    generics
        .params
        .iter()
        .filter_map(|param| match param {
            GenericParam::Type(t) => {
                let ident = &t.ident;
                Some(quote!(#ident))
            }
            GenericParam::Const(c) => {
                let ident = &c.ident;
                Some(quote!(#ident))
            }
            GenericParam::Lifetime(_) => None,
        })
        .collect()
}

/// The reverse of the checks, for the checks against the Leaker's witnesses.
///
/// A check (`check_fn()`) calls the witness, which has the original's generics, from a function
/// with the copy's: a copy with weaker bounds fails, one with stronger bounds does not. The other
/// way round needs a function with the original's generics that takes the copy's, and the
/// original's names are the Leaker's. So the witnesses are also the methods of a trait there
/// (`reverse_trait()`), and the Referrer's crate implements it for a type of its own with the
/// methods of the copy. rustc compares each method with the trait's: the bounds of the copy must
/// follow from the original's (E0276, outlives bounds included: a copy that turns `?Sized` into
/// `Sized`, or adds `'b: 'a` or a predicate, fails), the generics must be the same (E0049, E0195),
/// and the type it returns the same (`*mut`). For a trait the implementor is the host, as in the
/// checks: the copy is an impl for it.
fn reverse_impls(id: u64, target: &Path, checks: &[Check], host: Option<&Type>) -> TokenStream {
    let mut groups: Vec<(Vec<Ident>, Vec<TokenStream>)> = Vec::new();
    for check in checks
        .iter()
        .filter(|c| c.expected.is_none() && c.reverse)
    {
        let pos = &check.pos;
        let method = method(id, check.j, pos, Implementor::Fixed, TokenStream::new());
        match groups.iter_mut().find(|(mods, _)| *mods == pos.mods) {
            Some((_, methods)) => methods.push(method),
            None => groups.push((pos.mods.clone(), vec![method])),
        }
    }
    let mut out = TokenStream::new();
    for (k, (mods, methods)) in groups.into_iter().enumerate() {
        let mut trait_path = target.clone();
        trait_path
            .segments
            .extend(mods.into_iter().map(PathSegment::from));
        trait_path.segments.push(reverse_trait(id).into());
        let implementor = host.map(|host| quote!(<#host>));
        let probe = ident(&format!("__TlProbe_{}_{}", id, k));
        out.extend(quote! {
            #[allow(non_camel_case_types, dead_code)]
            struct #probe;
            impl #trait_path #implementor for #probe {
                #(#methods)*
            }
        });
    }
    out
}

/// `ty` with each lifetime of `generics` in it: `(&'a (), &'b (), ty)`. The functions have no
/// inputs, so a lifetime that the signature does not name would be free for a call to choose (a
/// bound on it would not be compared), and one that only a bound names is early-bound where one in
/// the type of the copy may not be (E0195 against the trait of the reverse check).
fn pinned(generics: &Generics, ty: Type) -> Type {
    let lifetimes: Vec<&Lifetime> = generics
        .params
        .iter()
        .filter_map(|param| match param {
            GenericParam::Lifetime(l) => Some(&l.lifetime),
            _ => None,
        })
        .collect();
    if lifetimes.is_empty() {
        return ty;
    }
    parse_quote!((#(&#lifetimes (),)* #ty))
}

/// The three functions that compare a position with the original written for the Referrer's crate
/// (`Check::expected`), in the copy's module, where the names that the module declares are the
/// copy's: `__tl_ex_<id>_<j>` is the original, `__tl_chk_<id>_<j>` has the copy's generics and type
/// and returns what the original returns (the original's bounds follow from the copy's), and
/// `__tl_rv_<id>_<j>` has the original's and returns what the copy returns (the copy's follow from
/// the original's). The types are under `*mut`, so they must be equal, and each lifetime is in the
/// type (`pinned()`).
fn reroot_fns(id: u64, check: &Check) -> TokenStream {
    let expected = check.expected.as_ref().unwrap();
    let (ex_generics, ex_ty) = lowered(expected, Implementor::Fixed);
    let (generics, ty) = lowered(&check.pos, Implementor::Fixed);
    let (ex_ty, ty) = (invariant_in(ex_ty), invariant_in(ty));
    let (ex, chk, rv) = (
        ident(&format!("__tl_ex_{}_{}", id, check.j)),
        ident(&format!("__tl_chk_{}_{}", id, check.j)),
        ident(&format!("__tl_rv_{}_{}", id, check.j)),
    );
    let targs = type_args(&generics);
    let (ex_params, ex_where) = (&ex_generics.params, &ex_generics.where_clause);
    let (params, where_clause) = (&generics.params, &generics.where_clause);
    quote! {
        #[allow(non_snake_case, dead_code)]
        fn #ex<#ex_params>() -> ::core::marker::PhantomData<#ex_ty> #ex_where {
            ::core::marker::PhantomData
        }
        #[allow(non_snake_case, dead_code)]
        fn #chk<#params>() -> ::core::marker::PhantomData<#ty> #where_clause {
            #ex::<#(#targs),*>()
        }
        #[allow(non_snake_case, dead_code)]
        fn #rv<#ex_params>() -> ::core::marker::PhantomData<#ex_ty> #ex_where {
            #chk::<#(#targs),*>()
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Entry points.
// ---------------------------------------------------------------------------------------------

fn module_items<'a>(m: &'a mut ItemMod, path: &[Ident]) -> Option<&'a mut Vec<Item>> {
    let items = &mut m.content.as_mut()?.1;
    match path.split_first() {
        None => Some(items),
        Some((first, rest)) => {
            for item in items.iter_mut() {
                if let Item::Mod(sub) = item {
                    if sub.ident == *first {
                        return module_items(sub, rest);
                    }
                }
            }
            None
        }
    }
}

fn dollar() -> Punct {
    Punct::new('$', Spacing::Alone)
}

/// The errors of a rejection (`__tl_errors_<id>!`, after the report): each is its message, the
/// tokens it is about, its first note, its `Reason` and the kind of its cause, which `\u{1f}`
/// separates, and `\u{1d}` separates the errors.
fn rejection_text(errors: &[ErrorItem]) -> String {
    let clean = |text: &str| -> String { text.replace(['\n', '|', '\u{1f}', '\u{1d}'], " ") };
    errors
        .iter()
        .map(|item| {
            let cause = format!("{:?}", item.cause);
            format!(
                "{}\u{1f}{}\u{1f}{}\u{1f}{:?}\u{1f}{}",
                clean(&item.message),
                clean(&item.cause.to_token_stream().to_string()),
                clean(item.notes.first().map(|(_, n)| n.as_str()).unwrap_or("")),
                item.reason,
                cause.split('(').next().unwrap_or("")
            )
        })
        .collect::<Vec<_>>()
        .join("\u{1d}")
}

/// [`crate::allow_every_cause()`], which must drop every error (they are a flat list). If one
/// remains the macro panics, so the Leaker's crate does not build and the run fails (`ALLOW LEFT
/// ERRORS`, which `tests/prop_cross_crate` reads).
fn allow_all(result: LeakerResult, what: &str) -> LeakerResult {
    let allowed = crate::allow_every_cause(result);
    if !allowed.errors().is_empty() {
        panic!(
            "ALLOW LEFT ERRORS: {}: allow() of the cause of every error left {}: {}",
            what,
            allowed.errors().len(),
            crate::messages(allowed.errors()).replace('\n', " ; ")
        );
    }
    allowed
}

/// What [`leak()`] does with an item that the Leaker rejects.
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum Mode {
    /// `#[prop_leak]`: the item is not leaked (`rej`).
    Plain,
    /// `#[prop_leak_accept]`: accepted if `allow()` of every cause drops every error.
    Accept,
    /// `#[prop_leak_result]`: accepted with its errors, which the Referrer's macro allows or reports.
    Result,
}

/// `#[prop_leak(..)]`, and, with `accept`, `#[prop_leak_accept(..)]` (`TYPE_LEAK_PROP_ACCEPT`).
///
/// With `accept`, an item that the Leaker rejects is emitted with the recovered Referrer
/// (`LeakerResult::into_referrer()`) as an accepted item (the Repeater's impls, the witnesses, the
/// macro of the Referrer, which copies the part that conflicts as written), with the status `acc`.
/// `LeakerResult::allow(cause)` for the cause of every error must drop every error
/// ([`allow_all()`]: the macro panics otherwise). The errors are what the report says (and what
/// `family()` of the harness explains).
///
/// With `Mode::Result` the same item is emitted whatever the errors, and the macro of the Referrer
/// carries the `LeakerResult` with them in place of the Referrer.
pub(crate) fn leak(attr: TokenStream, item: TokenStream, mode: Mode) -> Result<TokenStream> {
    let accept = mode != Mode::Plain;
    let mut args: Args = parse2(attr)?;
    let original: Item = parse2(item)?;
    let id = args.id;
    let kind = kind_name(&original);
    let report_name = ident(&format!("__tl_report_{}", id));
    let errors_name = ident(&format!("__tl_errors_{}", id));
    let mac_name = ident(&format!("__tl_prop_{}", id));
    let dollar = dollar();
    let (result, context, assertions) = analyze(&original, &args);
    // The results of `View::check()` (`via = "check"`), which the copy must agree with.
    let checked = CHECKED.with(|checked| checked.take());
    // Another call of the same API gives the same analysis as the plain scope. (Not the scope of
    // a type opened in the one of a module: its contexts and its `Self` are other ones.)
    if !args.via.is_empty() && args.via != "nested" {
        let mut plain_args = args.clone();
        plain_args.via = if args.via == "check" {
            "unchecked".to_string()
        } else {
            String::new()
        };
        let (plain, plain_context, plain_assertions) = analyze(&original, &plain_args);
        same_analysis(
            (&result, context, &assertions),
            (&plain, plain_context, &plain_assertions),
            &args,
        )?;
    }
    // `allow_stages`: for each rule what the two stages leave, and the copies after them (nothing is
    // emitted otherwise).
    let mut stages_macro = TokenStream::new();
    let mut stage_copies: Vec<StageCopy> = Vec::new();
    if args.stages {
        for rule in Rule::ALL {
            let (copies, stages) = rule_stage(&original, &args, rule);
            stage_copies.extend(copies);
            stages_macro.extend(stages);
        }
    }
    // The bare spelling of a trait of the prelude: its twin in full gives the same entries if the
    // names of the prelude are allowed, and it is a relative path if they are not.
    let first = result
        .clone()
        .finish()
        .map(|referrer| (referrer, context, assertions.clone()));
    let twin = crate::bare::check(
        &original,
        args.scope == "impl",
        args.prelude,
        &first,
        |twin| analysis(twin, &args),
    )?;
    // The errors of the analysis (none if it succeeded).
    let errors: Vec<ErrorItem> = result.errors().to_vec();
    // `__tl_carry_<id>!` hands the result, with its errors, to a macro of another crate
    // ([`carry()`]), with the tokens of the errors as the Leaker has them.
    let carry_macro = {
        let carry_name = ident(&format!("__tl_carry_{}", id));
        let carried = result.to_token_stream();
        let expected = errors
            .iter()
            .map(|e| e.to_token_stream().to_string())
            .collect::<Vec<_>>()
            .join("\u{1d}");
        quote! {
            #[doc(hidden)]
            #[macro_export]
            macro_rules! #carry_name {
                (#dollar(#dollar t:tt)*) => {
                    #dollar crate::__macro::__prop_carry! {
                        { #carried } { #expected } #dollar(#dollar t)*
                    }
                };
            }
        }
    };
    if !errors.is_empty() {
        // With `accept`, the cause of every error is allowed, which must drop every error: the
        // Referrer is the recovered one either way.
        if accept {
            allow_all(result.clone(), &format!("item {}", id));
        }
        if mode == Mode::Plain {
            // The item is not leaked: the Leaker's crate still has it.
            let reason = crate::messages(&errors).replace(['\n', '|'], " ");
            let report = format!("{}|{}|rej|{}", id, kind, reason);
            let errors = rejection_text(&errors);
            // The copy of `allow_stages` is made all the same, and compares with the witnesses,
            // which the item has no other output for.
            let (head, tail, stage_macros) = if stage_copies.is_empty() {
                (quote!(#original), TokenStream::new(), TokenStream::new())
            } else {
                let (mut impls, mut assertions, mut macros) =
                    (Vec::new(), TokenStream::new(), TokenStream::new());
                for stage in stage_copies {
                    impls.extend(stage.impls);
                    assertions.extend(stage.assertions);
                    macros.extend(stage.macros);
                }
                let (head, tail) = place(
                    &original,
                    impls,
                    &assertions,
                    witnesses_of(&original, &args),
                )?;
                (head, tail, macros)
            };
            return Ok(quote! {
                #head
                #tail
                #stage_macros
                #stages_macro
                #carry_macro
                #[doc(hidden)]
                #[macro_export]
                macro_rules! #report_name { () => { #report }; }
                #[doc(hidden)]
                #[macro_export]
                macro_rules! #errors_name { () => { #errors }; }
                #[doc(hidden)]
                #[macro_export]
                macro_rules! #mac_name { (#dollar(#dollar t:tt)*) => {}; }
            });
        }
        args.accepted = true;
    }
    // What the macro of the Referrer carries: the result with its errors, or the Referrer.
    let carried = (mode == Mode::Result).then(|| result.to_token_stream());
    // The recovered Referrer: the parts that conflict are copied as written.
    let referrer = result.clone().into_referrer();
    let carried = carried.unwrap_or_else(|| referrer.to_token_stream());

    let status = if args.accepted { "acc" } else { "ok" };
    let errors = rejection_text(&errors);
    // Count what the Referrer will check, by rewriting a copy here too.
    let counted = count_copy(&result, context, &original, &args, status, twin, &checked)?;
    let mut impls = counted.impls;
    let report = counted.report;
    let witnesses = witnesses_of(&original, &args);
    let mut assertions = assertions;
    let mut stage_macros = TokenStream::new();
    for stage in stage_copies {
        impls.extend(stage.impls);
        assertions.extend(stage.assertions);
        stage_macros.extend(stage.macros);
    }

    let (head, tail) = place(&original, impls, &assertions, witnesses)?;
    let args_tokens = args.tokens();
    Ok(quote! {
        #head
        #tail
        #stage_macros
        #stages_macro
        #carry_macro
        #[doc(hidden)]
        #[macro_export]
        macro_rules! #report_name { () => { #report }; }
        #[doc(hidden)]
        #[macro_export]
        macro_rules! #errors_name { () => { #errors }; }
        #[doc(hidden)]
        #[macro_export]
        macro_rules! #mac_name {
            (#dollar(#dollar t:tt)*) => {
                #dollar crate::__macro::__prop_apply! {
                    { #carried , #context }
                    { #args_tokens }
                    { #original }
                    #dollar(#dollar t)*
                }
            };
        }
    })
}

/// The analysis through `via` and the one of the plain scope are the same: the context, the
/// Referrer, the errors (their tokens, spans aside) and the assertions. `View::check()` leaves
/// nothing in the Leaker, and the scopes given by parts or by other calls give what the walk gives.
fn same_analysis(
    got: (&LeakerResult, ContextId, &TokenStream),
    plain: (&LeakerResult, ContextId, &TokenStream),
    args: &Args,
) -> Result<()> {
    let errors = |result: &LeakerResult| -> Vec<String> {
        result
            .errors()
            .iter()
            .map(|e| e.to_token_stream().to_string())
            .collect()
    };
    let what = if got.1 != plain.1 {
        Some("the context")
    } else if got.0.clone().into_referrer() != plain.0.clone().into_referrer() {
        Some("the Referrer")
    } else if errors(got.0) != errors(plain.0) {
        Some("the errors")
    } else if got.2.to_string() != plain.2.to_string() {
        Some("the assertions")
    } else {
        None
    };
    match what {
        None => Ok(()),
        Some(what) => Err(Error::new(
            Span::call_site(),
            format!(
                "VIA DIFFERS: the analysis through `via = \"{}\"` of the scope `{}` gives another {} than the plain scope",
                args.via, args.scope, what
            ),
        )),
    }
}

/// The type with each projection through the Repeater (`<S as Repeater<ARGS, KEY>>::Type`)
/// replaced by the entry that its `KEY` names (the index is in the low half).
fn unreplaced(referrer: &Referrer, ty: &Type) -> Type {
    struct Back<'a>(&'a Referrer);
    impl VisitMut for Back<'_> {
        fn visit_type_mut(&mut self, ty: &mut Type) {
            if let Type::Path(TypePath {
                qself: Some(qself),
                path,
            }) = ty
            {
                let at = qself.position;
                if at > 0 && path.segments.len() == at + 1 && path.segments[at].ident == "Type" {
                    if let PathArguments::AngleBracketed(arguments) = &path.segments[at - 1].arguments
                    {
                        if let Some(GenericArgument::Const(Expr::Lit(ExprLit {
                            lit: Lit::Int(key),
                            ..
                        }))) = arguments.args.last()
                        {
                            let key = key.base10_parse::<u128>().unwrap_or(0);
                            let ix = (key & u128::from(u64::MAX)) as usize;
                            if let Some((_, entry, _)) = self.0.iter().nth(ix) {
                                *ty = entry.clone();
                                return;
                            }
                        }
                    }
                }
            }
            syn::visit_mut::visit_type_mut(self, ty);
        }
    }
    let mut back = ty.clone();
    Back(referrer).visit_type_mut(&mut back);
    back
}

/// What `View::check()` said of the positions it checked agrees with the copy of the item: a type
/// with no cause is as written in the copy, but for the entries in it (a type of another crate)
/// that the copy replaces; a type leaked whole is a projection; one with causes in its parts is
/// another type. A type with a cause that cannot be leaked, an elided lifetime and `Self` (the
/// copy replaces it) are not compared.
fn check_against_copy(
    referrer: &Referrer,
    checked: &[Checked],
    original: &Item,
    copy: &Item,
    args: &Args,
    host: Option<&Host>,
) -> Result<()> {
    if checked.is_empty() {
        return Ok(());
    }
    let orig = collect_positions(original, None, args);
    let copied = collect_positions(copy, host, args);
    let text = |ty: &Type| quote!(#ty).to_string();
    for c in checked {
        let (o, k) = (&orig[c.j], &copied[c.j]);
        let oty = &o.ty;
        if c.impossible > 0
            || has_elided(&o.ty)
            || tokens_name_self(quote!(#oty))
        {
            continue;
        }
        let same = text(&o.ty) == text(&k.ty);
        let whole = matches!(&k.ty, Type::Path(TypePath { qself: Some(_), .. }));
        let says = if c.must == 0 {
            (text(&o.ty) != text(&unreplaced(referrer, &k.ty)))
                .then_some("no cause, and the copy changed it")
        } else if c.is_own {
            (!whole).then_some("a type leaked whole, and the copy is no projection")
        } else {
            same.then_some("causes in its parts, and the copy is as written")
        };
        if let Some(says) = says {
            return Err(Error::new(
                Span::call_site(),
                format!(
                    "CHECK DISAGREES: `View::check()` of `{}` found {}: {}; the copy has `{}`",
                    text(&o.ty),
                    c.must,
                    says,
                    text(&k.ty)
                ),
            ));
        }
    }
    Ok(())
}

/// The witnesses of the type positions of `original` and of the header of an impl, each with the path
/// of the module it is written in, and the reverse trait of each module (`reverse_trait()`).
fn witnesses_of(original: &Item, args: &Args) -> Vec<(Vec<Ident>, TokenStream)> {
    let mut positions = collect_positions(original, None, args);
    let header = header_position(original, args).filter(|pos| !has_no_witness(pos));
    let n = positions.len();
    positions.extend(header.clone());
    let witnessed: Vec<(usize, &Position)> = positions
        .iter()
        .enumerate()
        .filter(|(j, pos)| *j >= n || !has_no_witness(pos))
        .collect();
    let mut out: Vec<(Vec<Ident>, TokenStream)> = witnessed
        .iter()
        .map(|(j, pos)| (pos.mods.clone(), witness_fn(args.id, *j, pos)))
        .collect();
    let mut modules: Vec<&Vec<Ident>> = Vec::new();
    for (_, pos) in &witnessed {
        if !modules.contains(&&pos.mods) {
            modules.push(&pos.mods);
        }
    }
    let name = reverse_trait(args.id);
    let implementor = matches!(original, Item::Trait(_)).then(|| quote!(<__Self: ?Sized>));
    for mods in modules {
        let methods = witnessed
            .iter()
            .filter(|(_, pos)| pos.mods == *mods)
            .map(|(j, pos)| method(args.id, *j, pos, Implementor::Where, TokenStream::new()));
        out.push((
            mods.clone(),
            quote!(pub trait #name #implementor { #(#methods)* }),
        ));
    }
    out
}

/// The item and what is written next to it, as the tokens before the item and the tokens after it:
/// the Repeater's impls (in the module each is keyed by, for a module), the assertions and the
/// witnesses (in the module they are of).
fn place(
    original: &Item,
    impls: Impls,
    assertions: &TokenStream,
    witnesses: Vec<(Vec<Ident>, TokenStream)>,
) -> Result<(TokenStream, TokenStream)> {
    let mut emitted = original.clone();
    Ok(match &mut emitted {
        Item::Mod(m) => {
            // The Repeater's implementations and the witnesses go into the modules they belong
            // to. The module's own name is the first of the path of a position.
            let mut extra: Vec<(Vec<Ident>, TokenStream)> = impls;
            extra.push((Vec::new(), assertions.clone()));
            for (mods, tokens) in extra.into_iter().chain(
                witnesses
                    .into_iter()
                    .map(|(mods, t)| (mods[1..].to_vec(), t)),
            ) {
                let span = m.ident.span();
                let items = module_items(m, &mods)
                    .ok_or_else(|| Error::new(span, "test macro: no such module"))?;
                items.push(Item::Verbatim(tokens));
            }
            (quote!(#m), TokenStream::new())
        }
        other => {
            let impls = impls.into_iter().map(|(_, t)| t);
            let witnesses = witnesses.into_iter().map(|(_, t)| t);
            (
                quote!(#other),
                quote!(#assertions #(#impls)* #(#witnesses)*),
            )
        }
    })
}

/// What the macro writes for the copy that a Referrer makes of `original` with `result`: the
/// Repeater's impls of its entries, and the report of what the macro counted.
struct Counted {
    impls: Impls,
    report: String,
}

/// Counts what the Referrer will check of the copy of `original` by rewriting a copy here too: the
/// entries, the type positions, those that a witness compares (`status` is `ok`, or `acc` for an
/// item accepted by `allow()`). Fails if the item has a shape that the macro cannot write.
fn count_copy(
    result: &LeakerResult,
    context: ContextId,
    original: &Item,
    args: &Args,
    status: &str,
    twin: bool,
    checked: &[Checked],
) -> Result<Counted> {
    let referrer = &result.clone().into_referrer();
    let (id, kind) = (args.id, kind_name(original));
    let impls = repeater_impls(result, referrer, args.repeater_id(), args.marker.as_ref())?;
    let positions = collect_positions(original, None, args);
    let dummy: Path = parse_quote!(::__dummy);
    // The result is read back from its tokens, as the Referrer's macro gets it.
    let plain: LeakerResult = parse2(result.to_token_stream())?;
    let copy = rewrite(&plain.into_referrer(), context, original, args, &dummy, false);
    let report = match copy {
        Ok(rewritten) => {
            let host = host_arg(&rewritten, args, &dummy);
            // The copy of an item accepted by `allow()` keeps the part that conflicts as written, and
            // so a primitive in it that the Leaker does not allow.
            if !args.accepted {
                check_primitives_replaced(&rewritten.item, original, args, host.as_ref())?;
                check_against_copy(
                    referrer,
                    checked,
                    original,
                    &rewritten.item,
                    args,
                    host.as_ref(),
                )?;
            }
            let checks = comparable(&rewritten.item, original, args, host.as_ref(), &dummy);
            let headers = checks.iter().filter(|c| c.header).count();
            let left_out = usize::from(
                header_of_copy(&rewritten.item, original, args).is_some_and(|(_, cycles)| cycles),
            );
            let rerooted = checks.iter().filter(|c| c.expected.is_some()).count();
            format!(
                "{}|{}|{}|entries={} positions={} checked={} twin={} rerooted={} headers={} header_cycles={}",
                id,
                kind,
                status,
                referrer.iter().count(),
                positions.len(),
                checks.len() - headers,
                u8::from(twin),
                rerooted,
                headers,
                left_out
            )
        }
        Err(e) => format!(
            "{}|{}|{}|entries={} positions={} rewrite-error={}",
            id,
            kind,
            status,
            referrer.iter().count(),
            positions.len(),
            e.to_string().replace(['\n', '|'], " ")
        ),
    };
    Ok(Counted { impls, report })
}

/// What `TYPE_LEAK_PROP_FEATURES=allow_stages` adds to the output of an item for each [`Rule`] and
/// [`Side`]: the copy that a macro makes
///
/// - `Side::Result`: when it analyzes the item without the rule (the primitive types allowed, and the
///   names of the prelude too but for `Rule::Prelude`), gives the rule to the result, and allows the
///   cause of every error that remains, as `prop_leak_accept` does;
/// - `Side::Config`: when it analyzes the item with a config that has the rule, and allows the cause
///   of every error that remains.
struct StageCopy {
    /// The Repeater's impls of its entries, with an `ID` of their own ([`Args::repeater_id()`]).
    impls: Impls,
    /// The assertions of its analysis.
    assertions: TokenStream,
    /// `__tl_st_<rule>_<side>_report_<id>!`, the report in the form of `__tl_report_<id>!` and the
    /// errors that were allowed after it, and `__tl_st_<rule>_<side>_<id>!`, the macro that the
    /// Referrer's crates of the stage copies invoke (`gen::Case::stages`): it expands to nothing in
    /// the other crates, as the macro of an item accepted by `allow()` does. The names are those of
    /// [`Rule::names()`]. A crate of no Leaker's name invokes it with `clean` first, and gets the
    /// copy if the rule left no error and nothing otherwise.
    macros: TokenStream,
}

/// What the macro of an item makes for the rule `rule` of `allow_stages`: the two copies, and
/// `__tl_stages_<rule>_<id>!`, which tells what the stages leave. The text of the latter is
///
/// ```text
/// MATTERS \u{1f} PLAIN \u{1f} BEFORE \u{1f} AFTER \u{1f} BEFORE_CAUSES \u{1f} AFTER_CAUSES \u{1f} SAME_ERRORS \u{1f} SAME_REFERRER
/// ```
///
/// the number of errors without the rule, with the rule given to the config (`BEFORE`) and to the
/// result (`AFTER`), their causes, and whether the config and the result give the same errors
/// (their text, spans left out) and an equal Referrer.
///
/// `MATTERS` is whether the rule changes the analysis, which is whether the copies are made (the rule
/// `Prelude` is made for every item, and `Values` for the items that read values): it drops or trims
/// an error that the analysis without it found, or the config with the rule finds other errors than
/// the analysis without it, or another Referrer than the result with the rule.
fn rule_stage(original: &Item, args: &Args, rule: Rule) -> (Vec<StageCopy>, TokenStream) {
    let id = args.id;
    // The primitive types are allowed, so that the rule is the only path rule that differs: with
    // `u8` relative, `Into<u8>` is an error with a relative path that no rule leaves out. The names
    // of the prelude are allowed too, but for the rule that is about them.
    let mut plain_args = args.clone();
    plain_args.primitive = true;
    plain_args.prelude = rule != Rule::Prelude;
    // The relative `std`, `core` and `alloc` of the generated items are allowed in the attribute, so
    // that the other analyses accept them: here it is the rule.
    if rule == Rule::StdCrates {
        plain_args.allow.retain(|path| !is_std_crate(path));
    }
    let (plain, plain_context, plain_assertions) = analyze(original, &plain_args);
    let (before, before_context, before_assertions) =
        analyze_with(original, &plain_args, Some(rule));
    let plain_errors = plain.errors().to_vec();
    let after = rule.on_result(plain.clone());
    let text = |errors: &[ErrorItem]| -> Vec<String> {
        errors
            .iter()
            .map(|e| e.to_token_stream().to_string())
            .collect()
    };
    let causes = |result: &LeakerResult| -> String {
        let text = |error: &ErrorItem| error.cause.to_token_stream().to_string();
        let joined = result
            .errors()
            .iter()
            .map(text)
            .collect::<Vec<_>>()
            .join(" ; ");
        joined.replace(['\n', '|', '\u{1f}', '\u{1d}', '\u{1e}'], " ")
    };
    let applies = rule != Rule::Values || args.values;
    let same_errors = text(before.errors()) == text(after.errors());
    let same_referrer = before.clone().into_referrer() == after.clone().into_referrer();
    let matters = applies
        && (rule == Rule::Prelude
            || text(plain.errors()) != text(after.errors())
            || text(plain.errors()) != text(before.errors())
            || !same_referrer);
    let stages_text = format!(
        "{}\u{1f}{}\u{1f}{}\u{1f}{}\u{1f}{}\u{1f}{}\u{1f}{}\u{1f}{}",
        u8::from(matters),
        plain.errors().len(),
        before.errors().len(),
        after.errors().len(),
        causes(&before),
        causes(&after),
        u8::from(same_errors),
        u8::from(same_referrer)
    );
    let name = rule.stages_name(id);
    let stages = quote! {
        #[doc(hidden)]
        #[macro_export]
        macro_rules! #name { () => { #stages_text }; }
    };
    let copies = vec![
        stage_copy(
            original,
            args,
            rule,
            Side::Result,
            &plain_errors,
            matters.then_some((after, plain_context, plain_assertions)),
        ),
        stage_copy(
            original,
            args,
            rule,
            Side::Config,
            &plain_errors,
            matters.then_some((before, before_context, before_assertions)),
        ),
    ];
    (copies, stages)
}

/// The copy of `allow_stages` of `original` on a side after the rule, from the analysis made for the
/// side (none: the rule does not matter for the item, and no copy is made): see [`StageCopy`].
///
/// The copy is made in the crate of its own, where the names of the prelude mean the prelude: that is
/// what allowing them is a promise of, and the generated crates declare no name of the prelude
/// (`tests/prop_cross_crate` checks it). The copy keeps the parts that conflict as written, the ones
/// that the rule drops and those that the cause allowed, so the Referrer's crate must resolve them:
/// it is the crate of the items accepted by `allow()`, or the one that makes only the names that the
/// rule allows resolve (`clean`).
///
/// The status of its report is `ok` if the analysis leaves no error, `acc` if the causes of the
/// errors were allowed (and dropped them all: [`allow_all()`]) and `skip` if the rule does not
/// matter. A macro error of the copy is no error of the Leaker's crate: it is a `rewrite-error` of
/// the report.
fn stage_copy(
    original: &Item,
    args: &Args,
    rule: Rule,
    side: Side,
    plain: &[ErrorItem],
    analysis: Option<(LeakerResult, ContextId, TokenStream)>,
) -> StageCopy {
    let (id, kind) = (args.id, kind_name(original));
    let (mac_name, report_name) = rule.names(side, id);
    let dollar = dollar();
    let noop = quote! {
        #[doc(hidden)]
        #[macro_export]
        macro_rules! #mac_name { (#dollar(#dollar t:tt)*) => {}; }
    };
    // After the errors of the report, the errors of the analysis without the rule, which tell what the
    // rule dropped (`\u{1b}`): a copy that does not compile is explained by them.
    let plain_text = rejection_text(plain);
    let report_macro = |report: String, errors: String| {
        let text = format!("{}\u{1e}{}\u{1b}{}", report, errors, plain_text);
        quote! {
            #[doc(hidden)]
            #[macro_export]
            macro_rules! #report_name { () => { #text }; }
        }
    };
    let nothing = |report: String, errors: String| StageCopy {
        impls: Vec::new(),
        assertions: TokenStream::new(),
        macros: {
            let report = report_macro(report, errors);
            quote!(#report #noop)
        },
    };
    let (result, context, assertions) = match analysis {
        Some(analysis) => analysis,
        None => return nothing(format!("{}|{}|skip|", id, kind), String::new()),
    };
    // The errors that the rule does not drop are allowed by their causes, and so are those that
    // come up from inside them.
    let errors: Vec<ErrorItem> = result.errors().to_vec();
    let result = allow_all(
        result,
        &format!("item {}, the copy after the rule `{}`", id, rule.name()),
    );
    let status = if errors.is_empty() { "ok" } else { "acc" };
    // The Referrer's crate of the stage copies: the arguments of the macro say so.
    let mut copy_args = args.clone();
    copy_args.primitive = true;
    copy_args.prelude = true;
    copy_args.stages = false;
    copy_args.stage = rule.number(side);
    copy_args.accepted = true;
    let referrer = result.clone().into_referrer();
    let text = rejection_text(&errors);
    let counted = match count_copy(&result, context, original, &copy_args, status, false, &[]) {
        Ok(counted) => counted,
        Err(e) => {
            let report = format!(
                "{}|{}|{}|entries={} positions=0 rewrite-error={}",
                id,
                kind,
                status,
                referrer.iter().count(),
                e.to_string().replace(['\n', '|'], " ")
            );
            return nothing(report, text);
        }
    };
    let (carried, args_tokens) = (referrer.to_token_stream(), copy_args.tokens());
    let report = report_macro(counted.report, text);
    let expansion = |more: TokenStream| {
        quote! {
            #dollar crate::__macro::__prop_apply! {
                { #carried , #context }
                { #args_tokens }
                { #original }
                #dollar(#dollar t)* #more
            }
        }
    };
    // `clean` first in the invocation: the copy is made only if no error remains after the rule,
    // and the module copies drop their imports (`apply()`).
    let clean = if status == "ok" && matches!(rule, Rule::Values | Rule::Default) {
        if side == Side::Config {
            expansion(quote!(, clean, constants, imports))
        } else {
            expansion(quote!(, clean, constants))
        }
    } else if status == "ok" {
        expansion(quote!(, clean))
    } else {
        TokenStream::new()
    };
    let ordinary = expansion(TokenStream::new());
    StageCopy {
        impls: counted.impls,
        assertions,
        macros: quote! {
            #report
            #[doc(hidden)]
            #[macro_export]
            macro_rules! #mac_name {
                (clean #dollar(#dollar t:tt)*) => { #clean };
                (#dollar(#dollar t:tt)*) => { #ordinary };
            }
        },
    }
}

/// What the Referrer's invocation allows of the errors it got (`allow(std_crates, trait Path, type
/// Ty)`).
enum AllowArg {
    /// `Rules::std_crates()`
    StdCrates,
    /// `LeakerResult::allow(cause)`: `trait Path` is a `Cause::Trait`, `type Ty` a `Cause::Type`.
    Cause(Cause),
    /// `all`: the cause of every error ([`crate::allow_every_cause()`]), as `prop_leak_accept` does.
    AllCauses,
}

impl Parse for AllowArg {
    fn parse(input: ParseStream) -> Result<Self> {
        if input.parse::<Option<Token![trait]>>()?.is_some() {
            return Ok(AllowArg::Cause(Cause::Trait(input.parse()?)));
        }
        if input.parse::<Option<Token![type]>>()?.is_some() {
            return Ok(AllowArg::Cause(Cause::Type(input.parse()?)));
        }
        let word: Ident = input.parse()?;
        match word.to_string().as_str() {
            "std_crates" => Ok(AllowArg::StdCrates),
            "all" => Ok(AllowArg::AllCauses),
            _ => Err(Error::new(
                word.span(),
                "expected `std_crates`, `trait <path>` or `type <type>`",
            )),
        }
    }
}

/// Input of `__prop_apply!`.
struct ApplyInput {
    /// A Referrer without errors reads as a result too.
    result: LeakerResult,
    context: ContextId,
    args: Args,
    item: Item,
    /// The module that contains the item, in the Leaker's crate.
    path: Path,
    /// The host of a trait is `dyn Host`.
    dyn_host: bool,
    /// The invocation is in the Referrer's crate of the items accepted by `allow()` (`accept`),
    /// where the names that such an item copies as written resolve as in the Leaker's crate: the
    /// copy is made there and not in the other crate, see [`apply()`].
    accept: bool,
    /// What the invocation allows of the errors of the result.
    allow: Vec<AllowArg>,
    /// The crate of the invocation has none of the Leaker's names (`allow_stages`): a module copy
    /// drops its imports as the copy of an ordinary item does, and the names that the rule allows
    /// are the only ones to resolve.
    clean: bool,
    /// With `clean`: the rule allows the paths of values and defaults, so the module copy keeps the
    /// import of the constant they name (`ILEN`).
    constants: bool,
    /// With `clean`: the config wrote every relative path of a value or a default as it is, so the
    /// module copy keeps all its imports.
    imports: bool,
    /// `inst(N)`: the copy is also used at an instantiation of the generics of the item, which `N`
    /// seeds ([`Instance`]). The ordinary Referrer's crate only.
    inst: Option<u64>,
    /// `shadow(N)`: the copy is made again in a module and in a block that declare names the item
    /// leaks ([`shadow_check()`]). The ordinary Referrer's crate only.
    shadow: Option<u64>,
    /// `lifetime_args` (with `inst(N)`): every type parameter that [`friendly_type()`] allows is
    /// given a type that holds a lifetime ([`lifetime_type()`]), so the copy writes it into
    /// whatever signature of the item names the parameter, which a fixture asks for.
    lifetime_args: bool,
}

/// `(N)`: the number a flag of the invocation carries.
fn seed_arg(input: ParseStream) -> Result<u64> {
    let content;
    parenthesized!(content in input);
    content.parse::<LitInt>()?.base10_parse()
}

fn braced_parse<T: Parse>(input: ParseStream) -> Result<T> {
    let content;
    braced!(content in input);
    content.parse()
}

impl Parse for ApplyInput {
    fn parse(input: ParseStream) -> Result<Self> {
        let (result, context) = {
            let content;
            braced!(content in input);
            let result = content.parse()?;
            content.parse::<Token![,]>()?;
            (result, content.parse()?)
        };
        let args = braced_parse(input)?;
        let item = braced_parse(input)?;
        let key: Ident = input.parse()?;
        if key != "path" {
            return Err(Error::new(key.span(), "expected `path`"));
        }
        input.parse::<Token![=]>()?;
        let path = Path::parse_mod_style(input)?;
        // Flags after the path: `dyn_host` (the trait is implemented for `dyn Host`, an unsized
        // type, instead of a unit struct), `accept` and `allow(..)`.
        let (mut dyn_host, mut accept, mut clean) = (false, false, false);
        let (mut constants, mut imports) = (false, false);
        let (mut inst, mut shadow) = (None, None);
        let mut lifetime_args = false;
        let mut allow = Vec::new();
        while input.parse::<Option<Token![,]>>()?.is_some() {
            let keyword: Ident = input.parse()?;
            match keyword.to_string().as_str() {
                "dyn_host" => dyn_host = true,
                "accept" => accept = true,
                "clean" => clean = true,
                "constants" => constants = true,
                "imports" => imports = true,
                "inst" => inst = Some(seed_arg(input)?),
                "shadow" => shadow = Some(seed_arg(input)?),
                "lifetime_args" => lifetime_args = true,
                "allow" => {
                    let content;
                    parenthesized!(content in input);
                    allow.extend(Punctuated::<AllowArg, Token![,]>::parse_terminated(&content)?);
                }
                _ => return Err(Error::new(
                    keyword.span(),
                    "expected `dyn_host`, `accept`, `clean`, `constants`, `imports`, `inst(..)`, `lifetime_args`, `shadow(..)` or `allow(..)`",
                )),
            }
        }
        Ok(Self {
            result,
            context,
            args,
            item,
            path,
            dyn_host,
            accept,
            allow,
            clean,
            constants,
            inst,
            shadow,
            imports,
            lifetime_args,
        })
    }
}

/// `__prop_apply!`: the Referrer's side.
///
/// An item that the Leaker accepted only by `allow()` of the causes of its errors
/// (`Args::accepted`, `TYPE_LEAK_PROP_ACCEPT`) is copied in the crate that `tests/prop_cross_crate`
/// makes for those (the invocation says `accept`), and expands to nothing in the other one; every
/// other item is copied in the other one and expands to nothing in that one. The parts of an
/// accepted item that the copy keeps as written name the Leaker's items by the paths it has in the
/// Leaker's crate (`R0`, `self::R1`, `super::base::B1`, `crate::base::B1`, an import `I1`), which
/// the crate of those items has the modules for (`gen::Case::accept_referrer`). The other crate
/// has none of them: the copy of an ordinary item that kept such a path would not compile there.
/// The imports of a copied module stay, as the part that is copied as written may name them.
///
/// The errors of the result (`#[prop_leak_result]`) that `allow(..)` of the invocation does not drop
/// are reported here, in the Referrer's crate, and the copy is made with the recovered Referrer
/// either way.
pub(crate) fn apply(input: TokenStream) -> Result<TokenStream> {
    let ApplyInput {
        mut result,
        context,
        args,
        item,
        path,
        dyn_host,
        accept,
        allow,
        clean,
        constants,
        inst,
        shadow,
        imports,
        lifetime_args,
    } = parse2(input)?;
    if args.accepted != accept {
        return Ok(TokenStream::new());
    }
    for allowed in allow {
        result = match allowed {
            AllowArg::StdCrates => result.allow(Rules::new().std_crates()),
            AllowArg::Cause(cause) => result.allow(|c: &Cause, _: Place| *c == cause),
            AllowArg::AllCauses => crate::allow_every_cause(result),
        };
    }
    let errors = result.errors().to_vec();
    let rewritten = rewrite(
        &result.clone().into_referrer(),
        context,
        &item,
        &args,
        &path,
        dyn_host,
    )?;
    let id = args.id;

    // The checks call the witnesses by the path of the module of the item, in the Leaker's crate
    // (the position of an item of a module starts with its name), and the reverse trait of the
    // witnesses is implemented ([`reverse_impls()`]). The positions of a module that name what it
    // declares are compared in the copy's module with the original written for this crate
    // ([`Reroot`]).
    let host = host_arg(&rewritten, &args, &path);
    let host_ty = host.as_ref().map(|(ty, _)| ty);
    let all = comparable(&rewritten.item, &item, &args, host.as_ref(), &path);
    let mut checks = TokenStream::new();
    let mut in_module: Vec<(Vec<Ident>, TokenStream)> = Vec::new();
    for check in &all {
        if check.expected.is_some() {
            in_module.push((check.pos.mods[1..].to_vec(), reroot_fns(id, check)));
        } else {
            checks.extend(check_fn(id, check.j, &check.pos, &path, host_ty));
        }
    }
    // A trait is the one with a host, and its reverse trait is generic over the implementor.
    checks.extend(reverse_impls(
        id,
        &path,
        &all,
        rewritten.host.as_ref().and(host_ty),
    ));
    let mut copy = rewritten.item.clone();
    let body = match &mut copy {
        Item::Trait(t) => {
            let host = rewritten.host.as_ref().unwrap();
            // The module of a scope `mod` declares the trait: its signatures name it as written
            // (`<Self as Tr<N>>::A`).
            let name = &t.ident;
            let import = args.nested().then(|| quote!(use #path::#name;));
            match &item {
                Item::Trait(original) => {
                    let host_impl = host_impl(t, original, host, dyn_host, &path, args.bodies, None);
                    // The Referrer's crate declares the copy, which has the defaults of the
                    // parameters rewritten (`rewrite_defaults()`).
                    let declared = args.redeclare.then(|| {
                        let mut declared = t.clone();
                        declared.ident = ident(&format!("{}Copy", t.ident));
                        quote!(#[allow(dead_code)] #declared)
                    });
                    quote!(#import #host_impl #declared)
                }
                _ => unreachable!(),
            }
        }
        Item::Impl(_) => TokenStream::new(),
        Item::Mod(m) => {
            if let (Some((_, items)), false) = (&mut m.content, args.accepted && !clean) {
                drop_relative_uses(items, constants, imports);
            }
            for (mods, tokens) in in_module {
                let span = m.ident.span();
                let items = module_items(m, &mods)
                    .ok_or_else(|| Error::new(span, "test macro: no such module"))?;
                items.push(Item::Verbatim(tokens));
            }
            quote!(#m)
        }
        other => quote!(#other),
    };
    // Reported last: nothing is emitted (nightly) if the macro fails above.
    let errors = crate::report(&errors);
    // The copy at an instantiation of its generics, and inside a module and a block that declare the
    // names it leaks: the ordinary Referrer's crate only (the others do not ask for it).
    let mut more = TokenStream::new();
    if inst.is_some() || shadow.is_some() {
        let referrer = result.clone().into_referrer();
        if let Some(seed) = inst {
            let ask = InstAsk {
                seed,
                lifetime_args,
            };
            let (tokens, n, [elided, entry, elided_pairs, entry_pairs]) =
                instance_part(&referrer, context, &item, &args, &path, dyn_host, ask)?;
            more.extend(tokens);
            counted("inst", id, n);
            counted("inst_elided", id, elided);
            counted("inst_elided_entry", id, entry);
            counted("inst_elided_candidates", id, elided_pairs);
            counted("inst_elided_entry_candidates", id, entry_pairs);
        }
        if let Some(seed) = shadow {
            let hidden = shadow_check(&referrer, context, &item, &args, &path, dyn_host, seed)?;
            counted("shadow", id, hidden);
        }
    }
    // `TYPE_LEAK_PROP_DUMP=<id>` prints what the copy at an instantiation adds, to debug it.
    if std::env::var("TYPE_LEAK_PROP_DUMP").map_or(false, |dump| dump == id.to_string()) {
        eprintln!("TYPE_LEAK_PROP_DUMP {}: {}", id, more);
    }
    Ok(quote! {
        #errors
        #body
        #checks
        #more
    })
}

/// `__prop_carry!`: the Referrer's side of the errors that the Leaker carries (`__tl_carry_<id>!`),
/// in a crate of their own (`tlp_e_NNN`, `gen::Case::errors_referrer`), where the errors that remain
/// are reported and the crate does not compile.
///
/// The result is read from its tokens (`LeakerResult`, the Referrer and the `@ [ {error} .. ]` list),
/// and its errors must be those that the Leaker had, tokens for tokens (spans aside: they are
/// compared as text). Then, by the number of the invocation (`carry(N)`), the errors are
///
/// - kept: all remain;
/// - allowed one: `LeakerResult::allow()` of the paths of the first error drops that error, and
///   no other that has a path of its own;
/// - allowed all: `allow()` of the cause of every error drops every one.
///
/// Each error that remains is reported with `ErrorItem::emit()`, whose tokens must be those of
/// the pure `ErrorItem::to_compile_error()` (a `compile_error!`, on the stable compiler). The
/// numbers go to the harness (`TLCOUNT carry_errors <n>` `TLCOUNT carry_reported <n>`), which counts
/// the diagnostics of the crate.
pub(crate) fn carry(input: TokenStream) -> Result<TokenStream> {
    struct CarryInput {
        result: LeakerResult,
        expected: String,
        seed: u64,
    }
    impl Parse for CarryInput {
        fn parse(input: ParseStream) -> Result<Self> {
            let result = braced_parse(input)?;
            let expected = braced_parse::<LitStr>(input)?.value();
            let mut seed = 0;
            if input.parse::<Option<Ident>>()?.is_some() {
                seed = seed_arg(input)?;
            }
            Ok(CarryInput {
                result,
                expected,
                seed,
            })
        }
    }
    let CarryInput {
        mut result,
        expected,
        seed,
    } = parse2(input)?;
    let fail = |what: String| Error::new(Span::call_site(), format!("CARRY: {}", what));
    let texts = |result: &LeakerResult| -> Vec<String> {
        result
            .errors()
            .iter()
            .map(|e| e.to_token_stream().to_string())
            .collect()
    };
    let got = texts(&result);
    let want: Vec<String> = expected
        .split('\u{1d}')
        .filter(|text| !text.is_empty())
        .map(String::from)
        .collect();
    // The spacing of the tokens depends on where they come from (`-32768` read back is `- 32768`),
    // so the texts are compared without whitespace.
    let bare = |text: &String| text.split_whitespace().collect::<String>();
    if got.len() != want.len() || got.iter().zip(&want).any(|(g, w)| bare(g) != bare(w)) {
        let first = got
            .iter()
            .zip(&want)
            .find(|(g, w)| bare(g) != bare(w))
            .map_or(String::new(), |(g, w)| format!(":\n  read back: {}\n  Leaker's:  {}", g, w));
        return Err(fail(format!(
            "the errors after the round trip differ from the Leaker's: {} against {}{}",
            got.len(),
            want.len(),
            first
        )));
    }
    let total = got.len();
    let mode = seed % 3;
    match mode {
        1 => {
            if let Some(first) = result.errors().first() {
                let paths = if first.paths.is_empty() {
                    vec![first.cause.clone()]
                } else {
                    first.paths.clone()
                };
                let text = got[0].clone();
                result = result.allow(move |c: &Cause, _: Place| paths.contains(c));
                if texts(&result).contains(&text) {
                    return Err(fail(format!(
                        "`allow()` of the paths of the first error did not drop it: {}",
                        text
                    )));
                }
            }
        }
        2 => {
            result = crate::allow_every_cause(result);
            if !result.errors().is_empty() {
                return Err(fail(
                    "`allow()` of the cause of every error left some".to_string(),
                ));
            }
        }
        _ => (),
    }
    let mut out = TokenStream::new();
    for error in result.errors() {
        let pure = error.to_compile_error();
        if pure.to_string() != error.to_compile_error().to_string()
            || !pure.to_string().contains("compile_error")
        {
            return Err(fail(
                "`ErrorItem::to_compile_error()` is not pure, or no `compile_error!`".to_string(),
            ));
        }
        let emitted = error.emit();
        // The stable compiler gets the same tokens (the nightly one has emitted a diagnostic).
        if !emitted.is_empty() && emitted.to_string() != pure.to_string() {
            return Err(fail(
                "`ErrorItem::emit()` gives other tokens than `to_compile_error()`".to_string(),
            ));
        }
        out.extend(emitted);
    }
    // A `LeakerResult` with no error carries the Referrer only.
    let reported = result.errors().len();
    if total == 0 && reported != 0 {
        return Err(fail("errors out of none".to_string()));
    }
    eprintln!("TLCOUNT carry_errors 0 {}", total);
    eprintln!("TLCOUNT carry_reported 0 {}", reported);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A parameter that has marker bounds only, and is named by no other bound, can be given a type.
    #[test]
    fn friendly_parameters_are_those_no_other_bound_names() {
        let item: Item = parse_quote! {
            struct S<T: Clone + 'static, U: Into<T>, const N: usize, const M: usize, V>(T, U, V) where
                [u8; M]: Default;
        };
        let generics = match &item {
            Item::Struct(i) => &i.generics,
            _ => unreachable!(),
        };
        let types: Vec<&TypeParam> = generics
            .params
            .iter()
            .filter_map(|p| match p {
                GenericParam::Type(t) => Some(t),
                _ => None,
            })
            .collect();
        // `T` is named by the bound of `U`; `U` has a bound that is no marker; `V` has none.
        assert!(friendly_type(&item, types[0]).is_none());
        assert!(friendly_type(&item, types[1]).is_none());
        assert!(friendly_type(&item, types[2]).is_some());
        let consts: Vec<&ConstParam> = generics
            .params
            .iter()
            .filter_map(|p| match p {
                GenericParam::Const(c) => Some(c),
                _ => None,
            })
            .collect();
        assert!(friendly_const(&item, consts[0]));
        assert!(!friendly_const(&item, consts[1]));
        // A method that bounds a parameter by a trait that no type has makes it unfit.
        let item: Item = parse_quote! {
            trait Tr<T> { fn f<X>(&self) where T: Fn(X); }
        };
        let t = match &item {
            Item::Trait(i) => match i.generics.params.first() {
                Some(GenericParam::Type(t)) => t.clone(),
                _ => unreachable!(),
            },
            _ => unreachable!(),
        };
        assert!(friendly_type(&item, &t).is_none());
    }

    /// The parameters that no type with a lifetime can be given to: those a signature names whose
    /// lifetime a path hides, and all of them if it names `Self` (or `Self::A`), which holds them.
    #[test]
    fn a_signature_that_names_self_hides_every_parameter() {
        let hidden = |src: &str| {
            let item: Item = parse_str(src).unwrap();
            let generics = match &item {
                Item::Struct(i) => &i.generics,
                _ => unreachable!(),
            };
            let mut hidden: Vec<String> = hidden_inputs(&item, generics).into_iter().collect();
            hidden.sort();
            hidden
        };
        // The inputs show no lifetime and name `U` only.
        assert_eq!(
            hidden("struct S<T, U>(fn(::core::str::Chars, U) -> &B1<T>);"),
            ["U"]
        );
        for input in ["Self", "Self::A", "Box<Self>", "(u8, <Self as Tr>::A)"] {
            let item = format!(
                "struct S<T, U>(fn(::core::str::Chars, {}) -> &B1<T>);",
                input
            );
            assert_eq!(hidden(&item), ["T", "U"], "{}", input);
        }
        // A lifetime in the inputs is the one the output takes: the copy can name it.
        assert!(hidden("struct S<T, U>(fn(&u8, Self) -> &B1<T>);").is_empty());
        // No lifetime in the output, nothing to elide.
        assert!(hidden("struct S<T, U>(fn(::core::str::Chars, Self) -> B1<T>);").is_empty());
    }

    /// The pairs where an argument with a lifetime meets a signature that relies on elision (with a
    /// replaced entry in the inputs, or any), but not the parameters that cannot be given any type
    /// or that a path hides a lifetime from.
    #[test]
    fn candidates_are_the_parameters_of_a_signature_that_relies_on_elision() {
        let candidates = |src: &str| {
            let item: Item = parse_str(src).unwrap();
            let generics = match &item {
                Item::Struct(i) => &i.generics,
                _ => unreachable!(),
            };
            let replaced = [quote!(::std::vec::Vec<T>).to_string()];
            elided_candidates(&item, generics, Some(&replaced))
        };
        // `V` is named by a signature without the entry, `W` by none.
        let item =
            "struct S<T, U, V, W>(fn(&::std::vec::Vec<T>, U) -> &u8, fn(&V) -> &u8, Option<W>);";
        assert_eq!(candidates(item), ["T", "U"]);
        let parsed: Item = parse_str(item).unwrap();
        let generics = match &parsed {
            Item::Struct(i) => &i.generics,
            _ => unreachable!(),
        };
        assert_eq!(elided_candidates(&parsed, generics, None), ["T", "U", "V"]);
        // One pair for each signature.
        assert_eq!(
            candidates("struct S<T>(fn(&::std::vec::Vec<T>) -> &u8, Box<dyn Fn(T, &::std::vec::Vec<T>) -> &u8>);"),
            ["T", "T"]
        );
        // `U` needs more than a type has, and a path hides a lifetime of the signature `V` is in.
        assert_eq!(
            candidates(
                "struct S<T, U: Iterator, V>(fn(&::std::vec::Vec<T>, U, V) -> &u8, fn(::core::str::Chars, V) -> &u8);"
            ),
            ["T"]
        );
        // And an output with no elided lifetime has nothing to name.
        assert!(candidates("struct S<T>(fn(&::std::vec::Vec<T>) -> u8);").is_empty());
    }

    /// An item put in a module or a block that declares names is found there again.
    #[test]
    fn a_wrapped_item_is_the_item_inside() {
        let item: Item = parse_quote!(pub struct S(R0););
        for block in [false, true] {
            let wrap = Wrap {
                block,
                names: vec![ident("R0"), ident("R1")],
            };
            let wrapped = wrap.around(item.clone());
            assert_ne!(wrapped, item);
            assert_eq!(wrap.inside(wrapped), item);
        }
    }

    /// The choices of the macro are those of the seed.
    #[test]
    fn the_draws_follow_the_seed() {
        let (mut a, mut b) = (Draw(7), Draw(7));
        for _ in 0..8 {
            assert_eq!(a.next(), b.next());
        }
        assert!((0..100).any(|_| a.chance(50)) && (0..100).any(|_| !a.chance(50)));
    }

    const MODULE: &str = "pub mod mm {
        use super::base::B0 as I0;
        use ::krate::base::B1 as J1;
        use super::base::BLEN as ILEN;
        pub struct R0;
        pub struct RC<const N: usize>;
        pub mod n { use super::super::base::BL as IL; pub struct R1; }
    }";

    /// What the original type of a position in `mm` (or in `mm::n` with `nested`) must equal in the
    /// Referrer's crate, when the copy has `copy` there.
    fn expected(orig: &str, copy: &str, nested: bool) -> Option<String> {
        let module: Item = parse_str(MODULE).unwrap();
        let args: Args = parse_str("id = 0, prelude = false").unwrap();
        let reroot = Reroot::new(&module, &parse_quote!(::krate), &args).unwrap();
        let mods = if nested { ["mm", "n"].to_vec() } else { ["mm"].to_vec() };
        let position = |ty: &str| Position {
            mods: mods.iter().map(|m| ident(m)).collect(),
            layers: Vec::new(),
            self_sub: None,
            ty: parse_str(ty).unwrap(),
        };
        let found = reroot.position(&position(orig), &position(copy))?.ty;
        Some(quote!(#found).to_string().replace(' ', ""))
    }

    const ENTRY: &str = "<() as ::krate::Repeater<(), 0>>::Type";

    #[test]
    fn a_declared_name_is_the_copys_and_an_import_is_the_leakers() {
        assert_eq!(expected("R0", "R0", false).unwrap(), "R0");
        assert_eq!(expected("self::R0", "self::R0", false).unwrap(), "self::R0");
        assert_eq!(expected("I0", ENTRY, false).unwrap(), "::krate::base::B0");
        assert_eq!(expected("J1<R0>", "J1<R0>", false).unwrap(), "::krate::base::B1<R0>");
        assert_eq!(
            expected("crate::base::B2<u8>", "crate::base::B2<u8>", false),
            Some("::krate::base::B2<u8>".to_string())
        );
        // Out of the module by `super`, and into it.
        assert_eq!(
            expected("super::super::base::B0", ENTRY, true).unwrap(),
            "::krate::base::B0"
        );
        assert_eq!(expected("super::R0", "super::R0", true).unwrap(), "super::R0");
        assert_eq!(expected("IL<u8>", ENTRY, true).unwrap(), "::krate::base::BL<u8>");
    }

    #[test]
    fn a_name_the_module_declares_is_the_leakers_only_inside_an_entry() {
        // Wrapped by a path of another module: the entry names the Leaker's `R0`.
        assert_eq!(
            expected("crate::base::B1<R0>", ENTRY, false).unwrap(),
            "::krate::base::B1<::krate::mm::R0>"
        );
        // A bare name the module declares that the copy replaced: the copy's own, so that the
        // comparison fails if the Visitor took it for an entry.
        assert_eq!(expected("R0", ENTRY, false).unwrap(), "R0");
        assert_eq!(
            expected("Option<R0>", "Option<R0>", false),
            None,
            "a bare name of the prelude is not allowed here"
        );
        // A constant that is not the module's makes the whole type the entry, `RC` included.
        assert_eq!(
            expected("RC<ILEN>", ENTRY, false).unwrap(),
            "::krate::mm::RC<{::krate::base::BLEN}>"
        );
        assert_eq!(expected("RC<3>", ENTRY, false).unwrap(), "RC<3>");
    }

    #[test]
    fn the_header_of_an_impl_assumes_its_trait_reference() {
        let args: Args = parse_str("id = 0, scope = \"\"").unwrap();
        let item: Item = parse_str("impl<'a, T: Clone> ::krate::m::Tr<'a, T> for &'static fn() {}").unwrap();
        let header = header_position(&item, &args).unwrap();
        let (generics, ty) = lowered(&header, Implementor::Fixed);
        let where_clause = &generics.where_clause;
        assert_eq!(
            quote!(#where_clause).to_string().replace(' ', ""),
            "where(&'staticfn()):::krate::m::Tr<'a,T>"
        );
        assert_eq!(quote!(#ty).to_string().replace(' ', ""), "(&'a(),&'staticfn())");
        // An inherent impl has no trait reference, and the fragments of a scope are not the header.
        let inherent: Item = parse_str("impl Foo {}").unwrap();
        assert!(header_position(&inherent, &args).is_none());
        let scoped: Args = parse_str("id = 0, scope = \"impl\"").unwrap();
        assert!(header_position(&item, &scoped).is_none());
    }

    /// The header with an entry that has arguments is left out only if the generics state a bound
    /// on one of its arguments through another entry (E0277 in rustc 1.99 otherwise on the impl's
    /// bounds).
    #[test]
    fn a_header_is_left_out_only_where_rustc_cannot_normalize_it() {
        let args: Args = parse_str("id = 0, scope = \"\"").unwrap();
        let cycles = |bound: &str, selfty: &str| {
            let src = format!(
                "impl<T: Clone + {}, E> ::k::Tr<T> for ({}, E) {{}}",
                bound, selfty
            );
            let item: Item = parse_str(&src).unwrap();
            header_cycles(&header_position(&item, &args).unwrap())
        };
        // The projection of the entry `ix` of the Leaker 1, through `::k::Repeater`.
        let entry = |host: &str, args: &str, ix: EntryId| {
            let key = repeater_key(1, ix);
            format!("<{} as ::k::Repeater<{}, {}>>::Type", host, args, key)
        };
        let with_args = entry("u8", "(T,)", parse_quote!(2));
        let no_args = entry("u8", "()", parse_quote!(2));
        let entry_bound = format!("AsRef<{}>", entry("u64", "()", parse_quote!(1)));
        let (with_args, no_args, entry_bound) = (&*with_args, &*no_args, &*entry_bound);
        assert!(cycles(entry_bound, with_args));
        // A bound on a generic that the entry is not given does not matter, a where predicate does.
        let other = |generics: &str, where_clause: &str| {
            let src = format!(
                "impl<T: Clone, {}, E> ::k::Tr<T> for ({}, E) {} {{}}",
                generics, with_args, where_clause
            );
            let item: Item = parse_str(&src).unwrap();
            header_cycles(&header_position(&item, &args).unwrap())
        };
        assert!(!other(&format!("U: {}", entry_bound), ""));
        assert!(other("U", &format!("where T: {}", entry_bound)));
        let vec_entry = format!("where Vec<(T, {})>: Clone", entry("u64", "()", parse_quote!(1)));
        assert!(other("U", &vec_entry));
        // The associated types are taken as they are, and so are the bounds without an entry.
        let item = format!("Iterator<Item = {}>", entry("u16", "()", parse_quote!(3)));
        assert!(!cycles(&item, with_args));
        assert!(!cycles("AsRef<Vec<u8>>", with_args));
        assert!(!cycles(entry_bound, no_args));
        // But the inputs of `Fn` are arguments.
        let call = format!("Fn({}) -> u8", entry("u16", "()", parse_quote!(3)));
        assert!(cycles(&call, with_args));
    }
}
