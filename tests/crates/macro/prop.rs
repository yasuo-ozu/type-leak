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
//!   invariant, so that a check compares them for equality and not as subtypes), and
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
//!   for each type position, which compiles only if the rewritten type is exactly the original.
//!
//! The `Self` of a position is the type itself (a struct, an enum or a union), the self type (an
//! impl) or the host type (a trait). A module cannot name `Self`, so its items do not.
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
//! given the rule (`LeakerResult::allow_prelude()`, ...) and then `allow(cause)` for the causes of
//! the errors that remain, and after the analysis of a config that has the rule. When no error
//! remains after the rule, the copy is also made in a crate with none of the Leaker's names (the
//! invocation says `clean`). Their Repeater's impls have an `ID` of their own
//! ([`Args::repeater_id()`]).
//!
//! With `#[prop_leak_result]` the Leaker does not decide: an item with errors is emitted with its
//! `LeakerResult` (the Referrer, then the errors, as tokens). The Referrer's invocation says what
//! it allows (`allow(std_crates, trait Path, type Ty)`: `LeakerResult::allow_std_crates()`,
//! `allow(cause)`), [`apply()`] reports every error that remains with
//! `ErrorItem::to_compile_error()`, and copies the item with the recovered Referrer either way.

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
    /// The `ID` of the Repeater, the index of the item in the Leaker's crate.
    id: u64,
    allow_self: bool,
    /// Allow the names of the standard prelude (`Vec`, `Option`, `Clone`, ...).
    prelude: bool,
    /// Allowed crates (`::std`).
    allow: Vec<Path>,
    /// How the item is analyzed: `""` by the `intern_*` method of its kind, or by a scope (see
    /// [`analyze()`]): `"mod"`, `"generics"`, `"trait"` or `"impl"`.
    scope: String,
    /// The Referrer replaces the optional entries too (`allow_ambiguous_crate_name(false)`), so it
    /// needs the Repeater's impl of every entry. A trusting Referrer needs those of the required
    /// entries only, and that is all this macro emits then, as a downstream crate does. Not said:
    /// the impls of all the entries, for any Referrer; the Referrer's invocation says which it is
    /// (`strict`), and the two must agree if it is said.
    strict: Option<bool>,
    /// The Leaker allows the primitive types (`LeakerConfig::allow_primitive()`).
    primitive: bool,
    /// The expectation about a bare `?Sized` without the prelude: `true` if the Leaker accepts it
    /// whatever the prelude (the `?` applies to the real `Sized` only), `false` if it is a relative
    /// path like any other bare name of the prelude (`gen::BARE_RELAXED_SIZED_ALWAYS_OK`).
    relaxed_sized: bool,
    /// A trait: the Referrer's crate declares the rewritten trait too (`<Name>Copy`), not only an
    /// impl of the Leaker's trait for the host. Its parameter defaults are rewritten for a
    /// declaration, which cannot name the defaulted parameter itself (E0128): see
    /// [`rewrite_defaults()`].
    redeclare: bool,
    /// The Leaker reads the values of the item (`LeakerConfig::check_values()`): the default of an
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
    /// that does (`LeakerConfig::nested_self_entries()`).
    nested_self_entries: bool,
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
}

impl Parse for Args {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut args = Args {
            id: 0,
            allow_self: true,
            prelude: false,
            allow: Vec::new(),
            scope: String::new(),
            strict: None,
            primitive: true,
            relaxed_sized: false,
            redeclare: false,
            values: false,
            bodies: false,
            nested_self_entries: false,
            no_nested_self: false,
            stages: false,
            accepted: false,
            stage: 0,
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
                ("strict", Lit::Bool(b)) => args.strict = Some(b.value),
                ("primitive", Lit::Bool(b)) => args.primitive = b.value,
                ("relaxed_sized", Lit::Bool(b)) => args.relaxed_sized = b.value,
                ("redeclare", Lit::Bool(b)) => args.redeclare = b.value,
                ("values", Lit::Bool(b)) => args.values = b.value,
                ("bodies", Lit::Bool(b)) => args.bodies = b.value,
                ("nested_self_entries", Lit::Bool(b)) => args.nested_self_entries = b.value,
                ("no_nested_self", Lit::Bool(b)) => args.no_nested_self = b.value,
                ("stages", Lit::Bool(b)) => args.stages = b.value,
                ("accepted", Lit::Bool(b)) => args.accepted = b.value,
                ("stage", Lit::Int(i)) => args.stage = i.base10_parse()?,
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
        let (primitive, relaxed_sized) = (self.primitive, self.relaxed_sized);
        let (redeclare, values) = (self.redeclare, self.values);
        let bodies = self.bodies;
        let (nested_self_entries, accepted) = (self.nested_self_entries, self.accepted);
        let no_nested_self = self.no_nested_self;
        let (stages, stage) = (self.stages, self.stage);
        let strict = self.strict.map(|strict| quote!(strict = #strict,));
        let scope = &self.scope;
        let allow = self
            .allow
            .iter()
            .map(|p| quote!(#p).to_string().replace(' ', ""))
            .collect::<Vec<_>>()
            .join(",");
        quote!(id = #id, allow_self = #allow_self, prelude = #prelude, scope = #scope, allow = #allow, #strict primitive = #primitive, relaxed_sized = #relaxed_sized, redeclare = #redeclare, values = #values, bodies = #bodies, nested_self_entries = #nested_self_entries, no_nested_self = #no_nested_self, stages = #stages, accepted = #accepted, stage = #stage)
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
    /// The analysis without the rule, whose result is given the rule (`LeakerResult::allow_*()`).
    Result,
    /// The analysis of a config that has the rule (`LeakerConfig::allow_*()`). Its copy is what the
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
    /// `allow_prelude()`: for every item, with the analysis not allowing the prelude.
    Prelude,
    /// `allow_crate()`.
    Crate,
    /// `allow_path(super::base)`: the module `base` as the modules at depth 1 (`m0`, `mm0`) name it.
    /// The paths of the deeper ones (`super::super::base`) are other paths.
    Path,
    /// `allow_std_crates()`.
    StdCrates,
    /// `allow_values()`: the items that the Leaker reads the values of.
    Values,
    /// `allow_default()`.
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

    fn on_config(self, config: &mut LeakerConfig) {
        match self {
            Rule::Prelude => config.allow_prelude(),
            Rule::Crate => config.allow_crate(),
            Rule::Path => config.allow_path(parse_quote!(super::base)),
            Rule::StdCrates => config.allow_std_crates(),
            Rule::Values => config.allow_values(),
            Rule::Default => config.allow_default(),
        };
    }

    fn on_result(self, result: &mut LeakerResult) {
        match self {
            Rule::Prelude => result.allow_prelude(),
            Rule::Crate => result.allow_crate(),
            Rule::Path => result.allow_path(parse_quote!(super::base)),
            Rule::StdCrates => result.allow_std_crates(),
            Rule::Values => result.allow_values(),
            Rule::Default => result.allow_default(),
        };
    }
}

/// `std`, `core` or `alloc` as a relative path (`LeakerConfig::allow_std_crates()`).
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
/// - `impl`: an impl in `enter_impl(..)`, where only the self type and the arguments of the trait
///   are visited (the shape of a macro that resolves the trait by its name). The generics and the
///   items of the impl are not analyzed, and not compared.
fn analyze(item: &Item, args: &Args) -> (LeakerResult, ContextId, TokenStream) {
    analyze_with(item, args, |_| ())
}

/// [`analyze()`] with `rule` given to the config as well: `allow_stages` compares it with the same
/// rule given to the result.
fn analyze_with(
    item: &Item,
    args: &Args,
    rule: impl FnOnce(&mut LeakerConfig),
) -> (LeakerResult, ContextId, TokenStream) {
    let mut config = LeakerConfig::new();
    if args.primitive {
        config.allow_primitive();
    }
    if args.prelude {
        config.allow_prelude();
    }
    if args.values {
        config.check_values();
    }
    if args.nested_self_entries {
        config.nested_self_entries();
    }
    for path in &args.allow {
        config.allow_path(path.clone());
    }
    rule(&mut config);
    let mut leaker = Leaker::from_config(config);
    let allow_self = args.allow_self;
    let context = match (args.scope.as_str(), item) {
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
                .enter_impl(&i.generics, &i.self_ty, allow_self, |view| {
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
    };
    let assertions = leaker.assertions();
    (leaker.result(), context, assertions)
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

/// The Repeater's implementation of each entry with the path of the module it is keyed by,
/// relative to the analyzed module (`[]` for it and for the entries of no module).
///
/// An item that a trusting Referrer copies (`strict` is `false`) gets the implementations of the
/// required entries only, as a downstream crate writes them (`Referrer::required()`): a Referrer
/// that replaced an optional entry anyway fails to compile. A strict one gets all the entries.
fn repeater_impls(referrer: &Referrer, id: u64, strict: bool) -> Result<(Impls, usize)> {
    let repeater: Path = parse_quote!(crate::Repeater);
    let slots = RepeaterSlots::new(repeater.clone());
    // For `Self` bounded by the trait, the implementation is for `__S` (the implementor).
    let self_param = ident("__S");
    check_entry_helpers(referrer)?;
    let mut impls = Vec::new();
    let mut skipped = 0;
    for ix in 0..referrer.iter().count() {
        crate::check_plain(referrer, ix, &slots)?;
        let r = referrer
            .repeater_impl_with(ix, ImplSelf::Param(&self_param), |value, param| {
                slots.encode_const(value, &param.ty)
            })
            .map_err(Error::from)?;
        check_impl_pieces(referrer, ix, &r, &self_param)?;
        if !strict && !referrer.must(ix) {
            skipped += 1;
            continue;
        }
        let module = r
            .module
            .and_then(|m| referrer.module_path(m))
            .map(|p| p.to_vec())
            .unwrap_or_default();
        let (impl_generics, _, where_clause) = r.generics.split_for_impl();
        let (args, self_ty, ty) = (&r.args, &r.self_ty, &r.ty);
        impls.push((
            module,
            quote! {
                impl #impl_generics #repeater<#args, #id, #ix> for #self_ty #where_clause {
                    type Type = #ty;
                }
            },
        ));
    }
    Ok((impls, skipped))
}

/// A struct, an enum or a union nested in a module (scope `mod`) has a `Self` of its own: only a
/// type that must be leaked has it in its context (`SelfRequirement::Type`). An optional entry that
/// names it (an absolute type, `::p::W<Self>`) would be replaced by a strict Referrer with the
/// Leaker's type, where a trusting one copies the copy's own `Self`, and the check cannot tell: it
/// writes the Leaker's type for every `Self` of the copy (`host_arg()`). So it is checked here.
fn check_nested_self_is_required(referrer: &Referrer, item: &Item, args: &Args) -> Result<()> {
    if args.scope != "mod" || !matches!(item, Item::Struct(_) | Item::Enum(_) | Item::Union(_)) {
        return Ok(());
    }
    for ix in 0..referrer.iter().count() {
        let nested_type = matches!(
            referrer.context_self(referrer.context(ix)),
            Some(SelfRequirement::Type(_))
        );
        if !referrer.must(ix) && nested_type {
            let ty = referrer.iter().nth(ix).unwrap();
            return Err(Error::new(
                Span::call_site(),
                format!(
                    "NESTED SELF OPTIONAL: the optional entry `{}` of an item nested in a module names its own `Self`: a strict Referrer would replace it with the Leaker's type, where the copy's `Self` is its own",
                    quote!(#ty)
                ),
            ));
        }
    }
    Ok(())
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

/// The entries and their helpers agree: `required()` is the entries with `must()`, and they are the
/// first ones; `generics(ix)` is the generics of the context of the entry, as
/// `iter_with_generics()` has them, and the context is one of `contexts()`.
fn check_entry_helpers(referrer: &Referrer) -> Result<()> {
    let n = referrer.iter().count();
    let musts: Vec<usize> = (0..n).filter(|&ix| referrer.must(ix)).collect();
    let required: Vec<usize> = referrer.required().map(|(ix, _)| ix).collect();
    if musts != required {
        return Err(helpers_disagree(format!(
            "`must()` holds for {:?} and `required()` has {:?}",
            musts, required
        )));
    }
    if musts != (0..musts.len()).collect::<Vec<_>>() {
        return Err(helpers_disagree(format!(
            "the required entries {:?} are not the first {} ones",
            musts,
            musts.len()
        )));
    }
    if referrer.has_required() != !musts.is_empty() {
        return Err(helpers_disagree(
            "`has_required()` and `required()` differ".to_string(),
        ));
    }
    if referrer.is_empty() != (n == 0) {
        return Err(helpers_disagree(
            "`is_empty()` and `iter()` differ".to_string(),
        ));
    }
    for (ix, ty) in referrer.required() {
        if referrer.iter().nth(ix) != Some(ty) {
            return Err(helpers_disagree(format!(
                "`required()` has another type than `iter()` at {}",
                ix
            )));
        }
    }
    let contexts: Vec<ContextId> = referrer.contexts().collect();
    let mut with_generics = referrer.iter_with_generics();
    for (ix, ty) in referrer.iter().enumerate() {
        let context = referrer.context(ix);
        if !contexts.contains(&context) {
            return Err(helpers_disagree(format!(
                "the context of the entry {} is not in `contexts()`",
                ix
            )));
        }
        if generics_text(referrer.generics(ix)) != generics_text(referrer.context_generics(context))
        {
            return Err(helpers_disagree(format!(
                "`generics({})` is not the `context_generics()` of its context",
                ix
            )));
        }
        match with_generics.next() {
            Some((other, generics))
                if other == ty
                    && generics_text(generics) == generics_text(referrer.generics(ix)) => {}
            _ => {
                return Err(helpers_disagree(format!(
                    "`iter_with_generics()` differs from `iter()` and `generics()` at {}",
                    ix
                )))
            }
        }
    }
    if with_generics.next().is_some() {
        return Err(helpers_disagree(
            "`iter_with_generics()` has more entries than `iter()`".to_string(),
        ));
    }
    Ok(())
}

/// Normalizes the text of generics that the impl of an entry declares, or that
/// `Referrer::impl_generics()` returns, so that the two can be compared: the implementor `__S`
/// replaces `Self` (and `<__S>::A` is `Self::A`), and the parentheses that rustc needs around a
/// first qualified path are dropped.
fn impl_text(text: &str, self_param: &Ident) -> String {
    text.replace(char::is_whitespace, "")
        .replace(&self_param.to_string(), "Self")
        .replace("<Self>::", "Self::")
        .replace(['(', ')'], "")
}

/// Takes the last parameter off the generics. (They are collected again, so that no comma is left
/// after the parameter before it.)
fn implementor_param(generics: &mut Generics) -> Option<GenericParam> {
    let mut params: Vec<GenericParam> = generics.params.iter().cloned().collect();
    let last = params.pop();
    generics.params = params.into_iter().collect();
    last
}

/// The generics of the impl of an entry are `Referrer::impl_generics()` of its context, with the
/// parameter of the implementor added (`__S: ?Sized`, and the trait for a `Self` that a trait
/// bounds) unless the impl is for a type; its module is the context's.
fn check_impl_pieces(
    referrer: &Referrer,
    ix: usize,
    r: &RepeaterImpl,
    self_param: &Ident,
) -> Result<()> {
    let context = referrer.context(ix);
    let declared = referrer.impl_generics(context);
    let mut got = r.generics.clone();
    match referrer.context_self(context) {
        Some(SelfRequirement::Type(_)) => (),
        _ => match implementor_param(&mut got) {
            Some(GenericParam::Type(t)) if t.ident == *self_param => {
                // `?Sized`, and the trait of a `Self` that is bound by it. Any type that the
                // bounds hold for is a `Sized` one (a struct, an enum or a union), so it has no
                // bound if a bound of the generics needs it `Sized`, and `?Sized` if not.
                let expected = match referrer.context_self(context) {
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
                    "the impl of the entry {} does not end with the parameter `{}`: {}",
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
            "the impl of the entry {} declares `{}` and `impl_generics()` is `{}`",
            ix,
            generics_text(&got),
            generics_text(&declared)
        )));
    }
    if r.module != referrer.context_module(context) {
        return Err(helpers_disagree(format!(
            "the impl of the entry {} has another module than `context_module()`",
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

/// `pub fn __tl_w_<id>_<j><generics>() -> PhantomData<ty>`, with the generics around the position.
pub(crate) fn witness_fn(id: u64, j: usize, pos: &Position) -> TokenStream {
    let mut ty = pos.ty.clone();
    let mut generics = merge_generics(&pos.layers);
    if let Some(sub) = &pos.self_sub {
        SelfSubst(sub).visit_type_mut(&mut ty);
        SelfSubst(sub).visit_generics_mut(&mut generics);
        if sub.bound {
            let trait_ref = sub.trait_ref.as_ref().unwrap();
            generics
                .params
                .push(parse_quote!(__Self: ?::core::marker::Sized + #trait_ref));
        }
    }
    let name = witness_name(id, j);
    let params = &generics.params;
    let where_clause = &generics.where_clause;
    let invariant = invariant_in(ty);
    quote! {
        #[allow(non_snake_case, dead_code)]
        pub fn #name<#params>() -> ::core::marker::PhantomData<#invariant> #where_clause {
            ::core::marker::PhantomData
        }
    }
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
        const PRIMITIVES: &[&str] = &[
            "u8", "u16", "u32", "u64", "u128", "usize", "i8", "i16", "i32", "i64", "i128", "isize",
            "f32", "f64", "bool", "char", "str",
        ];
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
            "FromIterator",
            "ToOwned",
            "ToString",
            "drop",
        ];
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
}

impl Projector<'_> {
    fn project(&self, ix: usize) -> Result<Type> {
        let context = self.referrer.context(ix);
        let self_ty: Type = match self.referrer.context_self(context) {
            Some(SelfRequirement::Bound(_)) => self
                .bound_self
                .clone()
                .ok_or_else(|| Error::new(Span::call_site(), "test macro: no host for Self"))?,
            Some(SelfRequirement::Type(_)) => self
                .type_self
                .clone()
                .ok_or_else(|| Error::new(Span::call_site(), "test macro: no type for Self"))?,
            // The default of a parameter of a trait analyzed as the item itself, whose generics
            // have a bound that names `Self` (`trait Tr<T: AsRef<Self>, U = Rel<T>>`): the
            // Leaker analyzed it without `Self`, and the entry is for any type that the bound holds
            // for. The implementor is the host, which has it (`T: AsRef<Host>`), not `()`.
            _ if self.bound_self.is_some()
                && names_self(self.referrer.context_generics(context)) =>
            {
                self.bound_self.clone().unwrap()
            }
            _ => parse_quote!(()),
        };
        self.project_with(ix, self_ty)
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
    fn project_nested(&self, nested: &NestedSelf, base: &Path, ix: usize) -> Result<Type> {
        if let Some(SelfRequirement::Any) = self.referrer.context_self(self.referrer.context(ix)) {
            return self.project_with(ix, parse_quote!(Self));
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
        self.project_with(ix, self_ty)
    }

    fn project_with(&self, ix: usize, self_ty: Type) -> Result<Type> {
        let slots = RepeaterSlots::new(self.repeater.clone());
        let site =
            UseSite::substituting(&self.root_params, &self.root_args).with_self(self_ty.clone());
        let encoded = self
            .referrer
            .entry_args_with(ix, &site, |value, param| {
                slots.encode_const(value, &param.ty)
            })
            .map_err(Error::from)?;
        let (repeater, id) = (&self.repeater, self.id);
        Ok(parse_quote!(<#self_ty as #repeater<#encoded, #id, #ix>>::Type))
    }
}

struct Rewritten {
    item: Item,
    /// The name of the host type of a trait.
    host: Option<Ident>,
    /// The host is `dyn <host>`, an unsized type (the implementor of the trait is not `Sized`).
    dyn_host: bool,
}

/// The host type: a unit struct `Host`, or `dyn Host + 'static` if the implementor must be unsized.
fn host_type(host: &Ident, dyn_host: bool) -> Type {
    if dyn_host {
        parse_quote!(dyn #host + 'static)
    } else {
        parse_quote!(#host)
    }
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
    let mut repeater = leaker_path.clone();
    repeater.segments = repeater.segments.iter().take(1).cloned().collect();
    repeater.segments.push(ident("Repeater").into());
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
        root_args: generics.map(identity_args).unwrap_or_default(),
        bound_self: None,
        type_self: None,
    };
    let host = matches!(item, Item::Trait(_)).then(|| ident(&format!("Host{}", args.id)));
    let mut self_ty = None;
    match item {
        Item::Struct(_) | Item::Enum(_) | Item::Union(_) => {
            let (name, generics) = match item {
                Item::Struct(i) => (&i.ident, &i.generics),
                Item::Enum(i) => (&i.ident, &i.generics),
                Item::Union(i) => (&i.ident, &i.generics),
                _ => unreachable!(),
            };
            let args = identity_args(generics);
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
            let rewritten = referrer.expand(context, (*i.self_ty).clone(), |_, ix| {
                projector.project(ix).unwrap_or_else(|e| {
                    *error.borrow_mut() = Some(e);
                    parse_quote!(())
                })
            });
            if let Some(e) = error.into_inner() {
                return Err(e);
            }
            projector.type_self = Some(rewritten);
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
        let visitor = referrer.clone().visitor(context, |_, ix| {
            projector.project(ix).unwrap_or_else(|e| {
                *error.borrow_mut() = Some(e);
                parse_quote!(())
            })
        });
        // The item of a scope `mod` is a nested item, which has a `Self` of its own.
        let visitor = match (self_ty, args.scope == "mod") {
            (Some(self_ty), false) => visitor.with_self(self_ty),
            _ => visitor,
        };
        // A struct, an enum, a union or a trait nested in the module (the item of a scope `mod`)
        // has a `Self` of its own, whose entries the visitor finds.
        let mut visitor: Box<dyn VisitMut + '_> = if args.no_nested_self {
            Box::new(visitor)
        } else {
            Box::new(visitor.with_nested_self(|nested, _, ix| {
                projector
                    .project_nested(nested, &base, ix)
                    .unwrap_or_else(|e| {
                        *error.borrow_mut() = Some(e);
                        parse_quote!(())
                    })
            }))
        };
        match &mut rewritten {
            other if args.scope == "mod" => {
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
    if let Some(e) = error.into_inner() {
        return Err(e);
    }
    if let (true, Item::Trait(original), Item::Trait(copy)) = (args.redeclare, item, &mut rewritten)
    {
        if args.scope != "mod" {
            return Err(Error::new(
                Span::call_site(),
                "test macro: `redeclare` is for the traits of a scope `mod`",
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
    })
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
        .clone()
        .visitor(context, |_, ix| {
            placeholder.project(ix).unwrap_or_else(|e| {
                *error.borrow_mut() = Some(e);
                parse_quote!(())
            })
        })
        .with_nested_self(|nested, _, ix| {
            placeholder
                .project_nested(nested, base, ix)
                .unwrap_or_else(|e| {
                    *error.borrow_mut() = Some(e);
                    parse_quote!(())
                })
        })
        .visit_file_mut(&mut file);
    if let Some(e) = error.into_inner() {
        return Err(e);
    }
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
/// name it (`clean` copies of `allow_values()` and `allow_default()`).
fn drop_relative_uses(items: &mut Vec<Item>, constants: bool) {
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
                || (constants && renames_ilen(&u.tree))
        }
        _ => true,
    });
    for item in items.iter_mut() {
        if let Item::Mod(m) = item {
            if let Some((_, items)) = &mut m.content {
                drop_relative_uses(items, constants);
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
/// tell the copy's `Self` from the Leaker's; `check_nested_self_is_required()` and the fixtures of
/// `tests/cross_crate.rs` do.) The copy of a struct that is not nested has its `Self` replaced
/// (`Visitor::with_self()`).
fn host_arg(rewritten: &Rewritten, args: &Args, leaker_path: &Path) -> Option<Host> {
    if let Some(host) = &rewritten.host {
        return Some((host_type(host, rewritten.dyn_host), leaker_path.clone()));
    }
    if args.scope != "mod" {
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
) -> TokenStream {
    let (impl_generics, ty_generics, where_clause) = item.generics.split_for_impl();
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

/// Without `LeakerConfig::allow_primitive()`, `u8` is a relative path like any other (a module may
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

/// The type positions of the copy that can be checked, with their index.
///
/// `item` is the original, and `copy` the rewritten item. A position with an elided lifetime has no
/// witness. In a copy of a module, the names it declares mean its own items: only the types
/// which are fully rewritten or absolute are compared.
fn comparable(
    copy: &Item,
    item: &Item,
    args: &Args,
    host: Option<&Host>,
) -> Vec<(usize, Position)> {
    let in_module = matches!(copy, Item::Mod(_));
    let original = collect_positions(item, None, args);
    let copied = collect_positions(copy, host, args);
    assert_eq!(original.len(), copied.len(), "test macro: positions differ");
    copied
        .into_iter()
        .zip(original)
        .enumerate()
        .filter(|(_, (pos, orig))| {
            if has_no_witness(orig) {
                return false;
            }
            if !in_module {
                return true;
            }
            let generics: HashSet<String> = pos
                .layers
                .iter()
                .flat_map(|g| g.params.iter())
                .map(|param| match param {
                    GenericParam::Lifetime(l) => l.lifetime.to_string(),
                    GenericParam::Type(t) => t.ident.to_string(),
                    GenericParam::Const(c) => c.ident.to_string(),
                })
                .collect();
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
        })
        .map(|(j, (pos, _))| (j, pos))
        .collect()
}

pub(crate) fn check_fn(
    id: u64,
    j: usize,
    pos: &Position,
    target: &Path,
    host: Option<&Type>,
) -> TokenStream {
    let mut generics = merge_generics(&pos.layers);
    let mut ty = pos.ty.clone();
    if let Some(sub) = &pos.self_sub {
        SelfSubst(sub).visit_type_mut(&mut ty);
        SelfSubst(sub).visit_generics_mut(&mut generics);
    }
    let mut targs: Vec<TokenStream> = generics
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
        .collect();
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
                clean(item.note().unwrap_or("")),
                item.reason,
                cause.split('(').next().unwrap_or("")
            )
        })
        .collect::<Vec<_>>()
        .join("\u{1d}")
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
/// macro of the Referrer, which copies the part that conflicts as written), with the status `acc`,
/// if `LeakerResult::allow(cause)` for the cause of every error, and then of the nested errors that
/// come up, drops every error ([`crate::allow_every_cause()`]; else `accfail`, and the remaining
/// errors follow the first ones: it cannot happen). The errors are what the
/// report says (and what `family()` of the harness explains).
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
        args.relaxed_sized,
        &first,
        |twin| analysis(twin, &args),
    )?;
    // The errors of the analysis (none if it succeeded).
    let errors: Vec<ErrorItem> = result.errors().to_vec();
    if !errors.is_empty() {
        // With `accept`, the cause of every error is allowed, which must drop every error: the
        // Referrer is the recovered one either way.
        let mut allowed = result.clone();
        crate::allow_every_cause(&mut allowed);
        let remaining = allowed.errors().to_vec();
        if mode != Mode::Result && (!accept || !remaining.is_empty()) {
            // The item is not leaked: the Leaker's crate still has it.
            let status = if accept { "accfail" } else { "rej" };
            let reason = crate::messages(&errors).replace(['\n', '|'], " ");
            let report = format!("{}|{}|{}|{}", id, kind, status, reason);
            let mut errors = rejection_text(&errors);
            if accept {
                errors.push('\u{1c}');
                errors.push_str(&rejection_text(&remaining));
            }
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
    let referrer = result.into_referrer();
    let carried = carried.unwrap_or_else(|| referrer.to_token_stream());

    let status = if args.accepted { "acc" } else { "ok" };
    let errors = rejection_text(&errors);
    // Count what the Referrer will check, by rewriting a copy here too.
    let counted = count_copy(&referrer, context, &original, &args, status, twin)?;
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

/// The witnesses of the type positions of `original`, each with the path of the module it is written
/// in.
fn witnesses_of(original: &Item, args: &Args) -> Vec<(Vec<Ident>, TokenStream)> {
    collect_positions(original, None, args)
        .iter()
        .enumerate()
        .filter(|(_, pos)| !has_no_witness(pos))
        .map(|(j, pos)| (pos.mods.clone(), witness_fn(args.id, j, pos)))
        .collect()
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

/// What the macro writes for the copy that a Referrer makes of `original` with `referrer`: the
/// Repeater's impls of its entries, and the report of what the macro counted.
struct Counted {
    impls: Impls,
    report: String,
}

/// Counts what the Referrer will check of the copy of `original` by rewriting a copy here too: the
/// entries, the type positions, those that a witness compares (`status` is `ok`, or `acc` for an
/// item accepted by `allow()`). Fails if the item has a shape that the macro cannot write.
fn count_copy(
    referrer: &Referrer,
    context: ContextId,
    original: &Item,
    args: &Args,
    status: &str,
    twin: bool,
) -> Result<Counted> {
    let (id, kind) = (args.id, kind_name(original));
    check_nested_self_is_required(referrer, original, args)?;
    let (impls, skipped) =
        repeater_impls(referrer, args.repeater_id(), args.strict.unwrap_or(true))?;
    let positions = collect_positions(original, None, args);
    let dummy: Path = parse_quote!(::__dummy);
    let counted = if args.strict == Some(true) {
        referrer.clone().allow_ambiguous_crate_name(false)
    } else {
        referrer.clone()
    };
    let report = match rewrite(&counted, context, original, args, &dummy, false) {
        Ok(rewritten) => {
            let host = host_arg(&rewritten, args, &dummy);
            // The copy of an item accepted by `allow()` keeps the part that conflicts as written, and
            // so a primitive in it that the Leaker does not allow.
            if !args.accepted {
                check_primitives_replaced(&rewritten.item, original, args, host.as_ref())?;
            }
            let rewritten_positions = comparable(&rewritten.item, original, args, host.as_ref());
            format!(
                "{}|{}|{}|entries={} positions={} checked={} skipped={} twin={}",
                id,
                kind,
                status,
                referrer.iter().count(),
                positions.len(),
                rewritten_positions.len(),
                skipped,
                u8::from(twin)
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
        analyze_with(original, &plain_args, |config| rule.on_config(config));
    let plain_errors = plain.errors().to_vec();
    let mut after = plain.clone();
    rule.on_result(&mut after);
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
/// errors were allowed (and dropped them all), `accfail` if an error remains (the macro emits no
/// copy then) and `skip` if the rule does not matter. A macro error of the copy is no error of the
/// Leaker's crate: it is a `rewrite-error` of the report.
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
    let (mut result, context, assertions) = match analysis {
        Some(analysis) => analysis,
        None => return nothing(format!("{}|{}|skip|", id, kind), String::new()),
    };
    // The errors that the rule does not drop are allowed by their causes, and so are those that
    // come up from inside them.
    let errors: Vec<ErrorItem> = result.errors().to_vec();
    crate::allow_every_cause(&mut result);
    let remaining = result.errors().to_vec();
    if !remaining.is_empty() {
        let reason = crate::messages(&remaining).replace(['\n', '|'], " ");
        let mut text = rejection_text(&errors);
        text.push('\u{1c}');
        text.push_str(&rejection_text(&remaining));
        return nothing(format!("{}|{}|accfail|{}", id, kind, reason), text);
    }
    let status = if errors.is_empty() { "ok" } else { "acc" };
    // The Referrer's crate of the stage copies: the arguments of the macro say so.
    let mut copy_args = args.clone();
    copy_args.primitive = true;
    copy_args.prelude = true;
    copy_args.stages = false;
    copy_args.stage = rule.number(side);
    copy_args.accepted = true;
    let referrer = result.into_referrer();
    let text = rejection_text(&errors);
    let counted = match count_copy(&referrer, context, original, &copy_args, status, false) {
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
        expansion(quote!(, clean, constants))
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
enum Allow {
    /// `LeakerResult::allow_std_crates()`
    StdCrates,
    /// `LeakerResult::allow(cause)`: `trait Path` is a `Cause::Trait`, `type Ty` a `Cause::Type`.
    Cause(Cause),
}

impl Parse for Allow {
    fn parse(input: ParseStream) -> Result<Self> {
        if input.parse::<Option<Token![trait]>>()?.is_some() {
            return Ok(Allow::Cause(Cause::Trait(input.parse()?)));
        }
        if input.parse::<Option<Token![type]>>()?.is_some() {
            return Ok(Allow::Cause(Cause::Type(input.parse()?)));
        }
        let word: Ident = input.parse()?;
        match word.to_string().as_str() {
            "std_crates" => Ok(Allow::StdCrates),
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
    strict: bool,
    /// The host of a trait is `dyn Host`.
    dyn_host: bool,
    /// The invocation is in the Referrer's crate of the items accepted by `allow()` (`accept`),
    /// where the names that such an item copies as written resolve as in the Leaker's crate: the
    /// copy is made there and not in the other crate, see [`apply()`].
    accept: bool,
    /// What the invocation allows of the errors of the result.
    allow: Vec<Allow>,
    /// The crate of the invocation has none of the Leaker's names (`allow_stages`): a module copy
    /// drops its imports as the copy of an ordinary item does, and the names that the rule allows
    /// are the only ones to resolve.
    clean: bool,
    /// With `clean`: the rule allows the paths of values and defaults, so the module copy keeps the
    /// import of the constant they name (`ILEN`).
    constants: bool,
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
        // Flags after the path: `strict`, `dyn_host` (the trait is implemented for `dyn Host`,
        // an unsized type, instead of a unit struct), `accept` and `allow(..)`.
        let (mut strict, mut dyn_host, mut accept, mut clean) = (false, false, false, false);
        let mut constants = false;
        let mut allow = Vec::new();
        while input.parse::<Option<Token![,]>>()?.is_some() {
            let keyword: Ident = input.parse()?;
            match keyword.to_string().as_str() {
                "strict" => strict = true,
                "dyn_host" => dyn_host = true,
                "accept" => accept = true,
                "clean" => clean = true,
                "constants" => constants = true,
                "allow" => {
                    let content;
                    parenthesized!(content in input);
                    allow.extend(Punctuated::<Allow, Token![,]>::parse_terminated(&content)?);
                }
                _ => return Err(Error::new(
                    keyword.span(),
                    "expected `strict`, `dyn_host`, `accept`, `clean`, `constants` or `allow(..)`",
                )),
            }
        }
        Ok(Self {
            result,
            context,
            args,
            item,
            path,
            strict,
            dyn_host,
            accept,
            allow,
            clean,
            constants,
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
        strict,
        dyn_host,
        accept,
        allow,
        clean,
        constants,
    } = parse2(input)?;
    if args.accepted != accept {
        return Ok(TokenStream::new());
    }
    for allowed in allow {
        match allowed {
            Allow::StdCrates => result.allow_std_crates(),
            Allow::Cause(cause) => result.allow(cause),
        };
    }
    let errors = crate::report(result.errors());
    let referrer = result.into_referrer();
    // `strict` makes the Referrer replace the optional entries as well.
    if args.strict.is_some_and(|said| said != strict) {
        return Err(Error::new(
            Span::call_site(),
            "test macro: the item is invoked with another `strict` than the one it was analyzed for",
        ));
    }
    let referrer = if strict {
        referrer.allow_ambiguous_crate_name(false)
    } else {
        referrer
    };
    let rewritten = rewrite(&referrer, context, &item, &args, &path, dyn_host)?;
    let id = args.id;

    // The checks call the witnesses by the path of the module of the item, in the Leaker's crate
    // (the position of an item of a module starts with its name).
    let host = host_arg(&rewritten, &args, &path);
    let mut checks = TokenStream::new();
    for (j, pos) in comparable(&rewritten.item, &item, &args, host.as_ref()) {
        checks.extend(check_fn(
            id,
            j,
            &pos,
            &path,
            host.as_ref().map(|(ty, _)| ty),
        ));
    }
    let mut copy = rewritten.item.clone();
    let body = match &mut copy {
        Item::Trait(t) => {
            let host = rewritten.host.as_ref().unwrap();
            // The module of a scope `mod` declares the trait: its signatures name it as written
            // (`<Self as Tr<N>>::A`).
            let name = &t.ident;
            let import = (args.scope == "mod").then(|| quote!(use #path::#name;));
            match &item {
                Item::Trait(original) => {
                    let host_impl = host_impl(t, original, host, dyn_host, &path, args.bodies);
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
                drop_relative_uses(items, constants);
            }
            quote!(#m)
        }
        other => quote!(#other),
    };
    Ok(quote! {
        #errors
        #body
        #checks
    })
}
