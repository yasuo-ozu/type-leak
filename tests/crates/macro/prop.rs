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
//! - a witness `pub fn __tl_w_17_<j><generics>() -> PhantomData<OriginalType>` for each type
//!   position `j` of the item (a field, a parameter, a return type, ...), and
//! - the exported `macro_rules! __tl_prop_17`, which carries the serialized `Referrer`, the
//!   `ContextId` and the item, and `__tl_report_17`, a report of the analysis.
//!
//! In the Referrer's crate, `leaker::__tl_prop_17!(path = ::leaker::m)` (`path` is the module that
//! contains the item) expands `__prop_apply!`, which rewrites a copy of the item with
//! `Referrer::visitor(context, ..)` and emits
//!
//! - the copy, for a struct, an enum, a union and a module,
//! - an implementation of the trait for a host type, with the rewritten signatures, for a trait,
//! - `fn __tl_chk_17_<j><generics>() -> PhantomData<RewrittenType> { path::__tl_w_17_<j>::<..>() }`
//!   for each type position, which compiles only if the rewritten type is exactly the original.
//!
//! The `Self` of a position is the type itself (a struct, an enum or a union), the self type (an
//! impl) or the host type (a trait). A module cannot name `Self`, so its items do not.

use proc_macro2::{Punct, Spacing, Span, TokenStream};
use std::collections::HashSet;
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;
use syn::visit::Visit;
use syn::visit_mut::VisitMut;
use syn::*;
use template_quote::quote;
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
}

impl Parse for Args {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut args = Args {
            id: 0,
            allow_self: true,
            prelude: false,
            allow: Vec::new(),
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
        let allow = self
            .allow
            .iter()
            .map(|p| quote!(#p).to_string().replace(' ', ""))
            .collect::<Vec<_>>()
            .join(",");
        quote!(id = #id, allow_self = #allow_self, prelude = #prelude, allow = #allow)
    }
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
fn analyze(
    item: &Item,
    args: &Args,
) -> std::result::Result<(Referrer, ContextId, TokenStream), NotInternableError> {
    let mut config = LeakerConfig::new();
    config.allow_primitive();
    if args.prelude {
        config.allow_std_prelude();
    }
    config.allowed_paths.extend(args.allow.iter().cloned());
    let mut leaker = Leaker::from_config(config);
    let allow_self = args.allow_self;
    let context = match item {
        Item::Struct(i) => leaker.intern_struct(i, allow_self)?,
        Item::Enum(i) => leaker.intern_enum(i, allow_self)?,
        Item::Union(i) => leaker.intern_union(i, allow_self)?,
        Item::Trait(i) => leaker.intern_trait(i, allow_self)?,
        Item::Impl(i) => leaker.intern_impl(i, allow_self)?,
        Item::Mod(i) => leaker.intern_mod(i)?,
        _ => panic!("unsupported item"),
    };
    let assertions = leaker.assertions();
    Ok((leaker.finish(), context, assertions))
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

/// The Repeater's implementation of each entry with the path of the module it is keyed by,
/// relative to the analyzed module (`[]` for it and for the entries of no module).
fn repeater_impls(referrer: &Referrer, id: u64) -> Result<Vec<(Vec<Ident>, TokenStream)>> {
    let repeater: Path = parse_quote!(crate::Repeater);
    let slots = RepeaterSlots::new(repeater.clone());
    // For `Self` bounded by the trait, the implementation is for `__S` (the implementor).
    let self_param = ident("__S");
    let mut impls = Vec::new();
    for ix in 0..referrer.iter().count() {
        crate::check_plain(referrer, ix, &slots)?;
        let r = referrer
            .repeater_impl_with(ix, ImplSelf::Param(&self_param), |value, param| {
                slots.encode_const(value, &param.ty)
            })
            .map_err(Error::from)?;
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
    Ok(impls)
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

fn collect_positions(item: &Item, host: Option<&Host>) -> Vec<Position> {
    let mut out = Vec::new();
    collect_item(item, &mut Vec::new(), &mut out, true, host);
    out
}

fn collect_item(
    item: &Item,
    mods: &mut Vec<Ident>,
    out: &mut Vec<Position>,
    with_self: bool,
    host: Option<&Host>,
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
    match item {
        Item::Struct(i) => {
            let sub = with_self.then(|| SelfSub {
                ty: type_with_params(&i.ident, &i.generics),
                trait_ref: None,
                bound: false,
            });
            for ty in fields_of(&i.fields) {
                push(&[&i.generics], &sub, &ty);
            }
        }
        Item::Enum(i) => {
            let sub = with_self.then(|| SelfSub {
                ty: type_with_params(&i.ident, &i.generics),
                trait_ref: None,
                bound: false,
            });
            for variant in &i.variants {
                for ty in fields_of(&variant.fields) {
                    push(&[&i.generics], &sub, &ty);
                }
            }
        }
        Item::Union(i) => {
            let sub = with_self.then(|| SelfSub {
                ty: type_with_params(&i.ident, &i.generics),
                trait_ref: None,
                bound: false,
            });
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
            for trait_item in &i.items {
                match trait_item {
                    TraitItem::Fn(f) => {
                        let layers = [&i.generics, &f.sig.generics];
                        for ty in sig_types(&f.sig) {
                            push(&layers, &sub, &ty);
                        }
                    }
                    TraitItem::Const(c) => push(&[&i.generics, &c.generics], &sub, &c.ty),
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
            for impl_item in &i.items {
                match impl_item {
                    ImplItem::Fn(f) => {
                        let layers = [&i.generics, &f.sig.generics];
                        for ty in sig_types(&f.sig) {
                            push(&layers, &sub, &ty);
                        }
                    }
                    ImplItem::Const(c) => push(&[&i.generics, &c.generics], &sub, &c.ty),
                    ImplItem::Type(t) => push(&[&i.generics, &t.generics], &sub, &t.ty),
                    _ => (),
                }
            }
        }
        Item::Fn(f) => {
            for ty in sig_types(&f.sig) {
                push(&[&f.sig.generics], &None, &ty);
            }
        }
        Item::Type(t) => push(&[&t.generics], &None, &t.ty),
        Item::Const(c) => push(&[], &None, &c.ty),
        Item::Mod(m) => {
            if let Some((_, items)) = &m.content {
                mods.push(m.ident.clone());
                for item in items {
                    collect_item(item, mods, out, false, host);
                }
                mods.pop();
            }
        }
        _ => (),
    }
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
    let mut predicates: Punctuated<WherePredicate, Token![,]> = Punctuated::new();
    for layer in layers {
        for param in &layer.params {
            if !matches!(param, GenericParam::Lifetime(_)) {
                params.push(param.clone());
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
    quote! {
        #[allow(non_snake_case, dead_code)]
        pub fn #name<#params>() -> ::core::marker::PhantomData<#ty> #where_clause {
            ::core::marker::PhantomData
        }
    }
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
            _ => parse_quote!(()),
        };
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
        id: args.id,
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
        let mut visitor = referrer.clone().visitor(context, |_, ix| {
            projector.project(ix).unwrap_or_else(|e| {
                *error.borrow_mut() = Some(e);
                parse_quote!(())
            })
        });
        if let Some(self_ty) = self_ty {
            visitor = visitor.with_self(self_ty);
        }
        match &mut rewritten {
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
    Ok(Rewritten {
        item: rewritten,
        host,
        dyn_host,
    })
}

/// Drops the imports of names the Leaker leaks: a copy of a module in another crate cannot resolve
/// them, and every use of them is rewritten. Imports of absolute paths and of the module's own
/// items stay.
fn drop_relative_uses(items: &mut Vec<Item>) {
    items.retain(|item| match item {
        Item::Use(u) => {
            u.leading_colon.is_some() || matches!(&u.tree, UseTree::Path(p) if p.ident == "self")
        }
        _ => true,
    });
    for item in items.iter_mut() {
        if let Item::Mod(m) = item {
            if let Some((_, items)) = &mut m.content {
                drop_relative_uses(items);
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

fn host_arg(rewritten: &Rewritten, leaker_path: &Path) -> Option<Host> {
    rewritten
        .host
        .as_ref()
        .map(|host| (host_type(host, rewritten.dyn_host), leaker_path.clone()))
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
) -> TokenStream {
    let (impl_generics, ty_generics, where_clause) = item.generics.split_for_impl();
    let name = &item.ident;
    let mut items = TokenStream::new();
    for (trait_item, original) in item.items.iter().zip(&original.items) {
        match (trait_item, original) {
            (TraitItem::Fn(f), _) => {
                let sig = &f.sig;
                items.extend(quote!(#sig { ::core::unimplemented!() }));
            }
            (TraitItem::Const(c), TraitItem::Const(original)) => {
                let (name, ty) = (&c.ident, &c.ty);
                let value = const_value(&original.ty);
                items.extend(quote!(const #name: #ty = #value;));
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
    let original = collect_positions(item, None);
    let copied = collect_positions(copy, host);
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

pub(crate) fn check_fn(id: u64, j: usize, pos: &Position, target: &Path, host: Option<&Type>) -> TokenStream {
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
    // for `&'a T`) that the original relies on.
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

/// `#[prop_leak(..)]`
pub(crate) fn leak(attr: TokenStream, item: TokenStream) -> Result<TokenStream> {
    let args: Args = parse2(attr)?;
    let original: Item = parse2(item)?;
    let id = args.id;
    let kind = kind_name(&original);
    let report_name = ident(&format!("__tl_report_{}", id));
    let mac_name = ident(&format!("__tl_prop_{}", id));
    let dollar = dollar();
    let (referrer, context, assertions) = match analyze(&original, &args) {
        Ok(result) => result,
        Err(e) => {
            // The item is not leaked: the Leaker's crate still has it.
            let reason = e.to_string().replace(['\n', '|'], " ");
            let report = format!("{}|{}|rej|{}", id, kind, reason);
            return Ok(quote! {
                #original
                #[doc(hidden)]
                #[macro_export]
                macro_rules! #report_name { () => { #report }; }
                #[doc(hidden)]
                #[macro_export]
                macro_rules! #mac_name { (#dollar(#dollar t:tt)*) => {}; }
            });
        }
    };

    let impls = repeater_impls(&referrer, id)?;
    let positions = collect_positions(&original, None);
    let witnesses: Vec<(Vec<Ident>, TokenStream)> = positions
        .iter()
        .enumerate()
        .filter(|(_, pos)| !has_no_witness(pos))
        .map(|(j, pos)| (pos.mods.clone(), witness_fn(id, j, pos)))
        .collect();

    // Count what the Referrer will check, by rewriting a copy here too.
    let dummy: Path = parse_quote!(::__dummy);
    let report = match rewrite(&referrer, context, &original, &args, &dummy, false) {
        Ok(rewritten) => {
            let host = host_arg(&rewritten, &dummy);
            let rewritten_positions = comparable(&rewritten.item, &original, &args, host.as_ref());
            format!(
                "{}|{}|ok|entries={} positions={} checked={}",
                id,
                kind,
                referrer.iter().count(),
                positions.len(),
                rewritten_positions.len()
            )
        }
        Err(e) => format!(
            "{}|{}|ok|entries={} positions={} rewrite-error={}",
            id,
            kind,
            referrer.iter().count(),
            positions.len(),
            e.to_string().replace(['\n', '|'], " ")
        ),
    };

    let mut emitted = original.clone();
    let (head, tail): (TokenStream, TokenStream) = match &mut emitted {
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
    };
    let args_tokens = args.tokens();
    Ok(quote! {
        #head
        #tail
        #[doc(hidden)]
        #[macro_export]
        macro_rules! #report_name { () => { #report }; }
        #[doc(hidden)]
        #[macro_export]
        macro_rules! #mac_name {
            (#dollar(#dollar t:tt)*) => {
                #dollar crate::__macro::__prop_apply! {
                    { #referrer , #context }
                    { #args_tokens }
                    { #original }
                    #dollar(#dollar t)*
                }
            };
        }
    })
}

/// Input of `__prop_apply!`.
struct ApplyInput {
    referrer: Referrer,
    context: ContextId,
    args: Args,
    item: Item,
    /// The module that contains the item, in the Leaker's crate.
    path: Path,
    strict: bool,
    /// The host of a trait is `dyn Host`.
    dyn_host: bool,
}

fn braced_parse<T: Parse>(input: ParseStream) -> Result<T> {
    let content;
    braced!(content in input);
    content.parse()
}

impl Parse for ApplyInput {
    fn parse(input: ParseStream) -> Result<Self> {
        let (referrer, context) = {
            let content;
            braced!(content in input);
            let referrer = content.parse()?;
            content.parse::<Token![,]>()?;
            (referrer, content.parse()?)
        };
        let args = braced_parse(input)?;
        let item = braced_parse(input)?;
        let key: Ident = input.parse()?;
        if key != "path" {
            return Err(Error::new(key.span(), "expected `path`"));
        }
        input.parse::<Token![=]>()?;
        let path = Path::parse_mod_style(input)?;
        // Flags after the path: `strict`, and `dyn_host` (the trait is implemented for `dyn Host`,
        // an unsized type, instead of a unit struct).
        let (mut strict, mut dyn_host) = (false, false);
        while input.parse::<Option<Token![,]>>()?.is_some() {
            let keyword: Ident = input.parse()?;
            match keyword.to_string().as_str() {
                "strict" => strict = true,
                "dyn_host" => dyn_host = true,
                _ => {
                    return Err(Error::new(
                        keyword.span(),
                        "expected `strict` or `dyn_host`",
                    ))
                }
            }
        }
        Ok(Self {
            referrer,
            context,
            args,
            item,
            path,
            strict,
            dyn_host,
        })
    }
}

/// `__prop_apply!`: the Referrer's side.
pub(crate) fn apply(input: TokenStream) -> Result<TokenStream> {
    let ApplyInput {
        referrer,
        context,
        args,
        item,
        path,
        strict,
        dyn_host,
    } = parse2(input)?;
    // `strict` makes the Referrer replace the optional entries as well.
    let referrer = if strict {
        referrer.allow_ambiguous_crate_name(false)
    } else {
        referrer
    };
    let rewritten = rewrite(&referrer, context, &item, &args, &path, dyn_host)?;
    let id = args.id;

    // The checks call the witnesses by the path of the module of the item, in the Leaker's crate
    // (the position of an item of a module starts with its name).
    let host = host_arg(&rewritten, &path);
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
            match &item {
                Item::Trait(original) => host_impl(t, original, host, dyn_host, &path),
                _ => unreachable!(),
            }
        }
        Item::Impl(_) => TokenStream::new(),
        Item::Mod(m) => {
            if let Some((_, items)) = &mut m.content {
                drop_relative_uses(items);
            }
            quote!(#m)
        }
        other => quote!(#other),
    };
    Ok(quote! {
        #body
        #checks
    })
}
