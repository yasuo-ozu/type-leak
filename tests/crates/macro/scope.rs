//! Macros which analyze a fragment of an impl with the scope API, as `coinduction` and `newer-type`
//! do (tests/cross_crate.rs).
//!
//! `scope_leak!` (the Leaker's side) is given the header of an impl,
//!
//! ```ignore
//! scope_leak! {
//!     id = 7001, marker = M7001, allow_self = false,
//!     impl<'a, T: ::core::clone::Clone> crate::m::Tr<Rel<T>, &'a Rel2> for crate::m::Holder<T, Rel3> {}
//! }
//! ```
//!
//! whose trait and self type are resolved by name, so only their arguments are leaked:
//! `Leaker::enter_impl(&generics, &self_ty, allow_self, |view| { view.visit_path_arguments(..) .. })`.
//! If the Referrer has a required entry, it emits the marker type, the Repeater's implementation of
//! each required entry (`ImplSelf::Marker(&marker)` for an entry which does not name `Self`, and
//! `ImplSelf::Param(&ident)` for one that does), a witness for each type argument, and the exported
//! `macro_rules! __scope_apply_7001`, which carries the Referrer and the top-level context.
//!
//! In the Referrer's crate, `leaker::__scope_apply_7001!(path = ::leaker::m)` expands
//! `__scope_apply!`: it rewrites the self type, the trait path and the type arguments with
//! `Referrer::visitor(context, ..)`, whose closure writes the path through the Repeater with
//! `Referrer::entry_args_with(ix, &UseSite::identity(), ..)`, and emits
//!
//! - `fn __tl_chk_<id>_<j>() -> PhantomData<rewritten>`, which compiles only if the rewritten type
//!   is the original one (the witness of the Leaker's crate returns the original),
//! - a function whose bound is `<rewritten self type>: <rewritten trait>`, called with arguments,
//!   so rustc has to find the Leaker's real impl for the rewritten types, and
//! - `pub const SCOPE_<id>_SELF/_TRAIT/_TYPES: &str`, the tokens of the rewritten self type, trait
//!   path and type arguments.
//!
//! `scope_errors!` runs the same analysis and reports the errors (`Vec<ErrorItem>`) of it in a
//! `pub const`.

use crate::prop::{check_fn, witness_fn, Position, SelfSub};
use proc_macro2::{Span, TokenStream};
use syn::parse::{Parse, ParseStream};
use syn::spanned::Spanned;
use syn::visit_mut::VisitMut;
use syn::*;
use template_quote::{quote, ToTokens};
use type_leak::*;

/// Which scope the analysis enters.
#[derive(Clone, Copy, PartialEq)]
enum Scope {
    /// `enter_impl(&generics, &self_ty, allow_self, ..)`
    Impl,
    /// `enter_generics(&generics, ..)`, where `Self` has no meaning.
    Generics,
}

/// `key = value, .. impl<..> Trait<..> for Ty {}`
struct ScopeInput {
    id: u64,
    marker: Ident,
    allow_self: bool,
    primitive: bool,
    prelude: bool,
    /// Emit the Repeater's implementation of the optional entries too (for a strict Referrer).
    all: bool,
    scope: Scope,
    /// The name of the const of `scope_errors!`.
    name: Option<Ident>,
    item: ItemImpl,
}

impl Parse for ScopeInput {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut parsed = ScopeInput {
            id: 0,
            marker: Ident::new("Marker", Span::call_site()),
            allow_self: false,
            primitive: true,
            prelude: false,
            all: false,
            scope: Scope::Impl,
            name: None,
            item: parse_quote!(impl A for B {}),
        };
        while !input.peek(Token![impl]) {
            let key: Ident = input.parse()?;
            input.parse::<Token![=]>()?;
            match key.to_string().as_str() {
                "id" => parsed.id = input.parse::<LitInt>()?.base10_parse()?,
                "marker" => parsed.marker = input.parse()?,
                "name" => parsed.name = Some(input.parse()?),
                "allow_self" => parsed.allow_self = input.parse::<LitBool>()?.value,
                "primitive" => parsed.primitive = input.parse::<LitBool>()?.value,
                "prelude" => parsed.prelude = input.parse::<LitBool>()?.value,
                "all" => parsed.all = input.parse::<LitBool>()?.value,
                "scope" => {
                    let scope: Ident = input.parse()?;
                    parsed.scope = match scope.to_string().as_str() {
                        "impl" | "impl_" => Scope::Impl,
                        "generics" => Scope::Generics,
                        _ => {
                            return Err(Error::new(scope.span(), "expected `impl_` or `generics`"))
                        }
                    }
                }
                _ => return Err(Error::new(key.span(), "unknown key")),
            }
            input.parse::<Option<Token![,]>>()?;
        }
        parsed.item = input.parse()?;
        if parsed.item.trait_.is_none() {
            return Err(Error::new(
                parsed.item.span(),
                "expected an impl of a trait",
            ));
        }
        Ok(parsed)
    }
}

impl ScopeInput {
    fn trait_path(&self) -> &Path {
        &self.item.trait_.as_ref().unwrap().1
    }

    /// The arguments of the self type (the last segment of its path) and then those of the segments
    /// of the trait: the fragments which are leaked, as their names are resolved elsewhere.
    fn fragments(&self) -> Vec<&PathArguments> {
        let mut fragments = Vec::new();
        if let Type::Path(path) = &*self.item.self_ty {
            fragments.extend(path.path.segments.last().map(|seg| &seg.arguments));
        }
        fragments.extend(self.trait_path().segments.iter().map(|seg| &seg.arguments));
        fragments
    }

    /// The type arguments of the fragments, in order.
    fn types(&self) -> Vec<Type> {
        self.fragments()
            .into_iter()
            .flat_map(|arguments| match arguments {
                PathArguments::AngleBracketed(args) => args
                    .args
                    .iter()
                    .filter_map(|arg| match arg {
                        GenericArgument::Type(ty) => Some(ty.clone()),
                        _ => None,
                    })
                    .collect(),
                _ => Vec::new(),
            })
            .collect()
    }

    /// The analysis, as `coinduction` does it: a scope, in which the arguments are visited.
    fn analyze(&self) -> std::result::Result<(Referrer, ContextId), Vec<ErrorItem>> {
        let mut config = LeakerConfig::new();
        if self.primitive {
            config.allow_primitive();
        }
        if self.prelude {
            config.allow_std_prelude();
        }
        let mut leaker = Leaker::from_config(config);
        let visit = |view: &mut View<'_>| {
            for arguments in self.fragments() {
                view.visit_path_arguments(arguments);
            }
        };
        let (context, ()) = match self.scope {
            Scope::Impl => leaker.enter_impl(
                &self.item.generics,
                &self.item.self_ty,
                self.allow_self,
                visit,
            ),
            Scope::Generics => leaker.enter_generics(&self.item.generics, visit),
        };
        Ok((leaker.result().finish()?, context))
    }

    fn pairs(&self) -> TokenStream {
        let (id, marker) = (self.id, &self.marker);
        let (allow_self, primitive, prelude) = (self.allow_self, self.primitive, self.prelude);
        let all = self.all;
        quote!(id = #id, marker = #marker, allow_self = #allow_self, primitive = #primitive, prelude = #prelude, all = #all,)
    }
}

fn ident(name: &str) -> Ident {
    Ident::new(name, Span::call_site())
}

/// The position of the `j`-th type argument, the generics around it being those of the impl.
fn position(item: &ItemImpl, self_ty: &Type, ty: Type) -> Position {
    Position {
        mods: Vec::new(),
        layers: vec![item.generics.clone()],
        self_sub: Some(SelfSub {
            ty: self_ty.clone(),
            trait_ref: None,
            bound: false,
        }),
        ty,
    }
}

/// `scope_leak!`, the Leaker's side.
pub(crate) fn leak(input: TokenStream) -> Result<TokenStream> {
    let input: ScopeInput = parse2(input)?;
    let (referrer, context) = input.analyze().map_err(|errors| to_syn_error(&errors))?;
    let ScopeInput {
        id, marker, item, ..
    } = &input;
    let marker_ty: Type = parse_quote!(#marker);
    let self_param = ident("__S");
    let repeater: Path = parse_quote!(crate::Repeater);
    let slots = RepeaterSlots::new(repeater.clone());

    // The Repeater's implementation of each required entry, as `coinduction` writes them: only if
    // there is one (`has_required()`). An optional entry (the type of another crate) needs none
    // unless a strict Referrer replaces it (`all = true`).
    let wanted: Vec<usize> = if input.all {
        (0..referrer.iter().count()).collect()
    } else if referrer.has_required() {
        referrer.required().map(|(ix, _)| ix).collect()
    } else {
        Vec::new()
    };
    let mut impls = TokenStream::new();
    // The head of each impl, as text, for the tests.
    let mut heads: Vec<String> = Vec::new();
    for &ix in &wanted {
        let target = match referrer.context_self(referrer.context(ix)) {
            // The entry does not name `Self`: the impl is for the marker.
            None => ImplSelf::Marker(&marker_ty),
            // It does: the impl is for the self type, or for a parameter bounded by the trait.
            Some(_) => ImplSelf::Param(&self_param),
        };
        crate::check_plain(&referrer, ix, &slots)?;
        let r = referrer
            .repeater_impl_with(ix, target, |value, param| {
                slots.encode_const(value, &param.ty)
            })
            .map_err(Error::from)?;
        let (impl_generics, _, where_clause) = r.generics.split_for_impl();
        let (args, self_ty, ty) = (&r.args, &r.self_ty, &r.ty);
        let head =
            quote!(impl #impl_generics #repeater<#args, #id, #ix> for #self_ty #where_clause);
        heads.push(head.to_string());
        impls.extend(quote! {
            #head {
                type Type = #ty;
            }
        });
    }
    let n_impls = wanted.len();

    let mut witnesses = TokenStream::new();
    let types = input.types();
    for (j, ty) in types.iter().enumerate() {
        witnesses.extend(witness_fn(
            *id,
            j,
            &position(item, &item.self_ty, ty.clone()),
        ));
    }

    let report = format!(
        "{}|entries={} required={} impls={} types={}",
        id,
        referrer.iter().count(),
        referrer.required().count(),
        n_impls,
        types.len()
    );
    let heads = heads.join(" ;; ");
    let report_name = ident(&format!("__scope_report_{}", id));
    let impls_name = ident(&format!("__scope_impls_{}", id));
    let apply_name = ident(&format!("__scope_apply_{}", id));
    let dollar = proc_macro2::Punct::new('$', proc_macro2::Spacing::Alone);
    let pairs = input.pairs();
    Ok(quote! {
        #[allow(dead_code)]
        pub struct #marker;
        #impls
        #witnesses
        #[doc(hidden)]
        #[macro_export]
        macro_rules! #report_name { () => { #report }; }
        #[doc(hidden)]
        #[macro_export]
        macro_rules! #impls_name { () => { #heads }; }
        #[doc(hidden)]
        #[macro_export]
        macro_rules! #apply_name {
            (#dollar(#dollar t:tt)*) => {
                #dollar crate::__macro::__scope_apply! {
                    { #referrer , #context }
                    { #pairs #item }
                    #dollar(#dollar t)*
                }
            };
        }
    })
}

/// `scope_errors!`: the same analysis, which must fail; emits `pub const NAME: &[(&str, &str,
/// &[(&str, &str)])]`, the message and the source text of the span of every
/// error, with its notes (the note and the source text of its span).
pub(crate) fn errors(input: TokenStream) -> Result<TokenStream> {
    let input: ScopeInput = parse2(input)?;
    let name = input
        .name
        .clone()
        .ok_or_else(|| Error::new(Span::call_site(), "`name = ..` is needed"))?;
    let err = match input.analyze() {
        Ok(_) => {
            return Err(Error::new(
                Span::call_site(),
                "the analysis was expected to fail",
            ))
        }
        Err(err) => err,
    };
    let mut rows = TokenStream::new();
    for item in &err {
        let message = &item.message;
        let source = item.span().source_text().unwrap_or_default();
        let tokens = item.cause.to_token_stream().to_string();
        let mut notes = TokenStream::new();
        for (tokens, note) in &item.notes {
            let at = tokens.span().source_text().unwrap_or_default();
            notes.extend(quote!((#note, #at),));
        }
        rows.extend(quote!((#message, #tokens, #source, &[#notes]),));
    }
    // The error is also what a macro reports as `compile_error!`s: one for each item.
    let rendered = to_syn_error(&err);
    let n_compile_errors = rendered.into_iter().count();
    let (names, count) = (ident(&format!("{}_COMPILE_ERRORS", name)), n_compile_errors);
    Ok(quote! {
        pub const #name: &[(&str, &str, &str, &[(&str, &str)])] = &[#rows];
        pub const #names: usize = #count;
    })
}

struct ApplyInput {
    referrer: Referrer,
    context: ContextId,
    scope: ScopeInput,
    path: Path,
    strict: bool,
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
        let scope = {
            let content;
            braced!(content in input);
            content.parse()?
        };
        let key: Ident = input.parse()?;
        if key != "path" {
            return Err(Error::new(key.span(), "expected `path`"));
        }
        input.parse::<Token![=]>()?;
        let path = Path::parse_mod_style(input)?;
        let strict = input.parse::<Option<Token![,]>>()?.is_some() && {
            let keyword: Ident = input.parse()?;
            if keyword != "strict" {
                return Err(Error::new(keyword.span(), "expected `strict`"));
            }
            true
        };
        Ok(Self {
            referrer,
            context,
            scope,
            path,
            strict,
        })
    }
}

/// Makes the paths that start with `crate` (the Leaker's crate) absolute, as the Referrer's crate
/// names the Leaker's.
struct Absolutize<'a>(&'a Ident);

impl VisitMut for Absolutize<'_> {
    fn visit_path_mut(&mut self, i: &mut Path) {
        if i.leading_colon.is_none() && i.segments.first().map_or(false, |seg| seg.ident == "crate")
        {
            i.leading_colon = Some(Default::default());
            i.segments[0].ident = self.0.clone();
        }
        syn::visit_mut::visit_path_mut(self, i)
    }
}

/// `__scope_apply!`, the Referrer's side.
pub(crate) fn apply(input: TokenStream) -> Result<TokenStream> {
    let ApplyInput {
        referrer,
        context,
        scope,
        path,
        strict,
    } = parse2(input)?;
    let referrer = if strict {
        referrer.allow_ambiguous_crate_name(false)
    } else {
        referrer
    };
    let krate = path
        .segments
        .first()
        .map(|seg| seg.ident.clone())
        .ok_or_else(|| Error::new(path.span(), "expected a path"))?;
    let (id, item) = (scope.id, &scope.item);
    let mut repeater = path.clone();
    repeater.segments = repeater.segments.iter().take(1).cloned().collect();
    repeater.segments.push(ident("Repeater").into());
    let mut marker_path = path.clone();
    marker_path.segments.push(scope.marker.clone().into());
    let marker_ty: Type = Type::Path(TypePath {
        qself: None,
        path: marker_path,
    });
    let slots = RepeaterSlots::new(repeater.clone());

    let mut absolutize = Absolutize(&krate);
    let mut self_ty: Type = (*item.self_ty).clone();
    absolutize.visit_type_mut(&mut self_ty);
    let mut trait_path = scope.trait_path().clone();
    absolutize.visit_path_mut(&mut trait_path);
    let mut types = scope.types();
    for ty in &mut types {
        absolutize.visit_type_mut(ty);
    }

    // The path of an entry through the Repeater: the impl is for the marker, or for the self type
    // if the entry names `Self`.
    let error = std::cell::RefCell::new(None);
    let project = |self_ty: Option<&Type>, ix: usize| -> Type {
        let result = (|| -> Result<Type> {
            let owner: Type = match referrer.context_self(referrer.context(ix)) {
                None => marker_ty.clone(),
                Some(_) => self_ty
                    .cloned()
                    .ok_or_else(|| Error::new(Span::call_site(), "the self type names `Self`"))?,
            };
            let site = UseSite::identity().with_self(owner.clone());
            let args = referrer.entry_args_with(ix, &site, |value, param| {
                slots.encode_const(value, &param.ty)
            })?;
            Ok(parse_quote!(<#owner as #repeater<#args, #id, #ix>>::Type))
        })();
        result.unwrap_or_else(|e| {
            *error.borrow_mut() = Some(e);
            parse_quote!(())
        })
    };

    // The self type first, as `Self` of an entry is written with it.
    referrer
        .clone()
        .visitor(context, |_, ix| project(None, ix))
        .visit_type_mut(&mut self_ty);
    let mut visitor = referrer
        .clone()
        .visitor(context, |_, ix| project(Some(&self_ty), ix));
    let mut types = types;
    for ty in &mut types {
        visitor.visit_type_mut(ty);
    }
    visitor.visit_path_mut(&mut trait_path);
    drop(visitor);
    if let Some(e) = error.into_inner() {
        return Err(e);
    }

    let mut checks = TokenStream::new();
    for (j, ty) in types.iter().enumerate() {
        checks.extend(check_fn(
            id,
            j,
            &position(item, &self_ty, ty.clone()),
            &path,
            None,
        ));
    }

    // The rewritten self type has the rewritten trait only if every type of the impl is the
    // original one: rustc finds the Leaker's impl.
    let (impl_generics, _, where_clause) = item.generics.split_for_impl();
    let predicates = where_clause.map(|w| &w.predicates);
    let arguments: Vec<TokenStream> = item
        .generics
        .params
        .iter()
        .map(|param| match param {
            GenericParam::Lifetime(_) => Ok(quote!('static)),
            GenericParam::Type(_) => Ok(quote!(u8)),
            GenericParam::Const(c) => {
                Err(Error::new(c.span(), "a const parameter is not supported"))
            }
        })
        .collect::<Result<_>>()?;
    let bound_fn = ident(&format!("__scope_bound_{}", id));
    let call_fn = ident(&format!("__scope_call_{}", id));

    let (self_text, trait_text) = (
        quote!(#self_ty).to_string(),
        quote!(#trait_path).to_string(),
    );
    let types_text = types
        .iter()
        .map(|ty| quote!(#ty).to_string())
        .collect::<Vec<_>>()
        .join(" ;; ");
    let (self_const, trait_const, types_const) = (
        ident(&format!("SCOPE_{}_SELF", id)),
        ident(&format!("SCOPE_{}_TRAIT", id)),
        ident(&format!("SCOPE_{}_TYPES", id)),
    );
    Ok(quote! {
        #checks
        #[allow(dead_code)]
        fn #bound_fn #impl_generics () where #self_ty: #trait_path, #predicates {}
        #[allow(dead_code)]
        fn #call_fn() { #bound_fn::<#(for argument in &arguments), { #argument }>() }
        pub const #self_const: &str = #self_text;
        pub const #trait_const: &str = #trait_text;
        pub const #types_const: &str = #types_text;
    })
}
