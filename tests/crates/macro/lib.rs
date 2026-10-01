//! A proc macro which uses `type-leak` as a downstream crate would (in the style of
//! `newer-type`), to test it across crate boundaries.
//!
//! In the Leaker's crate:
//!
//! ```ignore
//! pub use type_leak_test_macro as __macro;
//! type_leak_test_macro::define_repeater!(pub Repeater);
//!
//! pub mod m {
//!     pub struct MyType;
//!     #[type_leak_test_macro::leak(
//!         helper = crate::__macro, repeater = crate::Repeater, module = crate::m,
//!     )]
//!     pub trait MyTrait<const B: bool> { fn f(&self) -> MyType; }
//! }
//! ```
//!
//! The generics (including the local generics of each leaked type) are encoded into the
//! Repeater's trait arguments with [`RepeaterSlots`], so the Repeater (defined by
//! [`repeater_definition()`]) is the only item shared by the Leaker and the Referrer.
//!
//! In the Referrer's crate, `MyTrait!` implements `MyTrait` for a wrapper by forwarding each
//! method to the field `.0`, with the types in the signatures rewritten by the [`Referrer`]:
//!
//! ```ignore
//! pub struct Wrapper<const B: bool>(pub Inner);
//! leaker::m::MyTrait!(impl<const B: bool> MyTrait<B> for Wrapper<B> => 0);
//! ```

use proc_macro2::{Punct, Spacing, Span, TokenStream, TokenTree};
use std::collections::HashMap;
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;
use syn::visit_mut::VisitMut;
use syn::*;
use template_quote::quote;
use type_leak::*;

/// Arguments of `#[leak(..)]`.
struct LeakArgs {
    helper: Path,
    repeater: Path,
    module: Path,
}

impl Parse for LeakArgs {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut map: HashMap<String, Path> = HashMap::new();
        for kv in Punctuated::<MetaNameValue, Token![,]>::parse_terminated(input)? {
            let key = kv
                .path
                .get_ident()
                .map(|i| i.to_string())
                .unwrap_or_default();
            let value = match kv.value {
                Expr::Path(ExprPath { path, .. }) => path,
                other => return Err(Error::new(other.span(), "expected a path")),
            };
            map.insert(key, value);
        }
        let mut take = |key: &str| {
            map.remove(key)
                .ok_or_else(|| Error::new(Span::call_site(), format!("missing `{}`", key)))
        };
        Ok(Self {
            helper: take("helper")?,
            repeater: take("repeater")?,
            module: take("module")?,
        })
    }
}

/// Emits the Repeater defined by [`repeater_definition()`]: `define_repeater!(pub Repeater);`.
#[proc_macro]
pub fn define_repeater(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    struct Input(Visibility, Ident);
    impl Parse for Input {
        fn parse(input: ParseStream) -> Result<Self> {
            let vis = input.parse()?;
            let ident = input.parse()?;
            let _: Option<Token![;]> = input.parse()?;
            Ok(Self(vis, ident))
        }
    }
    let Input(vis, ident) = parse_macro_input!(input as Input);
    let def = repeater_definition(&vis, &ident);
    quote!(#def).into()
}

/// Returns a deterministic ID of the trait, used as the Repeater's `ID`.
fn trait_id(module: &Path, ident: &Ident) -> u64 {
    // FNV-1a
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in format!("{}::{}", quote!(#module), ident).bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Replaces the leading `crate` of `path` with `$crate`, so that it can be used in the exported
/// `macro_rules!`.
fn dollar_crate(path: &Path) -> TokenStream {
    let mut tokens = quote!(#path).into_iter().collect::<Vec<_>>();
    if matches!(tokens.first(), Some(TokenTree::Ident(i)) if i == "crate") {
        tokens.insert(0, TokenTree::Punct(Punct::new('$', Spacing::Alone)));
    }
    tokens.into_iter().collect()
}

/// Removes defaults from generic parameters, which are not allowed in `impl`s.
fn strip_defaults(param: &GenericParam) -> GenericParam {
    let mut param = param.clone();
    match &mut param {
        GenericParam::Type(tp) => {
            tp.eq_token = None;
            tp.default = None;
        }
        GenericParam::Const(cp) => {
            cp.eq_token = None;
            cp.default = None;
        }
        GenericParam::Lifetime(_) => (),
    }
    param
}

/// Sorts generic parameters so that lifetimes come first.
fn lifetimes_first<'a>(params: impl IntoIterator<Item = &'a GenericParam>) -> Vec<GenericParam> {
    let mut params: Vec<GenericParam> = params.into_iter().map(strip_defaults).collect();
    params.sort_by_key(|p| !matches!(p, GenericParam::Lifetime(_)));
    params
}

fn param_as_arg(param: &GenericParam) -> TokenStream {
    match param {
        GenericParam::Lifetime(l) => {
            let l = &l.lifetime;
            quote!(#l)
        }
        GenericParam::Type(t) => {
            let i = &t.ident;
            quote!(#i)
        }
        GenericParam::Const(c) => {
            let i = &c.ident;
            quote!(#i)
        }
    }
}

/// The Leaker side: `#[leak(..)]` on a trait.
#[proc_macro_attribute]
pub fn leak(
    attr: proc_macro::TokenStream,
    item: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let args = parse_macro_input!(attr as LeakArgs);
    let item_trait = parse_macro_input!(item as ItemTrait);
    match leak_impl(args, item_trait) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

fn leak_impl(args: LeakArgs, item_trait: ItemTrait) -> Result<TokenStream> {
    let mut leaker = Leaker::from_config({
        let mut config = LeakerConfig::new();
        config.allow_primitive().allow_std_prelude();
        config
    });
    leaker
        .intern_with(&item_trait.generics, |v| v.visit_item_trait(&item_trait))
        .map_err(Error::from)?;
    // Compile-time checks of the Leaker's assumptions (e.g. that `u8` is not shadowed).
    let assertions = leaker.assertions();
    leaker.reduce_roots();
    let referrer = leaker.finish();

    let LeakArgs {
        helper,
        repeater,
        module,
    } = &args;
    let id = trait_id(module, &item_trait.ident);
    let trait_ident = &item_trait.ident;
    let vis = &item_trait.vis;
    let params = &item_trait.generics.params;
    let trait_args: Vec<_> = params.iter().map(param_as_arg).collect();
    let slots = RepeaterSlots::new(repeater.clone());
    let encoded_params = slots
        .encode_generics_params_to_ty(params)
        .map_err(Error::from)?;
    let outer_predicates: Vec<WherePredicate> = item_trait
        .generics
        .where_clause
        .iter()
        .flat_map(|wc| wc.predicates.iter().cloned())
        .collect();

    let mut repeater_impls = TokenStream::new();
    for (ix, (ty, locals)) in referrer.iter_with_local_generics().enumerate() {
        let encoded_locals = slots
            .encode_generics_params_to_ty(&locals.params)
            .map_err(Error::from)?;
        let impl_params = lifetimes_first(params.iter().chain(locals.params.iter()));
        let local_predicates: Vec<&WherePredicate> = locals
            .where_clause
            .iter()
            .flat_map(|wc| wc.predicates.iter())
            .collect();
        repeater_impls.extend(quote! {
            impl<#(#impl_params,)* __S: ?::core::marker::Sized>
                #repeater<(#encoded_params, #encoded_locals), #id, #ix> for __S
            where
                __S: #module::#trait_ident<#(#trait_args),*>,
                #(#outer_predicates,)*
                #(#local_predicates,)*
            {
                type Type = #ty;
            }
        });
    }

    let mac_name = Ident::new(
        &format!("__type_leak_test_{}_{}", trait_ident, id),
        Span::call_site(),
    );
    let dollar = Punct::new('$', Spacing::Alone);
    let helper = dollar_crate(helper);
    let repeater = dollar_crate(repeater);
    let module = dollar_crate(module);
    Ok(quote! {
        #item_trait
        #assertions
        #repeater_impls
        #[doc(hidden)]
        #[macro_export]
        macro_rules! #mac_name {
            (#dollar(#dollar t:tt)*) => {
                #helper::__implement! {
                    { #item_trait }
                    { #referrer }
                    { #id }
                    { #repeater }
                    { #module }
                    #dollar(#dollar t)*
                }
            };
        }
        #[doc(hidden)]
        #vis use #mac_name as #trait_ident;
    })
}

/// Input of `__implement!`: the data passed by the exported `macro_rules!`, followed by
/// `impl<generics> Trait<args> for SelfTy => field`.
struct ImplementInput {
    item_trait: ItemTrait,
    referrer: Referrer,
    id: LitInt,
    repeater: Path,
    module: Path,
    generics: Generics,
    trait_args: AngleBracketedGenericArguments,
    self_ty: Type,
    field: Member,
}

fn braced_parse<T: Parse>(input: ParseStream) -> Result<T> {
    let content;
    braced!(content in input);
    content.parse()
}

impl Parse for ImplementInput {
    fn parse(input: ParseStream) -> Result<Self> {
        let item_trait: ItemTrait = braced_parse(input)?;
        let referrer = braced_parse(input)?;
        let id = braced_parse(input)?;
        let repeater = braced_parse::<Path>(input)?;
        let module = braced_parse::<Path>(input)?;
        input.parse::<Token![impl]>()?;
        let mut generics: Generics = input.parse()?;
        let trait_ident: Ident = input.parse()?;
        if trait_ident != item_trait.ident {
            return Err(Error::new(
                trait_ident.span(),
                format!("expected `{}`", item_trait.ident),
            ));
        }
        // `<..>` is optional for traits without generic parameters (then it is `<>`).
        let trait_args = if input.peek(Token![<]) {
            input.parse()?
        } else {
            AngleBracketedGenericArguments {
                colon2_token: None,
                lt_token: Default::default(),
                args: Punctuated::new(),
                gt_token: Default::default(),
            }
        };
        input.parse::<Token![for]>()?;
        let self_ty = input.parse()?;
        generics.where_clause = input.parse()?;
        input.parse::<Token![=>]>()?;
        let field = input.parse()?;
        Ok(Self {
            item_trait,
            referrer,
            id,
            repeater,
            module,
            generics,
            trait_args,
            self_ty,
            field,
        })
    }
}

/// Substitutes the trait's generic parameters with the arguments given by the Referrer.
struct Substitute {
    types: HashMap<Ident, Type>,
    consts: HashMap<Ident, Expr>,
    lifetimes: HashMap<Lifetime, Lifetime>,
}

impl VisitMut for Substitute {
    fn visit_type_mut(&mut self, i: &mut Type) {
        if let Type::Path(TypePath { qself: None, path }) = i {
            if let Some(ident) = path.get_ident() {
                if let Some(ty) = self.types.get(ident) {
                    *i = ty.clone();
                    return;
                }
            }
        }
        syn::visit_mut::visit_type_mut(self, i)
    }
    fn visit_generic_argument_mut(&mut self, i: &mut GenericArgument) {
        // A bare const parameter is parsed as a type argument, e.g. `N` in `Foo<N>`.
        if let GenericArgument::Type(Type::Path(TypePath { qself: None, path })) = i {
            if let Some(expr) = path.get_ident().and_then(|ident| self.consts.get(ident)) {
                *i = GenericArgument::Const(parse_quote!({ #expr }));
                return;
            }
        }
        syn::visit_mut::visit_generic_argument_mut(self, i)
    }
    fn visit_expr_mut(&mut self, i: &mut Expr) {
        if let Expr::Path(ExprPath {
            qself: None, path, ..
        }) = i
        {
            if let Some(expr) = path.get_ident().and_then(|ident| self.consts.get(ident)) {
                *i = parse_quote!({ #expr });
                return;
            }
        }
        syn::visit_mut::visit_expr_mut(self, i)
    }
    fn visit_lifetime_mut(&mut self, i: &mut Lifetime) {
        if let Some(lt) = self.lifetimes.get(i) {
            *i = lt.clone();
        }
    }
}

/// Replaces placeholder types `__TypeLeakTestPlaceholder<ix>` with the Repeater's projections.
struct Resolve<'a> {
    projections: &'a [Type],
}

impl VisitMut for Resolve<'_> {
    fn visit_type_mut(&mut self, i: &mut Type) {
        if let Type::Path(TypePath { qself: None, path }) = i {
            if path.segments.len() == 1 && path.segments[0].ident == "__TypeLeakTestPlaceholder" {
                if let PathArguments::AngleBracketed(args) = &path.segments[0].arguments {
                    if let Some(GenericArgument::Const(Expr::Lit(ExprLit {
                        lit: Lit::Int(ix),
                        ..
                    }))) = args.args.first()
                    {
                        *i = self.projections[ix.base10_parse::<usize>().unwrap()].clone();
                        return;
                    }
                }
            }
        }
        syn::visit_mut::visit_type_mut(self, i)
    }
}

/// Forwards the receiver of a method to the field `field`.
fn forward_receiver(receiver: &Receiver, field: &Member) -> Result<TokenStream> {
    let is_self = |ty: &Type| matches!(ty, Type::Path(TypePath { qself: None, path }) if path.is_ident("Self"));
    let ty = receiver.ty.as_ref();
    Ok(match ty {
        // `self`, `mut self`, `&self`, `&mut self` (also with a lifetime)
        _ if receiver.colon_token.is_none() => match &receiver.reference {
            Some(_) if receiver.mutability.is_some() => quote!(&mut self.#field),
            Some(_) => quote!(&self.#field),
            None => quote!(self.#field),
        },
        // `self: Self`, `self: &Self`, `self: &mut Self`
        _ if is_self(ty) => quote!(self.#field),
        Type::Reference(TypeReference {
            mutability, elem, ..
        }) if is_self(elem) => match mutability {
            Some(_) => quote!(&mut self.#field),
            None => quote!(&self.#field),
        },
        // `self: Box<Self>`
        Type::Path(TypePath { qself: None, path })
            if path.segments.last().map_or(false, |seg| {
                seg.ident == "Box"
                    && matches!(&seg.arguments, PathArguments::AngleBracketed(args)
                        if args.args.len() == 1
                            && matches!(&args.args[0], GenericArgument::Type(t) if is_self(t)))
            }) =>
        {
            quote!(::std::boxed::Box::new((*self).#field))
        }
        _ => {
            return Err(Error::new(
                ty.span(),
                "this receiver type is not supported by this test macro",
            ))
        }
    })
}

/// The Referrer side, invoked through the `macro_rules!` exported by `#[leak(..)]`.
#[doc(hidden)]
#[proc_macro]
pub fn __implement(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    match implement_impl(parse_macro_input!(input as ImplementInput)) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

fn implement_impl(input: ImplementInput) -> Result<TokenStream> {
    let ImplementInput {
        item_trait,
        referrer,
        id,
        repeater,
        module,
        generics,
        trait_args,
        self_ty,
        field,
    } = input;
    let params = &item_trait.generics.params;
    let trait_ident = &item_trait.ident;
    let slots = RepeaterSlots::new(repeater.clone());
    let encoded_args = slots
        .encode_generics_args_to_ty(&trait_args.args, params)
        .map_err(Error::from)?;

    // Map the trait's generic parameters to the given arguments.
    let mut substitute = Substitute {
        types: HashMap::new(),
        consts: HashMap::new(),
        lifetimes: HashMap::new(),
    };
    let mut lifetime_params = params.iter().filter_map(|p| match p {
        GenericParam::Lifetime(l) => Some(&l.lifetime),
        _ => None,
    });
    let mut other_params = params
        .iter()
        .filter(|p| !matches!(p, GenericParam::Lifetime(_)));
    for arg in &trait_args.args {
        match arg {
            GenericArgument::Lifetime(lt) => {
                if let Some(param) = lifetime_params.next() {
                    substitute.lifetimes.insert(param.clone(), lt.clone());
                }
            }
            GenericArgument::Type(ty) => match other_params.next() {
                Some(GenericParam::Type(tp)) => {
                    substitute.types.insert(tp.ident.clone(), ty.clone());
                }
                Some(GenericParam::Const(cp)) => {
                    substitute
                        .consts
                        .insert(cp.ident.clone(), parse_quote!(#ty));
                }
                _ => return Err(Error::new(arg.span(), "too many generic arguments")),
            },
            GenericArgument::Const(expr) => match other_params.next() {
                Some(GenericParam::Const(cp)) => {
                    substitute.consts.insert(cp.ident.clone(), expr.clone());
                }
                _ => return Err(Error::new(arg.span(), "unexpected const argument")),
            },
            _ => return Err(Error::new(arg.span(), "unsupported generic argument")),
        }
    }
    // Omitted arguments take the parameters' defaults.
    for param in other_params {
        match param {
            GenericParam::Type(TypeParam {
                ident,
                default: Some(ty),
                ..
            }) => {
                substitute.types.insert(ident.clone(), ty.clone());
            }
            GenericParam::Const(ConstParam {
                ident,
                default: Some(expr),
                ..
            }) => {
                substitute.consts.insert(ident.clone(), expr.clone());
            }
            _ => return Err(Error::new(trait_args.span(), "too few generic arguments")),
        }
    }

    // Projections to the leaked types. Each depends on the local generics of the method, which
    // are in scope with the same names.
    let mut projections = Vec::new();
    for (ix, (_, locals)) in referrer.iter_with_local_generics().enumerate() {
        // The local generics are in scope with the same names, so they are encoded as parameters.
        let encoded_locals = slots
            .encode_generics_params_to_ty(&locals.params)
            .map_err(Error::from)?;
        projections.push(parse_quote! {
            <Self as #repeater<(#encoded_args, #encoded_locals), #id, #ix>>::Type
        });
    }

    let mut methods = TokenStream::new();
    for trait_item in &item_trait.items {
        let trait_fn = match trait_item {
            TraitItem::Fn(trait_fn) => trait_fn,
            _ => {
                return Err(Error::new(
                    trait_item.span(),
                    "only methods are supported by this test macro",
                ))
            }
        };
        // 1. Replace the leaked types with placeholders, 2. substitute the trait's generic
        // parameters, and 3. replace the placeholders with the projections. Substituting before
        // the replacement would break the lookup of leaked types (keyed by the parameters'
        // names), and substituting after it would rewrite the projections.
        let mut sig = trait_fn.sig.clone();
        let mut visitor = referrer
            .clone()
            .into_visitor(|_, ix| parse_quote!(__TypeLeakTestPlaceholder<#ix>));
        visitor.visit_signature_mut(&mut sig);
        substitute.visit_signature_mut(&mut sig);
        Resolve {
            projections: &projections,
        }
        .visit_signature_mut(&mut sig);

        // Forward to the field: `Trait::<args>::method::<generics>(&self.field, args..)`.
        let receiver = match sig.receiver() {
            Some(receiver) => forward_receiver(receiver, &field)?,
            None => {
                return Err(Error::new(
                    sig.span(),
                    "methods without a receiver are not supported",
                ))
            }
        };
        // The arguments may be patterns (e.g. `_` or `mut x`), so they are renamed.
        let mut arg_pats = Vec::new();
        for (n, input) in sig.inputs.iter_mut().enumerate() {
            if let FnArg::Typed(PatType { pat, .. }) = input {
                let ident = Ident::new(&format!("__arg{}", n), Span::call_site());
                *pat = Box::new(parse_quote!(#ident));
                arg_pats.push(ident);
            }
        }
        let turbofish: Vec<_> = sig
            .generics
            .params
            .iter()
            .filter(|p| !matches!(p, GenericParam::Lifetime(_)))
            .map(param_as_arg)
            .collect();
        let method = &sig.ident;
        methods.extend(quote! {
            #sig {
                #module::#trait_ident::#trait_args::#method::<#(#turbofish),*>(
                    #receiver, #(#arg_pats),*
                )
            }
        });
    }

    let (impl_generics, _, where_clause) = generics.split_for_impl();
    Ok(quote! {
        impl #impl_generics #module::#trait_ident #trait_args for #self_ty #where_clause {
            #methods
        }
    })
}
