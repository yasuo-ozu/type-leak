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
//! The generics each leaked type uses (those of the trait and those of its methods, see
//! [`Referrer::generics()`]) are encoded into the Repeater's trait arguments with
//! [`RepeaterSlots`], so the Repeater (defined by [`repeater_definition()`]) is the only item
//! shared by the Leaker and the Referrer. The lifetime bounds in them (`T: 'a`) are encoded too
//! (`&'a T`), so the Repeater's implementations declare the generics without them, as
//! [`Referrer::impl_generics()`] returns them.
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

/// Analyzes a trait without allowing any relative path, reporting a `NotInternableError` with
/// its `ToTokens` implementation (for tests/diagnostic.rs).
#[proc_macro_attribute]
pub fn leak_strict(
    _attr: proc_macro::TokenStream,
    item: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let item_trait = parse_macro_input!(item as ItemTrait);
    match Leaker::from_trait(&item_trait) {
        Ok(_) => quote!(#item_trait).into(),
        Err(e) => quote!(#e).into(),
    }
}

/// Input of `check_leak!`: `{ expected } { config } item`.
struct CheckLeakInput {
    /// The leaked types, or the message of the error.
    expected: std::result::Result<Vec<String>, String>,
    allow_external_crates: bool,
    item: Item,
}

/// Prints a type as `to_string()` does, after parsing, so that the spacing is normalized.
fn type_string(ty: &Type) -> String {
    quote!(#ty).to_string()
}

impl Parse for CheckLeakInput {
    fn parse(input: ParseStream) -> Result<Self> {
        let expected_tokens: TokenStream = braced_parse(input)?;
        let expected = syn::parse::Parser::parse2(
            |input: ParseStream| -> Result<std::result::Result<Vec<String>, String>> {
                if input.peek(Ident) && input.peek2(Token![:]) && !input.peek2(Token![::]) {
                    let key: Ident = input.parse()?;
                    if key != "error" {
                        return Err(Error::new(key.span(), "expected `error`"));
                    }
                    input.parse::<Token![:]>()?;
                    let lit: LitStr = input.parse()?;
                    return Ok(Err(lit.value()));
                }
                let types = Punctuated::<Type, Token![,]>::parse_terminated(input)?;
                Ok(Ok(types.iter().map(type_string).collect()))
            },
            expected_tokens,
        )?;
        let config_tokens: TokenStream = braced_parse(input)?;
        let mut allow_external_crates = true;
        syn::parse::Parser::parse2(
            |input: ParseStream| -> Result<()> {
                while !input.is_empty() {
                    let key: Ident = input.parse()?;
                    input.parse::<Token![=]>()?;
                    let value: LitBool = input.parse()?;
                    match key.to_string().as_str() {
                        "allow_external_crates" => allow_external_crates = value.value,
                        _ => return Err(Error::new(key.span(), "unknown flag")),
                    }
                    let _: Option<Token![,]> = input.parse()?;
                }
                Ok(())
            },
            config_tokens,
        )?;
        Ok(Self {
            expected,
            allow_external_crates,
            item: input.parse()?,
        })
    }
}

/// Checks the leaked types (or the error) of the Leaker for an item, which may contain real
/// `$crate` tokens when invoked from a `macro_rules!`:
///
/// ```ignore
/// check_leak! {
///     { Rel, Other<u8> }                     // the leaked types, sorted, or `error: "message"`
///     { allow_external_crates = false }      // the config (`allow_primitive` is always set)
///     trait Tr { fn f(&self) -> $crate::m::Wrap<Rel>; } // a trait, a struct or a module
/// }
/// ```
///
/// It expands to nothing if the result is as expected, and to a `compile_error!` otherwise.
#[proc_macro]
pub fn check_leak(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let CheckLeakInput {
        expected,
        allow_external_crates,
        item,
    } = parse_macro_input!(input as CheckLeakInput);
    let mut leaker = Leaker::from_config({
        let mut config = LeakerConfig::new();
        config
            .allow_primitive()
            .allow_external_crates(allow_external_crates);
        config
    });
    let result = match &item {
        Item::Trait(item) => leaker.intern_with(&item.generics, |v| v.visit_item_trait(item)),
        Item::Struct(item) => leaker.intern_with(&item.generics, |v| v.visit_item_struct(item)),
        Item::Mod(item) => leaker.intern_with(&Generics::default(), |v| v.visit_item_mod(item)),
        _ => return quote!(compile_error!("unsupported item");).into(),
    };
    let got: std::result::Result<Vec<String>, String> = match result {
        Ok(_) => {
            let mut types: Vec<String> = leaker.finish().iter().map(type_string).collect();
            types.sort();
            Ok(types)
        }
        Err(e) => Err(e.to_string()),
    };
    let expected = match expected {
        Ok(mut types) => {
            types.sort();
            Ok(types)
        }
        Err(e) => Err(e),
    };
    if got == expected {
        return proc_macro::TokenStream::new();
    }
    let message = format!("expected {:?}, got {:?}", expected, got);
    quote!(compile_error!(#message);).into()
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
    let slots = RepeaterSlots::new(repeater.clone());

    // The Repeater's implementation of each type declares exactly the generics the type uses, and
    // encodes them. If the type mentions `Self`, it is implemented for the implementor of the
    // trait (bounded by the trait with its parameters) or for the type `Self` is.
    let mut repeater_impls = TokenStream::new();
    // For `Self` bounded by the trait, the implementation is for `__S` (the implementor), where
    // `Self` of the trait is not available: so the types and generics are those with `Self`
    // replaced by `__S` (see `Referrer::iter_with_self()`), e.g. `Self::Item` is `<__S>::Item`
    // and `Self: Sized` is `__S: Sized`. For `Self` as a type, the implementation is for that
    // type, where `Self` means it.
    let marker: Type = parse_quote!(__S);
    let replaced: Vec<(Type, Generics)> = referrer.iter_with_self(&marker).collect();
    for (ix, (ty, generics)) in referrer.iter_with_generics().enumerate() {
        let context = referrer.context(ix);
        let (ty, generics) = match referrer.context_self(context) {
            Some(SelfRequirement::Bound(_)) => (&replaced[ix].0, &replaced[ix].1),
            _ => (ty, generics),
        };
        let encoded = slots
            .encode_generics_params_to_ty(&generics.params)
            .map_err(Error::from)?;
        // The lifetime bounds in the trait arguments are implied by the header, so they are not
        // declared. For `Bound`, `generics` has `Self` replaced (see `iter_with_self()`).
        let impl_params: Vec<GenericParam> = match referrer.context_self(context) {
            Some(SelfRequirement::Bound(_)) => without_encoded_bounds(generics),
            _ => referrer.impl_generics(context),
        }
        .params
        .into_iter()
        .collect();
        let predicates = generics.where_clause.iter().flat_map(|wc| wc.predicates.iter());
        let header = match referrer.context_self(context) {
            Some(SelfRequirement::Bound(path)) => quote! {
                impl<#(#impl_params,)* __S: ?::core::marker::Sized + #module::#path>
                    #repeater<#encoded, #id, #ix> for __S
            },
            Some(SelfRequirement::Type(self_ty)) => quote! {
                impl<#(#impl_params),*> #repeater<#encoded, #id, #ix> for #self_ty
            },
            // `Call` is not tied to a type.
            _ => quote! {
                impl<#(#impl_params,)* __S: ?::core::marker::Sized>
                    #repeater<#encoded, #id, #ix> for __S
            },
        };
        repeater_impls.extend(quote! {
            #header
            where
                #(#predicates,)*
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
    /// Items of the implementation given verbatim: the associated types and consts of the trait,
    /// and the methods which are not forwarded.
    extra: Vec<ImplItem>,
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
        let mut extra = Vec::new();
        if input.peek(token::Brace) {
            let content;
            braced!(content in input);
            while !content.is_empty() {
                extra.push(content.parse()?);
            }
        }
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
            extra,
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
        extra,
    } = input;
    let params = &item_trait.generics.params;
    let trait_ident = &item_trait.ident;
    let slots = RepeaterSlots::new(repeater.clone());

    // The projection to the leaked type `ix`: the Repeater is encoded with the arguments of the
    // generics the type uses, which are those of the trait or are in scope with the same names (the
    // generics of the method).
    let projection = |ix: usize,
                      root_params: &Punctuated<GenericParam, Token![,]>,
                      root_args: &[GenericArgument]|
     -> Result<Type> {
        let args = referrer.generic_args(ix, root_params, root_args)?;
        let encoded = slots.encode_generics_args_to_ty(&args, &referrer.generics(ix).params)?;
        Ok(parse_quote!(<Self as #repeater<#encoded, #id, #ix>>::Type))
    };

    // Complete the arguments: an omitted argument takes the default, which is rewritten in the
    // Leaker's type context (it may be relative to the Leaker's module), and then the arguments
    // are encoded as in the Leaker.
    let mut encode_error = None;
    let completed = referrer
        .complete_args(&trait_args.args, params, |_, ix, prefix| {
            let before: Punctuated<GenericParam, Token![,]> =
                params.iter().take(prefix.len()).cloned().collect();
            projection(ix, &before, prefix).unwrap_or_else(|e| {
                encode_error = Some(e);
                parse_quote!(())
            })
        })
        .map_err(Error::from)?;
    if let Some(e) = encode_error {
        return Err(e);
    }
    // Map the trait's generic parameters to the complete arguments.
    let mut substitute = Substitute {
        types: HashMap::new(),
        consts: HashMap::new(),
        lifetimes: HashMap::new(),
    };
    for (param, arg) in params.iter().zip(&completed) {
        match (param, arg) {
            (GenericParam::Lifetime(param), GenericArgument::Lifetime(lt)) => {
                substitute.lifetimes.insert(param.lifetime.clone(), lt.clone());
            }
            (GenericParam::Type(param), GenericArgument::Type(ty)) => {
                substitute.types.insert(param.ident.clone(), ty.clone());
            }
            (GenericParam::Const(param), GenericArgument::Const(expr)) => {
                substitute.consts.insert(param.ident.clone(), expr.clone());
            }
            _ => return Err(Error::new(arg.span(), "unexpected generic argument")),
        }
    }

    // Projections to the leaked types. The generics of a method are in scope with the same names.
    let mut projections = Vec::new();
    for ix in 0..referrer.iter().count() {
        projections.push(projection(ix, params, &completed)?);
    }
    // What `Self` is in the trait: the implementor, bounded by the trait with its parameters.
    let trait_params: Vec<_> = params.iter().map(param_as_arg).collect();
    let self_path: Path = if trait_params.is_empty() {
        parse_quote!(#trait_ident)
    } else {
        parse_quote!(#trait_ident<#(#trait_params),*>)
    };

    let mut methods = TokenStream::new();
    for trait_item in &item_trait.items {
        let trait_fn = match trait_item {
            TraitItem::Fn(trait_fn) => trait_fn,
            // Given verbatim in `extra`.
            TraitItem::Type(_) | TraitItem::Const(_) => continue,
            _ => {
                return Err(Error::new(
                    trait_item.span(),
                    "only methods, types and consts are supported by this test macro",
                ))
            }
        };
        // A method given verbatim is not forwarded.
        if extra.iter().any(|item| matches!(item, ImplItem::Fn(f) if f.sig.ident == trait_fn.sig.ident)) {
            continue;
        }
        // 1. Replace the leaked types with placeholders, 2. substitute the trait's generic
        // parameters, and 3. replace the placeholders with the projections. Substituting before
        // the replacement would break the lookup of leaked types (keyed by the parameters'
        // names), and substituting after it would rewrite the projections.
        let mut sig = trait_fn.sig.clone();
        let mut visitor = referrer
            .clone()
            .into_visitor(Some(SelfRequirement::Bound(self_path.clone())), |_, ix| {
                parse_quote!(__TypeLeakTestPlaceholder<#ix>)
            });
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
            #(#extra)*
            #methods
        }
    })
}
