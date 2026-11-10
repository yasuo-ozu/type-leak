//! The proc macros of the two-macro example of `type-leak`.
//!
//! The example is a trait defined in one crate and implemented for a wrapper in another:
//!
//! ```text
//!  repo (crate A)                              app (crate B)
//!  ---------------                             -------------
//!  #[export_trait]                             #[repo::forward(repo::Repo)]
//!  pub trait Repo<K> { .. }                    struct Cached(InMemory);
//!    |  analyzes the trait with type-leak         |  asks `repo` for the Referrer
//!    |  emits the Repeater impls                  v
//!    |  emits macro_rules! __tl_Repo  ----->  repo::__tl_Repo! { .. }
//!    v                                            |  adds `$crate::Repeater`, the id, the
//!  (the trait, as written)                        |  Referrer and the top-level ContextId
//!                                                 v
//!                                              repo_macros::__forward_finish! { .. }
//!                                                 writes `impl Repo<K> for Cached`, with
//!                                                 the types of the signatures replaced
//! ```
//!
//! What a macro of this kind has to decide, and where this example decides it:
//!
//! - The Repeater is a trait of crate A (`repo::Repeater`, at its root). It cannot be in the
//!   proc-macro crate (which exports only macros), and it cannot be in a third crate: crate A
//!   implements it for any `Self` (`impl<S: ?Sized> Repeater<..> for S`), which the orphan rule
//!   allows only for a local trait. The macros write the Repeater impls with `ToTokens` of
//!   `RepeaterImpl` (which writes the `KEY` itself) and the projection with
//!   `repeater_projection()`, not by hand.
//! - Crate B cannot be given the Referrer by crate A directly, because crate A does not know the
//!   input of B's macro. So crate A defines a `macro_rules!` which B calls; it adds what only A
//!   knows (`$crate::Repeater`, the id, the Referrer, the top-level `ContextId`, the trait) to the
//!   input and calls the proc macro of crate B's side. `repo` re-exports this crate, so that B
//!   needs `repo` alone.
//! - The id of a trait (the high half of the `KEY` of the Repeater, `repeater_key(id, ix)`) is a
//!   hash of the name of the crate and of the trait. A proc macro does not know the path of the
//!   item, and it needs only a number that no other exported trait of the crate has. It is written
//!   into both the impls and the macro of A, so both sides agree.
//! - The Repeater impls are written for all the entries (`LeakerResult::repeater_impls()`): crate
//!   B's macro replaces every one. A type of another crate (`::foo::X`) is an entry too; to have
//!   the copy write it as it is, give the Leaker `Rules::external_crates()` (the crate must be the
//!   same crate on both sides).

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use syn::parse::{Parse, ParseStream};
use syn::visit_mut::VisitMut;
use syn::{
    parse_macro_input, parse_quote, GenericArgument, GenericParam, Ident, ImplItem, ItemImpl,
    ItemStruct, ItemTrait, Path, PathArguments, Token, TraitItem, Type,
};
use type_leak::{
    array_const, complete_params, repeater_projection, Context, ContextId, EncodeError, EntryId,
    ErrorItem, ImplSelf, Leaker, Referrer, Substitute,
};

// ---------------------------------------------------------------------------------------------
// Crate A: `#[export_trait]`

/// Analyzes the trait, and writes it back with the Repeater impls for its leaked types and the
/// macro `__tl_<Trait>!` that crate B calls (see the documentation of the crate).
#[proc_macro_attribute]
pub fn export_trait(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let item_trait = parse_macro_input!(item as ItemTrait);

    // 1. Analyze the trait. The `ContextId` names the scope of its generics and `Self`: crate B
    //    needs it to rewrite a copy. A type that names `Self` is leaked whole.
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_trait(&item_trait);
    let result = leaker.result();
    if !result.errors().is_empty() {
        return report(result.errors()).into();
    }

    // 2. One Repeater impl for each entry, written next to the trait, where the relative paths of
    //    the entries mean what they mean in the trait.
    let id = id_of(&item_trait);
    let repeater: Path = parse_quote!(crate::Repeater);
    let self_param: syn::Ident = parse_quote!(__S);
    let mut impls: Vec<ItemImpl> = vec![];
    let repeater_impls =
        match result.repeater_impls(&repeater, id, ImplSelf::Param(&self_param), array_const) {
            Ok(impls) => impls,
            Err(error) => {
                return syn::Error::new(error.span(), error)
                    .to_compile_error()
                    .into()
            }
        };
    // `ToTokens` writes each impl with its `KEY`.
    for r in repeater_impls {
        impls.push(parse_quote!(#r));
    }

    // The Referrer goes to crate B.
    let referrer = result.into_referrer();

    // 3. The macro that crate B calls. `$crate` is crate A, whatever the name B gives it.
    let macro_name = format_ident!("__tl_{}", item_trait.ident);
    quote! {
        #item_trait
        #(#impls)*
        #[doc(hidden)]
        #[macro_export]
        macro_rules! #macro_name {
            ($($input:tt)*) => {
                $crate::__macros::__forward_finish! {
                    $crate::Repeater, #id, #referrer, #top_level, #item_trait ; $($input)*
                }
            };
        }
    }
    .into()
}

/// Reports every offending path of the trait at its own span, with the notes (why) and the help
/// (what to write instead).
fn report(items: &[ErrorItem]) -> TokenStream2 {
    let mut errors: Option<syn::Error> = None;
    for item in items {
        let mut message = item.message.clone();
        for (_at, note) in &item.notes {
            message += &format!("\n  note: {}", note);
        }
        if let Some((_at, help)) = &item.help {
            message += &format!("\n  help: {}", help);
        }
        let error = syn::Error::new(item.span(), message);
        match &mut errors {
            Some(errors) => errors.combine(error),
            None => errors = Some(error),
        }
    }
    errors.expect("the error has items").to_compile_error()
}

fn id_of(item: &ItemTrait) -> u64 {
    let mut hasher = DefaultHasher::new();
    std::env::var("CARGO_CRATE_NAME")
        .unwrap_or_default()
        .hash(&mut hasher);
    item.ident.to_string().hash(&mut hasher);
    hasher.finish()
}

// ---------------------------------------------------------------------------------------------
// Crate B: `#[forward(path::Trait)]` or `#[forward(path::Trait<Args>)]`

/// Implements the trait for a wrapper `struct W(Inner);` by calling `Inner`.
///
/// With `#[forward(repo::Repo)]` the impl is generic over the generics of the trait
/// (`impl<K> Repo<K> for W`). With `#[forward(repo::Repo<u8>)]` it is for these arguments.
#[proc_macro_attribute]
pub fn forward(attr: TokenStream, item: TokenStream) -> TokenStream {
    let trait_path = parse_macro_input!(attr as Path);
    let item = TokenStream2::from(item);
    // `repo::Repo<u8>` is exported by `repo::__tl_Repo!`.
    let mut macro_path = trait_path.clone();
    let last = macro_path
        .segments
        .last_mut()
        .expect("a path has a segment");
    last.ident = format_ident!("__tl_{}", last.ident);
    last.arguments = PathArguments::None;
    quote!(#macro_path! { #trait_path ; #item }).into()
}

/// The input that `repo::__tl_<Trait>!` gives, then the arguments of `#[forward(..)]` and the item.
struct Finish {
    repeater: Path,
    id: u64,
    referrer: Referrer,
    top_level: ContextId,
    item_trait: ItemTrait,
    trait_path: Path,
    wrapper: ItemStruct,
}

impl Parse for Finish {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let repeater = input.parse()?;
        input.parse::<Token![,]>()?;
        let id = input.parse::<syn::LitInt>()?.base10_parse()?;
        input.parse::<Token![,]>()?;
        // `Referrer` and `ContextId` read what `ToTokens` wrote.
        let referrer = input.parse()?;
        input.parse::<Token![,]>()?;
        let top_level = input.parse()?;
        input.parse::<Token![,]>()?;
        let item_trait = input.parse()?;
        input.parse::<Token![;]>()?;
        let trait_path = input.parse()?;
        input.parse::<Token![;]>()?;
        let wrapper = input.parse()?;
        Ok(Finish {
            repeater,
            id,
            referrer,
            top_level,
            item_trait,
            trait_path,
            wrapper,
        })
    }
}

#[doc(hidden)]
#[proc_macro]
pub fn __forward_finish(input: TokenStream) -> TokenStream {
    let finish = parse_macro_input!(input as Finish);
    match forward_impl(finish) {
        Ok(tokens) => tokens.into(),
        Err(error) => error.to_compile_error().into(),
    }
}

fn forward_impl(finish: Finish) -> syn::Result<TokenStream2> {
    let Finish {
        repeater,
        id,
        referrer,
        top_level,
        item_trait,
        trait_path,
        wrapper,
    } = finish;
    let name = &wrapper.ident;
    let inner = match wrapper.fields.iter().next() {
        Some(field) if wrapper.fields.len() == 1 => &field.ty,
        _ => {
            return Err(syn::Error::new_spanned(
                &wrapper,
                "expected a struct of one field",
            ))
        }
    };
    let generics = &item_trait.generics;

    // 1. Write each leaked type with its path through the Repeater, by the names of the generics of
    //    the entry. The paths come out as the impls of crate A have them, because both encode the
    //    generics of the same context.
    let project = |_leaked: Type, ix: EntryId, context: &Context| -> Type {
        let self_ty: Type = parse_quote!(Self);
        repeater_projection(&repeater, &self_ty, context, id, ix, array_const)
            .expect("the arguments can be encoded")
    };

    // Where the copy is: `impl<K> Repo<K> for W` declares the generics of the trait itself (nothing
    // is substituted). `impl Repo<u8> for W` gives them arguments, which `Substitute` writes into
    // the signatures of the copy (`t: K`, `&'a self`, and the paths through the Repeater).
    let explicit: Option<Vec<GenericArgument>> =
        match &trait_path.segments.last().unwrap().arguments {
            PathArguments::AngleBracketed(args) => Some(args.args.iter().cloned().collect()),
            _ => None,
        };
    let (impl_generics, trait_path, site_args) = match &explicit {
        Some(args) => {
            // 2. Arguments the copy omits take the defaults of the trait, which the visitor writes
            //    in the trait itself.
            let mut visited = item_trait.clone();
            visited.items.clear();
            referrer
                .visitor(top_level, project)
                .visit_item_trait_mut(&mut visited);
            let args = enc(complete_params(&visited.generics.params, args))?;
            (quote!(), trait_path, args)
        }
        None => {
            let args: Vec<GenericArgument> = generics.params.iter().map(own_name).collect();
            let mut path = trait_path;
            if !args.is_empty() {
                path.segments.last_mut().unwrap().arguments =
                    PathArguments::AngleBracketed(parse_quote!(<#(#args),*>));
            }
            let (impl_generics, _, _) = generics.split_for_impl();
            (quote!(#impl_generics), path, Vec::new())
        }
    };
    // What the site answers for a name: the argument of the trait's parameter of that name.
    let site = |name: &Ident| {
        let at = generics
            .params
            .iter()
            .position(|param| param_ident(param) == name)?;
        site_args.get(at).cloned()
    };

    let mut items: Vec<ImplItem> = vec![];
    for item in &item_trait.items {
        match item {
            TraitItem::Type(ty) => {
                let assoc = &ty.ident;
                items.push(parse_quote!(type #assoc = <#inner as #trait_path>::#assoc;));
            }
            TraitItem::Fn(method) => {
                let mut sig = method.sig.clone();
                // 1. Replace each leaked type with its path through the Repeater. The visitor
                //    needs the whole signature, and the `ContextId` of the trait.
                referrer
                    .visitor(top_level, project)
                    .visit_signature_mut(&mut sig);
                // 3. Write the arguments of the impl into the signature: its types, and the lifetimes
                //    outside them (`&'a self`, `T: 'a`). The names that the method declares itself
                //    get no answer (`site` knows the trait's parameters only).
                if explicit.is_some() {
                    let mut written = Substitute::new(&site);
                    written.visit_signature_mut(&mut sig);
                    enc(written.finish())?;
                }
                let receiver = match sig.receiver() {
                    Some(r) if r.reference.is_some() && r.mutability.is_some() => {
                        quote!(&mut self.0)
                    }
                    Some(r) if r.reference.is_some() => quote!(&self.0),
                    _ => {
                        return Err(syn::Error::new_spanned(
                            &sig,
                            "expected `&self` or `&mut self`",
                        ))
                    }
                };
                let call = &sig.ident;
                let args = sig.inputs.iter().filter_map(|arg| match arg {
                    syn::FnArg::Typed(arg) => Some(&arg.pat),
                    syn::FnArg::Receiver(_) => None,
                });
                items.push(
                    parse_quote!(#sig { <#inner as #trait_path>::#call(#receiver, #(#args),*) }),
                );
            }
            _ => {}
        }
    }
    Ok(quote! {
        #wrapper
        impl #impl_generics #trait_path for #name where #inner: #trait_path {
            #(#items)*
        }
    })
}

fn enc<T>(result: Result<T, EncodeError>) -> syn::Result<T> {
    result.map_err(|error| syn::Error::new(error.span(), error))
}

/// The argument that names a generic parameter itself (`K` for `K: Clone`).
fn own_name(param: &GenericParam) -> GenericArgument {
    match param {
        GenericParam::Type(p) => {
            let ident = &p.ident;
            parse_quote!(#ident)
        }
        GenericParam::Lifetime(p) => {
            let lifetime = &p.lifetime;
            parse_quote!(#lifetime)
        }
        GenericParam::Const(p) => {
            let ident = &p.ident;
            parse_quote!(#ident)
        }
    }
}

/// The name of a generic parameter, as `Substitute` asks for it.
fn param_ident(param: &GenericParam) -> &Ident {
    match param {
        GenericParam::Type(p) => &p.ident,
        GenericParam::Lifetime(p) => &p.lifetime.ident,
        GenericParam::Const(p) => &p.ident,
    }
}
