//! A macro that analyzes several items with one Leaker, and the copy of each item alone in another
//! crate, as a downstream crate does that expands one macro for each item of a module.
//!
//! `shared_leak!` (the Leaker's side) is given the items:
//!
//! ```ignore
//! shared_leak! {
//!     id = 9202;
//!     items [
//!         pub trait Tr { fn f<'b>(&self, x: Held<'b>) -> &'b u8; fn g<'b>(&self, x: Held<'b>); }
//!         pub struct S<'b> { pub held: Held<'b> }
//!     ];
//! }
//! ```
//!
//! It emits the items, allows the cause of every error (the part that conflicts is copied as
//! written), and emits the Repeater's implementation of every entry and the exported macros
//! `__shared_report_<id>!` (what it found) and `__shared_apply_<id>!`.
//!
//! In the Referrer's crate, `__shared_apply_<id>!(path = ::leaker::kinds, copy(1))` copies the
//! item 1 alone, with the types rewritten by `Referrer::visitor(context, ..)`: a struct, an enum
//! or a union as a new item (and a function that moves the fields of the Leaker's into it, which
//! compiles only if their types are the same), a trait as an implementation for a host type (which
//! rustc compares with the trait). The module that expands it names what the parts that are copied
//! as written name, and nothing else.

use proc_macro2::{Punct, Spacing, Span, TokenStream};
use syn::ext::IdentExt;
use syn::parse::{Parse, ParseStream};
use syn::visit_mut::VisitMut;
use syn::*;
use template_quote::quote;
use type_leak::*;

fn ident(name: &str) -> Ident {
    Ident::new(name, Span::call_site())
}

fn keyword(input: ParseStream, word: &str) -> Result<()> {
    let found = input.call(Ident::parse_any)?;
    if found != word {
        return Err(Error::new(found.span(), format!("expected `{}`", word)));
    }
    Ok(())
}

/// `id = 7; items [ item .. ];`
struct Shared {
    id: u64,
    items: Vec<Item>,
}

impl Parse for Shared {
    fn parse(input: ParseStream) -> Result<Self> {
        keyword(input, "id")?;
        input.parse::<Token![=]>()?;
        let id = input.parse::<LitInt>()?.base10_parse()?;
        input.parse::<Token![;]>()?;
        keyword(input, "items")?;
        let content;
        bracketed!(content in input);
        let mut items = Vec::new();
        while !content.is_empty() {
            items.push(content.parse()?);
        }
        input.parse::<Token![;]>()?;
        Ok(Self { id, items })
    }
}

/// `shared_leak!`, the Leaker's side.
pub(crate) fn leak(input: TokenStream) -> Result<TokenStream> {
    let Shared { id, items } = parse2(input)?;
    let mut config = LeakerConfig::new();
    config.allow_primitive().allow_std_prelude();
    let mut leaker = Leaker::from_config(config);
    let mut contexts = Vec::new();
    for item in &items {
        contexts.push(match item {
            Item::Trait(i) => leaker.intern_trait(i, true),
            Item::Struct(i) => leaker.intern_struct(i, true),
            Item::Enum(i) => leaker.intern_enum(i, true),
            Item::Union(i) => leaker.intern_union(i, true),
            other => return Err(Error::new_spanned(other, "unsupported item")),
        });
    }
    let mut result = leaker.result();
    let errors: Vec<ErrorItem> = result.errors().to_vec();
    for error in &errors {
        result.allow(error.cause.clone());
    }
    if let Some(error) = result.errors().first() {
        return Err(Error::new(Span::call_site(), &error.message));
    }
    let referrer = result.into_referrer();
    let self_param = ident("__S");
    let mut impls = TokenStream::new();
    for ix in 0..referrer.iter().count() {
        let r = referrer
            .repeater_impl(ix, ImplSelf::Param(&self_param))
            .map_err(Error::from)?;
        let (impl_generics, _, where_clause) = r.generics.split_for_impl();
        let (args, self_ty, ty) = (&r.args, &r.self_ty, &r.ty);
        impls.extend(quote! {
            impl #impl_generics crate::Repeater<#args, #id, #ix> for #self_ty #where_clause {
                type Type = #ty;
            }
        });
    }
    let reasons: Vec<String> = errors.iter().map(|e| format!("{:?}", e.reason)).collect();
    let report = format!(
        "{}|entries={}|errors={}",
        id,
        referrer.iter().count(),
        reasons.join(",")
    );
    let report_name = ident(&format!("__shared_report_{}", id));
    let apply_name = ident(&format!("__shared_apply_{}", id));
    let dollar = Punct::new('$', Spacing::Alone);
    Ok(quote! {
        #(#items)*
        #impls
        #[doc(hidden)]
        #[macro_export]
        macro_rules! #report_name { () => { #report }; }
        #[doc(hidden)]
        #[macro_export]
        macro_rules! #apply_name {
            (#dollar(#dollar t:tt)*) => {
                #dollar crate::__macro::__shared_apply! {
                    { #referrer } { #(#contexts),* } { #id #(#items)* }
                    #dollar(#dollar t)*
                }
            };
        }
    })
}

/// Input of `__shared_apply!`: the Referrer, the contexts, the id and the items, `path =
/// ::leaker::mod`, and the items to copy (`, copy(1)`).
struct Apply {
    referrer: Referrer,
    contexts: Vec<ContextId>,
    id: u64,
    items: Vec<Item>,
    path: Path,
    copies: Vec<usize>,
}

impl Parse for Apply {
    fn parse(input: ParseStream) -> Result<Self> {
        let content;
        braced!(content in input);
        let referrer = content.parse()?;
        let content;
        braced!(content in input);
        let contexts = content
            .parse_terminated(ContextId::parse, Token![,])?
            .into_iter()
            .collect();
        let content;
        braced!(content in input);
        let id = content.parse::<LitInt>()?.base10_parse()?;
        let mut items = Vec::new();
        while !content.is_empty() {
            items.push(content.parse()?);
        }
        keyword(input, "path")?;
        input.parse::<Token![=]>()?;
        let path = Path::parse_mod_style(input)?;
        let mut copies = Vec::new();
        while input.parse::<Option<Token![,]>>()?.is_some() {
            keyword(input, "copy")?;
            let content;
            parenthesized!(content in input);
            copies.push(content.parse::<LitInt>()?.base10_parse()?);
        }
        Ok(Self {
            referrer,
            contexts,
            id,
            items,
            path,
            copies,
        })
    }
}

/// `__shared_apply!`, the Referrer's side.
pub(crate) fn apply(input: TokenStream) -> Result<TokenStream> {
    let Apply {
        referrer,
        contexts,
        id,
        items,
        path,
        copies,
    } = parse2(input)?;
    let krate = path
        .segments
        .first()
        .map(|seg| seg.ident.clone())
        .ok_or_else(|| Error::new(Span::call_site(), "expected a path"))?;
    let repeater: Path = parse_quote!(::#krate::Repeater);
    let mut out = TokenStream::new();
    for n in copies {
        let (mut item, context) = (items[n].clone(), contexts[n]);
        let error = std::cell::RefCell::new(None);
        referrer
            .clone()
            .visitor(context, |_, ix| {
                match referrer.entry_args(ix, &UseSite::identity()) {
                    Ok(args) => parse_quote!(<() as #repeater<#args, #id, #ix>>::Type),
                    Err(e) => {
                        *error.borrow_mut() = Some(e);
                        parse_quote!(())
                    }
                }
            })
            .visit_item_mut(&mut item);
        if let Some(e) = error.into_inner() {
            return Err(Error::from(e));
        }
        match &item {
            Item::Trait(t) => {
                // An implementation for a host type, which rustc compares with the trait.
                let (host, name) = (ident(&format!("Host{}x{}", id, n)), &t.ident);
                let (impl_generics, ty_generics, where_clause) = t.generics.split_for_impl();
                let methods = t.items.iter().filter_map(|member| match member {
                    TraitItem::Fn(f) => {
                        let sig = &f.sig;
                        Some(quote!(#sig { loop {} }))
                    }
                    _ => None,
                });
                out.extend(quote! {
                    pub struct #host;
                    impl #impl_generics #path::#name #ty_generics for #host #where_clause {
                        #(#methods)*
                    }
                });
            }
            Item::Struct(s) => {
                // The fields of the Leaker's struct go into the copy: their types are the same.
                let (name, check) = (&s.ident, ident(&format!("__shared_check_{}_{}", id, n)));
                let (impl_generics, ty_generics, where_clause) = s.generics.split_for_impl();
                let moved = s
                    .fields
                    .members()
                    .map(|member| quote!(#member: original.#member));
                out.extend(quote! {
                    #item
                    #[allow(dead_code)]
                    fn #check #impl_generics (original: #path::#name #ty_generics) -> #name #ty_generics
                    #where_clause
                    {
                        #name { #(#moved),* }
                    }
                });
            }
            other => out.extend(quote!(#other)),
        }
    }
    Ok(out)
}
