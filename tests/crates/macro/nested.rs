//! A macro that analyzes types in an item nested in the scope of another item, with the scope API
//! (`Leaker::enter_trait()`, then `View::enter_impl()`), and the copy of it in another crate.
//!
//! The nested impl has a `Self` of its own, so a bound of the trait's generics that names the
//! `Self` of the trait has no name in it, and a type that uses such a generic has no context
//! (`Reason::UnnameableSelf`): the Leaker copies it as written. No item of Rust nests an impl in a
//! trait with the generics of the trait, so this is the only way to reach it.
//!
//! `nested_leak!` (the Leaker's side) is given the two scopes and the types, each with the
//! generics it is written in:
//!
//! ```ignore
//! nested_leak! {
//!     id = 9199;
//!     scope trait Tr<T: ?Sized + ::core::convert::AsRef<Self>>;
//!     nested impl<'a> ::path::Host;
//!     types [
//!         <'a, T: ?Sized> => ::path::ObjOf<'a, T, dyn SharedTr>;
//!         <> => ::path::Bare<dyn SharedTr>;
//!     ];
//! }
//! ```
//!
//! It allows the cause of every error, and emits the Repeater's implementation of every entry, a
//! witness `__nw_<id>_<j>` for each type, and the exported macros `__nested_report_<id>!` (what it
//! found) and `__nested_apply_<id>!`.
//!
//! In the Referrer's crate, `__nested_apply_<id>!(path = ::leaker::kinds)` copies the scopes as a
//! trait whose method has the impl in its body (the copy nests the items as the Leaker did), rewrites
//! the types there with `Referrer::visitor(context, ..)`, and emits `fn __nc_<id>_<j><generics>()`
//! for each, which returns the witness of the Leaker's crate, so it compiles only if the rewritten
//! type is the original one.

use proc_macro2::{Punct, Spacing, Span, TokenStream};
use syn::ext::IdentExt;
use syn::parse::{Parse, ParseStream};
use syn::visit_mut::VisitMut;
use syn::*;
use template_quote::{quote, ToTokens};
use type_leak::*;

fn ident(name: &str) -> Ident {
    Ident::new(name, Span::call_site())
}

/// `id = 7; scope trait Tr<..>; nested impl<..> Ty; types [ <..> => Ty; .. ];`
struct Nested {
    id: u64,
    trait_ident: Ident,
    trait_generics: Generics,
    impl_generics: Generics,
    self_ty: Type,
    types: Vec<(Generics, Type)>,
}

fn keyword(input: ParseStream, word: &str) -> Result<()> {
    let found = input.call(Ident::parse_any)?;
    if found != word {
        return Err(Error::new(found.span(), format!("expected `{}`", word)));
    }
    Ok(())
}

impl Parse for Nested {
    fn parse(input: ParseStream) -> Result<Self> {
        keyword(input, "id")?;
        input.parse::<Token![=]>()?;
        let id = input.parse::<LitInt>()?.base10_parse()?;
        input.parse::<Token![;]>()?;
        keyword(input, "scope")?;
        input.parse::<Token![trait]>()?;
        let trait_ident = input.parse()?;
        let trait_generics = input.parse()?;
        input.parse::<Token![;]>()?;
        keyword(input, "nested")?;
        input.parse::<Token![impl]>()?;
        let impl_generics = input.parse()?;
        let self_ty = input.parse()?;
        input.parse::<Token![;]>()?;
        keyword(input, "types")?;
        let content;
        bracketed!(content in input);
        let mut types = Vec::new();
        while !content.is_empty() {
            let generics = content.parse()?;
            content.parse::<Token![=>]>()?;
            types.push((generics, content.parse()?));
            content.parse::<Token![;]>()?;
        }
        input.parse::<Token![;]>()?;
        Ok(Self {
            id,
            trait_ident,
            trait_generics,
            impl_generics,
            self_ty,
            types,
        })
    }
}

impl ToTokens for Nested {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let (id, trait_ident, trait_generics) = (self.id, &self.trait_ident, &self.trait_generics);
        let (impl_generics, self_ty) = (&self.impl_generics, &self.self_ty);
        let types = self
            .types
            .iter()
            .map(|(generics, ty)| quote!(#generics => #ty;));
        tokens.extend(quote! {
            id = #id;
            scope trait #trait_ident #trait_generics;
            nested impl #impl_generics #self_ty;
            types [ #(#types)* ];
        });
    }
}

/// `nested_leak!`, the Leaker's side.
pub(crate) fn leak(input: TokenStream) -> Result<TokenStream> {
    let nested: Nested = parse2(input)?;
    let mut config = LeakerConfig::new();
    config.allow_primitive().allow_prelude();
    let mut leaker = Leaker::from_config(config);
    let (context, ()) =
        leaker.enter_trait(&nested.trait_ident, &nested.trait_generics, true, |view| {
            view.enter_impl(&nested.impl_generics, &nested.self_ty, true, |view| {
                for (_, ty) in &nested.types {
                    view.visit_type(ty);
                }
            });
        });
    let mut result = leaker.result();
    // Every error is accepted: the part that conflicts is copied as written.
    let errors: Vec<ErrorItem> = result.errors().to_vec();
    crate::allow_every_cause(&mut result);
    if let Some(error) = result.errors().first() {
        return Err(Error::new(Span::call_site(), &error.message));
    }
    let referrer = result.into_referrer();
    let id = nested.id;
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
    let witnesses = nested.types.iter().enumerate().map(|(j, (generics, ty))| {
        let name = ident(&format!("__nw_{}_{}", id, j));
        let (params, where_clause) = (&generics.params, &generics.where_clause);
        quote! {
            #[allow(non_snake_case, dead_code)]
            pub fn #name<#params>() -> ::core::marker::PhantomData<*mut #ty> #where_clause {
                ::core::marker::PhantomData
            }
        }
    });
    let reasons: Vec<String> = errors.iter().map(|e| format!("{:?}", e.reason)).collect();
    let report = format!(
        "{}|entries={}|errors={}",
        id,
        referrer.iter().count(),
        reasons.join(",")
    );
    let report_name = ident(&format!("__nested_report_{}", id));
    let apply_name = ident(&format!("__nested_apply_{}", id));
    let dollar = Punct::new('$', Spacing::Alone);
    Ok(quote! {
        #impls
        #(#witnesses)*
        #[doc(hidden)]
        #[macro_export]
        macro_rules! #report_name { () => { #report }; }
        #[doc(hidden)]
        #[macro_export]
        macro_rules! #apply_name {
            (#dollar(#dollar t:tt)*) => {
                #dollar crate::__macro::__nested_apply! {
                    { #referrer , #context }
                    { #nested }
                    #dollar(#dollar t)*
                }
            };
        }
    })
}

/// Input of `__nested_apply!`: the Referrer and the context, the scopes, `path = ::leaker::mod`,
/// and `, strict`.
struct Apply {
    referrer: Referrer,
    context: ContextId,
    nested: Nested,
    path: Path,
    strict: bool,
}

impl Parse for Apply {
    fn parse(input: ParseStream) -> Result<Self> {
        let content;
        braced!(content in input);
        let referrer = content.parse()?;
        content.parse::<Token![,]>()?;
        let context = content.parse()?;
        let content;
        braced!(content in input);
        let nested = content.parse()?;
        keyword(input, "path")?;
        input.parse::<Token![=]>()?;
        let path = Path::parse_mod_style(input)?;
        let strict = input.parse::<Option<Token![,]>>()?.is_some() && {
            keyword(input, "strict")?;
            true
        };
        Ok(Self {
            referrer,
            context,
            nested,
            path,
            strict,
        })
    }
}

/// Removes the attribute-less body of the probe functions: the rewritten return types.
fn probe_types(wrapper: &ItemTrait) -> Vec<Type> {
    struct Find(Vec<Type>);
    impl<'ast> syn::visit::Visit<'ast> for Find {
        fn visit_impl_item_fn(&mut self, i: &'ast ImplItemFn) {
            if let ReturnType::Type(_, ty) = &i.sig.output {
                self.0.push((**ty).clone());
            }
        }
    }
    let mut find = Find(Vec::new());
    syn::visit::Visit::visit_item_trait(&mut find, wrapper);
    find.0
}

/// `__nested_apply!`, the Referrer's side.
pub(crate) fn apply(input: TokenStream) -> Result<TokenStream> {
    let Apply {
        referrer,
        context,
        nested,
        path,
        strict,
    } = parse2(input)?;
    let referrer = referrer.allow_ambiguous_crate_name(!strict);
    let id = nested.id;
    let krate = path
        .segments
        .first()
        .map(|seg| seg.ident.clone())
        .ok_or_else(|| Error::new(Span::call_site(), "expected a path"))?;
    let repeater: Path = parse_quote!(::#krate::Repeater);

    // The copy nests the impl in the body of a method of the trait, as the Leaker nested the scopes.
    let (trait_generics, impl_generics, self_ty) = (
        &nested.trait_generics,
        &nested.impl_generics,
        &nested.self_ty,
    );
    let probes = nested.types.iter().enumerate().map(|(j, (_, ty))| {
        let name = ident(&format!("__probe{}", j));
        quote!(fn #name() -> #ty { loop {} })
    });
    let mut wrapper: ItemTrait = parse_quote! {
        trait __Wrap #trait_generics {
            fn __wrap() {
                impl #impl_generics #self_ty { #(#probes)* }
            }
        }
    };
    let error = std::cell::RefCell::new(None);
    referrer
        .clone()
        .visitor(context, |_, ix| {
            let args = referrer.entry_args(ix, &UseSite::identity());
            match args {
                Ok(args) => parse_quote!(<() as #repeater<#args, #id, #ix>>::Type),
                Err(e) => {
                    *error.borrow_mut() = Some(e);
                    parse_quote!(())
                }
            }
        })
        .visit_item_trait_mut(&mut wrapper);
    if let Some(e) = error.into_inner() {
        return Err(Error::from(e));
    }
    let rewritten = probe_types(&wrapper);
    assert_eq!(rewritten.len(), nested.types.len());

    let mut checks = TokenStream::new();
    for (j, ((generics, _), ty)) in nested.types.iter().zip(&rewritten).enumerate() {
        let name = ident(&format!("__nc_{}_{}", id, j));
        let witness = ident(&format!("__nw_{}_{}", id, j));
        let (params, where_clause) = (&generics.params, &generics.where_clause);
        let targs = generics.params.iter().filter_map(|param| match param {
            GenericParam::Type(t) => Some(t.ident.clone()),
            GenericParam::Const(c) => Some(c.ident.clone()),
            GenericParam::Lifetime(_) => None,
        });
        checks.extend(quote! {
            #[allow(non_snake_case, dead_code)]
            fn #name<#params>() -> ::core::marker::PhantomData<*mut #ty> #where_clause {
                #path::#witness::<#(#targs),*>()
            }
        });
    }
    // The rewritten types, as text, for the tests.
    let text = rewritten
        .iter()
        .map(|ty| quote!(#ty).to_string())
        .collect::<Vec<_>>()
        .join(" ;; ");
    let text_name = ident(&format!("NESTED_{}_TYPES", id));
    Ok(quote! {
        #checks
        pub const #text_name: &str = #text;
    })
}
