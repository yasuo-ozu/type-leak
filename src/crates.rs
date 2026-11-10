//! The absolute paths that an entry depends on, and the crates they name.

use crate::context::{Context, SelfRequirement};
use crate::error::Cause;
use crate::util::{ident_name, trait_head};
use proc_macro2::{Ident, Spacing, TokenStream, TokenTree};
use std::collections::BTreeSet;
use syn::punctuated::Punctuated;
use syn::visit::Visit;
use syn::*;

/// Returns the absolute paths that the entry `ty` of `context` names, without generic arguments,
/// once each, in order: those in the type, in the bounds of the generics and in the `Self`
/// requirement. `::core`, `::crate`, `::self`, `::super`, `::Self` and `::_` name no crate and are
/// left out.
///
/// Each is the [`Cause`] that the analysis asks an [`Allow`](crate::Allow) rule about: a path of a
/// type is a [`Cause::Type`], the trait of a bound or of a qualified path a [`Cause::Trait`], the
/// path of a macro a [`Cause::Macro`], and any other (a path in an expression, in the tokens of a
/// macro, or in a qualified path with no trait) a [`Cause::Unknown`].
pub(crate) fn absolute_paths(ty: &Type, context: &Context) -> Vec<Cause> {
    let mut finder = Finder::default();
    finder.visit_type(ty);
    finder.visit_generics(context.generics());
    match context.self_ty() {
        Some(SelfRequirement::Bound(path)) => {
            finder.add(Cause::Trait(path.clone()));
            finder.arguments(path);
        }
        Some(SelfRequirement::Type(ty)) => finder.visit_type(ty),
        _ => (),
    }
    finder.0
}

/// Returns the sorted names of the external crates (not `core`) that `ty` and `context` name as
/// `::name` ([`absolute_paths()`]).
///
/// `::crate`, `::self`, `::super`, `::Self` and `::_` name no crate (rustc rejects them in the
/// Leaker's crate), and cannot be written as a name in a serialized Referrer.
pub(crate) fn external_crates(ty: &Type, context: &Context) -> Vec<String> {
    absolute_paths(ty, context)
        .iter()
        .filter_map(|cause| cause.path()?.segments.first())
        .map(|seg| ident_name(&seg.ident))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

#[derive(Default)]
struct Finder(Vec<Cause>);

impl Finder {
    /// Records the path of `cause` without its arguments, if it is absolute and names a crate.
    fn add(&mut self, cause: Cause) {
        let cause = cause.bare();
        let named = cause.path().map_or(false, |path| {
            path.leading_colon.is_some()
                && path.segments.first().map_or(false, |seg| {
                    !matches!(
                        ident_name(&seg.ident).as_str(),
                        "core" | "crate" | "self" | "super" | "Self" | "_"
                    )
                })
        });
        if named && !self.0.contains(&cause) {
            self.0.push(cause);
        }
    }

    /// Visits the generic arguments of the segments of `path`.
    fn arguments(&mut self, path: &Path) {
        for seg in &path.segments {
            self.visit_path_arguments(&seg.arguments);
        }
    }

    /// Records the path of a qualified type or expression: its trait if it has one, else it is of
    /// unknown kind.
    fn qualified(&mut self, path: &Path, qself: &QSelf) {
        self.visit_qself(qself);
        if qself.as_token.is_some() && qself.position > 0 {
            self.add(Cause::Trait(trait_head(path, qself.position)));
        } else {
            self.add(Cause::Unknown(path.clone()));
        }
        self.arguments(path);
    }

    /// Scans macro tokens for a leading `::` followed by a path.
    fn scan(&mut self, tokens: TokenStream) {
        let trees: Vec<TokenTree> = tokens.into_iter().collect();
        let colons = |at: usize| -> bool {
            matches!(trees.get(at), Some(TokenTree::Punct(p))
                if p.as_char() == ':' && p.spacing() == Spacing::Joint)
                && matches!(trees.get(at + 1), Some(TokenTree::Punct(p)) if p.as_char() == ':')
        };
        let mut at = 0;
        while at < trees.len() {
            match &trees[at] {
                TokenTree::Group(group) => self.scan(group.stream()),
                TokenTree::Punct(_) if colons(at) => {
                    let after_name = at > 0
                        && match &trees[at - 1] {
                            TokenTree::Ident(_) => true,
                            TokenTree::Punct(p) => p.as_char() == '>',
                            _ => false,
                        };
                    if !after_name {
                        let mut names: Vec<&Ident> = Vec::new();
                        let mut next = at + 2;
                        while let Some(TokenTree::Ident(ident)) = trees.get(next) {
                            names.push(ident);
                            next += 1;
                            if colons(next) {
                                next += 2;
                            } else {
                                break;
                            }
                        }
                        if !names.is_empty() {
                            let segments: Punctuated<PathSegment, Token![::]> = names
                                .into_iter()
                                .map(|ident| PathSegment::from(ident.clone()))
                                .collect();
                            self.add(Cause::Unknown(Path {
                                leading_colon: Some(Default::default()),
                                segments,
                            }));
                        }
                        // The names after it are read; groups after them are scanned next.
                        at = next;
                        continue;
                    }
                    at += 2;
                    continue;
                }
                _ => (),
            }
            at += 1;
        }
    }
}

impl<'ast> Visit<'ast> for Finder {
    fn visit_path(&mut self, i: &'ast Path) {
        self.add(Cause::Unknown(i.clone()));
        self.arguments(i);
    }

    fn visit_type_path(&mut self, i: &'ast TypePath) {
        match &i.qself {
            Some(qself) => self.qualified(&i.path, qself),
            None => {
                self.add(Cause::type_path(&i.path));
                self.arguments(&i.path);
            }
        }
    }

    fn visit_expr_path(&mut self, i: &'ast ExprPath) {
        for attr in &i.attrs {
            self.visit_attribute(attr);
        }
        match &i.qself {
            Some(qself) => self.qualified(&i.path, qself),
            None => self.visit_path(&i.path),
        }
    }

    fn visit_trait_bound(&mut self, i: &'ast TraitBound) {
        self.add(Cause::Trait(i.path.clone()));
        self.arguments(&i.path);
    }

    fn visit_macro(&mut self, i: &'ast Macro) {
        self.add(Cause::Macro(i.path.clone()));
        self.arguments(&i.path);
        self.scan(i.tokens.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn paths(ty: Type, generics: Generics, self_ty: Option<SelfRequirement>) -> Vec<Cause> {
        absolute_paths(&ty, &Context::new(generics, self_ty))
    }

    fn of_type(ty: Type) -> Vec<Cause> {
        paths(ty, Generics::default(), None)
    }

    fn type_cause(path: Path) -> Cause {
        Cause::Type(Type::Path(TypePath { qself: None, path }))
    }

    fn crates_of(ty: Type) -> String {
        external_crates(&ty, &Context::none()).join(",")
    }

    #[test]
    fn a_path_of_a_type_is_asked_bare_and_once() {
        assert_eq!(
            of_type(parse_quote!(::a::W<::b::X, ::a::W<u8>, ::b::X>)),
            [
                type_cause(parse_quote!(::a::W)),
                type_cause(parse_quote!(::b::X)),
            ]
        );
    }

    #[test]
    fn core_and_the_names_of_no_crate_are_left_out() {
        assert_eq!(
            of_type(parse_quote!((::core::option::Option<u8>, Rel, ::a::X))),
            [type_cause(parse_quote!(::a::X))]
        );
        assert_eq!(crates_of(parse_quote!(::core::marker::PhantomData<u8>)), "");
    }

    #[test]
    fn the_trait_of_a_bound_a_dyn_type_and_a_qualified_type_is_a_trait() {
        let mut generics: Generics = parse_quote!(<U: ::b::B + ::core::clone::Clone, V>);
        generics.where_clause = Some(parse_quote!(where V: ::c::C));
        assert_eq!(
            paths(parse_quote!(::a::W<U, V>), generics, None),
            [
                type_cause(parse_quote!(::a::W)),
                Cause::Trait(parse_quote!(::b::B)),
                Cause::Trait(parse_quote!(::c::C)),
            ]
        );
        assert_eq!(
            of_type(parse_quote!(Box<dyn ::d::D + Send>)),
            [Cause::Trait(parse_quote!(::d::D))]
        );
        assert_eq!(
            of_type(parse_quote!(<u8 as ::e::E<::f::F>>::Out)),
            [
                Cause::Trait(parse_quote!(::e::E)),
                type_cause(parse_quote!(::f::F)),
            ]
        );
    }

    #[test]
    fn a_path_in_an_expression_a_macro_and_its_tokens() {
        assert_eq!(
            of_type(parse_quote!(
                [u8; ::a::N + ::b::m!(::c::N, x::y, ::d::p::<u8>::q)]
            )),
            [
                Cause::Unknown(parse_quote!(::a::N)),
                Cause::Macro(parse_quote!(::b::m)),
                Cause::Unknown(parse_quote!(::c::N)),
                Cause::Unknown(parse_quote!(::d::p)),
            ]
        );
        assert_eq!(
            of_type(parse_quote!(::a::m!(::b::X<u8>))),
            [
                Cause::Macro(parse_quote!(::a::m)),
                Cause::Unknown(parse_quote!(::b::X)),
            ]
        );
    }

    #[test]
    fn the_self_requirement_counts() {
        let bound = SelfRequirement::Bound(parse_quote!(::a::Tr<::b::X>));
        assert_eq!(
            paths(parse_quote!(Rel<Self>), Generics::default(), Some(bound)),
            [
                Cause::Trait(parse_quote!(::a::Tr)),
                type_cause(parse_quote!(::b::X)),
            ]
        );
        let ty = SelfRequirement::Type(parse_quote!(::a::W<::b::X>));
        assert_eq!(
            paths(parse_quote!(Rel<Self>), Generics::default(), Some(ty)),
            [
                type_cause(parse_quote!(::a::W)),
                type_cause(parse_quote!(::b::X)),
            ]
        );
        assert_eq!(
            paths(
                parse_quote!(Rel<Self>),
                Generics::default(),
                Some(SelfRequirement::Any)
            ),
            []
        );
    }

    #[test]
    fn crates_are_the_sorted_first_segments() {
        assert_eq!(
            crates_of(parse_quote!((::z::A, ::a::B, ::z::C<::a::D>))),
            "a,z"
        );
        assert_eq!(crates_of(parse_quote!([u8; ::r#fn::N])), "fn");
    }
}
