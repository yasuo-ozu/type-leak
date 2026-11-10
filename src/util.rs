//! Helpers for paths and types and the names and `Self` in them, the text of messages (`tidy()`,
//! `or_list()`), the prelude and primitive names (`KNOWN_NAMES`), `ReplaceSelf` and `Sorted`.

use crate::error::Cause;
use proc_macro2::{Span, TokenStream};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::ops::{Deref, DerefMut};
use syn::ext::IdentExt;
use syn::punctuated::Punctuated;
use syn::visit::Visit;
use syn::*;
use template_quote::{quote, ToTokens};

/// A `HashMap` or a `HashSet` whose `Debug` lists the entries sorted by their text, so a value
/// prints the same in every run (each map has its own random seed). It derefs to the collection:
/// lookups and updates are as without it. The derives of the type that has it stay derives.
#[derive(Clone, Default)]
pub(crate) struct Sorted<T>(pub(crate) T);

impl<T> Deref for Sorted<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T> DerefMut for Sorted<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

impl<K: fmt::Debug, V: fmt::Debug, S> fmt::Debug for Sorted<HashMap<K, V, S>> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Equal texts print equal, so the order of the ones that tie does not show.
        let mut entries: Vec<(String, &K, &V)> = self
            .0
            .iter()
            .map(|(key, value)| (format!("{:?} {:?}", key, value), key, value))
            .collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        f.debug_map()
            .entries(entries.iter().map(|(_, key, value)| (key, value)))
            .finish()
    }
}

impl<T: fmt::Debug, S> fmt::Debug for Sorted<HashSet<T, S>> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut items: Vec<(String, &T)> = self
            .0
            .iter()
            .map(|item| (format!("{:?}", item), item))
            .collect();
        items.sort_by(|a, b| a.0.cmp(&b.0));
        f.debug_set()
            .entries(items.iter().map(|(_, item)| item))
            .finish()
    }
}

/// The cause of an error about `bound`: its trait path, or the bound as written (without `?`) if it
/// has a binder or parentheses, which no node of the syntax holds.
pub(crate) fn bound_cause(bound: &TraitBound) -> Cause {
    if bound.lifetimes.is_none() && bound.paren_token.is_none() {
        return Cause::Trait(bound.path.clone());
    }
    let bound = TraitBound {
        modifier: TraitBoundModifier::None,
        ..bound.clone()
    };
    Cause::Unreadable(quote!(#bound))
}

/// `path` without the arguments of its segments: `Vec`, not `Vec<Rel<'_>>`.
pub(crate) fn without_arguments(path: &Path) -> Path {
    let mut path = path.clone();
    for seg in path.segments.iter_mut() {
        seg.arguments = PathArguments::None;
    }
    path
}

/// The first `position` segments of `path`: the trait of a qualified path.
pub(crate) fn trait_head(path: &Path, position: usize) -> Path {
    Path {
        leading_colon: path.leading_colon,
        segments: path.segments.iter().take(position).cloned().collect(),
    }
}

/// The head of `path` if it may start an associated type path (`T::A<X>`): relative, with an item
/// after it and no arguments. The path does not tell if the head is a type parameter.
pub(crate) fn projection_head(path: &Path) -> Option<&Ident> {
    if path.leading_colon.is_some() || path.segments.len() < 2 {
        return None;
    }
    path.segments
        .first()
        .filter(|seg| seg.arguments.is_none())
        .map(|seg| &seg.ident)
}

/// `<value>`, the start of a path `<value>::A`, at `span` (that of the `T` of `T::A`, so that an
/// error points at the path).
pub(crate) fn qself_of(value: Type, span: Span) -> QSelf {
    QSelf {
        lt_token: Token![<](span),
        ty: Box::new(value),
        position: 0,
        as_token: None,
        gt_token: Token![>](span),
    }
}

/// The segments of `path` after the first, with a leading `::`: `::A` of `T::A`, the path after
/// `<value>` in `<value>::A`.
pub(crate) fn after_head(path: &Path) -> Path {
    Path {
        leading_colon: Some(Default::default()),
        segments: path.segments.iter().skip(1).cloned().collect(),
    }
}

/// `<value>::A` for the path `T::A` and the type `value` for `T`. `fold` writes the segments after
/// `T`, whose arguments may name generics too.
pub(crate) fn projection(value: Type, path: &Path, fold: impl FnOnce(Path) -> Path) -> TypePath {
    TypePath {
        qself: Some(qself_of(value, Span::call_site())),
        path: fold(after_head(path)),
    }
}

/// The first segment of `path` if it is `Self` (not after a leading `::`).
pub(crate) fn self_head(path: &Path) -> Option<&PathSegment> {
    path.segments
        .first()
        .filter(|seg| path.leading_colon.is_none() && seg.ident == "Self")
}

/// Like [`self_head()`], if the segment has no arguments: `Self` and `Self::A`, not `Self<..>`.
pub(crate) fn plain_self_head(path: &Path) -> Option<&PathSegment> {
    self_head(path).filter(|seg| seg.arguments.is_none())
}

/// How the Leaker treats a name of [`KNOWN_NAMES`].
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Known {
    /// A primitive type: `Rules::primitive()` allows it.
    Primitive,
    /// In the prelude of every edition, as of rustc 1.56: `Rules::prelude()` allows it.
    Prelude,
    /// In the prelude of some editions or of later versions of rustc only, so the copy's crate may
    /// not have it: no option allows it, and the help gives its absolute path.
    Newer,
}

/// The names that the help gives an absolute path for (`use ::core::.. instead of ..`), and the
/// option that allows each: the one source of the help, `Rules::primitive()` and
/// `Rules::prelude()`.
pub(crate) const KNOWN_NAMES: &[(&str, &str, Known)] = &[
    // The prelude of every edition (`std::prelude::v1`).
    ("Copy", "::core::marker::Copy", Known::Prelude),
    ("Send", "::core::marker::Send", Known::Prelude),
    ("Sized", "::core::marker::Sized", Known::Prelude),
    ("Sync", "::core::marker::Sync", Known::Prelude),
    ("Unpin", "::core::marker::Unpin", Known::Prelude),
    ("Drop", "::core::ops::Drop", Known::Prelude),
    ("Fn", "::core::ops::Fn", Known::Prelude),
    ("FnMut", "::core::ops::FnMut", Known::Prelude),
    ("FnOnce", "::core::ops::FnOnce", Known::Prelude),
    ("drop", "::core::mem::drop", Known::Prelude),
    ("Box", "::std::boxed::Box", Known::Prelude),
    ("ToOwned", "::std::borrow::ToOwned", Known::Prelude),
    ("Clone", "::core::clone::Clone", Known::Prelude),
    ("PartialEq", "::core::cmp::PartialEq", Known::Prelude),
    ("PartialOrd", "::core::cmp::PartialOrd", Known::Prelude),
    ("Eq", "::core::cmp::Eq", Known::Prelude),
    ("Ord", "::core::cmp::Ord", Known::Prelude),
    ("AsRef", "::core::convert::AsRef", Known::Prelude),
    ("AsMut", "::core::convert::AsMut", Known::Prelude),
    ("Into", "::core::convert::Into", Known::Prelude),
    ("From", "::core::convert::From", Known::Prelude),
    ("Default", "::core::default::Default", Known::Prelude),
    ("Iterator", "::core::iter::Iterator", Known::Prelude),
    ("Extend", "::core::iter::Extend", Known::Prelude),
    ("IntoIterator", "::core::iter::IntoIterator", Known::Prelude),
    (
        "DoubleEndedIterator",
        "::core::iter::DoubleEndedIterator",
        Known::Prelude,
    ),
    (
        "ExactSizeIterator",
        "::core::iter::ExactSizeIterator",
        Known::Prelude,
    ),
    ("Option", "::core::option::Option", Known::Prelude),
    ("Some", "::core::option::Option::Some", Known::Prelude),
    ("None", "::core::option::Option::None", Known::Prelude),
    ("Result", "::core::result::Result", Known::Prelude),
    ("Ok", "::core::result::Result::Ok", Known::Prelude),
    ("Err", "::core::result::Result::Err", Known::Prelude),
    ("String", "::std::string::String", Known::Prelude),
    ("ToString", "::std::string::ToString", Known::Prelude),
    ("Vec", "::std::vec::Vec", Known::Prelude),
    // The prelude of the 2021 edition.
    ("TryFrom", "::core::convert::TryFrom", Known::Newer),
    ("TryInto", "::core::convert::TryInto", Known::Newer),
    ("FromIterator", "::core::iter::FromIterator", Known::Newer),
    // The prelude of the 2024 edition (`IntoFuture` is from rustc 1.64).
    ("Future", "::core::future::Future", Known::Newer),
    ("IntoFuture", "::core::future::IntoFuture", Known::Newer),
    // The prelude since rustc 1.80.
    ("size_of", "::core::mem::size_of", Known::Newer),
    ("size_of_val", "::core::mem::size_of_val", Known::Newer),
    ("align_of", "::core::mem::align_of", Known::Newer),
    ("align_of_val", "::core::mem::align_of_val", Known::Newer),
    ("bool", "::core::primitive::bool", Known::Primitive),
    ("char", "::core::primitive::char", Known::Primitive),
    ("str", "::core::primitive::str", Known::Primitive),
    ("u8", "::core::primitive::u8", Known::Primitive),
    ("u16", "::core::primitive::u16", Known::Primitive),
    ("u32", "::core::primitive::u32", Known::Primitive),
    ("u64", "::core::primitive::u64", Known::Primitive),
    ("u128", "::core::primitive::u128", Known::Primitive),
    ("usize", "::core::primitive::usize", Known::Primitive),
    ("i8", "::core::primitive::i8", Known::Primitive),
    ("i16", "::core::primitive::i16", Known::Primitive),
    ("i32", "::core::primitive::i32", Known::Primitive),
    ("i64", "::core::primitive::i64", Known::Primitive),
    ("i128", "::core::primitive::i128", Known::Primitive),
    ("isize", "::core::primitive::isize", Known::Primitive),
    ("f32", "::core::primitive::f32", Known::Primitive),
    ("f64", "::core::primitive::f64", Known::Primitive),
];

/// Returns `true` if `path` is relative and starts with a name of [`KNOWN_NAMES`] of the kind
/// `known`: the rule of the option that allows that kind.
pub(crate) fn starts_with_known(path: &Path, known: Known) -> bool {
    path.leading_colon.is_none()
        && path.segments.first().map_or(false, |seg| {
            KNOWN_NAMES
                .iter()
                .any(|(name, _, kind)| seg.ident == name && *kind == known)
        })
}

/// The crates that `Rules::std_crates()` allows.
const STD_CRATES: &[&str] = &["std", "core", "alloc"];

/// Returns `true` if `path` starts with `std`, `core` or `alloc`, relative or absolute.
pub(crate) fn is_std_crate_path(path: &Path) -> bool {
    path.segments
        .first()
        .map_or(false, |seg| STD_CRATES.iter().any(|name| seg.ident == name))
}

pub(crate) fn is_primitive_name(ident: &Ident) -> bool {
    KNOWN_NAMES
        .iter()
        .any(|(name, _, kind)| ident == name && *kind == Known::Primitive)
}

/// Returns `true` if `ty` is `part` or has it as a component, compared as written.
pub(crate) fn type_contains(ty: &Type, part: &Type) -> bool {
    struct Contains {
        part: String,
        found: bool,
    }
    impl<'ast> Visit<'ast> for Contains {
        fn visit_type(&mut self, i: &'ast Type) {
            if self.found {
                return;
            }
            let ty = unwrap_type_group(i);
            if quote!(#ty).to_string() == self.part {
                self.found = true;
                return;
            }
            syn::visit::visit_type(self, i)
        }
    }
    let part = unwrap_type_group(part);
    let mut contains = Contains {
        part: quote!(#part).to_string(),
        found: false,
    };
    contains.visit_type(ty);
    contains.found
}

/// Collects the names by which `ty` can refer to a generic: the first segment of each relative
/// path, the lifetimes, and the names in unparsed tokens. The Leaker finds the generics of a type
/// the same way (`starts_with_generic()`), so the Referrer computes the same contexts by it.
pub(crate) fn type_names(ty: &Type, names: &mut HashSet<String>) {
    struct Names<'a>(&'a mut HashSet<String>);
    impl Names<'_> {
        fn tokens(&mut self, tokens: &TokenStream) {
            token_names(tokens.clone(), self.0);
        }
    }
    impl<'ast> Visit<'ast> for Names<'_> {
        fn visit_path(&mut self, i: &'ast Path) {
            // A later segment (`U` in `inner::U`) names an item, not a generic.
            if i.leading_colon.is_none() {
                if let Some(seg) = i.segments.first() {
                    self.0.insert(ident_name(&seg.ident));
                }
            }
            syn::visit::visit_path(self, i)
        }
        fn visit_lifetime(&mut self, i: &'ast Lifetime) {
            self.0.insert(i.to_string());
        }
        fn visit_macro(&mut self, i: &'ast Macro) {
            self.tokens(&i.tokens);
            syn::visit::visit_macro(self, i)
        }
        fn visit_type(&mut self, i: &'ast Type) {
            match i {
                Type::Verbatim(tokens) => self.tokens(tokens),
                _ => syn::visit::visit_type(self, i),
            }
        }
        fn visit_expr(&mut self, i: &'ast Expr) {
            match i {
                Expr::Verbatim(tokens) => self.tokens(tokens),
                _ => syn::visit::visit_expr(self, i),
            }
        }
        fn visit_pat(&mut self, i: &'ast Pat) {
            match i {
                Pat::Verbatim(tokens) => self.tokens(tokens),
                _ => syn::visit::visit_pat(self, i),
            }
        }
        fn visit_type_param_bound(&mut self, i: &'ast TypeParamBound) {
            match i {
                TypeParamBound::Verbatim(tokens) => self.tokens(tokens),
                _ => syn::visit::visit_type_param_bound(self, i),
            }
        }
        fn visit_item(&mut self, i: &'ast Item) {
            match i {
                Item::Verbatim(tokens) => self.tokens(tokens),
                _ => syn::visit::visit_item(self, i),
            }
        }
    }
    Names(names).visit_type(ty)
}

/// Returns `true` if `ty` refers to `idents` (as the first segment of a relative path).
pub(crate) fn mentions_idents(ty: &Type, idents: &[Ident]) -> bool {
    struct Visitor<'a> {
        idents: &'a [Ident],
        found: bool,
    }
    impl<'ast> Visit<'ast> for Visitor<'_> {
        fn visit_path(&mut self, i: &'ast Path) {
            if i.leading_colon.is_none()
                && i.segments
                    .first()
                    .map_or(false, |seg| self.idents.contains(&seg.ident))
            {
                self.found = true;
            }
            syn::visit::visit_path(self, i)
        }
        fn visit_macro(&mut self, i: &'ast Macro) {
            // The tokens of a macro are not parsed, so they are matched by name.
            let names = names_in(&i.tokens);
            if self
                .idents
                .iter()
                .any(|ident| names.contains(&ident.to_string()))
            {
                self.found = true;
            }
            syn::visit::visit_macro(self, i)
        }
    }
    if idents.is_empty() {
        return false;
    }
    let mut visitor = Visitor {
        idents,
        found: false,
    };
    visitor.visit_type(ty);
    visitor.found
}

/// Returns the name of `ident` without `r#`, by which names are compared.
pub(crate) fn ident_name(ident: &Ident) -> String {
    ident.unraw().to_string()
}

/// Returns the names that `item` declares in the block or module around it, with what a `use`
/// brings in ([`use_names()`]); an impl or a macro call declares none. The Leaker reads the values
/// of a block with it, and the Referrer hides the entries that name them.
pub(crate) fn declared_names(item: &Item) -> Vec<Ident> {
    let mut names = Vec::new();
    match item {
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
        }) => names.push(ident.clone()),
        Item::ExternCrate(ItemExternCrate { ident, rename, .. }) => {
            let name = rename.as_ref().map_or(ident, |(_, rename)| rename);
            if name != "_" {
                names.push(name.clone());
            }
        }
        Item::Use(ItemUse { tree, .. }) => use_names(tree, None, &mut names),
        _ => (),
    }
    names
}

/// Adds to `names` what the `use` tree brings into scope: `use a::b::{self}` brings `b`, and
/// `as _` and a glob bring no name. `parent` is the name before a group.
fn use_names(tree: &UseTree, parent: Option<&Ident>, names: &mut Vec<Ident>) {
    match tree {
        UseTree::Path(UsePath { ident, tree, .. }) => use_names(tree, Some(ident), names),
        UseTree::Name(UseName { ident }) if ident == "self" => names.extend(parent.cloned()),
        UseTree::Name(UseName { ident }) => names.push(ident.clone()),
        UseTree::Rename(UseRename { rename, .. }) if rename != "_" => names.push(rename.clone()),
        UseTree::Rename(_) | UseTree::Glob(_) => (),
        UseTree::Group(UseGroup { items, .. }) => {
            for tree in items {
                use_names(tree, parent, names);
            }
        }
    }
}

/// Collects the names of the identifiers and lifetimes in `tokens`, without `r#`.
pub(crate) fn token_names(tokens: TokenStream, names: &mut HashSet<String>) {
    let mut after_quote = false;
    for tt in tokens {
        match &tt {
            proc_macro2::TokenTree::Ident(ident) if after_quote => {
                names.insert(format!("'{}", ident));
            }
            proc_macro2::TokenTree::Ident(ident) => {
                names.insert(ident_name(ident));
            }
            proc_macro2::TokenTree::Group(group) => token_names(group.stream(), names),
            _ => (),
        }
        after_quote = false;
        if let proc_macro2::TokenTree::Punct(punct) = &tt {
            after_quote = punct.as_char() == '\'';
        }
    }
}

/// Returns `true` if `param` is a type parameter named `name` (without `r#`).
pub(crate) fn is_type_param_named(param: &GenericParam, name: &str) -> bool {
    matches!(param, GenericParam::Type(param) if ident_name(&param.ident) == name)
}

/// Returns `true` if `param` is the lifetime parameter `lifetime`.
pub(crate) fn is_lifetime_param(param: &GenericParam, lifetime: &Lifetime) -> bool {
    matches!(param, GenericParam::Lifetime(param) if param.lifetime == *lifetime)
}

/// The names of the identifiers and lifetimes in `tokens`, as [`token_names()`] collects them.
pub(crate) fn names_in(tokens: impl ToTokens) -> HashSet<String> {
    let mut names = HashSet::new();
    token_names(tokens.into_token_stream(), &mut names);
    names
}

pub(crate) fn generic_param_name(param: &GenericParam) -> String {
    match param {
        GenericParam::Type(TypeParam { ident, .. })
        | GenericParam::Const(ConstParam { ident, .. }) => ident_name(ident),
        GenericParam::Lifetime(LifetimeParam { lifetime, .. }) => lifetime.to_string(),
    }
}

/// The lifetimes that the binder `for<..>` declares.
pub(crate) fn binder_lifetimes(
    lifetimes: Option<&BoundLifetimes>,
) -> impl Iterator<Item = Lifetime> + '_ {
    lifetimes
        .into_iter()
        .flat_map(|lifetimes| lifetimes.lifetimes.iter())
        .filter_map(|param| match param {
            GenericParam::Lifetime(param) => Some(param.lifetime.clone()),
            _ => None,
        })
}

/// Returns `true` if the binder `lifetimes` declares `wanted`.
pub(crate) fn binder_declares(lifetimes: Option<&BoundLifetimes>, wanted: &Lifetime) -> bool {
    binder_lifetimes(lifetimes).any(|lifetime| lifetime == *wanted)
}

/// Pushes `item` to `items` unless it is there already.
pub(crate) fn push_once<T: PartialEq>(items: &mut Vec<T>, item: T) {
    if !items.contains(&item) {
        items.push(item);
    }
}

/// The path `path` read as a type, which stands for a node that is no type (a bound).
pub(crate) fn path_type(path: &Path) -> Type {
    Type::Path(TypePath {
        qself: None,
        path: path.clone(),
    })
}

/// Returns the type inside invisible groups (from `$t:ty` in `macro_rules!`) and parentheses.
pub(crate) fn unwrap_type_group(ty: &Type) -> &Type {
    match ty {
        Type::Group(TypeGroup { elem, .. }) | Type::Paren(TypeParen { elem, .. }) => {
            unwrap_type_group(elem)
        }
        _ => ty,
    }
}

/// Wraps `ty` in parentheses.
fn parenthesize(ty: &mut Type) {
    let inner = std::mem::replace(ty, Type::Verbatim(TokenStream::new()));
    *ty = Type::Paren(TypeParen {
        paren_token: Default::default(),
        elem: Box::new(inner),
    });
}

/// Returns the bounds of the trait object (or `impl Trait`) that `ty` ends in, through pointees and
/// function pointer outputs: rustc rejects a `+` after `ty`, which would join them
/// (`Fn() -> &dyn A + B`, E0225). A parenthesized type ends the search.
fn tail_object(ty: &Type) -> Option<&Punctuated<TypeParamBound, Token![+]>> {
    match ty {
        Type::Reference(TypeReference { elem, .. }) | Type::Ptr(TypePtr { elem, .. }) => {
            tail_object(elem)
        }
        Type::BareFn(TypeBareFn {
            output: ReturnType::Type(_, output),
            ..
        }) => tail_object(output),
        Type::TraitObject(TypeTraitObject { bounds, .. })
        | Type::ImplTrait(TypeImplTrait { bounds, .. }) => Some(bounds),
        _ => None,
    }
}

/// The output type of a bound that is the sugar of `Fn`: `R` in `Fn(A) -> R`.
fn bound_output_mut(bound: &mut TypeParamBound) -> Option<&mut Type> {
    match bound {
        // A parenthesized bound `(Fn() -> R)` is closed already.
        TypeParamBound::Trait(TraitBound {
            paren_token: None,
            path,
            ..
        }) => match &mut path.segments.last_mut()?.arguments {
            PathArguments::Parenthesized(ParenthesizedGenericArguments {
                output: ReturnType::Type(_, output),
                ..
            }) => Some(output),
            _ => None,
        },
        _ => None,
    }
}

/// Returns the bounds sorted, with parentheses around a trait object in the output of `Fn(A) -> R`
/// exactly where it needs them, so that the result does not depend on how they were written.
pub(crate) fn canonical_bounds(
    bounds: Punctuated<TypeParamBound, Token![+]>,
) -> Punctuated<TypeParamBound, Token![+]> {
    // Sorted without the parentheses, so that it is idempotent.
    let bounds = sorted_by_tokens(bounds.into_iter().map(|mut bound| {
        // Takes off the parentheses around a type that ends in a trait object.
        if let Some(output) = bound_output_mut(&mut bound) {
            while let Type::Paren(TypeParen { elem, .. }) = output {
                if tail_object(elem).is_none() {
                    break;
                }
                *output = std::mem::replace(&mut **elem, Type::Verbatim(TokenStream::new()));
            }
        }
        bound
    }));
    let count = bounds.len();
    bounds
        .into_iter()
        .enumerate()
        .map(|(ix, mut bound)| {
            let followed = ix + 1 < count;
            // The output of `Fn(A) -> R` is parsed without a `+`: a trait object there needs
            // parentheses if a bound follows it (`Fn() -> &dyn A + B`) or it has several bounds.
            if let Some(output) = bound_output_mut(&mut bound) {
                if tail_object(output).map_or(false, |object| followed || object.len() > 1) {
                    parenthesize(output);
                }
            }
            bound
        })
        .collect()
}

/// Parenthesizes the pointee of a reference or a pointer if it is a trait object with several
/// bounds: `&(dyn A + B)`. A type that replaces `Self` can be one.
pub(crate) fn parenthesize_pointee(mut ty: Type) -> Type {
    if let Type::Reference(TypeReference { elem, .. }) | Type::Ptr(TypePtr { elem, .. }) = &mut ty {
        let several = match &**elem {
            Type::TraitObject(TypeTraitObject { bounds, .. })
            | Type::ImplTrait(TypeImplTrait { bounds, .. }) => bounds.len() > 1,
            _ => false,
        };
        if several {
            parenthesize(elem);
        }
    }
    ty
}

/// Parenthesizes the bounded type of a where predicate if it is a function pointer without an
/// output behind a reference or a pointer (`&fn(A): 'a`), whose `:` rustc takes for the start of
/// an output. A type that replaces `Self` can be one.
pub(crate) fn parenthesize_bounded(ty: &mut Type) {
    let mut under_pointer = false;
    let mut current = &*ty;
    let ambiguous = loop {
        match current {
            Type::Reference(TypeReference { elem, .. }) | Type::Ptr(TypePtr { elem, .. }) => {
                under_pointer = true;
                current = elem;
            }
            Type::BareFn(TypeBareFn {
                output: ReturnType::Type(_, output),
                ..
            }) => current = output,
            Type::BareFn(_) => break under_pointer,
            _ => break false,
        }
    };
    if ambiguous {
        parenthesize(ty);
    }
}

/// Parenthesizes the bounded type of the first predicate of `clause` if it is a qualified path
/// without `as` (`where (<Host>::A): Clone`): rustc reads `where <Host>` as generics of the
/// `where`. A type that replaces `Self` or a parameter can be one.
pub(crate) fn parenthesize_leading_qualified_path(clause: &mut WhereClause) {
    // Only the first predicate follows the `where`, and a binder (`for<'a>`) would come between.
    if let Some(WherePredicate::Type(PredicateType {
        lifetimes: None,
        bounded_ty,
        ..
    })) = clause.predicates.first_mut()
    {
        if matches!(bounded_ty, Type::Path(TypePath { qself: Some(q), .. }) if q.as_token.is_none())
        {
            parenthesize(bounded_ty);
        }
    }
}

pub(crate) fn has_lifetime_bound(object: &TypeTraitObject) -> bool {
    object
        .bounds
        .iter()
        .any(|bound| matches!(bound, TypeParamBound::Lifetime(_)))
}

/// Returns the trait object that `ty` is, also inside parentheses and groups.
pub(crate) fn trait_object_of(ty: &Type) -> Option<&TypeTraitObject> {
    match unwrap_type_group(ty) {
        Type::TraitObject(object) => Some(object),
        _ => None,
    }
}

/// Returns `true` for a reference with a lifetime, which its components cannot stand for: the copy
/// needs a bound (`Rel: 'static`) that its referent's copy may lack. Only `'static` needs it, but
/// every lifetime gets it; a trait object referent is exempt (see `Inherits::Reference`).
pub(crate) fn is_whole_reference(ty: &Type) -> bool {
    match unwrap_type_group(ty) {
        Type::Reference(TypeReference {
            lifetime: Some(_),
            elem,
            ..
        }) => !matches!(unwrap_type_group(elem), Type::TraitObject(_)),
        _ => false,
    }
}

/// Returns `items` sorted by the text of their tokens, the first of equal ones kept: the order that
/// makes bounds, predicates and lifetimes independent of how they were written.
pub(crate) fn sorted_by_tokens<T: ToTokens>(items: impl IntoIterator<Item = T>) -> Vec<T> {
    let mut items: Vec<(String, T)> = items
        .into_iter()
        .map(|item| (quote!(#item).to_string(), item))
        .collect();
    items.sort_by(|a, b| a.0.cmp(&b.0));
    items.dedup_by(|a, b| a.0 == b.0);
    items.into_iter().map(|(_, item)| item).collect()
}

/// Prints tokens of a path or a type as one writes them (`<Self as Tr>::A`, `Wrap<Rel<T>>`,
/// `m!(Self)`), not with the spaces that a token stream has. For the messages that suggest what to
/// write. Other tokens may lose a space (`a < b` is `a<b`).
pub(crate) fn tidy(tokens: &TokenStream) -> String {
    tidy_str(&tokens.to_string())
}

/// Like [`tidy()`] for the text of tokens.
pub(crate) fn tidy_str(text: &str) -> String {
    let mut text = join_angles(&join_paths(text));
    for (from, to) in [
        (":: ", "::"),
        ("< ", "<"),
        (" >", ">"),
        ("& ", "&"),
        (" ,", ","),
        (" ;", ";"),
    ] {
        text = text.replace(from, to);
    }
    // A macro call: `m ! (..)` and `m! (..)` (how the compiler prints it) are `m!(..)`, but `a && !
    // (b)` keeps its space.
    for open in ['(', '[', '{'] {
        let from = format!("! {}", open);
        let mut joined = String::with_capacity(text.len());
        let mut rest = text.as_str();
        while let Some(at) = rest.find(&from) {
            joined.push_str(&rest[..at]);
            let name = joined.trim_end_matches(' ').len();
            let after_name = joined[..name]
                .chars()
                .next_back()
                .map_or(false, |c| c.is_alphanumeric() || c == '_');
            if after_name {
                joined.truncate(name);
                joined.push('!');
                joined.push(open);
            } else {
                joined.push_str(&from);
            }
            rest = &rest[at + from.len()..];
        }
        joined.push_str(rest);
        text = joined;
    }
    text
}

/// Joins the segments of the paths in `text` (`Rel :: new` is `Rel::new`), but keeps the space
/// before a path that starts after a keyword or a separator (`dyn ::p::Tr`, `A, ::p::X`, `T:
/// ::p::Tr`, `-> ::p::X`).
fn join_paths(text: &str) -> String {
    const KEYWORDS: [&str; 8] = ["as", "dyn", "impl", "for", "in", "mut", "const", "where"];
    let is_word = |c: char| c.is_alphanumeric() || c == '_';
    let mut joined = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find(" :: ") {
        joined.push_str(&rest[..at]);
        let word = joined.rsplit(|c: char| !is_word(c)).next().unwrap_or("");
        let after_path = match joined.chars().next_back() {
            Some(c) if is_word(c) => !KEYWORDS.contains(&word),
            Some('>') => !joined.ends_with("->") && !ends_with_binder(&joined),
            Some(')' | ']') => true,
            _ => false,
        };
        joined.push_str(if after_path { "::" } else { " ::" });
        rest = &rest[at + " :: ".len()..];
    }
    joined.push_str(rest);
    joined
}

/// Joins `<` to the name before it (`Rel < T >` is `Rel<T>`), but not to a lifetime (`&'a <T as
/// Tr>::A`).
fn join_angles(text: &str) -> String {
    let is_word = |c: char| c.is_alphanumeric() || c == '_';
    let mut joined = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find(" <") {
        joined.push_str(&rest[..at]);
        let name = joined.trim_end_matches(is_word).len();
        let lifetime = name < joined.len() && joined[..name].ends_with('\'');
        joined.push_str(if lifetime { " <" } else { "<" });
        rest = &rest[at + " <".len()..];
    }
    joined.push_str(rest);
    joined
}

/// Returns `true` if `text` ends with the binder of lifetimes `for < 'a >`.
fn ends_with_binder(text: &str) -> bool {
    let mut depth = 0usize;
    for (at, c) in text.char_indices().rev() {
        match c {
            '>' if !text[..at].ends_with('-') => depth += 1,
            '<' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    let head = text[..at].trim_end();
                    return head.strip_suffix("for").map_or(false, |before| {
                        !before.ends_with(|c: char| c.is_alphanumeric() || c == '_')
                    });
                }
            }
            _ => (),
        }
    }
    false
}

/// The path without the segments that name a place in the crate (`crate`, `self`, `super`). At
/// least one segment stays.
fn without_crate_prefix(path: &Path) -> Path {
    let skip = path
        .segments
        .iter()
        .take_while(|seg| seg.ident == "crate" || seg.ident == "self" || seg.ident == "super")
        .count()
        .min(path.segments.len().saturating_sub(1));
    Path {
        leading_colon: None,
        segments: path.segments.iter().skip(skip).cloned().collect(),
    }
}

/// An example of the absolute form of the relative `path`, for a message: `core::fmt::Debug` is
/// `::core::fmt::Debug`, and any other path is in the crate (`crate::m::Tr<u8>` is
/// `::my_crate::m::Tr<u8>`). The arguments stay.
pub(crate) fn absolute_example(path: &Path) -> String {
    if path.leading_colon.is_none() && path.segments.len() > 1 && is_std_crate_path(path) {
        let absolute = Path {
            leading_colon: Some(Default::default()),
            segments: path.segments.clone(),
        };
        tidy(&quote!(#absolute))
    } else {
        let rest = without_crate_prefix(path);
        format!("::my_crate::{}", tidy(&quote!(#rest)))
    }
}

/// Joins the ways to fix an error: `a`, `a, or b`, `a, b, or c`.
pub(crate) fn or_list(options: Vec<String>) -> String {
    match options.as_slice() {
        [] => String::new(),
        [only] => only.clone(),
        [init @ .., last] => format!("{}, or {}", init.join(", "), last),
    }
}

/// Returns `true` if `ty` is `Self`, also inside groups and parentheses.
pub(crate) fn is_bare_self(ty: &Type) -> bool {
    matches!(
        unwrap_type_group(ty),
        Type::Path(TypePath { qself: None, path }) if path.is_ident("Self")
    )
}

/// The name of the generic that a where predicate's bounded type is, if it is a bare name.
pub(crate) fn bounded_ident(ty: &Type) -> Option<String> {
    match unwrap_type_group(ty) {
        Type::Path(TypePath { qself: None, path }) => path.get_ident().map(ident_name),
        _ => None,
    }
}

/// Finds the mentions of `Self`: a path starting with it (`Self`, `Self::X`, the self type of
/// `<Self as ..>`), or the name in the tokens of a macro. `::p::Self` is not one.
#[derive(Default)]
struct SelfFinder {
    /// The mentions, as the syntax they are: a type, the trait of a bound or a value.
    found: Vec<Cause>,
    /// The paths of `found` with an item after `Self` (`Self::X`).
    associated: Vec<Cause>,
}

impl SelfFinder {
    /// Scans tokens that are not parsed, such as those of a macro, for the name.
    fn scan(&mut self, tokens: TokenStream) {
        if names_in(tokens).contains("Self") {
            self.found.push(Cause::self_type());
        }
    }

    /// Records `path` if it starts with `Self`, as the syntax `cause` makes of it, then visits its
    /// arguments.
    fn path<'ast>(&mut self, path: &'ast Path, cause: fn(&Path) -> Cause) {
        if self_head(path).is_some() {
            self.found.push(cause(path));
            if path.segments.len() > 1 {
                self.associated.push(cause(path));
            }
        }
        syn::visit::visit_path(self, path)
    }
}

impl<'ast> Visit<'ast> for SelfFinder {
    /// A path that is none of the following: a value (a pattern, a struct literal, an expression).
    fn visit_path(&mut self, i: &'ast Path) {
        self.path(i, Cause::expr_path)
    }
    fn visit_type_path(&mut self, i: &'ast TypePath) {
        if let Some(qself) = &i.qself {
            self.visit_qself(qself);
        }
        self.path(&i.path, Cause::type_path)
    }
    fn visit_trait_bound(&mut self, i: &'ast TraitBound) {
        self.visit_trait_bound_modifier(&i.modifier);
        if let Some(lifetimes) = &i.lifetimes {
            self.visit_bound_lifetimes(lifetimes);
        }
        self.path(&i.path, |path| Cause::Trait(path.clone()))
    }
    fn visit_macro(&mut self, i: &'ast Macro) {
        self.scan(i.tokens.clone());
        syn::visit::visit_macro(self, i)
    }
    fn visit_type(&mut self, i: &'ast Type) {
        if let Type::Verbatim(tokens) = i {
            self.scan(tokens.clone());
        }
        syn::visit::visit_type(self, i)
    }
}

/// The [`SelfFinder`] after `visit`.
fn find_self(visit: impl FnOnce(&mut SelfFinder)) -> SelfFinder {
    let mut finder = SelfFinder::default();
    visit(&mut finder);
    finder
}

pub(crate) fn type_mentions_self(ty: &Type) -> bool {
    !find_self(|f| f.visit_type(ty)).found.is_empty()
}

pub(crate) fn generic_param_mentions_self(param: &GenericParam) -> bool {
    !find_self(|f| f.visit_generic_param(param)).found.is_empty()
}

pub(crate) fn where_predicate_mentions_self(predicate: &WherePredicate) -> bool {
    !find_self(|f| f.visit_where_predicate(predicate))
        .found
        .is_empty()
}

pub(crate) fn bound_mentions_self(bound: &TypeParamBound) -> bool {
    !find_self(|f| f.visit_type_param_bound(bound))
        .found
        .is_empty()
}

/// Returns `predicate` without the bounds that mention `Self`, or `None` if nothing is left: a
/// predicate on a type that mentions `Self` (`Self: Sized`, `Vec<Self>: Clone`) goes whole.
pub(crate) fn without_self_bounds(predicate: &WherePredicate) -> Option<WherePredicate> {
    match predicate {
        WherePredicate::Type(predicate) => {
            if type_mentions_self(&predicate.bounded_ty) {
                return None;
            }
            let bounds: Punctuated<TypeParamBound, Token![+]> = predicate
                .bounds
                .iter()
                .filter(|bound| !bound_mentions_self(bound))
                .cloned()
                .collect();
            (!bounds.is_empty()).then(|| {
                WherePredicate::Type(PredicateType {
                    bounds,
                    ..predicate.clone()
                })
            })
        }
        other => Some(other.clone()).filter(|predicate| !where_predicate_mentions_self(predicate)),
    }
}

/// The mentions of `Self` in a generic parameter or a where predicate.
pub(crate) struct SelfMentions {
    pub(crate) first: Option<Cause>,
    /// The first `Self::X`, an associated item of `Self`.
    pub(crate) associated: Option<Cause>,
}

impl From<SelfFinder> for SelfMentions {
    fn from(finder: SelfFinder) -> Self {
        Self {
            first: finder.found.first().cloned(),
            associated: finder.associated.first().cloned(),
        }
    }
}

pub(crate) fn self_in_param(param: &GenericParam) -> SelfMentions {
    find_self(|f| f.visit_generic_param(param)).into()
}

pub(crate) fn self_in_predicate(predicate: &WherePredicate) -> SelfMentions {
    find_self(|f| f.visit_where_predicate(predicate)).into()
}

pub(crate) fn first_self_in_generics(generics: &Generics) -> Option<Cause> {
    selves_in_generics(generics).into_iter().next()
}

/// Returns the mentions of `Self` in the bounds, defaults and where predicates of `generics`.
pub(crate) fn selves_in_generics(generics: &Generics) -> Vec<Cause> {
    find_self(|f| f.visit_generics(generics)).found
}

pub(crate) fn first_self_in_type(ty: &Type) -> Option<Cause> {
    find_self(|f| f.visit_type(ty)).found.into_iter().next()
}

/// Returns the first `Self::X` in the bounds, defaults and where predicates of `generics`.
pub(crate) fn first_associated_self_in_generics(generics: &Generics) -> Option<Cause> {
    find_self(|f| f.visit_generics(generics))
        .associated
        .into_iter()
        .next()
}

/// Returns `param` without its default.
pub(crate) fn without_default(mut param: GenericParam) -> GenericParam {
    match &mut param {
        GenericParam::Type(param) => {
            param.eq_token = None;
            param.default = None;
        }
        GenericParam::Const(param) => {
            param.eq_token = None;
            param.default = None;
        }
        GenericParam::Lifetime(_) => (),
    }
    param
}

/// The where clause of `predicates`, or `None` if there is none.
pub(crate) fn where_clause(
    predicates: impl IntoIterator<Item = WherePredicate>,
) -> Option<WhereClause> {
    let predicates: Punctuated<WherePredicate, Token![,]> = predicates.into_iter().collect();
    (!predicates.is_empty()).then(|| WhereClause {
        where_token: Default::default(),
        predicates,
    })
}

/// Returns the first `n` parameters without defaults, and the where predicates that do not name the
/// others.
pub(crate) fn without_defaults(generics: &Generics, n: usize) -> Generics {
    let params: Punctuated<GenericParam, Token![,]> = generics
        .params
        .iter()
        .take(n)
        .cloned()
        .map(without_default)
        .collect();
    let later: HashSet<String> = generics
        .params
        .iter()
        .skip(n)
        .map(generic_param_name)
        .collect();
    let predicates = generics
        .where_clause
        .iter()
        .flat_map(|wc| wc.predicates.iter())
        .filter(|predicate| names_in(predicate).is_disjoint(&later))
        .cloned();
    let has_params = !params.is_empty();
    Generics {
        lt_token: has_params.then(Default::default),
        params,
        gt_token: has_params.then(Default::default),
        where_clause: where_clause(predicates),
    }
}

/// Replaces `Self` by a type, in the types, expressions and macros it folds.
#[derive(Clone, Copy)]
pub(crate) struct ReplaceSelf<'a> {
    pub(crate) self_ty: &'a Type,
}

impl<'a> ReplaceSelf<'a> {
    /// Replaces each `Self`. One that a `::` follows is `<host>`, as an expression needs, unless
    /// the host is a path with no arguments.
    pub(crate) fn replace_tokens(&self, tokens: TokenStream) -> TokenStream {
        use proc_macro2::{Group, Spacing, TokenTree};
        let plain_host = matches!(
            self.self_ty,
            Type::Path(TypePath { qself: None, path })
                if path.segments.iter().all(|seg| seg.arguments.is_none())
        );
        let tokens: Vec<TokenTree> = tokens.into_iter().collect();
        let mut written = TokenStream::new();
        for (ix, tt) in tokens.iter().enumerate() {
            match tt {
                TokenTree::Ident(ident) if ident == "Self" => {
                    let path_follows = match (tokens.get(ix + 1), tokens.get(ix + 2)) {
                        (Some(TokenTree::Punct(a)), Some(TokenTree::Punct(b))) => {
                            a.as_char() == ':'
                                && a.spacing() == Spacing::Joint
                                && b.as_char() == ':'
                        }
                        _ => false,
                    };
                    let self_ty = self.self_ty;
                    written.extend(if path_follows && !plain_host {
                        quote!(<#self_ty>)
                    } else {
                        self_ty.to_token_stream()
                    });
                }
                TokenTree::Group(group) => {
                    let mut new =
                        Group::new(group.delimiter(), self.replace_tokens(group.stream()));
                    new.set_span(group.span());
                    written.extend(Some(TokenTree::Group(new)));
                }
                tt => written.extend(Some(tt.clone())),
            }
        }
        written
    }

    pub(crate) fn fold_type_with(&self, folder: &mut impl syn::fold::Fold, ty: Type) -> Type {
        match ty {
            Type::Path(TypePath { qself: None, path }) if plain_self_head(&path).is_some() => {
                if path.segments.len() > 1 {
                    let value = self.self_ty.clone();
                    Type::Path(projection(value, &path, |rest| folder.fold_path(rest)))
                } else {
                    self.self_ty.clone()
                }
            }
            Type::Verbatim(tokens) => Type::Verbatim(self.replace_tokens(tokens)),
            ty => parenthesize_pointee(syn::fold::fold_type(folder, ty)),
        }
    }

    /// `Self` alone in an expression is a value, so only `Self::A` is replaced.
    pub(crate) fn fold_expr_with(&self, folder: &mut impl syn::fold::Fold, expr: Expr) -> Expr {
        match expr {
            Expr::Path(ExprPath {
                attrs,
                qself: None,
                path,
            }) if plain_self_head(&path).is_some() && path.segments.len() > 1 => {
                let value = self.self_ty.clone();
                let TypePath { qself, path } =
                    projection(value, &path, |rest| folder.fold_path(rest));
                Expr::Path(ExprPath { attrs, qself, path })
            }
            expr => syn::fold::fold_expr(folder, expr),
        }
    }
}

impl<'a> syn::fold::Fold for ReplaceSelf<'a> {
    fn fold_type(&mut self, ty: Type) -> Type {
        let this = *self;
        this.fold_type_with(self, ty)
    }

    fn fold_expr(&mut self, expr: Expr) -> Expr {
        let this = *self;
        this.fold_expr_with(self, expr)
    }

    fn fold_predicate_type(&mut self, i: PredicateType) -> PredicateType {
        let mut i = syn::fold::fold_predicate_type(self, i);
        parenthesize_bounded(&mut i.bounded_ty);
        i
    }

    fn fold_macro(&mut self, i: Macro) -> Macro {
        Macro {
            tokens: self.replace_tokens(i.tokens),
            ..i
        }
    }
}
