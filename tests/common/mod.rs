use proc_macro2::{TokenStream, TokenTree};
use std::collections::BTreeSet;
use syn::parse::{ParseStream, Parser};
use syn::punctuated::Punctuated;
use syn::*;
use template_quote::{quote, ToTokens};
use type_leak::{
    substitute_args, Allow, Cause, CheckResult, Context, ContextId, EncodeError, EntryId,
    ErrorItem, ImplSelf, Leaker, LeakerResult, Place, Projection, Referrer, RepeaterImpl, Rules,
    SelfRequirement, Substitute, View, Visitor,
};

/// The tokens of `x`, as text.
#[allow(dead_code)]
pub fn s(x: &impl ToTokens) -> String {
    quote!(#x).to_string()
}

/// The messages of the errors, one per line.
#[allow(dead_code)]
pub fn messages(errors: &[ErrorItem]) -> String {
    errors
        .iter()
        .map(|item| item.message.as_str())
        .collect::<Vec<_>>()
        .join("\n")
}

/// The leaked types with their generics (and where clause), in the order of the Referrer.
#[allow(dead_code)]
pub fn entries_with_generics(referrer: &Referrer) -> Vec<(String, String)> {
    referrer
        .iter()
        .map(|(_, ty, context)| (ty, &context.generics))
        .map(|(ty, generics)| (s(ty), s(&quote!(#generics #{&generics.where_clause}))))
        .collect()
}

/// The message of the panic of `f`.
#[allow(dead_code)]
pub fn panic_message(f: impl FnOnce()) -> String {
    let payload =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).expect_err("it did not panic");
    match payload.downcast_ref::<String>() {
        Some(message) => message.clone(),
        None => payload
            .downcast_ref::<&str>()
            .expect("the message is a string")
            .to_string(),
    }
}

/// Converts the types of a `Referrer` to a `BTreeSet<String>` for order-independent comparison.
#[allow(dead_code)]
pub fn finish_as_strings<A: Allow>(leaker: Leaker<A>) -> BTreeSet<String> {
    let referrer = leaker.result().finish().unwrap();
    referrer
        .iter()
        .map(|(_, t, _)| quote!(#t).to_string())
        .collect()
}

/// The types of the entries, by index (`Referrer::iter()` gives them with their contexts).
#[allow(dead_code)]
pub fn types(referrer: &Referrer) -> impl Iterator<Item = &Type> {
    referrer.iter().map(|(_, ty, _)| ty)
}

/// The leaked types, as text, in the order of the Referrer.
#[allow(dead_code)]
pub fn type_texts(referrer: &Referrer) -> Vec<String> {
    types(referrer).map(s).collect()
}

/// What `Self` means in the context of an entry, as text: `-` (nothing), `Self: Tr` (the implementor of
/// a trait) or `Self = T` (a type).
#[allow(dead_code)]
pub fn self_text(context: &Context) -> String {
    match context.self_ty.as_ref() {
        None => "-".to_string(),
        Some(SelfRequirement::Bound(path)) => format!("Self: {}", s(path)),
        Some(SelfRequirement::Type(ty)) => format!("Self = {}", s(ty)),
        Some(other) => format!("{:?}", other),
    }
}

/// What `Self` means in the `ix`-th entry ([`self_text()`]).
#[allow(dead_code)]
pub fn self_of(referrer: &Referrer, ix: usize) -> String {
    self_text(entry(referrer, ix))
}

/// What `Self` means in each entry ([`self_text()`]).
#[allow(dead_code)]
pub fn selfs(referrer: &Referrer) -> Vec<String> {
    referrer
        .iter()
        .map(|(_, _, context)| self_text(context))
        .collect()
}

/// The tag of the format of a serialized Referrer, which it starts with (as text, without quotes).
#[allow(dead_code)]
pub fn format_tag() -> String {
    format!("type-leak-format-{}", Referrer::FORMAT_VERSION)
}

/// The path of the Repeater of the tests that write an impl.
#[allow(dead_code)]
pub fn repeater() -> Path {
    parse_quote!(crate::Repeater)
}

/// A type written as it is printed, to compare with an encoding.
#[allow(dead_code)]
pub fn t(ty: Type) -> String {
    s(&ty)
}

/// The generic arguments of `texts`, parsed.
#[allow(dead_code)]
pub fn arguments(texts: &[&str]) -> Vec<GenericArgument> {
    texts.iter().map(|text| parse_str(text).unwrap()).collect()
}

/// The paths of an error, as text.
#[allow(dead_code)]
pub fn path_texts(error: &ErrorItem) -> Vec<String> {
    error.paths.iter().map(s).collect()
}

/// The parts that the Referrer carries, as it serializes them after the format tag: `KIND (CAUSE)
/// CONTEXT [predicate] [in (TYPE) CONTEXT]`, one string for each.
#[allow(dead_code)]
pub fn parts_of(referrer: &Referrer) -> Vec<String> {
    let mut tokens = template_quote::ToTokens::to_token_stream(referrer)
        .into_iter()
        .skip(1);
    let list = match tokens.next() {
        Some(TokenTree::Group(group)) if group.delimiter() == proc_macro2::Delimiter::Bracket => {
            group.stream()
        }
        _ => return Vec::new(),
    };
    let (mut parts, mut current) = (Vec::new(), TokenStream::new());
    for token in list {
        match &token {
            TokenTree::Punct(punct) if punct.as_char() == ',' => {
                parts.push(std::mem::take(&mut current).to_string());
            }
            _ => current.extend([token]),
        }
    }
    if !current.is_empty() {
        parts.push(current.to_string());
    }
    parts
}

/// The config of most tests: primitive types and the names of the standard prelude are allowed.
#[allow(dead_code)]
pub fn std_config() -> Rules {
    Rules::new().primitive().prelude()
}

/// A Leaker with [`std_config()`].
#[allow(dead_code)]
pub fn std_leaker() -> Leaker<Rules> {
    Leaker::with_rule(std_config())
}

/// A Leaker that allows the paths `allowed` (and what [`std_config()`] allows).
#[allow(dead_code)]
pub fn leaker_with(allowed: &[Path]) -> Leaker<impl Allow> {
    let allowed = allowed.to_vec();
    let config = Rules::new()
        .primitive()
        .prelude()
        .or_if(move |cause: &Cause, place: Place| {
            allowed.iter().any(|a| starts_with(a.clone())(cause, place))
        });
    Leaker::with_rule(config)
}

/// Like [`leaker_with()`], with the paths as text.
#[allow(dead_code)]
pub fn leaker_allowing(allowed: &[&str]) -> Leaker<impl Allow> {
    let allowed: Vec<Path> = allowed
        .iter()
        .map(|path| parse_str(path).unwrap())
        .collect();
    leaker_with(&allowed)
}

/// An entry as the pair of its type and its generics, as text.
#[allow(dead_code)]
pub fn e(ty: &str, generics: &str) -> (String, String) {
    (ty.to_string(), generics.to_string())
}

/// The index of the entry whose type is `text` (as `s()` prints it).
#[allow(dead_code)]
pub fn entry_index(referrer: &Referrer, text: &str) -> usize {
    types(referrer)
        .position(|ty| s(ty) == text)
        .unwrap_or_else(|| panic!("no entry `{}`", text))
}

/// An entry as the pair of its type and its generics (with the where clause), as text.
#[allow(dead_code)]
pub fn entry_with_generics(ty: Type, generics: Generics) -> (String, String) {
    (s(&ty), s(&quote!(#generics #{&generics.where_clause})))
}

/// An identifier with the span of the call site.
#[allow(dead_code)]
pub fn ident(name: &str) -> Ident {
    Ident::new(name, proc_macro2::Span::call_site())
}

/// Rewrites each leaked type to `P<index>` (the index has no suffix): the `replace` of a visitor.
#[allow(dead_code)]
pub fn placeholder_unsuffixed(_: Type, ix: EntryId, _: &Context) -> Type {
    let ix = proc_macro2::Literal::usize_unsuffixed(ix.index());
    parse_quote!(P<#ix>)
}

/// Rewrites each leaked type to `P<ix>` (the index has the suffix `usize`): the `replace` of a
/// visitor.
#[allow(dead_code)]
pub fn placeholder(_: Type, ix: EntryId, _: &Context) -> Type {
    parse_quote!(P<#ix>)
}

/// The causes of the errors, as text.
#[allow(dead_code)]
pub fn cause_texts(errors: &[ErrorItem]) -> Vec<String> {
    errors.iter().map(|error| s(&error.cause)).collect()
}

/// The generics with their where clause, as text.
#[allow(dead_code)]
pub fn with_where(generics: &Generics) -> String {
    format!("{} {}", s(generics), s(&generics.where_clause))
}

/// The config of the property test: [`std_config()`] and the paths of `::p` are allowed.
#[allow(dead_code)]
pub fn p_config() -> Rules<impl Allow + Clone> {
    std_config().or_if(starts_with(parse_quote!(::p)))
}

/// A Leaker with [`p_config()`].
#[allow(dead_code)]
pub fn p_leaker() -> Leaker<impl Allow> {
    Leaker::with_rule(p_config())
}

/// The Referrer of a new Leaker that `f` analyzes with. Panics if there is an error.
#[allow(dead_code)]
pub fn leak_with(f: impl FnOnce(&mut Leaker)) -> Referrer {
    let mut leaker = Leaker::new();
    f(&mut leaker);
    leaker.result().finish().unwrap()
}

/// Runs `analyze` on `leaker` and returns its result.
#[allow(dead_code)]
pub fn run<A: Allow, R>(
    mut leaker: Leaker<A>,
    analyze: impl FnOnce(&mut Leaker<A>) -> R,
) -> LeakerResult {
    analyze(&mut leaker);
    leaker.result()
}

/// The errors that `leaker` has after `analyze` ran on it.
#[allow(dead_code)]
pub fn errors_in<A: Allow, R>(
    leaker: Leaker<A>,
    analyze: impl FnOnce(&mut Leaker<A>) -> R,
) -> Vec<ErrorItem> {
    run(leaker, analyze).errors().to_vec()
}

/// The `EntryId` of the `n`-th entry, counted from 0 (`Referrer::iter()` gives it).
#[allow(dead_code)]
pub fn entry_id_at(referrer: &Referrer, n: usize) -> EntryId {
    referrer.iter().nth(n).expect("no such entry").0
}

/// The context of the `ix`-th entry (`Referrer::iter()` gives it with the type).
#[allow(dead_code)]
pub fn entry(referrer: &Referrer, ix: usize) -> &Context {
    referrer.iter().nth(ix).expect("no such entry").2
}

/// The name of a generic parameter, as `substitute_args()` asks for it.
#[allow(dead_code)]
fn param_ident(param: &GenericParam) -> &Ident {
    match param {
        GenericParam::Lifetime(param) => &param.lifetime.ident,
        GenericParam::Type(param) => &param.ident,
        GenericParam::Const(param) => &param.ident,
    }
}

/// The projection through the Repeater at `path` for the Leaker `id`, with the impls for a
/// parameter `__S`: for the tests where only the path, the id and the key matter.
#[allow(dead_code)]
pub fn projection(path: Path, id: u64) -> Projection {
    Projection::new(path, id, ImplSelf::Param(parse_quote!(__S)))
}

/// The `KEY` of the Repeater for `entry` of the Leaker `id`.
#[allow(dead_code)]
pub fn key(id: u64, entry: EntryId) -> u128 {
    projection(parse_quote!(Repeater), id).key(entry)
}

thread_local! {
    static PROJECTION: &'static Projection =
        Box::leak(Box::new(projection(parse_quote!(Repeater), 0)));
}

/// The projection that the tests give to `Referrer::visitor()`: the Repeater at `Repeater`, the
/// id 0, impls for a parameter `__S`. It is leaked (one for each thread), so a visitor may borrow it
/// as long as a test needs.
#[allow(dead_code)]
pub fn test_projection() -> &'static Projection {
    PROJECTION.with(|projection| *projection)
}

/// The entry of a path through the Repeater that the visitor wrote,
/// `<S as Repeater<ARGS, KEY>>::Type` (the entry is in the low half of the `KEY`,
/// `Projection::key()`). `None` for another type.
#[allow(dead_code)]
pub fn written_entry(ty: &Type) -> Option<EntryId> {
    let Type::Path(TypePath {
        qself: Some(qself),
        path,
    }) = ty
    else {
        return None;
    };
    let repeater = path.segments.iter().nth(qself.position.checked_sub(1)?)?;
    let PathArguments::AngleBracketed(args) = &repeater.arguments else {
        return None;
    };
    if repeater.ident != "Repeater" || args.args.len() != 2 {
        return None;
    }
    let key: u128 = match args.args.last()? {
        GenericArgument::Const(Expr::Lit(ExprLit {
            lit: Lit::Int(key), ..
        })) => key.base10_parse().ok()?,
        _ => return None,
    };
    let low = (key as u64) as usize;
    Some(parse_quote!(#low))
}

/// Turns each path through the Repeater that the visitor wrote back into what `replace` makes of
/// its entry: `replace(leaked, entry, context)` with the type of the entry as the Referrer has it,
/// as the closure of `Referrer::visitor()` was called before it held the `Projection`. Apply it to
/// the node that the visitor visited, with the method that visited it.
///
/// With `host` (`Visitor::with_self()`), `Self` in the type of the entry is the host.
#[allow(dead_code)]
struct Replace<'r, F> {
    referrer: &'r Referrer,
    replace: F,
    host: Option<Type>,
}

impl<'r, F: FnMut(Type, EntryId, &Context) -> Type> Replace<'r, F> {
    fn new(referrer: &'r Referrer, replace: F) -> Self {
        Self {
            referrer,
            replace,
            host: None,
        }
    }
}

impl<F: FnMut(Type, EntryId, &Context) -> Type> syn::visit_mut::VisitMut for Replace<'_, F> {
    fn visit_type_mut(&mut self, ty: &mut Type) {
        let entry = match written_entry(ty) {
            Some(entry) => entry,
            None => return syn::visit_mut::visit_type_mut(self, ty),
        };
        let (_, leaked, context) = self
            .referrer
            .iter()
            .find(|(ix, _, _)| *ix == entry)
            .expect("the visitor wrote the path of an entry of the referrer");
        let mut leaked = leaked.clone();
        if let Some(host) = &self.host {
            syn::visit_mut::VisitMut::visit_type_mut(&mut HostForSelf(host), &mut leaked);
        }
        *ty = (self.replace)(leaked, entry, context);
    }
}

/// `Self` is the host, where it is a type of its own.
#[allow(dead_code)]
struct HostForSelf<'a>(&'a Type);

impl syn::visit_mut::VisitMut for HostForSelf<'_> {
    fn visit_type_mut(&mut self, ty: &mut Type) {
        match ty {
            Type::Path(TypePath { qself: None, path }) if path.is_ident("Self") => {
                *ty = self.0.clone()
            }
            _ => syn::visit_mut::visit_type_mut(self, ty),
        }
    }
}

/// A visitor of `Referrer::visitor(context, test_projection())` that turns what it writes into what
/// `replace` makes of each entry (see [`Replace`]) after each visit: the closure that
/// `Referrer::visitor()` took before it held the `Projection`.
///
/// It forwards the methods of `VisitMut` that the tests call, and `with_outer_generics()`,
/// `with_self()` and `with_nested_self()`.
#[allow(dead_code)]
pub struct Replacing<'r, F, N = fn(&SelfRequirement) -> Option<Type>> {
    visitor: Option<Visitor<'r, N>>,
    replace: Replace<'r, F>,
}

/// [`Replacing`] for the entries of `referrer`, in the scope `context`.
#[allow(dead_code)]
pub fn visitor<'r, F: FnMut(Type, EntryId, &Context) -> Type>(
    referrer: &'r Referrer,
    context: ContextId,
    replace: F,
) -> Replacing<'r, F> {
    Replacing {
        visitor: Some(referrer.visitor(context, test_projection())),
        replace: Replace::new(referrer, replace),
    }
}

impl<'r, F, N: FnMut(&SelfRequirement) -> Option<Type>> Replacing<'r, F, N> {
    /// `Visitor::with_outer_generics()`.
    #[allow(dead_code)]
    pub fn with_outer_generics(mut self) -> Self {
        self.visitor = self.visitor.take().map(Visitor::with_outer_generics);
        self
    }

    /// `Visitor::with_self()`, which must not fail here.
    #[allow(dead_code)]
    pub fn with_self(mut self, host: Type) -> Self {
        let visitor = self.visitor.take().expect("the visitor");
        self.visitor = Some(visitor.with_self(host.clone()).expect("a host that fits"));
        self.replace.host = Some(host);
        self
    }

    /// `Visitor::with_nested_self()`.
    #[allow(dead_code)]
    pub fn with_nested_self<M: FnMut(&SelfRequirement) -> Option<Type>>(
        self,
        hook: M,
    ) -> Replacing<'r, F, M> {
        Replacing {
            visitor: self.visitor.map(|visitor| visitor.with_nested_self(hook)),
            replace: self.replace,
        }
    }
}

macro_rules! forward {
    ($($method:ident: $node:ty),* $(,)?) => {
        $(
            fn $method(&mut self, node: &mut $node) {
                let visitor = self.visitor.as_mut().expect("the visitor");
                syn::visit_mut::VisitMut::$method(visitor, node);
                syn::visit_mut::VisitMut::$method(&mut self.replace, node);
            }
        )*
    };
}

impl<F: FnMut(Type, EntryId, &Context) -> Type, N: FnMut(&SelfRequirement) -> Option<Type>>
    syn::visit_mut::VisitMut for Replacing<'_, F, N>
{
    forward! {
        visit_item_mut: Item,
        visit_item_trait_mut: ItemTrait,
        visit_signature_mut: Signature,
        visit_file_mut: File,
        visit_item_impl_mut: ItemImpl,
        visit_item_mod_mut: ItemMod,
        visit_item_struct_mut: ItemStruct,
        visit_type_mut: Type,
        visit_impl_item_fn_mut: ImplItemFn,
        visit_item_fn_mut: ItemFn,
        visit_generic_argument_mut: GenericArgument,
        visit_where_clause_mut: WhereClause,
        visit_type_param_bound_mut: TypeParamBound,
        visit_trait_item_type_mut: TraitItemType,
        visit_trait_item_mut: TraitItem,
        visit_path_mut: Path,
        visit_impl_item_mut: ImplItem,
        visit_fields_mut: Fields,
    }
}

/// The primitive type that `ImplSelf::Bucket` is for as the self type of the impl of the entry
/// `ix` for the Leaker `id`, or `None` if the entry needs `Self`: what `Projection::of()` writes the
/// entry for, which is the type it is given for an entry that needs `Self`.
#[allow(dead_code)]
pub fn bucket_self(referrer: &Referrer, id: u64, ix: EntryId) -> Option<Type> {
    let projection = Projection::new(
        parse_quote!(Repeater),
        id,
        ImplSelf::Bucket {
            param: parse_quote!(__S),
        },
    );
    let probe: Type = parse_quote!(__Probe);
    let (_, _, context) = referrer.iter().find(|(entry, _, _)| *entry == ix)?;
    match projection.of(&probe, context, ix) {
        Ok(Type::Path(TypePath {
            qself: Some(qself), ..
        })) if *qself.ty != probe => Some(*qself.ty),
        _ => None,
    }
}

/// What the Repeater at `repeater` makes of a const `value` of the type `ty`: the slot of the
/// entry `Wrap<{ N }>` of the scope `<const N: ty>`, as written at a site that gives `N` the
/// `value`.
#[allow(dead_code)]
pub fn slot(repeater: &Path, value: &Expr, ty: &Type) -> std::result::Result<Type, EncodeError> {
    let generics: Generics = parse_quote!(<const N: #ty>);
    let mut leaker = Leaker::new();
    leaker.enter_generics(&generics, |view| {
        view.visit_type(&parse_quote!(Wrap<{ N }>))
    });
    let referrer = leaker.result().finish().unwrap();
    let (entry, _, context) = referrer.iter().next().expect("the entry Wrap<{ N }>");
    let own = projection(repeater.clone(), 0).of(&parse_quote!(Self), context, entry)?;
    let site = |name: &Ident| (name == "N").then(|| GenericArgument::Const(value.clone()));
    match projection_args(&own) {
        Type::Tuple(tuple) => {
            let mut arg = GenericArgument::Type(tuple.elems.first().expect("one slot").clone());
            substitute_args(std::iter::once(&mut arg), site)?;
            match arg {
                GenericArgument::Type(ty) => Ok(ty),
                other => Err(EncodeError::UnexpectedArgument(Box::new(other))),
            }
        }
        other => panic!("not a tuple: {}", quote!(#other)),
    }
}

/// A site where every generic is passed by its own name (the scope of the analysis).
#[allow(dead_code)]
pub fn identity(_: &Ident) -> Option<GenericArgument> {
    None
}

/// What a site answers for a name: the argument at the position of the parameter of that name in
/// `params`, if `args` reach it. Pass it to `substitute_args()`.
#[allow(dead_code)]
pub fn answers<'a>(
    params: &'a Punctuated<GenericParam, Token![,]>,
    args: &'a [GenericArgument],
) -> impl Fn(&Ident) -> Option<GenericArgument> + 'a {
    move |name| {
        let at = params.iter().position(|param| param_ident(param) == name)?;
        args.get(at).cloned()
    }
}

/// The argument that passes `param` on by its own name.
#[allow(dead_code)]
pub fn own_name_arg(param: &GenericParam) -> GenericArgument {
    match param {
        GenericParam::Lifetime(param) => GenericArgument::Lifetime(param.lifetime.clone()),
        GenericParam::Type(param) => {
            let ident = &param.ident;
            GenericArgument::Type(parse_quote!(#ident))
        }
        GenericParam::Const(param) => {
            let ident = &param.ident;
            GenericArgument::Const(parse_quote!(#ident))
        }
    }
}

/// The arguments for `generics` at a site that answers names with `f`: the generics as arguments,
/// each by its own name, substituted (`substitute_args()`).
#[allow(dead_code)]
pub fn generic_args(
    generics: &Generics,
    f: impl FnMut(&Ident) -> Option<GenericArgument>,
) -> std::result::Result<Vec<GenericArgument>, EncodeError> {
    let mut args: Vec<GenericArgument> = generics.params.iter().map(own_name_arg).collect();
    substitute_args(args.iter_mut(), f)?;
    Ok(args)
}

/// `ty` written at a site that answers names with `f` and `Self` with `self_ty` (`None`: `Self`
/// stays): `substitute_args()` on `ty` as an argument.
#[allow(dead_code)]
pub fn substitute_type(
    ty: &Type,
    mut f: impl FnMut(&Ident) -> Option<GenericArgument>,
    self_ty: Option<&Type>,
) -> std::result::Result<Type, EncodeError> {
    let mut arg = GenericArgument::Type(ty.clone());
    substitute_args(std::iter::once(&mut arg), |name: &Ident| {
        if name == "Self" {
            self_ty.map(|ty| GenericArgument::Type(ty.clone()))
        } else {
            f(name)
        }
    })?;
    match arg {
        GenericArgument::Type(ty) => Ok(ty),
        other => Err(EncodeError::UnexpectedArgument(Box::new(other))),
    }
}

/// The first argument of the Repeater, `ARGS` of `<S as Repeater<ARGS, KEY>>::Type` (the Repeater
/// may be a path of several segments: it is the one before `Type`).
#[allow(dead_code)]
pub fn projection_args(projection: &Type) -> Type {
    match projection {
        Type::Path(TypePath {
            qself: Some(_),
            path,
        }) => match &path.segments[path.segments.len() - 2].arguments {
            PathArguments::AngleBracketed(args) => match &args.args[0] {
                GenericArgument::Type(ty) => ty.clone(),
                other => panic!("not a type: {}", quote!(#other)),
            },
            other => panic!("not the arguments of a trait: {}", quote!(#other)),
        },
        other => panic!("not a projection: {}", quote!(#other)),
    }
}

/// The self type that a projection `<S as Repeater<..>>::Type` is written for: `S`.
#[allow(dead_code)]
pub fn projection_self(projection: &Type) -> Type {
    match projection {
        Type::Path(TypePath {
            qself: Some(qself), ..
        }) => (*qself.ty).clone(),
        other => panic!("not a projection: {}", quote!(#other)),
    }
}

/// The errors as one `syn::Error`, one for each in order. Panics if there is none.
#[allow(dead_code)]
pub fn to_syn_error(items: &[ErrorItem]) -> syn::Error {
    let mut errors = items.iter().cloned().map(syn::Error::from);
    let mut combined = errors.next().expect("at least one error");
    for error in errors {
        combined.combine(error);
    }
    combined
}

/// The first argument of the Repeater in the path of an impl, `ARGS` of `Repeater<ARGS>`.
#[allow(dead_code)]
pub fn impl_args(r: &RepeaterImpl) -> Type {
    match &r
        .path
        .segments
        .last()
        .expect("the path has a segment")
        .arguments
    {
        PathArguments::AngleBracketed(args) => match args.args.first() {
            Some(GenericArgument::Type(ty)) => ty.clone(),
            _ => panic!("not a type: {}", quote!(#args)),
        },
        other => panic!("not the arguments of a trait: {}", quote!(#other)),
    }
}

/// The first argument of the Repeater for an entry with `context`, as written at a site that
/// answers names with `f`: the one of `Projection::of()` (the generics of the context by their own
/// names, encoded), with the site substituted (`substitute_args()`). `self_ty` is what `Self: 'a`
/// names (`None`: `Self`).
#[allow(dead_code)]
pub fn repeater_args(
    context: &Context,
    f: impl FnMut(&Ident) -> Option<GenericArgument>,
    self_ty: Option<&Type>,
) -> std::result::Result<Type, EncodeError> {
    repeater_args_at(&parse_quote!(Repeater), context, f, self_ty)
}

/// Like [`repeater_args()`], for the Repeater at the path `repeater`.
#[allow(dead_code)]
pub fn repeater_args_at(
    repeater: &Path,
    context: &Context,
    f: impl FnMut(&Ident) -> Option<GenericArgument>,
    self_ty: Option<&Type>,
) -> std::result::Result<Type, EncodeError> {
    let default: Type = parse_quote!(Self);
    let written = projection(repeater.clone(), 0).of(
        self_ty.unwrap_or(&default),
        context,
        parse_str::<EntryId>("0").unwrap(),
    )?;
    substitute_type(&projection_args(&written), f, None)
}

/// Replaces the leaked types in the type `ty`, which is inside the scope `context`, with what
/// `replace` makes of each entry: the visitor of `context` on the type.
#[allow(dead_code)]
pub fn expand(
    referrer: &Referrer,
    context: ContextId,
    mut ty: Type,
    replace: impl FnMut(Type, EntryId, &Context) -> Type,
) -> Type {
    use syn::visit_mut::VisitMut;
    visitor(referrer, context, replace).visit_type_mut(&mut ty);
    ty
}

/// The trait `tr` as the Referrer's crate visits it to write a copy at `context`, where `replace`
/// writes the entries: only its generics are visited, so the defaults of its parameters are the
/// entries written (step 1 of the recipe of `Referrer::visitor()`).
#[allow(dead_code)]
pub fn visited_trait(
    referrer: &Referrer,
    context: ContextId,
    tr: &ItemTrait,
    replace: impl FnMut(Type, EntryId, &Context) -> Type,
) -> ItemTrait {
    use syn::visit_mut::VisitMut;
    let mut shell = tr.clone();
    shell.items.clear();
    shell.supertraits.clear();
    let mut file = File {
        shebang: None,
        attrs: Vec::new(),
        items: vec![Item::Trait(shell)],
    };
    visitor(referrer, context, replace).visit_file_mut(&mut file);
    match file.items.remove(0) {
        Item::Trait(shell) => shell,
        _ => unreachable!(),
    }
}

/// The arguments of an impl of the item with the `generics` analyzed at `context`: `given`
/// completed by `Referrer::complete_args()` (the Repeater is `Repeater`, the id `0`, and a leaked
/// type in a default is `<Self as Repeater<..>>::Type`). Each parameter is read back by writing its
/// own name through the `Substitute`.
#[allow(dead_code)]
pub fn completed_args<'a>(
    referrer: &Referrer,
    context: ContextId,
    generics: &Generics,
    given: impl IntoIterator<Item = &'a GenericArgument>,
) -> std::result::Result<Vec<GenericArgument>, EncodeError> {
    completed_args_for(
        referrer,
        context,
        generics,
        given,
        &projection(parse_quote!(Repeater), 0),
    )
}

/// Like [`completed_args()`], for another `projection`.
#[allow(dead_code)]
pub fn completed_args_for<'a>(
    referrer: &Referrer,
    context: ContextId,
    generics: &Generics,
    given: impl IntoIterator<Item = &'a GenericArgument>,
    projection: &Projection,
) -> std::result::Result<Vec<GenericArgument>, EncodeError> {
    use syn::visit_mut::VisitMut;
    let mut substitute = referrer.complete_args(context, given, projection)?;
    Ok(generics
        .params
        .iter()
        .map(|param| {
            let mut arg = own_name_arg(param);
            substitute.visit_generic_argument_mut(&mut arg);
            arg
        })
        .collect())
}

/// The `Substitute` for an impl of an item that declares `generics`, given the arguments `given`
/// (`Referrer::complete_args()`): the scope is `enter_generics()` of a Leaker that writes every
/// relative path as it is, so a default has no entry and is what it says.
#[allow(dead_code)]
pub fn substitute_for(
    generics: &Generics,
    given: &[GenericArgument],
) -> std::result::Result<Substitute, EncodeError> {
    let mut leaker = Leaker::with_rule(|_: &Cause, _: Place| true);
    let (context, ()) = leaker.enter_generics(generics, |_| ());
    let referrer = leaker.result().finish().unwrap();
    referrer.complete_args(context, given, &projection(parse_quote!(Repeater), 0))
}

/// The arguments for the generics `generics` as an impl of an item that declares them has them:
/// `given` completed as [`substitute_for()`] does it, each parameter read back by writing its own
/// name through the `Substitute`.
#[allow(dead_code)]
pub fn complete_generics(
    generics: &Generics,
    given: &[GenericArgument],
) -> std::result::Result<Vec<GenericArgument>, EncodeError> {
    use syn::visit_mut::VisitMut;
    let mut substitute = substitute_for(generics, given)?;
    Ok(generics
        .params
        .iter()
        .map(|param| {
            let mut arg = own_name_arg(param);
            substitute.visit_generic_argument_mut(&mut arg);
            arg
        })
        .collect())
}

/// Writes what a site answers into the syntax of a copy (step 3 of the recipe of
/// `Referrer::visitor()`): one `substitute_args()` pass for each type, generic argument,
/// expression and lifetime that it visits (what a visited type holds is substituted with it, not
/// again). The first error is kept in `error`.
#[allow(dead_code)]
pub struct Written<F> {
    f: F,
    pub error: Option<EncodeError>,
}

#[allow(dead_code)]
impl<F: FnMut(&Ident) -> Option<GenericArgument>> Written<F> {
    pub fn new(f: F) -> Self {
        Written { f, error: None }
    }

    fn write(&mut self, arg: &mut GenericArgument) {
        if let Err(error) = substitute_args(std::iter::once(arg), &mut self.f) {
            self.error.get_or_insert(error);
        }
    }

    fn misfit(&mut self, arg: GenericArgument) {
        self.error
            .get_or_insert(EncodeError::UnexpectedArgument(Box::new(arg)));
    }

    /// Writes into the fields of `item` (its generics are the declaration of the copy), and returns
    /// it. Panics if a value does not fit.
    pub fn item(mut self, mut item: ItemStruct) -> ItemStruct {
        use syn::visit_mut::VisitMut;
        self.visit_fields_mut(&mut item.fields);
        if let Some(error) = self.error {
            panic!("{}", error);
        }
        item
    }
}

impl<F: FnMut(&Ident) -> Option<GenericArgument>> syn::visit_mut::VisitMut for Written<F> {
    fn visit_type_mut(&mut self, ty: &mut Type) {
        let mut arg = GenericArgument::Type(ty.clone());
        self.write(&mut arg);
        match arg {
            GenericArgument::Type(written) => *ty = written,
            other => self.misfit(other),
        }
    }

    // A bare const parameter is parsed as a type argument (`N` in `Foo<N>`), and becomes a const.
    fn visit_generic_argument_mut(&mut self, arg: &mut GenericArgument) {
        self.write(arg)
    }

    fn visit_lifetime_mut(&mut self, lifetime: &mut Lifetime) {
        let mut arg = GenericArgument::Lifetime(lifetime.clone());
        self.write(&mut arg);
        match arg {
            GenericArgument::Lifetime(written) => *lifetime = written,
            other => self.misfit(other),
        }
    }
}

/// The context id with the number `n`.
#[allow(dead_code)]
pub fn id(n: usize) -> ContextId {
    parse_str(&n.to_string()).unwrap()
}

/// The context ids that a Referrer writes in its serialized form ([`serialized_ids()`]).
#[allow(dead_code)]
pub struct SerializedIds {
    /// The contexts that the tokens declare, in the order of their first group.
    pub declared: Vec<ContextId>,
    /// The context of each entry, by index.
    pub entries: Vec<ContextId>,
}

/// Reads the context ids from `referrer.to_token_stream()`, as `Parse for Referrer` reads them.
///
/// The form is the format tag, the groups of parts (`[ .. ]`) if there are any, the number of groups
/// if there is more than one, and then the groups in the order of the entries. A group is `[# id]`,
/// then, if the context is not declared yet, its declaration (the items `<generics> where .. ;` or
/// `declared <params> where .. ;`, then `Self .. ;` and `type G .. ;`, in this order, each ending
/// with a `;` at the top level of the tokens), then the entries `( Type, .. )`. A group with no
/// `# id` is for the id after that of the group before. A declared context has a group of entries
/// only. It panics with the tokens for any other shape.
#[allow(dead_code)]
pub fn serialized_ids(referrer: &Referrer) -> SerializedIds {
    let tokens = template_quote::ToTokens::to_token_stream(referrer);
    let read = read_serialized_ids(tokens.clone()).unwrap_or_else(|error| {
        panic!(
            "the reader of the serialized form does not know this shape ({}): {}",
            error, tokens
        )
    });
    assert_eq!(
        read.entries.len(),
        referrer.iter().count(),
        "the number of entries that the serialized form has: {}",
        tokens
    );
    read
}

/// The reader of [`serialized_ids()`], on tokens: it fails for a shape it does not know, and for
/// tokens after the groups.
#[allow(dead_code)]
pub fn read_serialized_ids(tokens: TokenStream) -> syn::Result<SerializedIds> {
    parse_serialized_ids.parse2(tokens)
}

fn parse_serialized_ids(input: ParseStream) -> syn::Result<SerializedIds> {
    let tag: LitStr = input.parse()?;
    let expected = format_tag();
    if tag.value() != expected {
        return Err(Error::new(
            tag.span(),
            format!("the tag is not `{}`", expected),
        ));
    }
    if input.peek(token::Bracket) {
        let parts: parse::ParseBuffer<'_>;
        bracketed!(parts in input);
        parts.parse::<TokenStream>()?;
    }
    let n_groups = if input.peek(LitInt) {
        input.parse::<LitInt>()?.base10_parse::<usize>()?
    } else {
        1
    };
    let mut declared: Vec<ContextId> = Vec::new();
    let mut entries = Vec::new();
    let mut next = 0;
    for _ in 0..n_groups {
        let context = if input.peek(Token![#]) {
            input.parse::<Token![#]>()?;
            input.parse::<ContextId>()?
        } else {
            id(next)
        };
        next = context.index() + 1;
        if declared.contains(&context) {
            // Another group of a declared context: its entries only.
            if !input.peek(token::Paren) {
                return Err(input.error("expected the entries of a declared context"));
            }
        } else {
            declared.push(context);
            // The declaration: items that end with `;`, in the order of the stages 0 to 3. Only
            // generics (0) or the generics as declared (1), not both.
            let mut stage = 0;
            while !input.peek(token::Paren) {
                let at = if input.peek(Token![<]) || input.peek(Token![where]) {
                    0
                } else if input
                    .cursor()
                    .ident()
                    .map_or(false, |(word, _)| word == "declared")
                {
                    1
                } else if input.peek(Token![Self]) {
                    2
                } else if input.peek(Token![type]) {
                    3
                } else {
                    return Err(input.error("expected an item of a declaration or the entries"));
                };
                if at < stage {
                    return Err(input.error("an item of a declaration out of its place"));
                }
                stage = match at {
                    0 | 1 => 2,
                    _ => at,
                };
                loop {
                    let token: TokenTree = input.parse()?;
                    if matches!(&token, TokenTree::Punct(punct) if punct.as_char() == ';') {
                        break;
                    }
                }
            }
        }
        let content: parse::ParseBuffer<'_>;
        parenthesized!(content in input);
        let group = punctuated::Punctuated::<Type, Token![,]>::parse_terminated(&content)?;
        entries.extend(group.iter().map(|_| context));
    }
    Ok(SerializedIds { declared, entries })
}

/// The ids of the contexts of the Referrer, in ascending order: those that its serialized form
/// declares ([`serialized_ids()`]), also those with no entry.
#[allow(dead_code)]
pub fn context_ids(referrer: &Referrer) -> Vec<ContextId> {
    let mut ids = serialized_ids(referrer).declared;
    ids.sort();
    ids
}

/// The id of the context of the `ix`-th entry, as the serialized form of the Referrer says
/// ([`serialized_ids()`]). `Referrer::iter()` gives the context, not its id.
#[allow(dead_code)]
pub fn entry_id(referrer: &Referrer, ix: usize) -> ContextId {
    match serialized_ids(referrer).entries.get(ix) {
        Some(context) => *context,
        None => panic!("no such entry: {}", ix),
    }
}

/// Analyzes `item` with the `intern_*` of its kind (a struct, an enum, a union, a trait or an
/// impl). A `mod` stands for the items it holds, nested in a scope of no generics: each has its
/// own `Self`. Panics for another item.
#[allow(dead_code)]
pub fn intern_item<A: Allow>(leaker: &mut Leaker<A>, item: &Item) -> ContextId {
    match item {
        Item::Struct(item) => leaker.intern_struct(item),
        Item::Enum(item) => leaker.intern_enum(item),
        Item::Union(item) => leaker.intern_union(item),
        Item::Trait(item) => leaker.intern_trait(item),
        Item::Impl(item) => leaker.intern_impl(item),
        Item::Mod(item) => {
            leaker
                .enter_generics(&Generics::default(), |view| {
                    for item in &item.content.as_ref().expect("a mod with items").1 {
                        view.visit_item(item);
                    }
                })
                .0
        }
        _ => panic!("unsupported item"),
    }
}

/// The word that names the kind of a cause.
#[allow(dead_code)]
pub fn cause_kind(cause: &Cause) -> &'static str {
    match cause {
        Cause::Type(_) => "type",
        Cause::Trait(_) => "trait",
        Cause::Expr(_) => "expr",
        Cause::Macro(_) => "macro",
        Cause::Lifetime(_) => "lifetime",
        Cause::Unknown(_) => "unknown",
        Cause::Unreadable(_) => "unreadable",
        other => panic!("{:?}", other),
    }
}

/// A cause as `kind(tokens)`.
#[allow(dead_code)]
pub fn kinded(cause: &Cause) -> String {
    format!("{}({})", cause_kind(cause), s(cause))
}

/// What an error says: the headline, the tokens, the notes and the help.
#[allow(dead_code)]
#[derive(Debug, PartialEq, Eq)]
pub struct Shown {
    pub message: String,
    pub tokens: String,
    pub notes: Vec<String>,
    pub help: String,
}

#[allow(dead_code)]
impl Shown {
    pub fn new(message: &str, tokens: &str, notes: &[&str], help: &str) -> Self {
        Self {
            message: message.to_string(),
            tokens: tokens.to_string(),
            notes: notes.iter().map(|note| note.to_string()).collect(),
            help: help.to_string(),
        }
    }
}

/// What each of the errors says: the headline, the tokens, the notes and the help.
#[allow(dead_code)]
pub fn shown(err: &[ErrorItem]) -> Vec<Shown> {
    err.iter()
        .map(|item| Shown {
            message: item.message.clone(),
            tokens: s(&item.cause),
            notes: item.notes.iter().map(|(_, note)| note.clone()).collect(),
            help: item
                .help_text()
                .expect("every error has a help")
                .to_string(),
        })
        .collect()
}

/// Interns the items `items` of a trait `Tr` with the generics `generics` (e.g. `fn f(&self) ->
/// Rel<Self>;`). The errors are those of `Leaker::result()`.
#[allow(dead_code)]
pub fn intern_fns<A: Allow>(
    leaker: &mut Leaker<A>,
    generics: &Generics,
    items: TokenStream,
) -> ContextId {
    let where_clause = &generics.where_clause;
    let item: ItemTrait = parse_quote!(trait Tr #generics #where_clause { #items });
    leaker.intern_trait(&item)
}

/// Analyzes the types of the function `sig`, which is itself the analyzed item, in a scope that
/// declares its generics: its inputs, its output and its where clause (see `View`). The function
/// has no receiver.
#[allow(dead_code)]
pub fn visit_function<A: Allow>(view: &mut View<'_, '_, A>, sig: &Signature) {
    for input in &sig.inputs {
        if let FnArg::Typed(arg) = input {
            view.visit_type(&arg.ty);
        }
    }
    if let ReturnType::Type(_, ty) = &sig.output {
        view.visit_type(ty);
    }
    if let Some(where_clause) = &sig.generics.where_clause {
        view.visit_where_clause(where_clause);
    }
}

/// Checks `ty` in a scope that declares the generics `generics` and nothing else (see
/// `View::check()`). `Self` has no meaning there.
#[allow(dead_code)]
pub fn check_in<A: Allow>(leaker: &mut Leaker<A>, ty: &Type, generics: &Generics) -> CheckResult {
    leaker.enter_generics(generics, |view| view.check(ty)).1
}

/// The path of `cause`, if it is a path without generic arguments, as the analysis asks about one.
#[allow(dead_code)]
pub fn bare_path(cause: &Cause) -> Option<&Path> {
    cause
        .path()
        .filter(|path| path.segments.iter().all(|seg| seg.arguments.is_none()))
}

/// The rule that allows the paths that start with `prefix`: a cause that is a path without generic
/// arguments, as the analysis asks about one (`Cause::starts_with()`). A cause with arguments
/// (`Wrap<Rel>`) is not allowed whole: its paths are.
#[allow(dead_code)]
pub fn starts_with(prefix: Path) -> impl Fn(&Cause, Place) -> bool + Clone {
    move |cause: &Cause, _: Place| bare_path(cause).is_some() && cause.starts_with(&prefix)
}

/// Analyzes the trait `item` with `config`; `check_values` reads the values too.
#[allow(dead_code)]
pub fn analyze_trait<A: Allow>(item: &ItemTrait, config: A, check_values: bool) -> LeakerResult {
    let mut leaker = Leaker::with_rule(config);
    leaker.check_values = check_values;
    leaker.check_bodies = check_values;
    leaker.intern_trait(item);
    leaker.result()
}

/// Analyzes the trait `item` with `config`.
#[allow(dead_code)]
pub fn analyze<A: Allow>(item: &ItemTrait, config: A) -> LeakerResult {
    analyze_trait(item, config, false)
}

/// [`analyze()`] with `check_values`: the values are read too.
#[allow(dead_code)]
pub fn analyze_values<A: Allow>(item: &ItemTrait, config: A) -> LeakerResult {
    analyze_trait(item, config, true)
}

/// The causes of the errors of the result, as text.
#[allow(dead_code)]
pub fn causes_of(result: &LeakerResult) -> Vec<String> {
    cause_texts(result.errors())
}

/// The causes that a rule must allow to accept `error`: each of its paths, or its cause if it has
/// no path.
#[allow(dead_code)]
fn causes_to_allow(error: &ErrorItem) -> Vec<Cause> {
    if error.paths.is_empty() {
        vec![error.cause.clone()]
    } else {
        error.paths.clone()
    }
}

/// The causes that accept the errors of `errors` whose cause is `cause`, by their paths or by the
/// cause (`causes_to_allow()`). If there is no such error, `cause` alone: a rule that changes
/// nothing.
#[allow(dead_code)]
pub fn causes_accepting(errors: &[ErrorItem], cause: &Cause) -> Vec<Cause> {
    let mut found = Vec::new();
    for error in errors.iter().filter(|error| error.cause == *cause) {
        found.extend(causes_to_allow(error));
    }
    if found.is_empty() {
        found.push(cause.clone());
    }
    found
}

/// The rule that accepts the errors of `errors` whose cause is `cause` (`causes_accepting()`): what
/// `|c, _| *c == cause` was when a cause accepted an error whatever its paths were.
#[allow(dead_code)]
pub fn accepting(errors: &[ErrorItem], cause: &Cause) -> impl Fn(&Cause, Place) -> bool + Clone {
    let allowed = causes_accepting(errors, cause);
    move |c: &Cause, _: Place| allowed.contains(c)
}

/// `result.accept_cause(&cause)`: `result.allow(accepting(result.errors(), &cause))`.
#[allow(dead_code)]
pub trait AcceptCause {
    fn accept_cause(self, cause: &Cause) -> Self;
}

impl AcceptCause for LeakerResult {
    fn accept_cause(self, cause: &Cause) -> Self {
        let rule = accepting(self.errors(), cause);
        self.allow(rule)
    }
}

/// Allows every error that remains, until none does, and returns how many rounds it took: each by
/// its paths, or by its cause if it has none (`causes_to_allow()`). The errors are a flat list, so
/// one round is enough.
#[allow(dead_code)]
pub fn allow_every_cause(mut result: LeakerResult) -> (LeakerResult, usize) {
    let mut rounds = 0;
    while !result.errors().is_empty() {
        let each: Vec<Vec<Cause>> = result.errors().iter().map(causes_to_allow).collect();
        for allowed in each {
            result = result.allow(move |c: &Cause, _: Place| allowed.contains(c));
        }
        rounds += 1;
    }
    (result, rounds)
}

/// `result.recovered()`: the Referrer of a result after every error is allowed: the part of each
/// is copied as written. `allow()` drops errors and changes no entry, so it is the Referrer that the
/// analysis made.
#[allow(dead_code)]
pub trait Recovered {
    fn recovered(self) -> Referrer;
}

impl Recovered for LeakerResult {
    #[track_caller]
    fn recovered(self) -> Referrer {
        self.allow(|_: &Cause, _: Place| true)
            .finish()
            .unwrap_or_else(|errors| panic!("errors left: {}", messages(&errors)))
    }
}

/// One of the rules that a config and a result can be given, so that a test can give the same rule
/// to either.
#[allow(dead_code)]
#[derive(Clone)]
pub enum Rule {
    /// The causes that accept the errors of a cause (`causes_accepting()`).
    Cause(Vec<Cause>),
    Path(Path),
    Primitive,
    Prelude,
    StdCrates,
    Crate,
    Values,
    Default,
}

impl Allow for Rule {
    fn allow(&self, cause: &Cause, place: Place) -> bool {
        match self {
            Rule::Cause(allowed) => allowed.contains(cause),
            Rule::Path(prefix) => starts_with(prefix.clone())(cause, place),
            Rule::Primitive => Rules::new().primitive().allow(cause, place),
            Rule::Prelude => Rules::new().prelude().allow(cause, place),
            Rule::StdCrates => Rules::new().std_crates().allow(cause, place),
            Rule::Crate => Rules::new().crate_paths().allow(cause, place),
            Rule::Values => place.in_value,
            Rule::Default => place.in_default,
        }
    }
}

#[allow(dead_code)]
impl Rule {
    /// Analyzes the trait `item` with `config` and the rule given to the config: it is applied
    /// during the analysis. `check_values` reads the values too.
    pub fn analyze_with<A: Allow>(
        &self,
        item: &ItemTrait,
        config: A,
        check_values: bool,
    ) -> LeakerResult {
        analyze_trait(item, (config, self.clone()), check_values)
    }

    /// Gives the rule to the result: it is applied after the analysis.
    pub fn on_result(&self, result: LeakerResult) -> LeakerResult {
        result.allow(self.clone())
    }
}
