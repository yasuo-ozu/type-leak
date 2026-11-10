use proc_macro2::{TokenStream, TokenTree};
use std::collections::BTreeSet;
use syn::parse::{ParseStream, Parser};
use syn::punctuated::Punctuated;
use syn::*;
use template_quote::quote;
use type_leak::{
    array_const, complete_params, repeater_projection, slot_const, substitute_args, Allow, Cause,
    CheckResult, Context, ContextId, EncodeError, EntryId, ErrorItem, ImplSelf, Leaker,
    LeakerResult, ModuleScope, Place, Referrer, Rules, View,
};

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

/// The `EntryId` of the `n`-th entry, counted from 0 (`Referrer::iter()` gives it).
#[allow(dead_code)]
pub fn entry_id_at(referrer: &Referrer, n: usize) -> EntryId {
    referrer.iter().nth(n).expect("no such entry").0
}

/// The type of the `ix`-th entry (`Referrer::iter()` gives it with its context).
#[allow(dead_code)]
pub fn entry_ty(referrer: &Referrer, ix: usize) -> &Type {
    referrer.iter().nth(ix).expect("no such entry").1
}

/// The context of the `ix`-th entry (`Referrer::iter()` gives it with the type).
#[allow(dead_code)]
pub fn entry(referrer: &Referrer, ix: usize) -> &Context {
    referrer.iter().nth(ix).expect("no such entry").2
}

/// The name of a generic parameter, as `substitute_args()` asks for it.
#[allow(dead_code)]
pub fn param_ident(param: &GenericParam) -> &Ident {
    match param {
        GenericParam::Lifetime(param) => &param.lifetime.ident,
        GenericParam::Type(param) => &param.ident,
        GenericParam::Const(param) => &param.ident,
    }
}

/// `slot_const(repeater)` for a const `value` of a parameter of type `ty`.
#[allow(dead_code)]
pub fn slot(repeater: &Path, value: &Expr, ty: &Type) -> std::result::Result<Type, EncodeError> {
    let param: ConstParam = parse_quote!(const N: #ty);
    slot_const(repeater)(value, &param)
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

/// The first argument of the Repeater, `ARGS` of `<S as Repeater<ARGS, KEY>>::Type`.
#[allow(dead_code)]
pub fn projection_args(projection: &Type) -> Type {
    match projection {
        Type::Path(TypePath {
            qself: Some(_),
            path,
        }) => match &path.segments[0].arguments {
            PathArguments::AngleBracketed(args) => match &args.args[0] {
                GenericArgument::Type(ty) => ty.clone(),
                other => panic!("not a type: {}", quote!(#other)),
            },
            other => panic!("not the arguments of a trait: {}", quote!(#other)),
        },
        other => panic!("not a projection: {}", quote!(#other)),
    }
}

/// The first argument of the Repeater for an entry with `context`, as written at a site that
/// answers names with `f`: the one of `repeater_projection()` (the generics of the context by
/// their own names, encoded), with the site substituted (`substitute_args()`). `self_ty` is what
/// `Self: 'a` names (`None`: `Self`).
#[allow(dead_code)]
pub fn repeater_args(
    context: &Context,
    f: impl FnMut(&Ident) -> Option<GenericArgument>,
    self_ty: Option<&Type>,
    const_fn: impl FnMut(&Expr, &ConstParam) -> std::result::Result<Type, EncodeError>,
) -> std::result::Result<Type, EncodeError> {
    let default: Type = parse_quote!(Self);
    let repeater: Path = parse_quote!(Repeater);
    let projection = repeater_projection(
        &repeater,
        self_ty.unwrap_or(&default),
        context,
        0,
        parse_str::<EntryId>("0").unwrap(),
        const_fn,
    )?;
    substitute_type(&projection_args(&projection), f, None)
}

/// The trait `tr` as the Referrer's crate visits it to write a copy at `context`, where `type_fn`
/// writes the entries: only its generics are visited, so the defaults of its parameters are the
/// entries written (step 1 of the recipe of `Referrer::visitor()`).
#[allow(dead_code)]
pub fn visited_trait(
    referrer: &Referrer,
    context: ContextId,
    tr: &ItemTrait,
    type_fn: impl FnMut(Type, EntryId, &Context) -> Type,
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
    referrer.visitor(context, type_fn).visit_file_mut(&mut file);
    match file.items.remove(0) {
        Item::Trait(shell) => shell,
        _ => unreachable!(),
    }
}

/// The arguments of an impl of `tr`: `given` completed by `complete_params()` on the parameters of
/// the visited trait (steps 1 and 2 of the recipe).
#[allow(dead_code)]
pub fn completed_args<'a>(
    referrer: &Referrer,
    context: ContextId,
    tr: &ItemTrait,
    given: impl IntoIterator<Item = &'a GenericArgument>,
    type_fn: impl FnMut(Type, EntryId, &Context) -> Type,
) -> std::result::Result<Vec<GenericArgument>, EncodeError> {
    complete_params(
        &visited_trait(referrer, context, tr, type_fn)
            .generics
            .params,
        given,
    )
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
/// then, if the context is not declared yet, its declaration (the items `<generics> where .. ;`,
/// `Self .. ;`, `type G .. ;` and `mod .. ;`, in this order, each ending with a `;` at the top level
/// of the tokens), then the entries `( Type, .. )`. A group with no `# id` is for the id after that
/// of the group before. A declared context has a group of entries only. It panics with the tokens
/// for any other shape.
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
    let expected = format!("type-leak-format-{}", Referrer::FORMAT_VERSION);
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
            // generics (0) and a module (3) may come once.
            let mut stage = 0;
            while !input.peek(token::Paren) {
                let at = if input.peek(Token![<]) || input.peek(Token![where]) {
                    0
                } else if input.peek(Token![Self]) {
                    1
                } else if input.peek(Token![type]) {
                    2
                } else if input.peek(Token![mod]) {
                    3
                } else {
                    return Err(input.error("expected an item of a declaration or the entries"));
                };
                if at < stage {
                    return Err(input.error("an item of a declaration out of its place"));
                }
                stage = if at == 0 || at == 3 { at + 1 } else { at };
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

/// The module that `context` is in (`ModuleScope::In`). It is `None` for a context that depends on
/// no module and for the context of a module itself.
#[allow(dead_code)]
pub fn in_module(context: &Context) -> Option<ContextId> {
    match &context.module {
        Some(ModuleScope::In(module)) => Some(*module),
        _ => None,
    }
}

/// The module of the `ix`-th entry, as `LeakerResult::repeater_impls()` gives it in
/// `RepeaterImpl::module`: the `ModuleScope::Itself` of the module its impl goes in, or `None` if
/// the entry is in no module.
#[allow(dead_code)]
pub fn entry_module(result: &LeakerResult, ix: usize) -> Option<ModuleScope> {
    let param: Ident = parse_quote!(__S);
    let r = result
        .repeater_impls(
            &parse_quote!(crate::Repeater),
            7,
            ImplSelf::Param(&param),
            array_const,
        )
        .unwrap_or_else(|error| panic!("the impls: {}", error))
        .nth(ix)
        .unwrap_or_else(|| panic!("no such entry: {}", ix));
    r.module
}

/// The path from its root of the module of the `ix`-th entry (`ModuleScope::Itself::path`); `None`
/// if the entry is in no module, or in a module of `enter_mod()`.
#[allow(dead_code)]
pub fn entry_module_path(result: &LeakerResult, ix: usize) -> Option<Vec<Ident>> {
    match entry_module(result, ix) {
        Some(ModuleScope::Itself { path, .. }) => path,
        _ => None,
    }
}

/// The hint given to `Leaker::enter_mod()` for the module of the `ix`-th entry
/// (`ModuleScope::Itself::hint`).
#[allow(dead_code)]
pub fn entry_module_hint(result: &LeakerResult, ix: usize) -> Option<String> {
    match entry_module(result, ix) {
        Some(ModuleScope::Itself { hint, .. }) => hint,
        _ => None,
    }
}

/// The module in which the module of the `ix`-th entry is nested (`ModuleScope::Itself::parent`).
#[allow(dead_code)]
pub fn entry_module_parent(result: &LeakerResult, ix: usize) -> Option<ContextId> {
    match entry_module(result, ix) {
        Some(ModuleScope::Itself { parent, .. }) => parent,
        _ => None,
    }
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

/// Enters the generics of the function `sig` and analyzes it there ([`visit_function()`]).
#[allow(dead_code)]
pub fn enter_function<A: Allow>(leaker: &mut Leaker<A>, sig: &Signature) -> (ContextId, ()) {
    leaker.enter_generics(&sig.generics, |view| visit_function(view, sig))
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

/// Analyzes the trait `item` (`Self` allowed) with `config`; `check_values` reads the values too.
#[allow(dead_code)]
pub fn analyze_trait<A: Allow>(item: &ItemTrait, config: A, check_values: bool) -> LeakerResult {
    let mut leaker = Leaker::from_config(config);
    leaker.check_values = check_values;
    leaker.intern_trait(item);
    leaker.result()
}

/// The causes that a rule must allow to accept `error`: each of its paths, or its cause if it has
/// no path.
#[allow(dead_code)]
pub fn causes_to_allow(error: &ErrorItem) -> Vec<Cause> {
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
