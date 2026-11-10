use proc_macro2::TokenStream;
use std::collections::BTreeSet;
use syn::*;
use template_quote::quote;
use type_leak::{
    Allow, Cause, CheckResult, ContextId, ErrorItem, Leaker, LeakerResult, Options, Place, Rules,
    View,
};

/// Converts the required types of a `Referrer` (see `Referrer::must()`) to a `BTreeSet<String>`
/// for order-independent comparison. The optional ones, which only the paths of external crates
/// (`::std::vec::Vec<..>`) cause, are not among them.
#[allow(dead_code)]
pub fn finish_as_strings<A: Allow>(leaker: Leaker<A>) -> BTreeSet<String> {
    let referrer = leaker.result().finish().unwrap();
    referrer
        .iter()
        .enumerate()
        .filter(|(ix, _)| referrer.must(*ix))
        .map(|(_, t)| quote!(#t).to_string())
        .collect()
}

/// Interns the items `items` of a trait `Tr` with the generics `generics` (e.g. `fn f(&self) ->
/// Rel<Self>;`), where `Self` may be interned if `allow_self`. The errors are those of
/// `Leaker::result()`.
#[allow(dead_code)]
pub fn intern_fns<A: Allow>(
    leaker: &mut Leaker<A>,
    generics: &Generics,
    items: TokenStream,
    allow_self: bool,
) -> ContextId {
    let where_clause = &generics.where_clause;
    let item: ItemTrait = parse_quote!(trait Tr #generics #where_clause { #items });
    leaker.intern_trait(&item, allow_self)
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
    let mut leaker = Leaker::from_config(
        config,
        if check_values {
            Options::new().check_values()
        } else {
            Options::new()
        },
    );
    leaker.intern_trait(item, true);
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
