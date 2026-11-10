use proc_macro2::TokenStream;
use std::collections::BTreeSet;
use syn::*;
use template_quote::quote;
use type_leak::{
    path_starts_with, Allow, Cause, CheckResult, ContextId, ErrorItem, Leaker, LeakerResult, View,
};

/// Converts the required types of a `Referrer` (see `Referrer::must()`) to a `BTreeSet<String>`
/// for order-independent comparison. The optional ones, which only the paths of external crates
/// (`::std::vec::Vec<..>`) cause, are not among them.
#[allow(dead_code)]
pub fn finish_as_strings<P: Fn(&Path) -> bool, E: Fn(&ErrorItem) -> bool>(
    leaker: Leaker<P, E>,
) -> BTreeSet<String> {
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
pub fn intern_fns<P: Fn(&Path) -> bool, E: Fn(&ErrorItem) -> bool>(
    leaker: &mut Leaker<P, E>,
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
pub fn visit_function<P: Fn(&Path) -> bool>(view: &mut View<'_, P>, sig: &Signature) {
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
pub fn enter_function<P: Fn(&Path) -> bool, E: Fn(&ErrorItem) -> bool>(
    leaker: &mut Leaker<P, E>,
    sig: &Signature,
) -> (ContextId, ()) {
    leaker.enter_generics(&sig.generics, |view| visit_function(view, sig))
}

/// Checks `ty` in a scope that declares the generics `generics` and nothing else (see
/// `View::check()`). `Self` has no meaning there.
#[allow(dead_code)]
pub fn check_in<P: Fn(&Path) -> bool, E: Fn(&ErrorItem) -> bool>(
    leaker: &mut Leaker<P, E>,
    ty: &Type,
    generics: &Generics,
) -> CheckResult {
    leaker.enter_generics(generics, |view| view.check(ty)).1
}

/// The rule that accepts the paths that start with `prefix`: `path_starts_with()`.
#[allow(dead_code)]
pub fn starts_with(prefix: Path) -> impl Fn(&Path) -> bool {
    move |path: &Path| path_starts_with(path, &prefix)
}

/// Analyzes the trait `item` (`Self` allowed) with `config`; `check_values` reads the values too.
#[allow(dead_code)]
pub fn analyze_trait<P: Fn(&Path) -> bool, E: Fn(&ErrorItem) -> bool>(
    item: &ItemTrait,
    config: Allow<P, E>,
    check_values: bool,
) -> LeakerResult {
    let mut leaker = Leaker::from_config(config, check_values, false);
    leaker.intern_trait(item, true);
    leaker.result()
}

/// Allows the cause of every error that remains, until none does, and returns how many rounds it took.
/// Dropping an error puts its nested errors in its place ([`type_leak::ErrorItem::nested`]), whose
/// causes are not those of the errors that were there, so one round may not be enough.
#[allow(dead_code)]
pub fn allow_every_cause(result: &mut LeakerResult) -> usize {
    let mut rounds = 0;
    while !result.errors().is_empty() {
        let causes: Vec<Cause> = result.errors().iter().map(|e| e.cause.clone()).collect();
        for cause in causes {
            result.allow(|e| e.cause == cause);
        }
        rounds += 1;
    }
    rounds
}

/// One of the `allow*()` operations that `Allow` and `LeakerResult` share, so that a test can
/// give the same rule to either.
#[allow(dead_code)]
#[derive(Clone)]
pub enum Rule {
    Cause(Cause),
    Path(Path),
    Primitive,
    Prelude,
    StdCrates,
    Crate,
    Values,
    Default,
}

#[allow(dead_code)]
impl Rule {
    /// Analyzes the trait `item` with `config` and the rule given to the config: it is applied
    /// during the analysis. `check_values` reads the values too.
    pub fn analyze_with<P: Fn(&Path) -> bool, E: Fn(&ErrorItem) -> bool>(
        &self,
        item: &ItemTrait,
        config: Allow<P, E>,
        check_values: bool,
    ) -> LeakerResult {
        match self {
            Rule::Cause(cause) => {
                let cause = cause.clone();
                analyze_trait(item, config.allow(move |e| e.cause == cause), check_values)
            }
            Rule::Path(path) => analyze_trait(
                item,
                config.allow_path_if(starts_with(path.clone())),
                check_values,
            ),
            Rule::Primitive => analyze_trait(item, config.allow_primitive(), check_values),
            Rule::Prelude => analyze_trait(item, config.allow_prelude(), check_values),
            Rule::StdCrates => analyze_trait(item, config.allow_std_crates(), check_values),
            Rule::Crate => analyze_trait(item, config.allow_crate(), check_values),
            Rule::Values => analyze_trait(item, config.allow_values(), check_values),
            Rule::Default => analyze_trait(item, config.allow_default(), check_values),
        }
    }

    /// Whether the rule accepts `error` as a rule on errors does, for a config that is given a list
    /// of rules ([`Rule::accepts_path()`] for the paths). The path rules of the lists of names
    /// (`Primitive` and `Prelude`) are not made for a list: use [`Rule::analyze_with()`].
    pub fn accepts_error(&self, error: &ErrorItem) -> bool {
        match self {
            Rule::Cause(cause) => error.cause == *cause,
            Rule::Values => error.in_value,
            Rule::Default => error.in_default,
            Rule::Path(_) | Rule::StdCrates | Rule::Crate => false,
            Rule::Primitive | Rule::Prelude => panic!("use analyze_with()"),
        }
    }

    /// Whether the rule accepts `path` as a path rule does, for a config that is given a list of
    /// rules.
    pub fn accepts_path(&self, path: &Path) -> bool {
        match self {
            Rule::Path(prefix) => starts_with(prefix.clone())(path),
            Rule::StdCrates => ["std", "core", "alloc"]
                .iter()
                .any(|name| starts_with(parse_str(name).unwrap())(path)),
            Rule::Crate => starts_with(parse_quote!(crate))(path),
            Rule::Cause(_) | Rule::Values | Rule::Default => false,
            Rule::Primitive | Rule::Prelude => panic!("use analyze_with()"),
        }
    }

    /// Gives the rule to the result: it is applied after the analysis.
    pub fn on_result(&self, result: &mut LeakerResult) {
        match self {
            Rule::Cause(cause) => result.allow(|e| e.cause == *cause),
            Rule::Path(path) => result.allow_path_if(starts_with(path.clone())),
            Rule::Primitive => result.allow_primitive(),
            Rule::Prelude => result.allow_prelude(),
            Rule::StdCrates => result.allow_std_crates(),
            Rule::Crate => result.allow_crate(),
            Rule::Values => result.allow_values(),
            Rule::Default => result.allow_default(),
        };
    }
}
