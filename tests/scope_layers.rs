//! `enter_type()`, `enter_impl()`, `enter_trait()` and `enter_mod()` are the scope of the item (or
//! of the module) around `enter_generics()` of its generics: what they return, record and roll back
//! is what `enter_generics()` does, plus what the item declares (`Self`, and whether it may be
//! interned, or the module). The expected values below were taken from the scopes before they were
//! composed that way, so they pin what has not changed: the ids (the same contexts, and no other
//! one), the entries, the rollback of the call as one unit, the panics and what `Self` and
//! `allow_self` mean.

mod common;

use common::visit_function;
use syn::*;
use template_quote::{quote, ToTokens};
use type_leak::{ContextId, Leaker, NotInternableError, Referrer, SelfRequirement, View};

fn s(x: &impl ToTokens) -> String {
    quote!(#x).to_string()
}

fn ident(name: &str) -> Ident {
    Ident::new(name, proc_macro2::Span::call_site())
}

/// The kinds of the scopes with a layer around their generics, and the module.
const KINDS: [&str; 6] = ["struct", "enum", "union", "impl", "trait", "mod"];

/// The kinds which declare `Self` (a module does not, and has no generics).
const ITEMS: [&str; 5] = ["struct", "enum", "union", "impl", "trait"];

/// The self type of an impl with the generics `generics`: `::p::W` with their parameters as
/// arguments.
fn self_type(generics: &Generics) -> Type {
    let args: Vec<String> = generics
        .params
        .iter()
        .map(|param| match param {
            GenericParam::Lifetime(param) => s(&param.lifetime),
            GenericParam::Type(param) => param.ident.to_string(),
            GenericParam::Const(param) => param.ident.to_string(),
        })
        .collect();
    if args.is_empty() {
        parse_quote!(::p::W)
    } else {
        parse_str(&format!("::p::W<{}>", args.join(", "))).unwrap()
    }
}

/// Enters the scope of `$kind` (one of [`KINDS`]) with the generics `$generics` on `$on`, which is
/// a `Leaker` or a `View`: the item is `S`, and the self type of the impl is that of
/// [`self_type()`]. A module has no generics: it is entered as `m`.
macro_rules! enter {
    ($on:expr, $kind:expr, $generics:expr, $allow_self:expr, $f:expr) => {{
        let generics: &Generics = $generics;
        match $kind {
            "struct" | "enum" | "union" => $on.enter_type(&ident("S"), generics, $allow_self, $f),
            "impl" => $on.enter_impl(generics, &self_type(generics), $allow_self, $f),
            "trait" => $on.enter_trait(&ident("S"), generics, $allow_self, $f),
            "mod" => $on.enter_mod("m", [], $f),
            kind => panic!("{}", kind),
        }
    }};
}

/// Runs `f`, which panics, as the closure around it catches the panic.
fn caught(f: impl FnOnce()) {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
    assert!(result.is_err());
}

/// What a closure returned, as a string (the outcome of a nested call, if it is one).
trait Show {
    fn show(&self) -> String;
}

impl Show for () {
    fn show(&self) -> String {
        String::new()
    }
}

impl Show for String {
    fn show(&self) -> String {
        format!(" [{}]", self)
    }
}

/// What a call returned, as a string.
fn outcome<R: Show>(result: std::result::Result<(ContextId, R), NotInternableError>) -> String {
    match result {
        Ok((_, inner)) => format!("ok{}", inner.show()),
        Err(NotInternableError::SelfWithoutMeaning(selves)) => format!(
            "Self without meaning: {}",
            selves
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Err(err) => err.to_string().replace('\n', " / "),
    }
}

/// Everything the Referrer says of the contexts and the entries, with the ids the calls returned.
fn dump(referrer: &Referrer, ids: &[(&str, ContextId)]) -> String {
    let mut lines = Vec::new();
    for (name, id) in ids {
        lines.push(format!("id {} = {}", name, id.index()));
    }
    for context in referrer.contexts() {
        let self_ty = match referrer.context_self(context) {
            None => "-".to_string(),
            Some(SelfRequirement::Bound(path)) => format!("Self: {}", s(path)),
            Some(SelfRequirement::Type(ty)) => format!("Self = {}", s(ty)),
            Some(other) => format!("{:?}", other),
        };
        let module = match referrer.context_module(context) {
            None => "-".to_string(),
            Some(module) => module.index().to_string(),
        };
        let info = match (referrer.module_hint(context), referrer.module_path(context)) {
            (None, None) => String::new(),
            (hint, path) => format!(
                " is module {:?} {:?}",
                hint,
                path.map(|p| p.iter().map(|i| i.to_string()).collect::<Vec<_>>())
            ),
        };
        lines.push(format!(
            "context {}: {} ; {} ; in module {}{}",
            context.index(),
            s(referrer.context_generics(context)),
            self_ty,
            module,
            info
        ));
    }
    for (ix, ty) in referrer.iter().enumerate() {
        lines.push(format!(
            "entry {}: {} in {}",
            ix,
            s(ty),
            referrer.context(ix)
        ));
    }
    lines.join("\n")
}

/// Asserts that `actual` is `expected` (see the module).
fn pin(name: &str, actual: &str, expected: &str) {
    assert_eq!(actual, expected.trim(), "{}", name);
}

// ── what a scope returns and records ─────────────────────────────────────────────────────────

/// What a scope of an item returns and records when it is the outermost one.
fn root_item(kind: &str) -> String {
    let generics: Generics = parse_quote!(<'a, T: ::p::Tr<Self>, U = Rel0>);
    let mut leaker = Leaker::new();
    let body = |view: &mut View| {
        view.visit_type(&parse_quote!(Rel<Self, T>));
        view.visit_type(&parse_quote!(Rel2<&'a U>));
        view.visit_type(&parse_quote!(Rel3));
        view.visit_type_param_bound(&parse_quote!(::p::Tr<Rel4>));
        view.visit_where_clause(&parse_quote!(where T: ::p::Tr2<Rel5>));
        7
    };
    let (id, value) = enter!(leaker, kind, &generics, true, body).unwrap();
    assert_eq!(value, 7);
    dump(&leaker.finish(), &[("top", id)])
}

#[test]
fn the_outermost_scope_of_an_item_returns_the_generics_context() {
    // A struct, an enum and a union declare the same `Self`, and the same contexts.
    for kind in ["struct", "enum", "union"] {
        pin(&format!("root_item {}", kind), &root_item(kind), ROOT_ITEM);
    }
    pin("root_item impl", &root_item("impl"), ROOT_ITEM_IMPL);
    pin("root_item trait", &root_item("trait"), ROOT_ITEM_TRAIT);
}

const ROOT_ITEM: &str = r#"
id top = 0
context 0: < 'a , T : :: p :: Tr < Self > , U > ; Self = S < 'a , T , U > ; in module -
context 1: < 'a , U : 'a > ; - ; in module -
context 2:  ; - ; in module -
entry 0: Rel < Self , T > in 0
entry 1: Rel2 < & 'a U > in 1
entry 2: Rel0 in 2
entry 3: Rel3 in 2
entry 4: Rel4 in 2
entry 5: Rel5 in 2
"#;
const ROOT_ITEM_IMPL: &str = r#"
id top = 0
context 0: < 'a , T : :: p :: Tr < Self > , U > ; Self = :: p :: W < 'a , T , U > ; in module -
context 1: < 'a , U : 'a > ; - ; in module -
context 2:  ; - ; in module -
entry 0: Rel < Self , T > in 0
entry 1: Rel2 < & 'a U > in 1
entry 2: Rel0 in 2
entry 3: Rel3 in 2
entry 4: Rel4 in 2
entry 5: Rel5 in 2
"#;
const ROOT_ITEM_TRAIT: &str = r#"
id top = 0
context 0: < 'a , T : :: p :: Tr < Self > , U > ; Self: S < 'a , T , U > ; in module -
context 1: < 'a , U : 'a > ; - ; in module -
context 2:  ; - ; in module -
entry 0: Rel < Self , T > in 0
entry 1: Rel2 < & 'a U > in 1
entry 2: Rel0 in 2
entry 3: Rel3 in 2
entry 4: Rel4 in 2
entry 5: Rel5 in 2
"#;

/// The ids of several scopes in one: the contexts of the generics are those of the generics, and
/// the items and the modules add none, except `Self` if an entry of the scope names it.
fn nested_ids() -> String {
    let mut leaker = Leaker::new();
    let s_ident = ident("S");
    let mut ids = Vec::new();
    let (top, ()) = leaker
        .enter_generics(&parse_quote!(<X>), |view| {
            let generics: Generics = parse_quote!(<U>);
            let (a, ()) = view
                .enter_type(&s_ident, &generics, true, |view| {
                    view.visit_type(&parse_quote!(Rel<Self, U>))
                })
                .unwrap();
            // The same generics, but another `Self`, which an entry names: another context. (`d`
            // below has no entry that names `Self`, so its context is that of the generics.)
            let (b, ()) = view
                .enter_type(&ident("E"), &generics, true, |view| {
                    view.visit_type(&parse_quote!(Rel<Self, U>))
                })
                .unwrap();
            let (c, ()) = view
                .enter_type(&ident("Un"), &parse_quote!(<'a>), false, |view| {
                    view.visit_type(&parse_quote!(Rel2<&'a ()>))
                })
                .unwrap();
            let (m, inner) = view
                .enter_mod("m", [ident("S")], |view| {
                    view.visit_type(&parse_quote!(S<Rel3>));
                    view.enter_type(&ident("S2"), &parse_quote!(<V>), true, |view| {
                        view.visit_type(&parse_quote!(Rel4<Self, V>))
                    })
                    .unwrap()
                    .0
                })
                .unwrap();
            let (m2, ()) = view
                .enter_mod("m2", [], |view| view.visit_type(&parse_quote!(Rel3)))
                .unwrap();
            // After the modules, the same generics are still the same context.
            let (d, ()) = view.enter_type(&s_ident, &generics, true, |_| ()).unwrap();
            let (e, ()) = view
                .enter_impl(&generics, &parse_quote!(::p::Wrap<U>), true, |view| {
                    view.visit_type(&parse_quote!(Rel6<Self, U>))
                })
                .unwrap();
            let (f, ()) = view
                .enter_trait(&ident("Tr"), &generics, false, |view| {
                    view.visit_type(&parse_quote!(Rel7<U>));
                    // A trait inside a module inside a trait.
                    view.enter_mod("m4", [], |view| {
                        view.enter_trait(&ident("Tr2"), &parse_quote!(<W>), true, |view| {
                            view.visit_type(&parse_quote!(Rel8<Self, W>))
                        })
                        .unwrap();
                    })
                    .unwrap();
                })
                .unwrap();
            // A module has no generics of its own: the ones around stay in scope.
            let (m3, ()) = view
                .enter_mod("m3", [], |view| view.visit_type(&parse_quote!(Rel5<X>)))
                .unwrap();
            ids.extend([
                ("a", a),
                ("b", b),
                ("c", c),
                ("m", m),
                ("inner", inner),
                ("m2", m2),
                ("d", d),
                ("e", e),
                ("f", f),
                ("m3", m3),
            ]);
        })
        .unwrap();
    ids.push(("top", top));
    dump(&leaker.finish(), &ids)
}

const NESTED_IDS: &str = r#"
id a = 0
id b = 1
id c = 2
id m = 3
id inner = 4
id m2 = 5
id d = 6
id e = 7
id f = 10
id m3 = 11
id top = 15
context 0: < U > ; Self = S < U > ; in module -
context 1: < U > ; Self = E < U > ; in module -
context 2: < 'a > ; - ; in module -
context 3:  ; - ; in module - is module Some("m") None
context 4: < V > ; Self = S2 < V > ; in module 3
context 5:  ; - ; in module - is module Some("m2") None
context 6: < U > ; - ; in module -
context 7: < U > ; Self = :: p :: Wrap < U > ; in module -
context 8:  ; - ; in module - is module Some("m4") None
context 9: < W > ; Self: Tr2 < W > ; in module 8
context 10: < U > ; Self: Tr < U > ; in module -
context 11:  ; - ; in module - is module Some("m3") None
context 12:  ; - ; in module 3
context 13:  ; - ; in module 5
context 14: < X > ; - ; in module 11
context 15: < X > ; - ; in module -
entry 0: Rel < Self , U > in 0
entry 1: Rel < Self , U > in 1
entry 2: Rel2 < & 'a () > in 2
entry 3: Rel4 < Self , V > in 4
entry 4: Rel7 < U > in 6
entry 5: Rel6 < Self , U > in 7
entry 6: Rel8 < Self , W > in 9
entry 7: Rel3 in 12
entry 8: Rel3 in 13
entry 9: Rel5 < X > in 14
"#;

#[test]
fn nested_scopes_of_items_and_modules_take_the_same_ids() {
    pin("nested_ids", &nested_ids(), NESTED_IDS);
}

// ── a scope which fails is rolled back as one unit ───────────────────────────────────────────

/// The generics of a scope whose default is not internable (an error found when the scope is left).
fn bad_generics() -> Generics {
    parse_quote!(<A, B = Bad<'_, A>>)
}

/// Enters a module with a scope around which succeeds, one which does not if `fail` (`kind` is the
/// scope), and one which succeeds. Returns the errors of the one which fails, and what is left.
fn failing(kind: &str, fail: bool) -> (String, String) {
    let mut leaker = Leaker::new();
    let mut failure = String::new();
    let (outer, (first, last)) = leaker
        .enter_mod("outer", [], |view| {
            view.visit_type(&parse_quote!(Rel));
            let (first, ()) = view
                .enter_type(&ident("F"), &parse_quote!(<A>), true, |view| {
                    view.visit_type(&parse_quote!(RelA<Self, A>));
                })
                .unwrap();
            if fail {
                // A module, which takes an id, a scope which succeeds, a type which does not, and
                // the generics, whose default does not either (a module has no generics: only the
                // type fails it).
                let body = |view: &mut View| {
                    view.visit_type(&parse_quote!(RelU<Self, A>));
                    view.enter_mod("lost", [], |view| view.visit_type(&parse_quote!(RelM)))
                        .unwrap();
                    view.enter_type(&ident("Lost"), &parse_quote!(<W>), true, |_| ())
                        .unwrap();
                    view.visit_type(&parse_quote!(Bad1<'_>));
                };
                let result = enter!(view, kind, &bad_generics(), true, body);
                failure = result.unwrap_err().to_string();
            }
            let (last, ()) = view
                .enter_type(&ident("L"), &parse_quote!(<C>), true, |view| {
                    view.visit_type(&parse_quote!(RelC<Self, C>));
                    view.enter_mod("kept", [], |view| view.visit_type(&parse_quote!(RelK)))
                        .unwrap();
                })
                .unwrap();
            (first, last)
        })
        .unwrap();
    let dumped = dump(
        &leaker.finish(),
        &[("outer", outer), ("first", first), ("last", last)],
    );
    (failure, dumped)
}

#[test]
fn a_failing_scope_of_an_item_or_a_module_is_rolled_back_whole() {
    let (_, ok) = failing("struct", false);
    pin("failing ok", &ok, FAILING_OK);
    let mut errors = Vec::new();
    for kind in KINDS {
        let (failure, left) = failing(kind, true);
        // What is left is what there is without the scope which failed, with the ids it took (the
        // module, the contexts of the types and of the generics) not taken.
        assert_eq!(left, ok, "{}", kind);
        errors.push(failure);
    }
    // The errors are returned once: the type, and the default of the generics (a module has no
    // generics).
    pin("failing errors", &errors.join("\n--\n"), FAILING_ERRORS);
}

const FAILING_OK: &str = r#"
id outer = 0
id first = 1
id last = 3
context 0:  ; - ; in module - is module Some("outer") None
context 1: < A > ; Self = F < A > ; in module 0
context 2:  ; - ; in module - is module Some("kept") None
context 3: < C > ; Self = L < C > ; in module 0
context 4:  ; - ; in module 0
context 5:  ; - ; in module 2
entry 0: RelA < Self , A > in 1
entry 1: RelC < Self , C > in 3
entry 2: Rel in 4
entry 3: RelK in 5
"#;
const FAILING_ERRORS: &str = r#"
use absolute path: 'Bad1 < '_ >'
use absolute path: 'Bad < '_ , A >'
--
use absolute path: 'Bad1 < '_ >'
use absolute path: 'Bad < '_ , A >'
--
use absolute path: 'Bad1 < '_ >'
use absolute path: 'Bad < '_ , A >'
--
use absolute path: 'Bad1 < '_ >'
use absolute path: 'Bad < '_ , A >'
--
use absolute path: 'Bad1 < '_ >'
use absolute path: 'Bad < '_ , A >'
--
use absolute path: 'Bad1 < '_ >'
"#;

/// A failing outermost scope leaves no trace: the ids and the Referrer are those of the Leaker
/// without it.
#[test]
fn a_failing_outermost_scope_leaves_no_trace() {
    let reference = {
        let mut leaker = Leaker::new();
        leaker.enter_mod("kept", [], |_| ()).unwrap();
        leaker.finish().to_token_stream().to_string()
    };
    let mut leaker = Leaker::new();
    let mut errors = Vec::new();
    for kind in KINDS {
        let body = |view: &mut View| {
            view.visit_type(&parse_quote!(RelLost<Self, A>));
            view.enter_mod("lost", [], |view| view.visit_type(&parse_quote!(RelM)))
                .unwrap();
            view.visit_type(&parse_quote!(Bad1<'_>));
        };
        let result = enter!(leaker, kind, &bad_generics(), true, body);
        errors.push(result.unwrap_err().to_string());
    }
    // The module which is entered after them is the first one.
    let (module, ()) = leaker.enter_mod("kept", [], |_| ()).unwrap();
    assert_eq!(module.index(), 0);
    assert_eq!(leaker.finish().to_token_stream().to_string(), reference);
    pin("outermost errors", &errors.join("\n--\n"), OUTERMOST_ERRORS);
}

const OUTERMOST_ERRORS: &str = r#"
use absolute path: 'Bad1 < '_ >'
use absolute path: 'Bad < '_ , A >'
--
use absolute path: 'Bad1 < '_ >'
use absolute path: 'Bad < '_ , A >'
--
use absolute path: 'Bad1 < '_ >'
use absolute path: 'Bad < '_ , A >'
--
use absolute path: 'Bad1 < '_ >'
use absolute path: 'Bad < '_ , A >'
--
use absolute path: 'Bad1 < '_ >'
use absolute path: 'Bad < '_ , A >'
--
use absolute path: 'Bad1 < '_ >'
"#;

// ── a panic in a scope ───────────────────────────────────────────────────────────────────────

#[test]
fn a_panic_in_an_outermost_scope_leaves_no_trace() {
    let reference = {
        let mut leaker = Leaker::new();
        leaker.enter_mod("kept", [], |_| ()).unwrap();
        leaker.finish().to_token_stream().to_string()
    };
    let mut leaker = Leaker::new();
    for kind in KINDS {
        caught(|| {
            let body = |view: &mut View| {
                view.visit_type(&parse_quote!(RelLost<Self, T>));
                view.enter_mod("lost", [], |_| ()).unwrap();
                view.enter_type(&ident("Lost"), &parse_quote!(<W>), true, |_| ())
                    .unwrap();
                panic!("a closure panics");
            };
            let _ = enter!(leaker, kind, &parse_quote!(<T>), true, body);
        });
    }
    let (module, ()) = leaker.enter_mod("kept", [], |_| ()).unwrap();
    assert_eq!(module.index(), 0);
    assert_eq!(leaker.finish().to_token_stream().to_string(), reference);
}

/// A scope of kind `kind` which panics (if `panic`) in a scope of the kind `outer` (which is not
/// a module), which catches it.
fn panicking(outer: &str, kind: &str, panic: bool, mention_self: bool) -> String {
    let mut leaker = Leaker::new();
    let body = |view: &mut View| {
        view.visit_type(&parse_quote!(Rel<T>));
        if panic {
            caught(|| {
                let body = |view: &mut View| {
                    view.visit_type(&parse_quote!(RelU<Self, U>));
                    view.enter_mod("lost", [parse_quote!(Lost)], |_| ())
                        .unwrap();
                    view.visit_type(&parse_quote!(Bad<'_>));
                    panic!("a closure panics");
                };
                let _ = enter!(view, kind, &parse_quote!(<U>), true, body);
            });
        }
        // `U` is not a generic any more, `Lost` is not declared, `Self` is the one of the outer
        // scope again, and so is `allow_self` (which a module, or an item with `true`, sets).
        view.visit_type(&parse_quote!(Wrap<U, Lost>));
        if mention_self {
            view.visit_type(&parse_quote!(Wrap2<Self>));
        }
        view.enter_mod("kept", [], |view| view.visit_type(&parse_quote!(RelK)))
            .unwrap()
            .0
    };
    match enter!(leaker, outer, &parse_quote!(<T>), false, body) {
        Ok((top, module)) => {
            let referrer = leaker.finish();
            dump(&referrer, &[("top", top), ("module", module)])
        }
        Err(err) => outcome(Err::<(ContextId, ()), _>(err)),
    }
}

#[test]
fn a_panic_caught_in_a_scope_restores_the_scope_around() {
    for mention_self in [false, true] {
        // Whatever the scope around declares (`Self`, and `allow_self`), it is as if the scope
        // which panicked had not been entered.
        let mut first = None;
        for outer in ITEMS {
            let expected = panicking(outer, "struct", false, mention_self);
            for kind in KINDS {
                assert_eq!(
                    panicking(outer, kind, true, mention_self),
                    expected,
                    "{} {} {}",
                    outer,
                    kind,
                    mention_self
                );
            }
            assert_eq!(
                *first.get_or_insert(expected.clone()),
                expected,
                "{}",
                outer
            );
        }
        pin(
            &format!("panicking {}", mention_self),
            &first.unwrap(),
            if mention_self {
                PANICKING_SELF
            } else {
                PANICKING
            },
        );
    }
}

const PANICKING: &str = r#"
id top = 1
id module = 0
context 0:  ; - ; in module - is module Some("kept") None
context 1: < T > ; - ; in module -
context 2:  ; - ; in module -
context 3:  ; - ; in module 0
entry 0: Rel < T > in 1
entry 1: Wrap < U , Lost > in 2
entry 2: RelK in 3
"#;
const PANICKING_SELF: &str = r#"
use absolute path: 'Wrap2 < Self >'
"#;

// ── the generics are declared once ───────────────────────────────────────────────────────────

#[test]
fn a_scope_of_an_item_or_a_module_adds_no_scope_of_its_own() {
    // A function which is the analyzed item is visited in the scope of its generics (`as_item()`),
    // whose generics are those of the Leaker if it is the outermost one: a module has none of its
    // own, and an item is the scope of its generics. The same function nested in a scope has local
    // generics (`nested()`). It is a method of a trait there: the copy of a function item may have
    // `'b` early-bound, and only that of a method is an error for a lifetime that no argument
    // constrains (E0195, F22), which shows that `'b` is local.
    let sig: Signature = parse_quote!(fn f<'b>(x: ::p::W<'b, dyn RelTr>) -> &'b ());
    let plain: Signature = parse_quote!(fn g(x: ::p::W<dyn RelTr>));
    let as_item = |v: &mut View<'_>, sig: &Signature| visit_function(v, sig);
    let nested = |v: &mut View<'_>, sig: &Signature| {
        v.visit_item(&parse_quote!(
            trait Nested {
                #sig;
            }
        ));
    };
    let b: Generics = parse_quote!(<'b>);
    let none = Generics::default();
    let mut lines = Vec::new();
    let mut check = |name: String, run: &dyn Fn(&mut Leaker) -> String| {
        let mut leaker = Leaker::new();
        let result = run(&mut leaker);
        let count = leaker.finish().iter().count();
        lines.push(format!("{}: {} ({})", name, result, count));
    };
    for kind in ["struct", "impl", "trait"] {
        check(format!("{} of the generics", kind), &|l| {
            outcome(enter!(l, kind, &b, true, |v| as_item(v, &sig)))
        });
        check(format!("{} of others", kind), &|l| {
            outcome(enter!(l, kind, &none, true, |v| nested(v, &sig)))
        });
        check(format!("{}, mod", kind), &|l| {
            outcome(enter!(l, kind, &b, true, |v| {
                v.enter_mod("m", [], |v| as_item(v, &sig)).unwrap();
            }))
        });
        check(format!("generics, {} of the generics", kind), &|l| {
            outcome(l.enter_generics(&b, |v| {
                outcome(enter!(v, kind, &b, true, |v| nested(v, &sig)))
            }))
        });
        check(format!("mod, {} of the generics", kind), &|l| {
            outcome(l.enter_mod("m", [], |v| {
                outcome(enter!(v, kind, &b, true, |v| nested(v, &sig)))
            }))
        });
        check(format!("{}, {}, mod", kind, kind), &|l| {
            outcome(enter!(l, kind, &b, true, |v| {
                outcome(enter!(v, kind, &b, true, |v| {
                    outcome(v.enter_mod("m", [], |v| nested(v, &sig)))
                }))
            }))
        });
    }
    check("mod".to_string(), &|l| {
        outcome(l.enter_mod("m", [], |v| nested(v, &sig)))
    });
    check("mod, no generics".to_string(), &|l| {
        outcome(l.enter_mod("m", [], |v| as_item(v, &plain)))
    });
    check("generics, mod".to_string(), &|l| {
        outcome(l.enter_generics(&b, |v| {
            v.enter_mod("m", [], |v| as_item(v, &sig)).unwrap();
        }))
    });
    check("mod, mod, no generics".to_string(), &|l| {
        outcome(l.enter_mod("m", [], |v| {
            v.enter_mod("n", [], |v| as_item(v, &plain)).unwrap();
        }))
    });
    check("mod, generics".to_string(), &|l| {
        outcome(l.enter_mod("m", [], |v| {
            outcome(v.enter_generics(&b, |v| nested(v, &sig)))
        }))
    });
    pin("signatures", &lines.join("\n"), SIGNATURES);
}

const SIGNATURES: &str = r#"
struct of the generics: ok (1)
struct of others: `RelTr` leaves the lifetime `'b` of the method unconstrained in the copy (E0195) (0)
struct, mod: ok (1)
generics, struct of the generics: ok [`RelTr` leaves the lifetime `'b` of the method unconstrained in the copy (E0195)] (0)
mod, struct of the generics: ok [`RelTr` leaves the lifetime `'b` of the method unconstrained in the copy (E0195)] (0)
struct, struct, mod: ok [ok [`RelTr` leaves the lifetime `'b` of the method unconstrained in the copy (E0195)]] (0)
impl of the generics: ok (1)
impl of others: `RelTr` leaves the lifetime `'b` of the method unconstrained in the copy (E0195) (0)
impl, mod: ok (1)
generics, impl of the generics: ok [`RelTr` leaves the lifetime `'b` of the method unconstrained in the copy (E0195)] (0)
mod, impl of the generics: ok [`RelTr` leaves the lifetime `'b` of the method unconstrained in the copy (E0195)] (0)
impl, impl, mod: ok [ok [`RelTr` leaves the lifetime `'b` of the method unconstrained in the copy (E0195)]] (0)
trait of the generics: ok (1)
trait of others: `RelTr` leaves the lifetime `'b` of the method unconstrained in the copy (E0195) (0)
trait, mod: ok (1)
generics, trait of the generics: ok [`RelTr` leaves the lifetime `'b` of the method unconstrained in the copy (E0195)] (0)
mod, trait of the generics: ok [`RelTr` leaves the lifetime `'b` of the method unconstrained in the copy (E0195)] (0)
trait, trait, mod: ok [ok [`RelTr` leaves the lifetime `'b` of the method unconstrained in the copy (E0195)]] (0)
mod: `RelTr` leaves the lifetime `'b` of the method unconstrained in the copy (E0195) (0)
mod, no generics: ok (2)
generics, mod: ok (1)
mod, mod, no generics: ok (2)
mod, generics: ok [`RelTr` leaves the lifetime `'b` of the method unconstrained in the copy (E0195)] (0)
"#;

// ── `Self` ───────────────────────────────────────────────────────────────────────────────────

#[test]
fn self_in_the_generics_of_a_scope() {
    let mut lines = Vec::new();
    let mut check = |name: String, run: &dyn Fn(&mut Leaker) -> String| {
        let mut leaker = Leaker::new();
        let result = run(&mut leaker);
        let count = leaker.finish().iter().count();
        lines.push(format!("{}: {} ({})", name, result, count));
    };
    let none = Generics::default();
    let mut bound: Generics = parse_quote!(<'a, T: ::p::Tr<Self>>);
    bound.where_clause = Some(parse_quote!(where Self: 'a));
    let default: Generics = parse_quote!(<T = Rel<Self>>);
    // It is not where nothing declares it.
    check("generics bound".to_string(), &|l| {
        outcome(l.enter_generics(&bound, |_| ()))
    });
    check("generics default".to_string(), &|l| {
        outcome(l.enter_generics(&default, |_| ()))
    });
    check("mod".to_string(), &|l| {
        outcome(l.enter_mod("m", [], |v| v.visit_type(&parse_quote!(Rel<Self>))))
    });
    check("mod, generics".to_string(), &|l| {
        outcome(l.enter_mod("m", [], |v| outcome(v.enter_generics(&bound, |_| ()))))
    });
    for kind in ["struct", "impl", "trait"] {
        // `Self` is declared by the item: it may be in its generics.
        check(format!("{} bound", kind), &|l| {
            outcome(enter!(l, kind, &bound, true, |_| ()))
        });
        check(format!("{} bound, not allowed", kind), &|l| {
            outcome(enter!(l, kind, &bound, false, |_| ()))
        });
        check(format!("{} default", kind), &|l| {
            outcome(enter!(l, kind, &default, true, |_| ()))
        });
        check(format!("mod, {}", kind), &|l| {
            outcome(l.enter_mod("m", [], |v| {
                outcome(enter!(v, kind, &bound, true, |v| {
                    v.visit_type(&parse_quote!(Rel<Self>))
                }))
            }))
        });
        // In a scope nested in one which declares it, it is what it is there.
        check(format!("{}, generics", kind), &|l| {
            outcome(enter!(l, kind, &none, true, |v| {
                outcome(v.enter_generics(&bound, |v| v.visit_type(&parse_quote!(Rel<Self>))))
            }))
        });
        check(format!("{}, mod, generics", kind), &|l| {
            outcome(enter!(l, kind, &none, true, |v| {
                outcome(v.enter_mod("m", [], |v| outcome(v.enter_generics(&bound, |_| ()))))
            }))
        });
        // What the scope around says of `allow_self` is restored, and a module or an item sets its
        // own.
        check(format!("{} not allowed", kind), &|l| {
            outcome(enter!(l, kind, &none, false, |v| {
                v.visit_type(&parse_quote!(Rel<Self>))
            }))
        });
        check(format!("{} not allowed, {} allowed", kind, kind), &|l| {
            outcome(enter!(l, kind, &none, false, |v| {
                enter!(v, kind, &none, true, |v| {
                    v.visit_type(&parse_quote!(Rel<Self>))
                })
                .unwrap();
                outcome(enter!(v, kind, &none, false, |v| {
                    v.visit_type(&parse_quote!(Rel2<Self>))
                }))
            }))
        });
        check(format!("{} not allowed, then", kind), &|l| {
            outcome(enter!(l, kind, &none, false, |v| {
                enter!(v, kind, &none, true, |_| ()).unwrap();
                v.enter_mod("m", [], |_| ()).unwrap();
                v.visit_type(&parse_quote!(Rel<Self>))
            }))
        });
        check(format!("{} not allowed, mod, item", kind), &|l| {
            outcome(enter!(l, kind, &none, false, |v| {
                v.enter_mod("m", [], |v| {
                    v.visit_item(&parse_quote!(
                        struct S {
                            a: Rel<Self>,
                        }
                    ));
                })
                .unwrap();
            }))
        });
        check(format!("{} not allowed, item", kind), &|l| {
            outcome(enter!(l, kind, &none, false, |v| {
                v.visit_item(&parse_quote!(
                    struct S {
                        a: Rel<Self>,
                    }
                ));
            }))
        });
    }
    pin("self", &lines.join("\n"), SELF);
}

const SELF: &str = r#"
generics bound: Self without meaning: Self, Self (0)
generics default: Self without meaning: Self (0)
mod: Self without meaning: Self (0)
mod, generics: ok [Self without meaning: Self, Self] (0)
struct bound: ok (0)
struct bound, not allowed: ok (0)
struct default: `Rel < Self >` is in the default of a parameter, and names `Self` (0)
mod, struct: ok [ok] (1)
struct, generics: ok [ok] (1)
struct, mod, generics: ok [ok [Self without meaning: Self, Self]] (0)
struct not allowed: use absolute path: 'Rel < Self >' (0)
struct not allowed, struct allowed: ok [use absolute path: 'Rel2 < Self >'] (1)
struct not allowed, then: use absolute path: 'Rel < Self >' (0)
struct not allowed, mod, item: ok (1)
struct not allowed, item: ok (1)
impl bound: ok (0)
impl bound, not allowed: ok (0)
impl default: `Rel < Self >` is in the default of a parameter, and names `Self` (0)
mod, impl: ok [ok] (1)
impl, generics: ok [ok] (1)
impl, mod, generics: ok [ok [Self without meaning: Self, Self]] (0)
impl not allowed: use absolute path: 'Rel < Self >' (0)
impl not allowed, impl allowed: ok [use absolute path: 'Rel2 < Self >'] (1)
impl not allowed, then: use absolute path: 'Rel < Self >' (0)
impl not allowed, mod, item: ok (1)
impl not allowed, item: ok (1)
trait bound: ok (0)
trait bound, not allowed: ok (0)
trait default: `Rel < Self >` is in the default of a parameter, and names `Self` (0)
mod, trait: ok [ok] (1)
trait, generics: ok [ok] (1)
trait, mod, generics: ok [ok [Self without meaning: Self, Self]] (0)
trait not allowed: use absolute path: 'Rel < Self >' (0)
trait not allowed, trait allowed: ok [use absolute path: 'Rel2 < Self >'] (1)
trait not allowed, then: use absolute path: 'Rel < Self >' (0)
trait not allowed, mod, item: ok (1)
trait not allowed, item: ok (1)
"#;
