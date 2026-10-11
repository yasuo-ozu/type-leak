//! The elided lifetime in the output of a signature, next to a leaked type that has lifetimes.
//!
//! `Box<dyn Fn(&Rel<T>) -> &u8>` has one lifetime in the inputs of the sugar, the elided one of
//! `&`, which elision gives to the output. A leaked type that has lifetimes (`Rel<T>` in a trait
//! with `'a, T: 'a` has `<'a, T: 'a>`) is replaced by a projection that names them in its
//! arguments (`<S as Repeater<(&'a (), T, &'a T), ..>>::Type`), so the inputs of the copy have
//! several lifetimes, and rustc cannot resolve the output (E0106). rustc counts the lifetimes that
//! are distinct in the inputs: each elided one, and each name once. A function pointer or `Fn(..)`
//! inside an input has its own signature. A method takes the lifetime of its output from a
//! receiver that is a reference, else it counts the inputs like the others.
//!
//! So a type that adds lifetimes to such inputs is not leaked alone: the type holding the signature
//! (a function pointer, a trait object, or a reference to one that takes its lifetime) is leaked
//! as a whole instead, which keeps the elision as written in the impl of the Repeater. Where
//! nothing holds the signature (a bound, `impl Fn(..)`, a method that has no receiver which is a
//! reference), or it cannot be leaked (a trait object under an elided reference, which takes its
//! lifetime from it), it is an error.
//!
//! The elided lifetimes inside a function pointer or `Fn(..)` are its own (rustc makes them
//! lifetimes of its binder), so they do not stop it from being leaked as a whole. They stop the
//! types inside it, which are not the holder.
//!
//! The copies compile as the tests of `tests/findings.rs` (F12) and the property test show.

mod common;

use common::check_in;
use syn::*;
use template_quote::quote;
use type_leak::{Leaker, LeakerConfig, NotInternableError, Referrer};

fn leaker() -> Leaker {
    let mut config = LeakerConfig::new();
    config.allow_primitive().allow_std_prelude();
    Leaker::from_config(config)
}

fn s(x: &impl template_quote::ToTokens) -> String {
    quote!(#x).to_string()
}

fn analyze(item: &ItemTrait) -> std::result::Result<Referrer, NotInternableError> {
    let mut leaker = leaker();
    leaker.intern_trait(item, true)?;
    Ok(leaker.finish())
}

/// The leaked types of a trait with the generics each uses.
fn leaked(item: ItemTrait) -> Vec<(String, String)> {
    let referrer = analyze(&item).unwrap_or_else(|e| panic!("{}: {}", s(&item), e));
    referrer
        .iter_with_generics()
        .map(|(ty, generics)| (s(ty), s(generics)))
        .collect()
}

fn e(ty: &str, generics: &str) -> (String, String) {
    (ty.to_string(), generics.to_string())
}

/// The message and the first note of the error of a trait.
fn error(item: ItemTrait) -> (String, String) {
    let err = match analyze(&item) {
        Ok(_) => panic!("{} is accepted", s(&item)),
        Err(err) => err,
    };
    (err.to_string(), err.note().unwrap())
}

/// The help of the error of a trait.
fn help(item: &ItemTrait) -> String {
    match analyze(item) {
        Ok(_) => panic!("{} is accepted", s(item)),
        Err(err) => err.help().unwrap(),
    }
}

/// The note of the error where nothing holds the signature.
const NOTE: &str = "the output `& u8` has an elided lifetime, which elision takes from the only \
                    lifetime of the inputs, but a leaked type in the inputs has `'a` among its \
                    generics, and the copy names it in the arguments of its projection, so the \
                    inputs have more than one lifetime (E0106)";

/// What to write for the error of `NOTE` in `Fn(..)` or a function pointer.
const HELP: &str = "name the lifetime of the inputs and of the output, or write `Rel<T>` with an \
                    absolute path";

/// What to write for the error of `NOTE` in a method: it can have a receiver too.
const HELP_METHOD: &str = "name the lifetime of the inputs and of the output, give the method a \
                           receiver (`&self`), or write `Rel<T>` with an absolute path";

/// The note of the error where the holder cannot be leaked.
const NOTE_OWNED: &str = "the output `& u8` has an elided lifetime, which elision takes from the \
                          only lifetime of the inputs, but a leaked type in the inputs has `'a` \
                          among its generics, and the copy names it in the arguments of its \
                          projection, so the inputs have more than one lifetime (E0106), so a \
                          type using them cannot be leaked alone, and the type holding the \
                          signature would be leaked as a whole instead";

#[test]
fn the_holder_is_leaked_as_a_whole() {
    // The sugar of `Fn(..)` in a trait object, in a box, behind a reference, with other bounds.
    for (item, holder) in [
        (
            parse_quote! { trait Tr<'a, T: 'a> { fn f(&self, x: Box<dyn Fn(&Rel<T>) -> &u8>); } },
            "dyn Fn (& Rel < T >) -> & u8",
        ),
        (
            parse_quote! {
                trait Tr<'a, T: 'a> { fn f(&self, x: Box<dyn FnMut(&Rel<T>) -> &u8 + Send>); }
            },
            "dyn FnMut (& Rel < T >) -> & u8 + Send",
        ),
        (
            parse_quote! { trait Tr<'a, T: 'a> { fn f(&self, x: Vec<Box<dyn Fn(&Rel<T>) -> &u8>>); } },
            "dyn Fn (& Rel < T >) -> & u8",
        ),
        // The reference takes the lifetime of the trait object, so it is leaked with it.
        (
            parse_quote! { trait Tr<'a, T: 'a> { fn f(&self, x: &'a dyn Fn(&Rel<T>) -> &u8); } },
            "& 'a dyn Fn (& Rel < T >) -> & u8",
        ),
        (
            parse_quote! {
                trait Tr<'a, T: 'a> { fn f(&self, x: &'a (dyn Fn(&Rel<T>) -> &u8 + 'a)); }
            },
            "& 'a (dyn Fn (& Rel < T >) -> & u8 + 'a)",
        ),
        // A function pointer.
        (
            parse_quote! { trait Tr<'a, T: 'a> { fn f(&self, x: fn(&Rel<T>) -> &u8); } },
            "fn (& Rel < T >) -> & u8",
        ),
        (
            parse_quote! {
                trait Tr<'a, T: 'a> { fn f(&self, x: unsafe extern "C" fn(&Rel<T>) -> &u8); }
            },
            "unsafe extern \"C\" fn (& Rel < T >) -> & u8",
        ),
        // Other elided lifetimes in the output, and a lifetime argument in the input.
        (
            parse_quote! {
                trait Tr<'a, T: 'a> { fn f(&self, x: fn(&Rel<T>) -> ::p::Ref<'_, u8>); }
            },
            "fn (& Rel < T >) -> :: p :: Ref < '_ , u8 >",
        ),
        (
            parse_quote! {
                trait Tr<'a, T: 'a> {
                    fn f(&self, x: Box<dyn Fn(&Rel<T>) -> Box<dyn Iterator<Item = &u8> + '_>>);
                }
            },
            "dyn Fn (& Rel < T >) -> Box < dyn Iterator < Item = & u8 > + '_ >",
        ),
        // The input is the only lifetime, whatever it is: the entry adds `'a` to it.
        (
            parse_quote! {
                trait Tr<'a, T: 'a> { fn f(&self, x: Box<dyn Fn(&'static Rel<T>) -> &u8>); }
            },
            "dyn Fn (& 'static Rel < T >) -> & u8",
        ),
        (
            parse_quote! {
                trait Tr<'a, T: 'a> { fn f(&self, x: Box<dyn for<'x> Fn(&'x Rel<T>) -> &u8>); }
            },
            "dyn for < 'x > Fn (& 'x Rel < T >) -> & u8",
        ),
    ] {
        let item: ItemTrait = item;
        let found = leaked(item.clone());
        assert_eq!(found.len(), 1, "{}: {:?}", s(&item), found);
        assert_eq!(found[0].0, holder, "{}", s(&item));
        assert!(found[0].1.contains("'a"), "{}: {:?}", s(&item), found);
    }
    // The generics are those of the whole: the lifetimes and the parameters the holder uses.
    assert_eq!(
        leaked(parse_quote! {
            trait Tr<'a, 'b, T: 'a, U: 'b> { fn f(&self, x: Box<dyn Fn(&Rel<T>) -> &u8>, y: &'b U); }
        }),
        [e("dyn Fn (& Rel < T >) -> & u8", "< 'a , T : 'a >")]
    );
    // What else is in it is covered, so it is not leaked on its own.
    assert_eq!(
        leaked(parse_quote! {
            trait Tr<'a, T: 'a> { fn f(&self, x: Box<dyn Fn(&Rel<T>, u8) -> &Rel2>); }
        }),
        [e("dyn Fn (& Rel < T > , u8) -> & Rel2", "< 'a , T : 'a >")]
    );
    // `Self` in an entry: the holder is leaked with it.
    assert_eq!(
        leaked(parse_quote! {
            trait Tr<'a> where Self: Sized + 'a { fn f(&self, x: Box<dyn Fn(&Rel<Self>) -> &u8>); }
        }),
        [e("dyn Fn (& Rel < Self >) -> & u8", "< 'a >")]
    );
    // The fields of a struct.
    let item: ItemStruct = parse_quote! {
        struct S<'a, T: 'a> { f: fn(&Rel<T>) -> &u8, g: Box<dyn Fn(&Rel<T>) -> &u8>, h: &'a Rel<T> }
    };
    let mut leaker = leaker();
    leaker.intern_struct(&item, true).unwrap();
    let referrer = leaker.finish();
    let mut found: Vec<String> = referrer.iter().map(s).collect();
    found.sort();
    assert_eq!(
        found,
        [
            "& 'a Rel < T >",
            "dyn Fn (& Rel < T >) -> & u8",
            "fn (& Rel < T >) -> & u8"
        ]
    );
}

#[test]
fn a_holder_inside_a_signature_is_the_entry_for_the_one_around() {
    // The inner one is the holder of `Rel<T>`, and it adds `'a` to the inputs of the outer one,
    // which is the entry in turn.
    assert_eq!(
        leaked(parse_quote! {
            trait Tr<'a, T: 'a> {
                fn f(&self, x: Box<dyn Fn(&Box<dyn Fn(&Rel<T>) -> &u8>) -> &u8>);
            }
        }),
        [e(
            "dyn Fn (& Box < dyn Fn (& Rel < T >) -> & u8 >) -> & u8",
            "< 'a , T : 'a >"
        )]
    );
    // The lifetimes in a signature inside an input are not those of the one around: the outer one
    // has only `&`, and the inner one names its own lifetime, so `Rel<T>` is leaked alone.
    assert_eq!(
        leaked(parse_quote! {
            trait Tr<'a, T: 'a> {
                fn f(&self, x: Box<dyn Fn(&Box<dyn for<'x> Fn(&'x Rel<T>) -> &'x u8>) -> &u8>);
            }
        }),
        [e("Rel < T >", "< 'a , T : 'a >")]
    );
    // The output of a signature is not in its inputs: an inner one there is its own holder.
    assert_eq!(
        leaked(parse_quote! {
            trait Tr<'a, T: 'a> {
                fn f(&self, x: Box<dyn Fn(&u8) -> Box<dyn Fn(&Rel<T>) -> &u8>>);
            }
        }),
        [e("dyn Fn (& Rel < T >) -> & u8", "< 'a , T : 'a >")]
    );
    // A method with a receiver that is a reference: its own elision is not at stake.
    assert_eq!(
        leaked(parse_quote! {
            trait Tr<'a, T: 'a> {
                fn f(&self, x: &Rel<T>, y: Box<dyn Fn(&Rel<T>) -> &u8>) -> &u8;
            }
        }),
        [
            e("Rel < T >", "< 'a , T : 'a >"),
            e("dyn Fn (& Rel < T >) -> & u8", "< 'a , T : 'a >")
        ]
    );
}

/// rules.md, Elided lifetimes: "a function pointer or `Fn(..)` inside an input has its own
/// signature". The same holds for one in the OUTPUT of the sugar: its elided lifetimes are its
/// own and do not count for the outer one, whose only lifetime is that of `&`. So `Rel<T>` is
/// leaked alone, whatever the signature in the output elides. Audit mutant
/// `e_output_barefn_not_own` (a bare function pointer in the output counts for the outer
/// signature, and the whole object is leaked).
#[test]
fn an_inner_signature_in_the_output_has_its_own_elision() {
    for output in [
        // A function pointer, an `Fn(..)`, and one with a binder in the inputs.
        "fn(&u8) -> &u8",
        "Box<dyn Fn(&u8) -> &u8>",
        "for<'y> fn(&'y u8) -> &'y u8",
        // In a tuple with another type, and behind a reference.
        "(u8, fn(&u8) -> &u8)",
        "Box<fn(&u8) -> &u8>",
    ] {
        let tr: ItemTrait = parse_str(&format!(
            "trait Tr<'a, T: 'a> {{ fn f(&self, x: Box<dyn Fn(&Rel<T>) -> {}>); }}",
            output
        ))
        .unwrap();
        assert_eq!(
            leaked(tr),
            [e("Rel < T >", "< 'a , T : 'a >")],
            "{}",
            output
        );
    }
    // An inner signature in the inputs that names its lifetimes is its own too: `Rel<T>` is leaked
    // alone. Without the names, the inner one is the holder of `Rel<T>`, and adds `'a` to the
    // inputs of the one around, which is the entry in turn (see the test above).
    assert_eq!(
        leaked(parse_quote! {
            trait Tr<'a, T: 'a> {
                fn f(&self, x: Box<dyn Fn(for<'y> fn(&'y Rel<T>) -> &'y u8, &u8) -> &u8>);
            }
        }),
        [e("Rel < T >", "< 'a , T : 'a >")]
    );
    assert_eq!(
        leaked(parse_quote! {
            trait Tr<'a, T: 'a> {
                fn f(&self, x: Box<dyn Fn(fn(&Rel<T>) -> &u8, &u8) -> &u8>);
            }
        }),
        [e(
            "dyn Fn (fn (& Rel < T >) -> & u8 , & u8) -> & u8",
            "< 'a , T : 'a >"
        )]
    );
}

#[test]
fn a_type_with_no_lifetime_is_leaked_alone() {
    // No lifetime among the generics of the entry, so the copy has only the elided one.
    for item in [
        parse_quote! { trait Tr<T> { fn f(&self, x: Box<dyn Fn(&Rel<T>) -> &u8>); } },
        parse_quote! { trait Tr<'a, T> { fn f(&self, x: Box<dyn Fn(&Rel<T>) -> &u8>); } },
        parse_quote! { trait Tr<T> { fn f(&self, x: fn(&Rel<T>) -> &u8); } },
        parse_quote! { trait Tr { fn f(&self, x: Box<dyn Fn(&Rel) -> &u8>); } },
        parse_quote! { trait Tr<T> { fn g(x: &Rel<T>) -> &u8; } },
        parse_quote! { trait Tr<T, F: Fn(&Rel<T>) -> &u8> {} },
    ] {
        let item: ItemTrait = item;
        let found = leaked(item.clone());
        assert_eq!(found.len(), 1, "{}: {:?}", s(&item), found);
        assert!(found[0].0.starts_with("Rel"), "{}: {:?}", s(&item), found);
        assert!(!found[0].1.contains("'a"), "{}: {:?}", s(&item), found);
    }
}

#[test]
fn the_elision_is_not_at_stake_if_the_output_does_not_rely_on_it() {
    let alone = [e("Rel < T >", "< 'a , T : 'a >")];
    for item in [
        // The output has no lifetime, or names it.
        parse_quote! { trait Tr<'a, T: 'a> { fn f(&self, x: Box<dyn Fn(&Rel<T>) -> u8>); } },
        parse_quote! { trait Tr<'a, T: 'a> { fn f(&self, x: Box<dyn Fn(&Rel<T>) -> Rel<T>>); } },
        parse_quote! { trait Tr<'a, T: 'a> { fn f(&self, x: Box<dyn Fn(&Rel<T>) -> &'a u8>); } },
        parse_quote! { trait Tr<'a, T: 'a> { fn f(&self, x: fn(&Rel<T>) -> &'static u8); } },
        parse_quote! { trait Tr<'a, T: 'a> { fn f(&self, x: fn(&Rel<T>)); } },
        // The output of an inner signature is its own.
        parse_quote! {
            trait Tr<'a, T: 'a> { fn f(&self, x: Box<dyn Fn(&Rel<T>) -> Box<dyn Fn(&u8) -> &u8>>); }
        },
        // The output names the lifetime of the input.
        parse_quote! {
            trait Tr<'a, T: 'a> { fn f(&self, x: Box<dyn for<'x> Fn(&'x Rel<T>) -> &'x u8>); }
        },
        parse_quote! { trait Tr<'a, T: 'a> { fn f(&self, x: for<'x> fn(&'x Rel<T>) -> &'x u8); } },
        // The entry is in the output.
        parse_quote! { trait Tr<'a, T: 'a> { fn f(&self, x: Box<dyn Fn(&u8) -> &Rel<T>>); } },
        // More than one lifetime in the inputs, or none: there is no elision to keep.
        parse_quote! { trait Tr<'a, T: 'a> { fn f(&self, x: fn(&u8, &Rel<T>) -> u8); } },
        // A method takes the lifetime of the output from the receiver that is a reference.
        parse_quote! { trait Tr<'a, T: 'a> { fn g(&self, x: &Rel<T>) -> &u8; } },
        parse_quote! { trait Tr<'a, T: 'a> { fn g(&mut self, x: &Rel<T>) -> &u8; } },
        parse_quote! { trait Tr<'a, T: 'a> { fn g(&'a self, x: &Rel<T>) -> &u8; } },
        parse_quote! { trait Tr<'a, T: 'a> { fn g(self: &Self, x: &Rel<T>) -> &u8; } },
        parse_quote! { trait Tr<'a, T: 'a> { fn g(self: &mut Self, x: &Rel<T>) -> &u8; } },
        parse_quote! {
            trait Tr<'a, T: 'a> { fn g(self: ::core::pin::Pin<&mut Self>, x: &Rel<T>) -> &u8; }
        },
        // Or it has no elided lifetime.
        parse_quote! { trait Tr<'a, T: 'a> { fn g(x: &Rel<T>) -> &'a u8; } },
        parse_quote! { trait Tr<'a, T: 'a> { fn g(x: &Rel<T>) -> u8; } },
    ] {
        let item: ItemTrait = item;
        assert_eq!(leaked(item.clone()), alone, "{}", s(&item));
    }
    // The only lifetime in the inputs is the one the entry has: nothing is added.
    assert_eq!(
        leaked(parse_quote! { trait Tr<'a, T: 'a> { fn g(x: &'a Rel<T>) -> &u8; } }),
        [e("& 'a Rel < T >", "< 'a , T : 'a >")]
    );
    assert_eq!(
        leaked(parse_quote! { trait Tr<'a, T: 'a> { fn g(x: Rel2<'a>, y: &'a u8) -> &u8; } }),
        [e("Rel2 < 'a >", "< 'a >")]
    );
}

#[test]
fn a_method_with_no_lifetime_from_its_receiver_is_an_error() {
    // Nothing holds a method, so the type that adds the lifetime is an error.
    for item in [
        parse_quote! { trait Tr<'a, T: 'a> { fn g(x: &Rel<T>) -> &u8; } },
        parse_quote! { trait Tr<'a, T: 'a> { fn g(x: &Rel<T>, y: u8) -> &u8; } },
        // A receiver that is not a reference.
        parse_quote! { trait Tr<'a, T: 'a>: Sized { fn g(self, x: &Rel<T>) -> &u8; } },
        parse_quote! { trait Tr<'a, T: 'a> { fn g(self: Box<Self>, x: &Rel<T>) -> &u8; } },
        parse_quote! { trait Tr<'a, T: 'a> { fn g(self: ::std::rc::Rc<Self>, x: &Rel<T>) -> &u8; } },
        // The type is not at the top of the input.
        parse_quote! { trait Tr<'a, T: 'a> { fn g(x: &(u8, Rel<T>)) -> &u8; } },
        parse_quote! { trait Tr<'a, T: 'a> { fn g(x: &[Rel<T>]) -> &u8; } },
    ] {
        let item: ItemTrait = item;
        let (message, note) = error(item.clone());
        assert_eq!(message, "use absolute path: 'Rel < T >'", "{}", s(&item));
        assert_eq!(note, NOTE, "{}", s(&item));
        assert_eq!(help(&item), HELP_METHOD, "{}", s(&item));
    }
    // Another output with an elided lifetime.
    for (item, output) in [
        (
            parse_quote! { trait Tr<'a, T: 'a> { fn g(x: &Rel<T>) -> ::p::Ref<'_, u8>; } },
            ":: p :: Ref < '_ , u8 >",
        ),
        (
            parse_quote! { trait Tr<'a, T: 'a> { fn g(x: &Rel<T>) -> Box<dyn Iterator<Item = &u8>>; } },
            "Box < dyn Iterator < Item = & u8 > >",
        ),
    ] {
        let item: ItemTrait = item;
        let (_, note) = error(item.clone());
        assert_eq!(
            note,
            NOTE.replace("`& u8`", &format!("`{}`", output)),
            "{}",
            s(&item)
        );
    }
    // The message, as rustc shows it: the lifetime that is elided is the cause.
    let err = analyze(&parse_quote! { trait Tr<'a, T: 'a> { fn g(x: &Rel<T>) -> &u8; } })
        .err()
        .unwrap();
    assert_eq!(
        syn::Error::from(err).to_string(),
        format!(
            "use absolute path: 'Rel < T >'\n\n  = note: {}\n  = note: it cannot be leaked as a \
             whole because of `& u8`\n  = help: {}\n",
            NOTE, HELP_METHOD
        )
    );
    // A function item of a module has no receiver either.
    let item: ItemMod =
        parse_quote! { mod m { pub fn g<'a, T: 'a>(x: &Rel<T>) -> &u8 { loop {} } } };
    let err = leaker().intern_mod(&item).err().unwrap();
    assert_eq!(err.note().as_deref(), Some(NOTE));
    // And an impl.
    let item: ItemImpl = parse_quote! {
        impl<'a, T: 'a> ::p::Tr<'a, T> for u8 { fn g(x: &Rel<T>) -> &u8 { loop {} } }
    };
    let err = leaker().intern_impl(&item, true).err().unwrap();
    assert_eq!(err.note().as_deref(), Some(NOTE));
}

#[test]
fn a_bound_with_the_elision_is_an_error() {
    // Nothing holds a bound: it is not a type.
    for item in [
        parse_quote! { trait Tr<'a, T: 'a, F: Fn(&Rel<T>) -> &u8> {} },
        parse_quote! { trait Tr<'a, T: 'a, F> where F: Fn(&Rel<T>) -> &u8 {} },
        parse_quote! { trait Tr<'a, T: 'a>: Fn(&Rel<T>) -> &u8 {} },
        parse_quote! { trait Tr<'a, T: 'a> { type A: Fn(&Rel<T>) -> &u8; } },
        parse_quote! { trait Tr<'a, T: 'a> { fn f<F: Fn(&Rel<T>) -> &u8>(&self, f: F); } },
        parse_quote! { trait Tr<'a, T: 'a> { fn f<F>(&self, f: F) where F: Fn(&Rel<T>) -> &u8; } },
        parse_quote! { trait Tr<'a, T: 'a> { fn f<F>(&self, f: F) where Box<F>: Fn(&Rel<T>) -> &u8; } },
    ] {
        let item: ItemTrait = item;
        let (message, note) = error(item.clone());
        assert_eq!(message, "use absolute path: 'Rel < T >'", "{}", s(&item));
        assert_eq!(note, NOTE, "{}", s(&item));
        assert_eq!(help(&item), HELP, "{}", s(&item));
    }
    // No lifetime in the entry, or no elision: fine.
    assert_eq!(
        leaked(parse_quote! { trait Tr<T, F: Fn(&Rel<T>) -> &u8> {} }),
        [e("Rel < T >", "< T >")]
    );
    assert_eq!(
        leaked(parse_quote! { trait Tr<'a, T: 'a, F: for<'x> Fn(&'x Rel<T>) -> &'x u8> {} }),
        [e("Rel < T >", "< 'a , T : 'a >")]
    );
}

#[test]
fn a_holder_that_cannot_be_leaked_is_an_error() {
    // `impl Trait` cannot be leaked at all.
    let (message, note) = error(parse_quote! {
        trait Tr<'a, T: 'a> { fn f(&self, x: impl Fn(&Rel<T>) -> &u8); }
    });
    assert_eq!(message, "use absolute path: 'Rel < T >'");
    assert_eq!(note, NOTE_OWNED);
    // A trait object under an elided reference takes its lifetime from it, so it cannot be leaked
    // alone, and the reference has no lifetime to name.
    let err = analyze(&parse_quote! {
        trait Tr<'a, T: 'a> { fn f(&self, x: &dyn Fn(&Rel<T>) -> &u8); }
    })
    .err()
    .unwrap();
    assert_eq!(
        syn::Error::from(err).to_string(),
        format!(
            "use absolute path: 'Rel < T >'\n\n  = note: {}\n  = note: it cannot be leaked as a \
             whole because of `dyn Fn (& Rel < T >) -> & u8`\n  = note: the lifetime of the trait \
             object defaults to that of the enclosing reference or path\n  = help: {}\n",
            NOTE_OWNED, HELP
        )
    );
    // Naming it is the fix.
    assert_eq!(
        leaked(parse_quote! {
            trait Tr<'a, T: 'a> { fn f(&self, x: &'a dyn Fn(&Rel<T>) -> &u8); }
        }),
        [e("& 'a dyn Fn (& Rel < T >) -> & u8", "< 'a , T : 'a >")]
    );
}

#[test]
fn an_elided_lifetime_of_the_holder_is_its_own() {
    // `&u8` is a lifetime of the function pointer (rustc makes it one of its binder), so it does
    // not stop it from being leaked as a whole: the impl of the Repeater writes it as it is.
    assert_eq!(
        leaked(parse_quote! {
            trait Tr { fn f(&self, f: for<'x> fn(Rel<'x>, &u8) -> &'x u8); }
        }),
        [e("for < 'x > fn (Rel < 'x > , & u8) -> & 'x u8", "")]
    );
    assert_eq!(
        leaked(parse_quote! {
            trait Tr { fn f(&self, f: Box<dyn for<'x> Fn(Rel<'x>, &u8) -> &'x u8>); }
        }),
        [e("dyn for < 'x > Fn (Rel < 'x > , & u8) -> & 'x u8", "")]
    );
    assert_eq!(
        leaked(parse_quote! {
            trait Tr { fn f(&self, f: for<'x> fn(Rel<'x>, ::p::Ref<'_, u8>) -> &'x u8); }
        }),
        [e(
            "for < 'x > fn (Rel < 'x > , :: p :: Ref < '_ , u8 >) -> & 'x u8",
            ""
        )]
    );
    // The types inside it are not the holder: a lifetime that is elided in them is not theirs, so
    // they cannot be leaked on their own.
    let (message, note) = error(parse_quote! {
        trait Tr { fn f(&self, f: Box<dyn Fn(Rel<&Rel2>) -> u8>); }
    });
    assert_eq!(message, "use absolute path: 'Rel < & Rel2 >'");
    assert_eq!(note, "it cannot be leaked as a whole because of `& Rel2`");
    // The holder is not an entry for that reason alone: with no cause in it, it is copied.
    let referrer = analyze(&parse_quote! {
        trait Tr { fn f(&self, f: Box<dyn Fn(&u8) -> &u8>, g: for<'x> fn(&'x u8, &u8)); }
    })
    .unwrap();
    assert_eq!(referrer.iter().count(), 0);
}

#[test]
fn check_tells_the_holder_is_interned() {
    let generics: Generics = parse_quote!(<'a, T: 'a>);
    let mut leaker = leaker();
    let mut check = |ty: Type| check_in(&mut leaker, &ty, &generics);
    let tokens = |tokens: &[proc_macro2::TokenStream]| -> Vec<String> {
        tokens.iter().map(|t| t.to_string()).collect()
    };
    // The holder is the cause of its own, and it has no impossible one: its elided lifetimes are
    // its own.
    let result = check(parse_quote!(dyn Fn(&Rel<T>) -> &u8));
    assert_eq!(tokens(&result.must), ["Rel < T >"]);
    assert!(result.is_own && result.impossible.is_empty());
    let result = check(parse_quote!(fn(&Rel<T>) -> &u8));
    assert_eq!(tokens(&result.must), ["Rel < T >"]);
    assert!(result.is_own && result.impossible.is_empty());
    // A box holds it: the cause is the trait object, and not the box.
    let result = check(parse_quote!(Box<dyn Fn(&Rel<T>) -> &u8>));
    assert_eq!(tokens(&result.must), ["Rel < T >"]);
    assert!(!result.is_own && result.impossible.is_empty());
    // Not where the entry has no lifetimes.
    let generics: Generics = parse_quote!(<T>);
    let result = check_in(
        &mut self::leaker(),
        &parse_quote!(dyn Fn(&Rel<T>) -> &u8),
        &generics,
    );
    assert_eq!(tokens(&result.must), ["Rel < T >"]);
    assert!(!result.is_own && result.impossible.is_empty());
}

#[test]
fn the_copy_is_rewritten_to_the_holder() {
    let item: ItemTrait = parse_quote! {
        trait Tr<'a, T: 'a> {
            fn f(&self, a: Box<dyn Fn(&Rel<T>) -> &u8>, b: fn(&Rel<T>) -> &u8, c: &Rel<T>);
        }
    };
    let mut leaker = leaker();
    let context = leaker.intern_trait(&item, true).unwrap();
    let referrer = leaker.finish();
    // The entries are ordered by context and then by their tokens.
    assert_eq!(
        referrer.iter().map(s).collect::<Vec<_>>(),
        [
            "Rel < T >",
            "dyn Fn (& Rel < T >) -> & u8",
            "fn (& Rel < T >) -> & u8"
        ]
    );
    let mut copy = item.clone();
    syn::visit_mut::VisitMut::visit_item_trait_mut(
        &mut referrer.visitor(context, |_, ix| parse_quote!(P<#ix>)),
        &mut copy,
    );
    let expected: ItemTrait = parse_quote! {
        trait Tr<'a, T: 'a> {
            fn f(&self, a: Box<P<1usize>>, b: P<2usize>, c: &P<0usize>);
        }
    };
    assert_eq!(s(&copy), s(&expected));
}

#[test]
fn a_view_walks_the_signature_like_the_leaker() {
    use syn::visit::Visit;
    // `visit_signature` of the view analyzes the arguments of a method with its elision.
    let sig: Signature = parse_quote!(fn g(x: &Rel<T>) -> &u8);
    let generics: Generics = parse_quote!(<'a, T: 'a>);
    let mut leaker = leaker();
    let result = leaker.enter_generics(&generics, |view| view.visit_signature(&sig));
    assert_eq!(result.err().unwrap().note().as_deref(), Some(NOTE));
    // And the sugar and the function pointers in a type visited alone.
    let ty: Type = parse_quote!(Box<dyn Fn(&Rel<T>) -> &u8>);
    let mut leaker = self::leaker();
    leaker
        .enter_generics(&generics, |view| view.visit_type(&ty))
        .unwrap();
    assert_eq!(
        leaker.finish().iter().map(s).collect::<Vec<_>>(),
        ["dyn Fn (& Rel < T >) -> & u8"]
    );
}

#[test]
fn an_elided_output_may_take_the_lifetime_of_the_binder() {
    // The output has the only lifetime of the inputs, `'x` of the binder, which `Rel<'x>` leaked
    // alone would have only in a projection: rustc rejects the copy (E0581, E0582) as it does for
    // an output that names `'x` (see `tests/binder_output.rs`), so the holder is the entry.
    assert_eq!(
        leaked(parse_quote! { trait Tr { fn f(&self, f: for<'x> fn(Rel<'x>) -> &u8); } }),
        [e("for < 'x > fn (Rel < 'x >) -> & u8", "")]
    );
    assert_eq!(
        leaked(parse_quote! { trait Tr { fn f(&self, f: Box<dyn for<'x> Fn(Rel<'x>) -> &u8>); } }),
        [e("dyn for < 'x > Fn (Rel < 'x >) -> & u8", "")]
    );
    // An input that keeps `'x` anchors it, whatever the output is.
    assert_eq!(
        leaked(parse_quote! { trait Tr { fn f(&self, f: for<'x> fn(&'x u8, Rel<'x>) -> &u8); } }),
        [e("Rel < 'x >", "< 'x >")]
    );
    // A bound or a where predicate cannot be leaked, so it is an error.
    for item in [
        parse_quote! { trait Tr { fn f<F: for<'x> Fn(Rel<'x>) -> &u8>(&self, f: F); } },
        parse_quote! { trait Tr { fn f<F>(&self, f: F) where F: for<'x> Fn(Rel<'x>) -> &u8; } },
        parse_quote! { trait Tr { fn f<F>(&self, f: F) where for<'x> F: Fn(Rel<'x>) -> &u8; } },
    ] {
        let item: ItemTrait = item;
        let (message, _) = error(item.clone());
        assert_eq!(message, "use absolute path: 'Rel < 'x >'", "{}", s(&item));
    }
    // The lifetime of the trait is not declared by the binder, so it is not at stake.
    assert_eq!(
        leaked(parse_quote! { trait Tr<'a> { fn f(&self, f: fn(Rel<'a>) -> &u8); } }),
        [e("Rel < 'a >", "< 'a >")]
    );
    assert_eq!(
        leaked(parse_quote! { trait Tr { fn f(&self, f: for<'x> fn(Rel<'x>, u8) -> u8); } }),
        [e("Rel < 'x >", "< 'x >")]
    );
}
