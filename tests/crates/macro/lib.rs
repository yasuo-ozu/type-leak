//! A proc macro which uses `type-leak` as a downstream crate would (in the style of
//! `newer-type`), to test it across crate boundaries.
//!
//! In the Leaker's crate:
//!
//! ```ignore
//! pub use type_leak_test_macro as __macro;
//! type_leak_test_macro::define_repeater!(pub Repeater);
//!
//! pub mod m {
//!     pub struct MyType;
//!     #[type_leak_test_macro::leak(
//!         helper = crate::__macro, repeater = crate::Repeater, module = crate::m,
//!     )]
//!     pub trait MyTrait<const B: bool> { fn f(&self) -> MyType; }
//! }
//! ```
//!
//! The generics each leaked type uses (those of the trait and those of its methods, see
//! [`Referrer::generics()`]) are encoded into the Repeater's trait arguments with
//! [`RepeaterSlots`], so the Repeater (defined by [`repeater_definition()`]) is the only item
//! shared by the Leaker and the Referrer. The lifetime bounds in them (`T: 'a`) are encoded too
//! (`&'a T`), so the Repeater's implementations declare the generics without them, as
//! [`Referrer::impl_generics()`] returns them.
//!
//! In the Referrer's crate, `MyTrait!` implements `MyTrait` for a wrapper by forwarding each
//! method to the field `.0`, with the types in the signatures rewritten by the [`Referrer`]:
//!
//! ```ignore
//! pub struct Wrapper<const B: bool>(pub Inner);
//! leaker::m::MyTrait!(impl<const B: bool> MyTrait<B> for Wrapper<B> => 0);
//! ```

mod bare;
mod nested;
mod prop;
mod scope;
mod shared;

use proc_macro2::{Punct, Spacing, Span, TokenStream, TokenTree};
use std::collections::HashMap;
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;
use syn::visit_mut::VisitMut;
use syn::*;
use template_quote::{quote, ToTokens};
use type_leak::*;

/// Arguments of `#[leak(..)]`.
struct LeakArgs {
    helper: Path,
    repeater: Path,
    module: Path,
    /// An external crate which the Leaker allows (`allow = ::std`), whose types are not optional
    /// entries.
    allow: Option<Path>,
}

impl Parse for LeakArgs {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut map: HashMap<String, Path> = HashMap::new();
        for kv in Punctuated::<MetaNameValue, Token![,]>::parse_terminated(input)? {
            let key = kv
                .path
                .get_ident()
                .map(|i| i.to_string())
                .unwrap_or_default();
            let value = match kv.value {
                Expr::Path(ExprPath { path, .. }) => path,
                other => return Err(Error::new(other.span(), "expected a path")),
            };
            map.insert(key, value);
        }
        let mut take = |key: &str| {
            map.remove(key)
                .ok_or_else(|| Error::new(Span::call_site(), format!("missing `{}`", key)))
        };
        let allow = take("allow").ok();
        Ok(Self {
            helper: take("helper")?,
            repeater: take("repeater")?,
            module: take("module")?,
            allow,
        })
    }
}

/// Emits the Repeater defined by [`repeater_definition()`]: `define_repeater!(pub Repeater);`.
#[proc_macro]
pub fn define_repeater(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    struct Input(Visibility, Ident);
    impl Parse for Input {
        fn parse(input: ParseStream) -> Result<Self> {
            let vis = input.parse()?;
            let ident = input.parse()?;
            let _: Option<Token![;]> = input.parse()?;
            Ok(Self(vis, ident))
        }
    }
    let Input(vis, ident) = parse_macro_input!(input as Input);
    let def = repeater_definition(&vis, &ident);
    quote!(#def).into()
}

/// Returns a deterministic ID of the trait, used as the Repeater's `ID`.
fn trait_id(module: &Path, ident: &Ident) -> u64 {
    // FNV-1a
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in format!("{}::{}", quote!(#module), ident).bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Replaces the leading `crate` of `path` with `$crate`, so that it can be used in the exported
/// `macro_rules!`.
fn dollar_crate(path: &Path) -> TokenStream {
    let mut tokens = quote!(#path).into_iter().collect::<Vec<_>>();
    if matches!(tokens.first(), Some(TokenTree::Ident(i)) if i == "crate") {
        tokens.insert(0, TokenTree::Punct(Punct::new('$', Spacing::Alone)));
    }
    tokens.into_iter().collect()
}

fn param_as_arg(param: &GenericParam) -> TokenStream {
    match param {
        GenericParam::Lifetime(l) => {
            let l = &l.lifetime;
            quote!(#l)
        }
        GenericParam::Type(t) => {
            let i = &t.ident;
            quote!(#i)
        }
        GenericParam::Const(c) => {
            let i = &c.ident;
            quote!(#i)
        }
    }
}

/// Checks `Referrer::is_plain()` of the `ix`-th entry against what the impl needs: an entry is
/// plain if and only if its impl for a marker type declares nothing (no generics, no where clause,
/// no argument, no module) and does not name `Self`.
pub(crate) fn check_plain(referrer: &Referrer, ix: usize, slots: &RepeaterSlots) -> Result<()> {
    let marker: Type = parse_quote!(__Marker);
    let r = referrer.repeater_impl_with(ix, ImplSelf::Marker(&marker), |value, param| {
        slots.encode_const(value, &param.ty)
    });
    let needs_nothing = match r {
        Ok(r) => {
            let predicates = r.generics.where_clause.iter().flat_map(|w| &w.predicates);
            r.generics.params.is_empty()
                && predicates.count() == 0
                && r.args == parse_quote!(())
                && r.module.is_none()
        }
        // The type names `Self`.
        Err(EncodeError::NeedsSelf(_)) => false,
        Err(e) => return Err(e.into()),
    };
    if referrer.is_plain(ix) == needs_nothing {
        return Ok(());
    }
    let ty = referrer.iter().nth(ix).unwrap();
    Err(Error::new(
        Span::call_site(),
        format!(
            "`Referrer::is_plain({})` is {}, but the impl for a marker needs {}: {}",
            ix,
            referrer.is_plain(ix),
            if needs_nothing { "nothing" } else { "more" },
            quote!(#ty)
        ),
    ))
}

/// The errors of the Leaker, reported with `ErrorItem::to_compile_error()` (a `Diagnostic` on
/// nightly), one by one.
pub(crate) fn report(errors: &[ErrorItem]) -> TokenStream {
    errors.iter().map(ErrorItem::to_compile_error).collect()
}

/// The messages of the errors of the Leaker, one per line.
pub(crate) fn messages(errors: &[ErrorItem]) -> String {
    let messages: Vec<&str> = errors.iter().map(|item| item.message.as_str()).collect();
    messages.join("\n")
}

/// Reports the errors of the Leaker as `report()` does, or, with `serialized`, after they were
/// written as tokens (their `ToTokens`) and read back (their `Parse`) through a
/// `proc_macro::TokenStream`, as a macro does that hands its errors to another one (the attribute
/// of `leak_strict` and `leak_generics`: `#[leak_strict(serialized)]`).
fn report_errors(errors: &[ErrorItem], serialized: bool) -> proc_macro::TokenStream {
    if !serialized {
        return report(errors).into();
    }
    let written: TokenStream = errors.iter().map(ToTokens::to_token_stream).collect();
    let read = |input: ParseStream| -> Result<Vec<ErrorItem>> {
        let mut items = Vec::new();
        while !input.is_empty() {
            items.push(input.parse()?);
        }
        Ok(items)
    };
    match syn::parse::Parser::parse(read, proc_macro::TokenStream::from(written)) {
        Ok(items) => report(&items).into(),
        Err(e) => e.to_compile_error().into(),
    }
}

/// Analyzes a trait without allowing any relative path, reporting the errors with
/// `ErrorItem::to_compile_error()` (for tests/diagnostic.rs). `#[leak_strict(serialized)]` reports
/// them after the serialization (`report_errors()`).
#[proc_macro_attribute]
pub fn leak_strict(
    attr: proc_macro::TokenStream,
    item: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let serialized = attr.to_string() == "serialized";
    let item_trait = parse_macro_input!(item as ItemTrait);
    let mut leaker = Leaker::new();
    leaker.intern_trait(&item_trait, true);
    match leaker.result().finish() {
        Ok(_) => quote!(#item_trait).into(),
        Err(e) => report_errors(&e, serialized),
    }
}

/// Like `leak_strict`, but enters only the generics of the trait (`Leaker::enter_generics()`), where
/// `Self` has no meaning, and visits the types of its methods: a mention of `Self` is an error
/// with the reason `SelfWithoutMeaning`, reported with `ErrorItem::to_compile_error()` (for
/// tests/diagnostic.rs). `#[leak_generics(serialized)]` reports them after the serialization.
#[proc_macro_attribute]
pub fn leak_generics(
    attr: proc_macro::TokenStream,
    item: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let serialized = attr.to_string() == "serialized";
    let item_trait = parse_macro_input!(item as ItemTrait);
    let mut leaker = Leaker::new();
    leaker.enter_generics(&item_trait.generics, |view| {
        for trait_item in &item_trait.items {
            if let TraitItem::Fn(method) = trait_item {
                for input in &method.sig.inputs {
                    if let FnArg::Typed(arg) = input {
                        view.visit_type(&arg.ty);
                    }
                }
                if let ReturnType::Type(_, ty) = &method.sig.output {
                    view.visit_type(ty);
                }
            }
        }
    });
    match leaker.result().finish() {
        Ok(_) => quote!(#item_trait).into(),
        Err(e) => report_errors(&e, serialized),
    }
}

/// `#[leak_tagged(tag = "current")]` on a trait analyzes it and exports `macro_rules!
/// __tagged_<Trait>` which hands the serialized Referrer to `__read_referrer!`, as the macro of a
/// Referrer's crate does with the one it gets from the Leaker's crate. The tag of the format
/// (`Referrer::FORMAT_VERSION`) it carries is `"current"` (the right one), `"stale"` (the previous
/// version), `"future"` (the next one), `"none"` (missing) or `"garbled"` (not a tag at all), so
/// that the error of a Referrer serialized by another version of `type-leak` can be tested through
/// a macro (tests/diagnostic.rs).
#[proc_macro_attribute]
pub fn leak_tagged(
    attr: proc_macro::TokenStream,
    item: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let tag = match &parse_macro_input!(attr as MetaNameValue).value {
        Expr::Lit(ExprLit {
            lit: Lit::Str(tag), ..
        }) => tag.value(),
        other => return Error::new(other.span(), "expected a string").to_compile_error().into(),
    };
    let item_trait = parse_macro_input!(item as ItemTrait);
    let mut leaker = Leaker::new();
    let context = leaker.intern_trait(&item_trait, true);
    let referrer = match leaker.result().finish() {
        Ok(referrer) => referrer,
        Err(e) => return report(&e).into(),
    };
    let mut tokens: Vec<TokenTree> = quote!(#referrer).into_iter().collect();
    // The first token of a Referrer is its tag, `"type-leak:<version>"` or the like.
    let current: LitStr = match parse2(tokens[0].clone().into()) {
        Ok(current) => current,
        Err(e) => return e.to_compile_error().into(),
    };
    let prefix = current.value();
    let prefix = prefix.trim_end_matches(|c: char| c.is_ascii_digit());
    let version = Referrer::FORMAT_VERSION;
    match tag.as_str() {
        "current" => (),
        "stale" => tokens[0] = proc_macro2::Literal::string(&format!("{}{}", prefix, version - 1)).into(),
        "future" => tokens[0] = proc_macro2::Literal::string(&format!("{}{}", prefix, version + 1)).into(),
        "garbled" => tokens[0] = proc_macro2::Literal::string("not-a-tag").into(),
        "none" => {
            tokens.remove(0);
        }
        _ => return Error::new(Span::call_site(), "unknown tag").to_compile_error().into(),
    }
    let mac_name = Ident::new(&format!("__tagged_{}", item_trait.ident), Span::call_site());
    let tokens: TokenStream = tokens.into_iter().collect();
    quote! {
        #item_trait
        macro_rules! #mac_name {
            () => {
                ::type_leak_test_macro::__read_referrer! { { #tokens , #context } }
            };
        }
    }
    .into()
}

/// The Referrer's side of `leak_tagged`: parses `{ referrer , context }`, which fails if the
/// format tag of the Referrer is not the one of this `type-leak`.
#[doc(hidden)]
#[proc_macro]
pub fn __read_referrer(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    struct Input;
    impl Parse for Input {
        fn parse(input: ParseStream) -> Result<Self> {
            let content;
            braced!(content in input);
            let _: Referrer = content.parse()?;
            content.parse::<Token![,]>()?;
            let _: ContextId = content.parse()?;
            Ok(Input)
        }
    }
    match syn::parse::<Input>(input) {
        Ok(Input) => quote!(const _: () = ();).into(),
        Err(e) => e.to_compile_error().into(),
    }
}

/// Input of `check_leak!`: `{ expected } { config } item`.
struct CheckLeakInput {
    /// The leaked types (an optional entry is preceded by `?`), or the message of the error.
    expected: std::result::Result<Vec<String>, String>,
    /// The paths allowed by the config (`allow = ::foo`).
    allowed: Vec<Path>,
    item: Item,
}

/// Prints a type as `to_string()` does, after parsing, so that the spacing is normalized.
fn type_string(ty: &Type) -> String {
    quote!(#ty).to_string()
}

impl Parse for CheckLeakInput {
    fn parse(input: ParseStream) -> Result<Self> {
        let expected_tokens: TokenStream = braced_parse(input)?;
        let expected = syn::parse::Parser::parse2(
            |input: ParseStream| -> Result<std::result::Result<Vec<String>, String>> {
                if input.peek(Ident) && input.peek2(Token![:]) && !input.peek2(Token![::]) {
                    let key: Ident = input.parse()?;
                    if key != "error" {
                        return Err(Error::new(key.span(), "expected `error`"));
                    }
                    input.parse::<Token![:]>()?;
                    let lit: LitStr = input.parse()?;
                    return Ok(Err(lit.value()));
                }
                let mut types = Vec::new();
                while !input.is_empty() {
                    let optional = input.parse::<Option<Token![?]>>()?.is_some();
                    let ty: Type = input.parse()?;
                    types.push(entry_string(&ty, !optional));
                    let _: Option<Token![,]> = input.parse()?;
                }
                Ok(Ok(types))
            },
            expected_tokens,
        )?;
        let config_tokens: TokenStream = braced_parse(input)?;
        let mut allowed = Vec::new();
        syn::parse::Parser::parse2(
            |input: ParseStream| -> Result<()> {
                while !input.is_empty() {
                    let key: Ident = input.parse()?;
                    input.parse::<Token![=]>()?;
                    match key.to_string().as_str() {
                        "allow" => allowed.push(input.parse()?),
                        _ => return Err(Error::new(key.span(), "unknown flag")),
                    }
                    let _: Option<Token![,]> = input.parse()?;
                }
                Ok(())
            },
            config_tokens,
        )?;
        Ok(Self {
            expected,
            allowed,
            item: input.parse()?,
        })
    }
}

/// Prints an entry: the type, preceded by `? ` if it is optional.
fn entry_string(ty: &Type, must: bool) -> String {
    if must {
        type_string(ty)
    } else {
        format!("? {}", type_string(ty))
    }
}

/// Checks the leaked types (or the error) of the Leaker for an item, which may contain real
/// `$crate` tokens when invoked from a `macro_rules!`:
///
/// ```ignore
/// check_leak! {
///     { Rel, ? ::foo::X, Other<u8> }         // the leaked types, sorted, or `error: "message"`
///     { allow = ::bar }                      // the config (`allow_primitive` is always set)
///     trait Tr { fn f(&self) -> $crate::m::Wrap<Rel>; } // a trait, a struct or a module
/// }
/// ```
///
/// The optional entries (see `Referrer::must()`) are preceded by `?`. It expands to nothing if the
/// result is as expected, and to a `compile_error!` otherwise.
#[proc_macro]
pub fn check_leak(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let CheckLeakInput {
        expected,
        allowed,
        item,
    } = parse_macro_input!(input as CheckLeakInput);
    let mut leaker = Leaker::from_config({
        let mut config = LeakerConfig::new();
        config.allow_primitive();
        for path in allowed {
            config.allow_path(path);
        }
        config
    });
    match &item {
        Item::Trait(item) => leaker.intern_trait(item, true),
        Item::Struct(item) => leaker.intern_struct(item, true),
        Item::Mod(item) => leaker.intern_mod(item),
        _ => return quote!(compile_error!("unsupported item");).into(),
    };
    let got: std::result::Result<Vec<String>, String> = match leaker.result().finish() {
        Ok(referrer) => {
            let mut types: Vec<String> = referrer
                .iter()
                .enumerate()
                .map(|(ix, ty)| entry_string(ty, referrer.must(ix)))
                .collect();
            types.sort();
            Ok(types)
        }
        Err(e) => Err(messages(&e)),
    };
    let expected = match expected {
        Ok(mut types) => {
            types.sort();
            Ok(types)
        }
        Err(e) => Err(e),
    };
    if got == expected {
        return proc_macro::TokenStream::new();
    }
    let message = format!("expected {:?}, got {:?}", expected, got);
    quote!(compile_error!(#message);).into()
}

/// The Leaker side: `#[leak(..)]` on a trait.
#[proc_macro_attribute]
pub fn leak(
    attr: proc_macro::TokenStream,
    item: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let args = parse_macro_input!(attr as LeakArgs);
    let item_trait = parse_macro_input!(item as ItemTrait);
    match leak_impl(args, item_trait) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

fn leak_impl(args: LeakArgs, item_trait: ItemTrait) -> Result<TokenStream> {
    let mut leaker = Leaker::from_config({
        let mut config = LeakerConfig::new();
        config.allow_primitive().allow_prelude();
        if let Some(path) = args.allow.clone() {
            config.allow_path(path);
        }
        config
    });
    let context = leaker.intern_trait(&item_trait, true);
    // Compile-time checks of the Leaker's assumptions (e.g. that `u8` is not shadowed).
    let assertions = leaker.assertions();
    leaker.reduce_roots();
    let referrer = leaker
        .result()
        .finish()
        .map_err(|errors| to_syn_error(&errors))?;

    let LeakArgs {
        helper,
        repeater,
        module,
        ..
    } = &args;
    let id = trait_id(module, &item_trait.ident);
    let trait_ident = &item_trait.ident;
    let vis = &item_trait.vis;
    let slots = RepeaterSlots::new(repeater.clone());

    // The Repeater's implementation of each entry declares exactly the generics the type uses, and
    // encodes them. If the type mentions `Self`, it is implemented for the implementor of the
    // trait (bounded by the trait with its parameters) or for the type `Self` is: that is what
    // `Referrer::repeater_impl()` describes, with the consts encoded by the slots.
    let mut repeater_impls = TokenStream::new();
    let self_param: Ident = parse_quote!(__S);
    for ix in 0..referrer.iter().count() {
        let r = referrer
            .repeater_impl_with(ix, ImplSelf::Param(&self_param), |value, param| {
                slots.encode_const(value, &param.ty)
            })
            .map_err(Error::from)?;
        check_plain(&referrer, ix, &slots)?;
        let (impl_generics, _, where_clause) = r.generics.split_for_impl();
        let (args, self_ty, ty) = (&r.args, &r.self_ty, &r.ty);
        repeater_impls.extend(quote! {
            impl #impl_generics #repeater<#args, #id, #ix> for #self_ty #where_clause {
                type Type = #ty;
            }
        });
    }

    let mac_name = Ident::new(
        &format!("__type_leak_test_{}_{}", trait_ident, id),
        Span::call_site(),
    );
    let dollar = Punct::new('$', Spacing::Alone);
    let helper = dollar_crate(helper);
    let repeater = dollar_crate(repeater);
    let module = dollar_crate(module);
    Ok(quote! {
        #item_trait
        #assertions
        #repeater_impls
        #[doc(hidden)]
        #[macro_export]
        macro_rules! #mac_name {
            (#dollar(#dollar t:tt)*) => {
                #helper::__implement! {
                    { #item_trait }
                    // The scope of the copy goes with the Referrer, as an integer.
                    { #referrer , #context }
                    { #id }
                    { #repeater }
                    { #module }
                    #dollar(#dollar t)*
                }
            };
        }
        #[doc(hidden)]
        #vis use #mac_name as #trait_ident;
    })
}

/// Input of `__implement!`: the data passed by the exported `macro_rules!`, followed by
/// `impl<generics> Trait<args> for SelfTy => field`, or by `copy [SelfTy] { items }`.
enum Input {
    Implement(ImplementInput),
    Copy(CopyInput),
}

/// The data passed by the exported `macro_rules!`.
struct Header {
    item_trait: ItemTrait,
    referrer: Referrer,
    /// The top-level context of the trait: the scope of the copies.
    context: ContextId,
    id: LitInt,
    repeater: Path,
    module: Path,
}

/// `copy [SelfTy] { items }`: the items written with the types of the Leaker (even a free function
/// or a struct, which have no `Self`: it is `SelfTy`, if given) are copied with the leaked types
/// rewritten.
///
/// `show NAME [SelfTy] { items }` does not emit the copies, but `pub const NAME: &str`, which holds
/// their tokens, so that a test can tell what a Referrer wrote (e.g. a strict one from a trusting
/// one).
struct CopyInput {
    header: Header,
    /// The name of the const of `show`.
    show: Option<Ident>,
    self_ty: Option<Type>,
    items: Vec<Item>,
}

struct ImplementInput {
    item_trait: ItemTrait,
    referrer: Referrer,
    context: ContextId,
    id: LitInt,
    repeater: Path,
    module: Path,
    generics: Generics,
    trait_args: AngleBracketedGenericArguments,
    self_ty: Type,
    field: Member,
    /// Items of the implementation given verbatim: the associated types and consts of the trait,
    /// and the methods which are not forwarded.
    extra: Vec<ImplItem>,
}

fn braced_parse<T: Parse>(input: ParseStream) -> Result<T> {
    let content;
    braced!(content in input);
    content.parse()
}

impl Parse for Input {
    fn parse(input: ParseStream) -> Result<Self> {
        let item_trait = braced_parse(input)?;
        let (referrer, context) = {
            let content;
            braced!(content in input);
            let referrer = content.parse()?;
            content.parse::<Token![,]>()?;
            (referrer, content.parse()?)
        };
        let header = Header {
            item_trait,
            referrer,
            context,
            id: braced_parse(input)?,
            repeater: braced_parse::<Path>(input)?,
            module: braced_parse::<Path>(input)?,
        };
        // `strict` makes the Referrer replace the optional entries as well (it does not trust the
        // names of the crates, see `Referrer::allow_ambiguous_crate_name()`).
        let mut header = header;
        if input.peek(Ident) && input.fork().parse::<Ident>()? == "strict" {
            input.parse::<Ident>()?;
            header.referrer = header.referrer.allow_ambiguous_crate_name(false);
        }
        if input.peek(Ident) && !input.peek(Token![impl]) {
            let keyword: Ident = input.parse()?;
            if keyword != "copy" && keyword != "show" {
                return Err(Error::new(
                    keyword.span(),
                    "expected `impl`, `copy` or `show`",
                ));
            }
            let show = if keyword == "show" {
                Some(input.parse::<Ident>()?)
            } else {
                None
            };
            let content;
            bracketed!(content in input);
            let self_ty = if content.is_empty() {
                None
            } else {
                Some(content.parse()?)
            };
            let content;
            braced!(content in input);
            let mut items = Vec::new();
            while !content.is_empty() {
                items.push(content.parse()?);
            }
            return Ok(Input::Copy(CopyInput {
                header,
                show,
                self_ty,
                items,
            }));
        }
        let Header {
            item_trait,
            referrer,
            context,
            id,
            repeater,
            module,
        } = header;
        input.parse::<Token![impl]>()?;
        let mut generics: Generics = input.parse()?;
        let trait_ident: Ident = input.parse()?;
        if trait_ident != item_trait.ident {
            return Err(Error::new(
                trait_ident.span(),
                format!("expected `{}`", item_trait.ident),
            ));
        }
        // `<..>` is optional for traits without generic parameters (then it is `<>`).
        let trait_args = if input.peek(Token![<]) {
            input.parse()?
        } else {
            AngleBracketedGenericArguments {
                colon2_token: None,
                lt_token: Default::default(),
                args: Punctuated::new(),
                gt_token: Default::default(),
            }
        };
        input.parse::<Token![for]>()?;
        let self_ty = input.parse()?;
        generics.where_clause = input.parse()?;
        input.parse::<Token![=>]>()?;
        let field = input.parse()?;
        let mut extra = Vec::new();
        if input.peek(token::Brace) {
            let content;
            braced!(content in input);
            while !content.is_empty() {
                extra.push(content.parse()?);
            }
        }
        Ok(Input::Implement(ImplementInput {
            item_trait,
            referrer,
            context,
            id,
            repeater,
            module,
            generics,
            trait_args,
            self_ty,
            field,
            extra,
        }))
    }
}

/// Substitutes the trait's generic parameters with the arguments given by the Referrer.
struct Substitute {
    types: HashMap<Ident, Type>,
    consts: HashMap<Ident, Expr>,
    lifetimes: HashMap<Lifetime, Lifetime>,
}

impl VisitMut for Substitute {
    fn visit_type_mut(&mut self, i: &mut Type) {
        if let Type::Path(TypePath { qself: None, path }) = i {
            if let Some(ident) = path.get_ident() {
                if let Some(ty) = self.types.get(ident) {
                    *i = ty.clone();
                    return;
                }
            }
        }
        syn::visit_mut::visit_type_mut(self, i)
    }
    fn visit_generic_argument_mut(&mut self, i: &mut GenericArgument) {
        // A bare const parameter is parsed as a type argument, e.g. `N` in `Foo<N>`.
        if let GenericArgument::Type(Type::Path(TypePath { qself: None, path })) = i {
            if let Some(expr) = path.get_ident().and_then(|ident| self.consts.get(ident)) {
                *i = GenericArgument::Const(parse_quote!({ #expr }));
                return;
            }
        }
        syn::visit_mut::visit_generic_argument_mut(self, i)
    }
    fn visit_expr_mut(&mut self, i: &mut Expr) {
        if let Expr::Path(ExprPath {
            qself: None, path, ..
        }) = i
        {
            if let Some(expr) = path.get_ident().and_then(|ident| self.consts.get(ident)) {
                *i = parse_quote!({ #expr });
                return;
            }
        }
        syn::visit_mut::visit_expr_mut(self, i)
    }
    fn visit_lifetime_mut(&mut self, i: &mut Lifetime) {
        if let Some(lt) = self.lifetimes.get(i) {
            *i = lt.clone();
        }
    }
}

/// Replaces placeholder types `__TypeLeakTestPlaceholder<ix>` with the Repeater's projections.
struct Resolve<'a> {
    projections: &'a [Type],
}

impl VisitMut for Resolve<'_> {
    fn visit_type_mut(&mut self, i: &mut Type) {
        if let Type::Path(TypePath { qself: None, path }) = i {
            if path.segments.len() == 1 && path.segments[0].ident == "__TypeLeakTestPlaceholder" {
                if let PathArguments::AngleBracketed(args) = &path.segments[0].arguments {
                    if let Some(GenericArgument::Const(Expr::Lit(ExprLit {
                        lit: Lit::Int(ix),
                        ..
                    }))) = args.args.first()
                    {
                        *i = self.projections[ix.base10_parse::<usize>().unwrap()].clone();
                        return;
                    }
                }
            }
        }
        syn::visit_mut::visit_type_mut(self, i)
    }
}

/// Forwards the receiver of a method to the field `field`.
fn forward_receiver(receiver: &Receiver, field: &Member) -> Result<TokenStream> {
    let is_self = |ty: &Type| matches!(ty, Type::Path(TypePath { qself: None, path }) if path.is_ident("Self"));
    let ty = receiver.ty.as_ref();
    Ok(match ty {
        // `self`, `mut self`, `&self`, `&mut self` (also with a lifetime)
        _ if receiver.colon_token.is_none() => match &receiver.reference {
            Some(_) if receiver.mutability.is_some() => quote!(&mut self.#field),
            Some(_) => quote!(&self.#field),
            None => quote!(self.#field),
        },
        // `self: Self`, `self: &Self`, `self: &mut Self`
        _ if is_self(ty) => quote!(self.#field),
        Type::Reference(TypeReference {
            mutability, elem, ..
        }) if is_self(elem) => match mutability {
            Some(_) => quote!(&mut self.#field),
            None => quote!(&self.#field),
        },
        // `self: Box<Self>`
        Type::Path(TypePath { qself: None, path })
            if path.segments.last().map_or(false, |seg| {
                seg.ident == "Box"
                    && matches!(&seg.arguments, PathArguments::AngleBracketed(args)
                        if args.args.len() == 1
                            && matches!(&args.args[0], GenericArgument::Type(t) if is_self(t)))
            }) =>
        {
            quote!(::std::boxed::Box::new((*self).#field))
        }
        _ => {
            return Err(Error::new(
                ty.span(),
                "this receiver type is not supported by this test macro",
            ))
        }
    })
}

/// The Referrer side, invoked through the `macro_rules!` exported by `#[leak(..)]`.
#[doc(hidden)]
#[proc_macro]
pub fn __implement(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let result = match parse_macro_input!(input as Input) {
        Input::Implement(input) => implement_impl(input),
        Input::Copy(input) => copy_impl(input),
    };
    match result {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

/// Copies the items of `copy [SelfTy] { items }` with the leaked types rewritten, as the Referrer
/// does the signatures of the trait: `<SelfTy as Repeater<..>>::Type` (`Self` if there is no
/// `SelfTy`) in place of the leaked types, where `Self` of the leaked types is `SelfTy`. The
/// generic parameters of the trait are in scope in the items with the same names (e.g. the
/// `'a` of an `impl<'a> Trait<'a> for Wrapper`).
fn copy_impl(input: CopyInput) -> Result<TokenStream> {
    let CopyInput {
        header:
            Header {
                item_trait,
                referrer,
                context,
                id,
                repeater,
                ..
            },
        show,
        self_ty,
        items,
    } = input;
    let slots = RepeaterSlots::new(repeater.clone());
    let params = &item_trait.generics.params;
    let trait_params: Vec<GenericArgument> = params
        .iter()
        .map(|param| parse2(param_as_arg(param)))
        .collect::<Result<_>>()?;
    let self_in_copy: Type = self_ty.clone().unwrap_or_else(|| parse_quote!(Self));
    let mut error = None;
    let mut copies = TokenStream::new();
    for mut item in items {
        // `Self` is what the context says: the implementor of the trait.
        let mut visitor = referrer.clone().visitor(context, |_, ix| {
            let projection = (|| -> Result<Type> {
                let site =
                    UseSite::substituting(params, &trait_params).with_self(self_in_copy.clone());
                let encoded = referrer.entry_args_with(ix, &site, |value, param| {
                    slots.encode_const(value, &param.ty)
                })?;
                Ok(parse_quote!(<#self_in_copy as #repeater<#encoded, #id, #ix>>::Type))
            })();
            projection.unwrap_or_else(|e| {
                error = Some(e);
                parse_quote!(())
            })
        });
        if let Some(self_ty) = &self_ty {
            visitor = visitor.with_self(self_ty.clone());
        }
        // An impl of the trait declares its parameters again (`impl<'a> Trait<'a> for ..`), with
        // the bounds it needs there, which are not those of the trait: they are the trait's.
        if matches!(item, Item::Impl(_)) && !params.is_empty() {
            visitor = visitor.with_outermost_generics_as_leakers();
        }
        visitor.visit_item_mut(&mut item);
        drop(visitor);
        copies.extend(quote!(#item));
    }
    match (error, show) {
        (Some(e), _) => Err(e),
        (None, Some(name)) => {
            let text = copies.to_string();
            Ok(quote!(pub const #name: &str = #text;))
        }
        (None, None) => Ok(copies),
    }
}

fn implement_impl(input: ImplementInput) -> Result<TokenStream> {
    let ImplementInput {
        item_trait,
        referrer,
        context,
        id,
        repeater,
        module,
        generics,
        trait_args,
        self_ty,
        field,
        extra,
    } = input;
    let params = &item_trait.generics.params;
    let trait_ident = &item_trait.ident;
    let slots = RepeaterSlots::new(repeater.clone());

    // The projection to the leaked type `ix`: the Repeater is encoded with the arguments of the
    // generics the type uses, which are those of the trait or are in scope with the same names (the
    // generics of the method).
    let projection = |ix: usize,
                      root_params: &Punctuated<GenericParam, Token![,]>,
                      root_args: &[GenericArgument]|
     -> Result<Type> {
        let encoded = referrer.entry_args_with(
            ix,
            &UseSite::substituting(root_params, root_args),
            |value, param| slots.encode_const(value, &param.ty),
        )?;
        Ok(parse_quote!(<Self as #repeater<#encoded, #id, #ix>>::Type))
    };

    // Complete the arguments: an omitted argument takes the default, which is rewritten in the
    // Leaker's type context (it may be relative to the Leaker's module), and then the arguments
    // are encoded as in the Leaker.
    let mut encode_error = None;
    let completed = referrer
        .complete_args(&trait_args.args, params, |_, ix, prefix| {
            let before: Punctuated<GenericParam, Token![,]> =
                params.iter().take(prefix.len()).cloned().collect();
            projection(ix, &before, prefix).unwrap_or_else(|e| {
                encode_error = Some(e);
                parse_quote!(())
            })
        })
        .map_err(Error::from)?;
    if let Some(e) = encode_error {
        return Err(e);
    }
    // Map the trait's generic parameters to the complete arguments.
    let mut substitute = Substitute {
        types: HashMap::new(),
        consts: HashMap::new(),
        lifetimes: HashMap::new(),
    };
    for (param, arg) in params.iter().zip(&completed) {
        match (param, arg) {
            (GenericParam::Lifetime(param), GenericArgument::Lifetime(lt)) => {
                substitute.lifetimes.insert(param.lifetime.clone(), lt.clone());
            }
            (GenericParam::Type(param), GenericArgument::Type(ty)) => {
                substitute.types.insert(param.ident.clone(), ty.clone());
            }
            (GenericParam::Const(param), GenericArgument::Const(expr)) => {
                substitute.consts.insert(param.ident.clone(), expr.clone());
            }
            _ => return Err(Error::new(arg.span(), "unexpected generic argument")),
        }
    }

    // Projections to the leaked types. The generics of a method are in scope with the same names.
    let mut projections = Vec::new();
    for ix in 0..referrer.iter().count() {
        projections.push(projection(ix, params, &completed)?);
    }

    let mut methods = TokenStream::new();
    for trait_item in &item_trait.items {
        let trait_fn = match trait_item {
            TraitItem::Fn(trait_fn) => trait_fn,
            // Given verbatim in `extra`.
            TraitItem::Type(_) | TraitItem::Const(_) => continue,
            _ => {
                return Err(Error::new(
                    trait_item.span(),
                    "only methods, types and consts are supported by this test macro",
                ))
            }
        };
        // A method given verbatim is not forwarded.
        if extra.iter().any(|item| matches!(item, ImplItem::Fn(f) if f.sig.ident == trait_fn.sig.ident)) {
            continue;
        }
        // 1. Replace the leaked types with placeholders, 2. substitute the trait's generic
        // parameters, and 3. replace the placeholders with the projections. Substituting before
        // the replacement would break the lookup of leaked types (keyed by the parameters'
        // names), and substituting after it would rewrite the projections.
        let mut sig = trait_fn.sig.clone();
        // The context of the trait says what `Self` is (the implementor) and which generics are in
        // scope.
        let mut visitor = referrer
            .clone()
            .visitor(context, |_, ix| parse_quote!(__TypeLeakTestPlaceholder<#ix>));
        visitor.visit_signature_mut(&mut sig);
        substitute.visit_signature_mut(&mut sig);
        Resolve {
            projections: &projections,
        }
        .visit_signature_mut(&mut sig);

        // Forward to the field: `Trait::<args>::method::<generics>(&self.field, args..)`.
        let receiver = match sig.receiver() {
            Some(receiver) => forward_receiver(receiver, &field)?,
            None => {
                return Err(Error::new(
                    sig.span(),
                    "methods without a receiver are not supported",
                ))
            }
        };
        // The arguments may be patterns (e.g. `_` or `mut x`), so they are renamed.
        let mut arg_pats = Vec::new();
        for (n, input) in sig.inputs.iter_mut().enumerate() {
            if let FnArg::Typed(PatType { pat, .. }) = input {
                let ident = Ident::new(&format!("__arg{}", n), Span::call_site());
                *pat = Box::new(parse_quote!(#ident));
                arg_pats.push(ident);
            }
        }
        let turbofish: Vec<_> = sig
            .generics
            .params
            .iter()
            .filter(|p| !matches!(p, GenericParam::Lifetime(_)))
            .map(param_as_arg)
            .collect();
        let method = &sig.ident;
        methods.extend(quote! {
            #sig {
                #module::#trait_ident::#trait_args::#method::<#(#turbofish),*>(
                    #receiver, #(#arg_pats),*
                )
            }
        });
    }

    let (impl_generics, _, where_clause) = generics.split_for_impl();
    Ok(quote! {
        impl #impl_generics #module::#trait_ident #trait_args for #self_ty #where_clause {
            #(#extra)*
            #methods
        }
    })
}

/// The Leaker's side of a fragment of an impl, analyzed with the scope API (`Leaker::enter_impl()`),
/// in the style of `coinduction`: see `scope.rs`.
#[proc_macro]
pub fn scope_leak(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    match scope::leak(input.into()) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

/// Like `scope_leak!`, for an analysis which fails: emits the errors (`Vec<ErrorItem>`) as a const,
/// see `scope.rs`.
#[proc_macro]
pub fn scope_errors(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    match scope::errors(input.into()) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

/// The Referrer's side of `scope_leak!`, invoked through the `macro_rules!` that it exports.
#[doc(hidden)]
#[proc_macro]
pub fn __scope_apply(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    match scope::apply(input.into()) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

/// Analyzes types in an impl nested in the scope of a trait, and exports the macro that copies them
/// in another crate: see `nested.rs`.
#[proc_macro]
pub fn nested_leak(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    match nested::leak(input.into()) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

/// The Referrer's side of `nested_leak!`, invoked through the `macro_rules!` that it exports.
#[doc(hidden)]
#[proc_macro]
pub fn __nested_apply(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    match nested::apply(input.into()) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

/// Analyzes several items with one Leaker, and exports the macro that copies each of them alone in
/// another crate: see `shared.rs`.
#[proc_macro]
pub fn shared_leak(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    match shared::leak(input.into()) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

/// The Referrer's side of `shared_leak!`, invoked through the `macro_rules!` that it exports.
#[doc(hidden)]
#[proc_macro]
pub fn __shared_apply(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    match shared::apply(input.into()) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

/// The Leaker's side of `tests/prop_cross_crate.rs`: analyzes the struct, enum, union, trait, impl
/// or module it is put on (`#[prop_leak(id = 3, allow_self = true, prelude = false, allow = "::std")]`),
/// and emits the Repeater's implementations, the witnesses of its type positions, and the exported
/// macros `__tl_prop_<id>!` (for the Referrer's crate) and `__tl_report_<id>!`. See `prop.rs`.
#[proc_macro_attribute]
pub fn prop_leak(
    attr: proc_macro::TokenStream,
    item: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    match prop::leak(attr.into(), item.into(), prop::Mode::Plain) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

/// Like `prop_leak`, but an item that the Leaker rejects is emitted with its recovered Referrer as
/// an accepted item, if `LeakerResult::allow(cause)` for the cause of every error drops every
/// error: `TYPE_LEAK_PROP_ACCEPT`. See `prop::leak()`.
#[proc_macro_attribute]
pub fn prop_leak_accept(
    attr: proc_macro::TokenStream,
    item: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    match prop::leak(attr.into(), item.into(), prop::Mode::Accept) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

/// Like `prop_leak_accept`, but the Leaker leaves the errors to the Referrer: an item that has some
/// is emitted as an accepted item whose macro carries the `LeakerResult` (the Referrer, then the
/// errors, as tokens). `__prop_apply!` drops the errors that its invocation allows
/// (`allow(prelude, std_crates, trait Path, type Ty)`), reports the others with
/// `ErrorItem::to_compile_error()`, and copies the item with the recovered Referrer. See `prop.rs`.
#[proc_macro_attribute]
pub fn prop_leak_result(
    attr: proc_macro::TokenStream,
    item: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    match prop::leak(attr.into(), item.into(), prop::Mode::Result) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

/// Like `prop_leak`, but leaves the item as it is: it builds the Leaker's crate without
/// `type-leak`, to tell a bug of the generator from a bug of `type-leak`.
#[proc_macro_attribute]
pub fn prop_leak_noop(
    _attr: proc_macro::TokenStream,
    item: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    item
}

/// The Referrer's side of `tests/prop_cross_crate.rs`, invoked through the `macro_rules!` that
/// `prop_leak` exports.
#[doc(hidden)]
#[proc_macro]
pub fn __prop_apply(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    match prop::apply(input.into()) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}
