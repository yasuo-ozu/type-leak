//! The check of the values of an item, for [`LeakerConfig::check_values()`]: the default of an
//! associated const, the value of a const or a static, and the body of a function.
//!
//! The copy writes a value as it is, since an expression cannot go through the Repeater. So the
//! check rejects every path of the value that depends on where it is written, with the error of a
//! relative path. It must not reject what is not relative, so it follows the scopes of a body: a
//! `let` binding, the parameter of a function or a closure, the names of a pattern, and the items
//! and `use` declarations of a block are local, and so are the generics and `Self` in scope.
//!
//! A type is not an expression: the Referrer's visitor rewrites it where it stands, through the
//! Repeater. So a type in a *type position* of a value is analyzed as any type of the item is,
//! with the generics and `Self` in scope, and it becomes an entry if it names a relative path (or
//! an optional one, for a path of another crate). The type positions are the ones the visitor
//! walks as types: the target of an `as` cast, the type of a `let` and of the parameters and the
//! output of a closure, the generic arguments of a path and of a method call (the turbofish), and
//! the self type of a qualified path (`<Ty as Trait>::f`, not the trait). A bare identifier there
//! (`size_of::<Local>()`) is read as a type, as in an enum discriminant.
//!
//! What stays rejected, as a relative path:
//!
//! - a path that is not a type position: a function, a constant, a struct literal or an enum
//!   variant (`inner::f()`, `Local::new()`, `Local { .. }`), the trait of a qualified path;
//! - a type that names something the value declares (an item, a `use`, a generic or a lifetime of
//!   an item in a block): the copy writes the value with its declaration, so it cannot be leaked,
//!   and a relative path beside it is rejected as in any other place of the value;
//! - a type in the arguments of a macro, which the visitor does not rewrite, and a type in an item
//!   that a block declares.
//!
//! What the check cannot see it rejects, because it would otherwise accept a path it never read:
//! the arguments of a macro which are not expressions, a `macro_rules!`, a glob import, and syntax
//! that `syn` leaves unparsed.

use super::{self_segment, without_arguments, AnalyzeVisitor};
use crate::analysis::{CheckResult, Effect, Part, Place};
#[cfg(doc)]
use crate::config::LeakerConfig;
use crate::error::{Cause, Reason};
use crate::util::{ident_name, is_primitive_name, tidy, token_names};
use proc_macro2::TokenStream;
use std::collections::HashSet;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use syn::parse::{ParseStream, Parser};
use syn::punctuated::Punctuated;
use syn::visit::Visit;
use syn::*;
use template_quote::{quote, ToTokens};

impl AnalyzeVisitor {
    /// Checks the value `expr` of the const or static `name`, if values are read. `default`: it is
    /// the default of an associated const of a trait.
    pub(super) fn check_value(&mut self, name: &Ident, default: bool, expr: &Expr) {
        if !self.check_values {
            return;
        }
        let what = format!(
            "the {}value of `{}`",
            if default { "default " } else { "" },
            name
        );
        let outer = self.reading(default);
        ValueWalk::new(self, what).visit_expr(expr);
        self.reading_done(outer);
    }

    /// Checks the body `block` of the function `sig`, if values are read. `default`: it is the
    /// default body of a method of a trait. The generics of `sig` are in scope.
    pub(super) fn check_body(&mut self, sig: &Signature, block: &Block, default: bool) {
        if !self.check_values {
            return;
        }
        let what = format!(
            "the {}body of `{}`",
            if default { "default " } else { "" },
            sig.ident
        );
        let outer = self.reading(default);
        let mut walk = ValueWalk::new(self, what);
        for input in &sig.inputs {
            if let FnArg::Typed(arg) = input {
                walk.visit_pat(&arg.pat);
            }
        }
        walk.visit_block(block);
        self.reading_done(outer);
    }

    /// Starts reading a value, the default of a trait item if `default`: the errors found until
    /// [`AnalyzeVisitor::reading_done()`] are in a value (and in a default). Returns what to put
    /// back.
    fn reading(&mut self, default: bool) -> (bool, bool) {
        (
            std::mem::replace(&mut self.in_value, true),
            std::mem::replace(&mut self.in_trait_default, default),
        )
    }

    /// Ends the reading that [`AnalyzeVisitor::reading()`] started.
    fn reading_done(&mut self, (value, default): (bool, bool)) {
        self.in_value = value;
        self.in_trait_default = default;
    }
}

/// The names that a scope of a value declares.
#[derive(Default)]
struct Names {
    /// The bindings of patterns: a `let`, a parameter, a match arm, a `for`.
    bindings: HashSet<String>,
    /// The items, the `use` declarations and the generics of items: named wherever they are seen.
    items: HashSet<String>,
}

struct ValueWalk<'a> {
    visitor: &'a mut AnalyzeVisitor,
    /// Where the walk is, as the notes say it: "the default value of `N`".
    what: String,
    /// Never empty. The first is the scope of the value itself.
    scopes: Vec<Names>,
    /// The first scope whose bindings are seen. An item in a block sees the items around it, but not
    /// the bindings.
    barrier: usize,
    /// Greater than 0 where the walk reads a type itself instead of leaking it: in an item that a
    /// block declares, in the arguments of a macro (the visitor does not rewrite tokens), and in
    /// a type that names something the value declares.
    reads: usize,
    /// While such a type is read: the name it has that the value declares.
    declared_name: Option<String>,
}

impl<'a> ValueWalk<'a> {
    fn new(visitor: &'a mut AnalyzeVisitor, what: String) -> Self {
        Self {
            visitor,
            what,
            scopes: vec![Names::default()],
            barrier: 0,
            reads: 0,
            declared_name: None,
        }
    }

    fn scope(&mut self) -> &mut Names {
        self.scopes.last_mut().expect("a scope")
    }

    fn bind(&mut self, ident: &Ident) {
        self.scope().bindings.insert(ident_name(ident));
    }

    fn declare(&mut self, ident: &Ident) {
        self.scope().items.insert(ident_name(ident));
    }

    /// Declares a name that is not an identifier: a lifetime (`'a`).
    fn declare_name(&mut self, name: String) {
        self.scope().items.insert(name);
    }

    /// Returns `true` if the value declares `name` (an item, a generic, a lifetime or a binding),
    /// wherever it appears: in a path, a lifetime or the tokens of a macro.
    fn declares(&self, name: &str) -> bool {
        self.scopes.iter().any(|scope| scope.items.contains(name))
            || self.scopes[self.barrier..]
                .iter()
                .any(|scope| scope.bindings.contains(name))
    }

    /// Runs `f` in a new scope.
    fn scoped(&mut self, f: impl FnOnce(&mut Self)) {
        self.scopes.push(Names::default());
        f(self);
        self.scopes.pop();
    }

    /// Returns `true` if the path starts with a name that the value declares: an item (a path of
    /// any length), or a binding (a single name).
    fn is_local(&self, path: &Path) -> bool {
        let first = match path
            .segments
            .first()
            .filter(|_| path.leading_colon.is_none())
        {
            Some(first) => first,
            None => return false,
        };
        let name = ident_name(&first.ident);
        self.scopes.iter().any(|scope| scope.items.contains(&name))
            || (path.segments.len() == 1
                && first.arguments.is_none()
                && self.scopes[self.barrier..]
                    .iter()
                    .any(|scope| scope.bindings.contains(&name)))
    }

    /// Returns `true` if the path means the same wherever the value is written: it starts with
    /// `Self`, a generic, a name the value declares, or is absolute or allowed.
    fn is_known(&mut self, path: &Path) -> bool {
        // The receiver.
        if path.leading_colon.is_none() && path.get_ident().map_or(false, |ident| ident == "self") {
            return true;
        }
        if self_segment(path).is_some() {
            self.visitor.check_self_has_meaning(Cause::expr_path(path));
            return true;
        }
        if self.is_local(path) || self.visitor.starts_with_generic(path) {
            return true;
        }
        if !self.visitor.is_context_free(path) {
            return false;
        }
        // A primitive name is copied as it is, and may be shadowed: `Leaker::assertions()`.
        if path.leading_colon.is_none() {
            if let Some(seg) = path
                .segments
                .first()
                .filter(|seg| is_primitive_name(&seg.ident))
            {
                self.visitor
                    .frame
                    .effects
                    .push(Effect::Primitive(seg.ident.clone()));
            }
        }
        true
    }

    fn error(&mut self, error: CheckResult) {
        self.visitor.push_error(error);
    }

    /// Rejects the relative path `cause`, with `help` for what to write if it has to say more than
    /// the absolute path.
    fn relative(&mut self, cause: Cause, help: Option<String>) {
        self.relative_part(cause, help, None)
    }

    /// Rejects the relative path `path` of a value, copied as written with the types in its
    /// arguments (see [`Part`]).
    fn relative_path(&mut self, path: &Path) {
        let cause = Cause::expr_path(path);
        let part = self.visitor.node_part(cause.clone(), path);
        self.relative_part(cause, None, Some(part));
        // What is wrong in the arguments of the path if it were allowed: they are written as they
        // are with it, and the walk reads them again.
        if let Some(mut conflict) = self.visitor.pop_error() {
            if let Some(probe) = self.visitor.begin_probe(&conflict) {
                self.visit_path(path);
                conflict.nested = self.visitor.end_probe(probe);
            }
            self.visitor
                .frame
                .effects
                .push(Effect::Error(Box::new(conflict)));
        }
    }

    /// Like `relative()`, for a node that the Referrer leaves as written: `part`.
    fn relative_part(&mut self, cause: Cause, help: Option<String>, part: Option<Part>) {
        // The error names a macro in its own way, with its `!`, whereas the path of a macro in a
        // type is shown by its path alone.
        let headline = match &cause {
            Cause::Macro(_) => Some(format!("use absolute path: '{}'", cause.to_token_stream())),
            _ => None,
        };
        let mut note = format!("it is in {}, which the copy writes as it is", self.what);
        if let Some(name) = &self.declared_name {
            note.push_str(&format!(
                "; the type names `{}`, which the value declares, so it is not leaked",
                name
            ));
        }
        // The path is relative: the walk knows no generic, no `Self` and no name of the value by it.
        let paths = cause.as_path().map(without_arguments).into_iter().collect();
        self.error(CheckResult {
            must: vec![cause.clone()],
            impossible: vec![cause],
            is_own: true,
            optional: false,
            note: Some(note),
            help,
            headline,
            cause: None,
            reason: Reason::RelativePath,
            seen: 0,
            part,
            place: Place::default(),
            paths,
            nested: Vec::new(),
        });
    }

    /// Rejects the syntax `tokens`, which the walk cannot read: `why` is what it is.
    fn cannot_read(&mut self, tokens: TokenStream, why: &str) {
        let shown = tidy(&tokens);
        let note = format!(
            "{}. The Leaker must see every path in {}, which the copy writes as it is",
            why, self.what
        );
        self.error(CheckResult {
            headline: Some(format!("cannot analyze `{}` in {}", shown, self.what)),
            cause: None,
            reason: Reason::Unreadable,
            seen: 0,
            part: None,
            place: Place::default(),
            paths: Vec::new(),
            nested: Vec::new(),
            must: vec![Cause::Unreadable(tokens.clone())],
            impossible: vec![Cause::Unreadable(tokens)],
            is_own: true,
            optional: false,
            note: Some(note),
            help: Some(
                "write it with syntax the Leaker can read (paths and expressions), or move it \
                 out of the item"
                    .to_string(),
            ),
        });
    }

    /// Declares the names that `item` introduces in the block around it, where they are visible
    /// from the first statement.
    fn declare_item(&mut self, item: &Item) {
        match item {
            Item::Const(i) => self.declare(&i.ident),
            Item::Enum(i) => self.declare(&i.ident),
            Item::ExternCrate(i) => match &i.rename {
                Some((_, rename)) => self.declare(rename),
                None => self.declare(&i.ident),
            },
            Item::Fn(i) => self.declare(&i.sig.ident),
            Item::Macro(i) => {
                if let Some(ident) = &i.ident {
                    self.declare(ident);
                }
            }
            Item::Mod(i) => self.declare(&i.ident),
            Item::Static(i) => self.declare(&i.ident),
            Item::Struct(i) => self.declare(&i.ident),
            Item::Trait(i) => self.declare(&i.ident),
            Item::TraitAlias(i) => self.declare(&i.ident),
            Item::Type(i) => self.declare(&i.ident),
            Item::Union(i) => self.declare(&i.ident),
            Item::Use(i) => {
                let mut names = Vec::new();
                use_names(&i.tree, None, &mut names);
                for name in names {
                    self.scope().items.insert(name);
                }
            }
            _ => (),
        }
    }

    /// Handles a type in a type position of the value: it is leaked as any type of the item is,
    /// unless it names something the value declares or holds syntax that `syn` leaves unparsed.
    /// Then it is read here, and a relative path in it is rejected.
    fn type_position(&mut self, ty: &Type) {
        let mut scan = Scan {
            walk: self,
            declared: None,
            unreadable: false,
        };
        scan.visit_type(ty);
        let (declared, unreadable) = (scan.declared, scan.unreadable);
        // A value inside a type (the length of an array) is part of that type, which is leaked
        // whole or not at all: its types are not entries of their own.
        let inside_type = self.visitor.type_depth > 0;
        if declared.is_none() && !unreadable && !inside_type {
            return self.leak(ty);
        }
        let outer = std::mem::replace(&mut self.declared_name, declared);
        self.reads += 1;
        self.visit_type(ty);
        self.reads -= 1;
        self.declared_name = outer;
    }

    /// Analyzes `ty` as a type of the item, where the visitor is.
    ///
    /// The signature around does not apply to a type in its body, whatever lifetime it names: the
    /// types of the body are no input of the signature, so they do not decide which of its
    /// lifetimes the copy leaves unconstrained (E0195).
    fn leak(&mut self, ty: &Type) {
        let signatures = std::mem::take(&mut self.visitor.signatures);
        let outcome = catch_unwind(AssertUnwindSafe(|| self.visitor.visit_type(ty)));
        self.visitor.signatures = signatures;
        if let Err(payload) = outcome {
            resume_unwind(payload);
        }
    }

    /// Checks a path that is a name alone, as a `use` root, a macro argument or a pattern.
    fn check_name(&mut self, ident: &Ident) {
        let path = Path::from(ident.clone());
        if !self.is_known(&path) {
            self.relative(Cause::expr_path(&path), None);
        }
    }

    /// Checks the path `<ty as Trait>::A` or `<ty>::A`: the trait and the arguments, and the type.
    fn qualified<'ast>(&mut self, qself: Option<&'ast QSelf>, path: &'ast Path) {
        let qself = match qself {
            Some(qself) => qself,
            None => return self.visit_path(path),
        };
        self.visit_type(&qself.ty);
        // The first `position` segments are the trait, the others are names of the type's items.
        let position = if qself.as_token.is_some() {
            qself.position
        } else {
            0
        };
        if position > 0 {
            let trait_path = Path {
                leading_colon: path.leading_colon,
                segments: path.segments.iter().take(position).cloned().collect(),
            };
            if self.is_known(&trait_path) {
                for seg in path.segments.iter().take(position) {
                    self.visit_path_arguments(&seg.arguments);
                }
            } else {
                self.relative(Cause::Trait(trait_path), None);
            }
        }
        for seg in path.segments.iter().skip(position) {
            self.visit_path_arguments(&seg.arguments);
        }
    }

    /// Checks the root of a `use` tree, which names a crate, a module or an item of the value.
    fn use_roots(&mut self, tree: &UseTree, absolute: bool) {
        let ident = match tree {
            UseTree::Path(p) => &p.ident,
            UseTree::Name(n) => &n.ident,
            UseTree::Rename(r) => &r.ident,
            UseTree::Glob(_) => return,
            UseTree::Group(g) => {
                for tree in &g.items {
                    self.use_roots(tree, absolute);
                }
                return;
            }
        };
        let path = Path::from(ident.clone());
        if !absolute && !self.is_local(&path) && !self.visitor.is_context_free(&path) {
            self.relative(Cause::expr_path(&path), None);
        }
    }

    /// Checks a macro call: the path, then the arguments, which are read as expressions. A type in
    /// them is read, not leaked: the visitor does not rewrite the tokens of a macro.
    fn macro_call(&mut self, mac: &Macro) {
        let path = &mac.path;
        if !self.is_known(path) {
            let help = macro_help(path);
            return self.relative(Cause::Macro(path.clone()), Some(help));
        }
        self.reads += 1;
        self.macro_arguments_of(mac);
        self.reads -= 1;
    }

    fn macro_arguments_of(&mut self, mac: &Macro) {
        let path = &mac.path;
        let tokens = &mac.tokens;
        // These name no item in their arguments (`cfg!(unix)`, `stringify!(a b)`).
        if is_std_macro(path, &["cfg", "stringify"]) {
            return;
        }
        if is_std_macro(path, &["matches"]) {
            if let Ok((expr, pat, guard)) = parse_matches.parse2(tokens.clone()) {
                self.visit_expr(&expr);
                self.scoped(|walk| {
                    walk.visit_pat(&pat);
                    if let Some(guard) = &guard {
                        walk.visit_expr(guard);
                    }
                });
                return;
            }
        }
        match macro_exprs(tokens) {
            Some(args) => self.macro_arguments(&args, format_string(path)),
            None => self.cannot_read(
                quote!(#path ! (#tokens)),
                "the arguments of the macro are not expressions",
            ),
        }
    }

    /// Checks the arguments of a macro. A name in `name = value` is the name of an argument, which
    /// a format string captures as `{name}`, as it captures a variable that is in scope.
    /// `format` says which string literal is a format string.
    fn macro_arguments(&mut self, args: &[Expr], format: FormatString) {
        let named: Vec<String> = args.iter().filter_map(argument_name).collect();
        for (at, arg) in args.iter().enumerate() {
            match arg {
                Expr::Assign(assign) if argument_name(arg).is_some() => {
                    self.visit_expr(&assign.right)
                }
                Expr::Lit(ExprLit {
                    lit: Lit::Str(literal),
                    ..
                }) if format.is_at(at) => {
                    for name in captured_names(&literal.value()) {
                        if named.contains(&name) {
                            continue;
                        }
                        self.check_name(&Ident::new(&name, literal.span()));
                    }
                }
                _ => self.visit_expr(arg),
            }
        }
    }
}

impl<'ast> Visit<'ast> for ValueWalk<'_> {
    fn visit_attribute(&mut self, _: &'ast Attribute) {}

    fn visit_visibility(&mut self, _: &'ast Visibility) {}

    fn visit_path(&mut self, i: &'ast Path) {
        if self.is_known(i) {
            syn::visit::visit_path(self, i)
        } else {
            self.relative_path(i)
        }
    }

    fn visit_expr_path(&mut self, i: &'ast ExprPath) {
        self.qualified(i.qself.as_ref(), &i.path)
    }

    fn visit_type_path(&mut self, i: &'ast TypePath) {
        self.qualified(i.qself.as_ref(), &i.path)
    }

    fn visit_macro(&mut self, i: &'ast Macro) {
        self.macro_call(i)
    }

    fn visit_expr(&mut self, i: &'ast Expr) {
        match i {
            Expr::Verbatim(tokens) => self.cannot_read(tokens.clone(), "syn does not parse it"),
            _ => syn::visit::visit_expr(self, i),
        }
    }

    fn visit_type(&mut self, i: &'ast Type) {
        if self.reads == 0 {
            return self.type_position(i);
        }
        match i {
            Type::Verbatim(tokens) => self.cannot_read(tokens.clone(), "syn does not parse it"),
            _ => syn::visit::visit_type(self, i),
        }
    }

    fn visit_type_param_bound(&mut self, i: &'ast TypeParamBound) {
        match i {
            TypeParamBound::Verbatim(tokens) => {
                self.cannot_read(tokens.clone(), "syn does not parse it")
            }
            _ => syn::visit::visit_type_param_bound(self, i),
        }
    }

    fn visit_pat(&mut self, i: &'ast Pat) {
        match i {
            Pat::Verbatim(tokens) => self.cannot_read(tokens.clone(), "syn does not parse it"),
            _ => syn::visit::visit_pat(self, i),
        }
    }

    /// A name in a pattern is a binding, unless it is a path to a constant, a unit struct or a
    /// variant. Those are written with a capital letter (the lint `non_snake_case` makes a binding
    /// with one a warning), and a `ref` or `mut` name or one with `@` is a binding for certain.
    fn visit_pat_ident(&mut self, i: &'ast PatIdent) {
        let name = ident_name(&i.ident);
        let capital = name.chars().next().map_or(false, char::is_uppercase);
        let binding =
            i.by_ref.is_some() || i.mutability.is_some() || i.subpat.is_some() || !capital;
        if binding {
            self.bind(&i.ident);
        } else {
            self.check_name(&i.ident);
        }
        if let Some((_, pat)) = &i.subpat {
            self.visit_pat(pat);
        }
    }

    fn visit_generics(&mut self, i: &'ast Generics) {
        for param in &i.params {
            match param {
                GenericParam::Type(param) => self.declare(&param.ident),
                GenericParam::Const(param) => self.declare(&param.ident),
                GenericParam::Lifetime(param) => self.declare_name(param.lifetime.to_string()),
            }
        }
        syn::visit::visit_generics(self, i)
    }

    fn visit_block(&mut self, i: &'ast Block) {
        self.scoped(|walk| {
            // An item is visible in the whole block.
            for stmt in &i.stmts {
                if let Stmt::Item(item) = stmt {
                    walk.declare_item(item);
                }
            }
            for stmt in &i.stmts {
                walk.visit_stmt(stmt);
            }
        })
    }

    /// The initializer does not see the bindings of the pattern.
    fn visit_local(&mut self, i: &'ast Local) {
        if let Some(init) = &i.init {
            self.visit_expr(&init.expr);
            if let Some((_, diverge)) = &init.diverge {
                self.visit_expr(diverge);
            }
        }
        self.visit_pat(&i.pat);
    }

    fn visit_item(&mut self, i: &'ast Item) {
        // The item sees the items around it, and the generics of its own, not the bindings.
        let outer = std::mem::replace(&mut self.barrier, self.scopes.len());
        // The types of an item that a block declares are read, not leaked.
        self.reads += 1;
        self.scoped(|walk| match i {
            Item::Use(item) => {
                if has_glob(&item.tree) {
                    walk.cannot_read(
                        quote!(#item),
                        "a glob import hides which names it brings into scope",
                    );
                }
                walk.use_roots(&item.tree, item.leading_colon.is_some());
            }
            Item::Macro(item) => walk.cannot_read(
                quote!(#item),
                "a macro in item position may declare any name",
            ),
            Item::Verbatim(tokens) => walk.cannot_read(tokens.clone(), "syn does not parse it"),
            _ => syn::visit::visit_item(walk, i),
        });
        self.reads -= 1;
        self.barrier = outer;
    }

    fn visit_item_mod(&mut self, i: &'ast ItemMod) {
        if let Some((_, items)) = &i.content {
            for item in items {
                self.declare_item(item);
            }
        }
        syn::visit::visit_item_mod(self, i)
    }

    fn visit_impl_item(&mut self, i: &'ast ImplItem) {
        self.scoped(|walk| syn::visit::visit_impl_item(walk, i))
    }

    fn visit_trait_item(&mut self, i: &'ast TraitItem) {
        self.scoped(|walk| syn::visit::visit_trait_item(walk, i))
    }

    fn visit_expr_closure(&mut self, i: &'ast ExprClosure) {
        self.scoped(|walk| {
            // `for<'a> |x: &'a u8| ..` declares a lifetime that only the closure knows.
            for param in i.lifetimes.iter().flat_map(|binder| &binder.lifetimes) {
                if let GenericParam::Lifetime(param) = param {
                    walk.declare_name(param.lifetime.to_string());
                }
            }
            for input in &i.inputs {
                walk.visit_pat(input);
            }
            walk.visit_return_type(&i.output);
            walk.visit_expr(&i.body);
        })
    }

    /// The bindings of the pattern are those of the guard and the body.
    fn visit_arm(&mut self, i: &'ast Arm) {
        self.scoped(|walk| {
            walk.visit_pat(&i.pat);
            if let Some((_, guard)) = &i.guard {
                walk.visit_expr(guard);
            }
            walk.visit_expr(&i.body);
        })
    }

    /// The bindings of `if let` are those of the first branch.
    fn visit_expr_if(&mut self, i: &'ast ExprIf) {
        self.scoped(|walk| {
            walk.visit_expr(&i.cond);
            walk.visit_block(&i.then_branch);
        });
        if let Some((_, branch)) = &i.else_branch {
            self.visit_expr(branch);
        }
    }

    fn visit_expr_while(&mut self, i: &'ast ExprWhile) {
        self.scoped(|walk| {
            walk.visit_expr(&i.cond);
            walk.visit_block(&i.body);
        })
    }

    fn visit_expr_let(&mut self, i: &'ast ExprLet) {
        self.visit_expr(&i.expr);
        self.visit_pat(&i.pat);
    }

    fn visit_expr_for_loop(&mut self, i: &'ast ExprForLoop) {
        self.visit_expr(&i.expr);
        self.scoped(|walk| {
            walk.visit_pat(&i.pat);
            walk.visit_block(&i.body);
        })
    }
}

/// What a type of a value names that decides how it is read: a name the value declares, and syntax
/// that `syn` leaves unparsed.
struct Scan<'w, 'a> {
    walk: &'w ValueWalk<'a>,
    /// The first name that the value declares (an item, a generic or a lifetime of an item in a
    /// block, a `use`).
    declared: Option<String>,
    unreadable: bool,
}

impl Scan<'_, '_> {
    fn name(&mut self, name: &str) {
        if self.declared.is_none() && self.walk.declares(name) {
            self.declared = Some(name.to_string());
        }
    }
}

impl<'ast> Visit<'ast> for Scan<'_, '_> {
    fn visit_path(&mut self, i: &'ast Path) {
        if self.declared.is_none() && self.walk.is_local(i) {
            self.declared = i.segments.first().map(|seg| ident_name(&seg.ident));
        }
        syn::visit::visit_path(self, i)
    }

    fn visit_lifetime(&mut self, i: &'ast Lifetime) {
        self.name(&i.to_string());
    }

    /// A type macro, or a macro in a length: the names in its tokens are not told apart.
    fn visit_macro(&mut self, i: &'ast Macro) {
        let mut names = HashSet::new();
        token_names(i.tokens.clone(), &mut names);
        let mut names: Vec<String> = names.into_iter().collect();
        names.sort();
        for name in names {
            self.name(&name);
        }
        syn::visit::visit_macro(self, i)
    }

    fn visit_type(&mut self, i: &'ast Type) {
        match i {
            Type::Verbatim(_) => self.unreadable = true,
            _ => syn::visit::visit_type(self, i),
        }
    }

    fn visit_type_param_bound(&mut self, i: &'ast TypeParamBound) {
        match i {
            TypeParamBound::Verbatim(_) => self.unreadable = true,
            _ => syn::visit::visit_type_param_bound(self, i),
        }
    }

    fn visit_expr(&mut self, i: &'ast Expr) {
        match i {
            Expr::Verbatim(_) => self.unreadable = true,
            _ => syn::visit::visit_expr(self, i),
        }
    }

    fn visit_pat(&mut self, i: &'ast Pat) {
        match i {
            Pat::Verbatim(_) => self.unreadable = true,
            _ => syn::visit::visit_pat(self, i),
        }
    }

    /// An item in the length of an array declares names this scan does not follow.
    fn visit_item(&mut self, _: &'ast Item) {
        self.unreadable = true;
    }
}

/// The names that a `use` tree brings into scope. `parent`: the name before the group
/// (`use a::b::{self}` brings `b`).
fn use_names(tree: &UseTree, parent: Option<&Ident>, names: &mut Vec<String>) {
    match tree {
        UseTree::Path(p) => use_names(&p.tree, Some(&p.ident), names),
        UseTree::Name(n) if n.ident == "self" => names.extend(parent.map(ident_name)),
        UseTree::Name(n) => names.push(ident_name(&n.ident)),
        UseTree::Rename(r) => names.push(ident_name(&r.rename)),
        UseTree::Glob(_) => (),
        UseTree::Group(g) => {
            for tree in &g.items {
                use_names(tree, parent, names);
            }
        }
    }
}

fn has_glob(tree: &UseTree) -> bool {
    match tree {
        UseTree::Path(p) => has_glob(&p.tree),
        UseTree::Glob(_) => true,
        UseTree::Group(g) => g.items.iter().any(has_glob),
        UseTree::Name(_) | UseTree::Rename(_) => false,
    }
}

/// The arguments of a macro call as expressions: `a, b` or `a; n` (`vec![0; n]`).
fn macro_exprs(tokens: &TokenStream) -> Option<Vec<Expr>> {
    if let Ok(args) = Punctuated::<Expr, Token![,]>::parse_terminated.parse2(tokens.clone()) {
        return Some(args.into_iter().collect());
    }
    let repeat = |input: ParseStream| -> Result<Vec<Expr>> {
        let value: Expr = input.parse()?;
        input.parse::<Token![;]>()?;
        let count: Expr = input.parse()?;
        Ok(vec![value, count])
    };
    repeat.parse2(tokens.clone()).ok()
}

/// Returns `true` if `path` is the macro of the standard library `names`: `cfg!` or
/// `::core::cfg!`. (Another path ending in the same name is a macro of its own.)
fn is_std_macro(path: &Path, names: &[&str]) -> bool {
    let name = match path.segments.last() {
        Some(seg) => seg.ident.to_string(),
        None => return false,
    };
    names.contains(&name.as_str())
        && match path.segments.len() {
            1 => true,
            2 => {
                path.leading_colon.is_some()
                    && matches!(path.segments.first(), Some(seg) if seg.ident == "core" || seg.ident == "std")
            }
            _ => false,
        }
}

/// The arguments of `matches!(expr, pattern if guard)`.
fn parse_matches(input: ParseStream) -> Result<(Expr, Pat, Option<Expr>)> {
    let expr: Expr = input.parse()?;
    input.parse::<Token![,]>()?;
    let pat = Pat::parse_multi_with_leading_vert(input)?;
    let guard = if input.peek(Token![if]) {
        input.parse::<Token![if]>()?;
        Some(input.parse()?)
    } else {
        None
    };
    input.parse::<Option<Token![,]>>()?;
    Ok((expr, pat, guard))
}

/// The name of an argument of the form `name = value`.
fn argument_name(arg: &Expr) -> Option<String> {
    match arg {
        Expr::Assign(assign) => match &*assign.left {
            Expr::Path(ExprPath {
                qself: None, path, ..
            }) => path.get_ident().map(ident_name),
            _ => None,
        },
        _ => None,
    }
}

/// Where the format string of a macro is among its arguments.
#[derive(Clone, Copy)]
enum FormatString {
    /// The macro is one of the standard library, and has none (`vec!`, `dbg!`).
    None,
    /// The macro is one of the standard library: the argument at this position (`println!("..")`
    /// is the first, `write!(f, "..")` the second).
    At(usize),
    /// Any macro: every string literal among the arguments may be one.
    Any,
}

impl FormatString {
    fn is_at(self, at: usize) -> bool {
        match self {
            FormatString::None => false,
            FormatString::At(position) => position == at,
            FormatString::Any => true,
        }
    }
}

/// Tells where the format string is in the macro `path`: for the macros of the standard library
/// that format, the position; for the others of the library, none; for a macro of the user, any
/// literal may be.
fn format_string(path: &Path) -> FormatString {
    const FIRST: &[&str] = &[
        "print",
        "println",
        "eprint",
        "eprintln",
        "format",
        "format_args",
        "panic",
        "unreachable",
        "todo",
        "unimplemented",
    ];
    const SECOND: &[&str] = &["write", "writeln", "assert", "debug_assert"];
    const THIRD: &[&str] = &[
        "assert_eq",
        "assert_ne",
        "debug_assert_eq",
        "debug_assert_ne",
    ];
    const NONE: &[&str] = &[
        "dbg",
        "vec",
        "concat",
        "env",
        "option_env",
        "include_str",
        "include_bytes",
        "file",
        "line",
        "column",
        "module_path",
        "compile_error",
    ];
    if is_std_macro(path, FIRST) {
        FormatString::At(0)
    } else if is_std_macro(path, SECOND) {
        FormatString::At(1)
    } else if is_std_macro(path, THIRD) {
        FormatString::At(2)
    } else if is_std_macro(path, NONE) {
        FormatString::None
    } else {
        FormatString::Any
    }
}

/// The names that a format string captures from the scope: `{name}`, `{name:?}`, and the `width`
/// of `{:width$}`. Only ASCII names are read.
fn captured_names(format: &str) -> Vec<String> {
    let chars: Vec<char> = format.chars().collect();
    let mut names = Vec::new();
    let mut at = 0;
    while at < chars.len() {
        if chars[at] != '{' {
            at += 1;
        } else if chars.get(at + 1) == Some(&'{') {
            at += 2;
        } else {
            let end = match chars[at + 1..].iter().position(|&c| c == '}') {
                Some(n) => at + 1 + n,
                None => break,
            };
            let inner: String = chars[at + 1..end].iter().collect();
            let (arg, spec) = match inner.find(':') {
                Some(n) => (&inner[..n], &inner[n + 1..]),
                None => (&inner[..], ""),
            };
            names.push(arg.trim().to_string());
            let spec: Vec<char> = spec.chars().collect();
            for n in 0..spec.len() {
                if spec[n] != '$' {
                    continue;
                }
                let start = spec[..n]
                    .iter()
                    .rposition(|&c| !(c.is_ascii_alphanumeric() || c == '_'))
                    .map_or(0, |p| p + 1);
                names.push(spec[start..n].iter().collect());
            }
            at = end + 1;
        }
    }
    names.retain(|name| {
        name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
            && name.chars().next().map_or(false, |c| !c.is_ascii_digit())
    });
    names
}

/// What to write for the relative macro `path`: the absolute path of a macro of the standard
/// library, else how to write it.
fn macro_help(path: &Path) -> String {
    const CORE: &[&str] = &[
        "assert",
        "assert_eq",
        "assert_ne",
        "cfg",
        "column",
        "compile_error",
        "concat",
        "debug_assert",
        "debug_assert_eq",
        "debug_assert_ne",
        "env",
        "file",
        "format_args",
        "include_bytes",
        "include_str",
        "line",
        "matches",
        "module_path",
        "option_env",
        "panic",
        "stringify",
        "todo",
        "unimplemented",
        "unreachable",
        "write",
        "writeln",
    ];
    const STD: &[&str] = &[
        "dbg", "eprint", "eprintln", "format", "print", "println", "vec",
    ];
    let name = tidy(&quote!(#path));
    if path.leading_colon.is_none() && path.segments.len() == 1 {
        let crate_name = if CORE.contains(&name.as_str()) {
            Some("core")
        } else if STD.contains(&name.as_str()) {
            Some("std")
        } else {
            None
        };
        if let Some(crate_name) = crate_name {
            return format!("use `::{}::{}!` instead of `{}!`", crate_name, name, name);
        }
    }
    format!(
        "write `{0}!` with an absolute path (e.g. `::my_crate::{0}!`), or allow it with \
         `LeakerConfig::allow_path()`",
        name
    )
}
