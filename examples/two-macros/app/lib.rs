//! Crate B of the two-macro example: it implements `repo::Repo` for wrappers of a type that
//! implements it. It names neither `Page<K>` nor `model::Record`, and has no `Page` in scope.

use repo::forward;

/// A type that implements the trait.
pub struct InMemory(pub Vec<u8>);

impl repo::Repo<u8> for InMemory {
    type Error = String;

    fn get(&self, key: u8) -> repo::Page<u8> {
        repo::Page(self.0.iter().copied().filter(|b| *b == key).collect())
    }
    fn schema(&self) -> repo::model::Record {
        repo::model::Record(self.0.len() as u32)
    }
    fn failures(&mut self) -> repo::Page<String> {
        repo::Page(vec!["no failure".to_string()])
    }
    fn many<U>(&self, items: Vec<U>) -> repo::Page<U> {
        repo::Page(items)
    }
}

/// The generics of the trait are the generics of the impl: `impl<K> Repo<K> for Cached`.
#[forward(repo::Repo)]
pub struct Cached(InMemory);

/// The trait is given its arguments: `impl Repo<u8> for Logged`.
#[forward(repo::Repo<u8>)]
pub struct Logged(InMemory);

#[cfg(test)]
mod tests {
    use super::*;
    use repo::{model::Record, Page, Repo};

    #[test]
    fn the_wrapper_with_the_generics_of_the_trait_forwards() {
        let mut cached = Cached(InMemory(vec![1, 2, 2]));
        // The types are the ones of `repo`, which this crate never named.
        let page: Page<u8> = cached.get(2);
        assert_eq!(page, Page(vec![2, 2]));
        let record: Record = cached.schema();
        assert_eq!(record, Record(3));
        let failures: Page<String> = cached.failures();
        assert_eq!(failures, Page(vec!["no failure".to_string()]));
        let many: Page<char> = cached.many(vec!['a']);
        assert_eq!(many, Page(vec!['a']));
    }

    #[test]
    fn the_wrapper_with_arguments_forwards() {
        let mut logged = Logged(InMemory(vec![5]));
        let page: Page<u8> = logged.get(5);
        assert_eq!(page, Page(vec![5]));
        let record: Record = logged.schema();
        assert_eq!(record, Record(1));
        let failures: Page<String> = logged.failures();
        assert_eq!(failures.0.len(), 1);
        let many: Page<u8> = logged.many(vec![9]);
        assert_eq!(many, Page(vec![9]));
    }
}
