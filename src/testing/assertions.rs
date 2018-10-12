use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

pub struct Assertion<'t, T>
where
    T: 't + Debug,
{
    pub actual: &'t T,
}

#[macro_export]
macro_rules! assert_that {
    ($actual:expr) => {{
        Assertion { actual: &$actual }
    }};
}

/**
 * Option Assertions
 */
impl<'a, T> Assertion<'a, Option<T>>
where
    T: 'a + Debug + PartialEq,
{
    pub fn contains(self, expected: T) {
        match self.actual {
            None => panic!("expected Option to contain {:?}, was None", expected),
            Some(actual) => assert_eq!(actual, &expected),
        }
    }

    pub fn is_empty(self) {
        match self.actual {
            None => {}
            Some(actual) => panic!(
                "expected Option to be None, actually contained {:?}",
                actual
            ),
        }
    }
}

/**
 * Vec Assertions
 */
impl<'a, T> Assertion<'a, Vec<T>>
where
    T: 'a + Debug + PartialEq,
{
    pub fn is_empty(self) {
        assert!(self.actual.is_empty(), "actual wasn't empty!");
    }
}

/**
 * HashMap Assertions
 */
impl<'a, K, V> Assertion<'a, HashMap<K, V>>
where
    K: 'a + Debug + Eq + Hash,
    V: 'a + Debug + PartialEq,
{
    pub fn is_empty(self) {
        assert!(self.actual.is_empty(), "actual wasn't empty!");
    }

    pub fn has_size(&self, size: usize) {
        assert_eq!(self.actual.len(), size);
    }
}
