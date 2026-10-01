//! GBNF grammars: parsing and the compiled rule form the matcher walks.
//!
//! The syntax is the common GBNF dialect servers accept in a `grammar` field:
//!
//! ```text
//! root   ::= object
//! object ::= "{" ws ( pair ( "," ws pair )* )? "}"
//! pair   ::= string ":" ws value     # comments run to the end of the line
//! digit  ::= [0-9]
//! ```
//!
//! Rules are `name ::= alternatives`, one per line unless a parenthesis is open.
//! Items are quoted literals, character classes (`[a-z]`, `[^"\\]`), `.` for any
//! character, rule references and parenthesized groups, each optionally followed by
//! `*`, `+`, `?`, `{m}`, `{m,}` or `{m,n}`. Escapes: `\n \r \t \\ \" \] \[ \-`,
//! `\xHH`, `\uHHHH`, `\UHHHHHHHH`.
//!
//! Compiled, a rule is a list of alternatives, each a sequence of [`Elem`]s.
//! Repetition and groups become helper rules (`x*` is `R ::= x R |`), so the
//! matcher only ever sees characters and rule references. Left recursion is
//! rejected at compile time, since the matcher expands rules top-down.

use anyhow::{bail, Context, Result};
use std::collections::HashMap;

/// One element of an alternative.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Elem {
    /// One code point in any of `ranges` (inclusive), or outside all of them when
    /// `negated`.
    Chars { ranges: Vec<(u32, u32)>, negated: bool },
    /// The rule with this id.
    Rule(u32),
}

impl Elem {
    pub(crate) fn matches(&self, c: u32) -> bool {
        match self {
            Elem::Chars { ranges, negated } => ranges.iter().any(|&(lo, hi)| lo <= c && c <= hi) != *negated,
            Elem::Rule(_) => false,
        }
    }
}

/// A compiled grammar: `rules[r][a]` is alternative `a` of rule `r`.
#[derive(Clone, Debug)]
pub struct Grammar {
    pub(crate) rules: Vec<Vec<Vec<Elem>>>,
    pub(crate) names: Vec<String>,
    pub(crate) root: u32,
}

impl Grammar {
    /// Parse GBNF text. The start rule is `root`.
    pub fn parse(src: &str) -> Result<Grammar> {
        let mut p = Parser { src, s: src.as_bytes(), i: 0, b: Builder::default() };
        p.grammar()?;
        let b = p.b;
        for (name, &id) in &b.ids {
            if b.rules[id as usize].is_none() {
                bail!("rule `{name}` is used but never defined");
            }
        }
        let root = *b.ids.get("root").context("grammar has no `root` rule")?;
        let mut names = vec![String::new(); b.rules.len()];
        for (n, &id) in &b.ids { names[id as usize] = n.clone(); }
        for (i, n) in names.iter_mut().enumerate() { if n.is_empty() { *n = format!("_{i}"); } }
        let g = Grammar { rules: b.rules.into_iter().map(|r| r.unwrap_or_default()).collect(), names, root };
        g.check_left_recursion()?;
        Ok(g)
    }

    /// Whether rule `r` can match the empty string.
    fn nullable(&self) -> Vec<bool> {
        let mut null = vec![false; self.rules.len()];
        loop {
            let mut changed = false;
            for (r, alts) in self.rules.iter().enumerate() {
                if null[r] { continue; }
                let n = alts.iter().any(|seq| seq.iter().all(|e| match e {
                    Elem::Rule(x) => null[*x as usize],
                    Elem::Chars { .. } => false,
                }));
                if n { null[r] = true; changed = true; }
            }
            if !changed { return null; }
        }
    }

    /// Reject a rule that can reach itself without consuming a character: the
    /// top-down expansion would never terminate.
    fn check_left_recursion(&self) -> Result<()> {
        let null = self.nullable();
        // left[r] = rules that can start r with nothing consumed before them.
        let left: Vec<Vec<u32>> = self.rules.iter().map(|alts| {
            let mut v = Vec::new();
            for seq in alts {
                for e in seq {
                    match e {
                        Elem::Rule(x) => { v.push(*x); if !null[*x as usize] { break; } }
                        Elem::Chars { .. } => break,
                    }
                }
            }
            v
        }).collect();
        for start in 0..self.rules.len() as u32 {
            let mut seen = vec![false; self.rules.len()];
            let mut pending = left[start as usize].clone();
            while let Some(r) = pending.pop() {
                if r == start {
                    bail!("rule `{}` is left-recursive", self.names[start as usize]);
                }
                if !std::mem::replace(&mut seen[r as usize], true) {
                    pending.extend(&left[r as usize]);
                }
            }
        }
        Ok(())
    }
}

#[derive(Default)]
struct Builder {
    ids: HashMap<String, u32>,
    rules: Vec<Option<Vec<Vec<Elem>>>>,
}

impl Builder {
    fn id(&mut self, name: &str) -> u32 {
        if let Some(&i) = self.ids.get(name) { return i; }
        let i = self.rules.len() as u32;
        self.rules.push(None);
        self.ids.insert(name.to_string(), i);
        i
    }

    /// A helper rule with no name.
    fn fresh(&mut self, alts: Vec<Vec<Elem>>) -> u32 {
        self.rules.push(Some(alts));
        (self.rules.len() - 1) as u32
    }
}

struct Parser<'a> {
    src: &'a str,
    s: &'a [u8],
    i: usize,
    b: Builder,
}

impl Parser<'_> {
    fn peek(&self) -> Option<u8> { self.s.get(self.i).copied() }

    /// Skip spaces and comments; newlines too when `newlines` (inside parentheses
    /// or between rules).
    fn ws(&mut self, newlines: bool) {
        while let Some(c) = self.peek() {
            match c {
                b' ' | b'\t' => self.i += 1,
                b'\r' | b'\n' if newlines => self.i += 1,
                b'#' => { while self.peek().is_some_and(|c| c != b'\n') { self.i += 1; } }
                _ => break,
            }
        }
    }

    fn name(&mut self) -> Result<String> {
        let start = self.i;
        while self.peek().is_some_and(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_') { self.i += 1; }
        if self.i == start { bail!("expected a rule name at byte {}", self.i); }
        Ok(String::from_utf8_lossy(&self.s[start..self.i]).into_owned())
    }

    fn grammar(&mut self) -> Result<()> {
        loop {
            self.ws(true);
            if self.peek().is_none() { return Ok(()); }
            let name = self.name()?;
            self.ws(false);
            if !self.s[self.i..].starts_with(b"::=") {
                bail!("expected `::=` after rule `{name}` at byte {}", self.i);
            }
            self.i += 3;
            let alts = self.alternatives(false)?;
            let id = self.b.id(&name);
            if self.b.rules[id as usize].is_some() { bail!("rule `{name}` is defined twice"); }
            self.b.rules[id as usize] = Some(alts);
        }
    }

    /// `seq ( "|" seq )*`, up to the end of the line (or the closing parenthesis
    /// when `nested`).
    fn alternatives(&mut self, nested: bool) -> Result<Vec<Vec<Elem>>> {
        let mut alts = vec![self.sequence(nested)?];
        loop {
            self.ws(nested);
            if self.peek() == Some(b'|') {
                self.i += 1;
                alts.push(self.sequence(nested)?);
            } else {
                return Ok(alts);
            }
        }
    }

    fn sequence(&mut self, nested: bool) -> Result<Vec<Elem>> {
        let mut seq = Vec::new();
        loop {
            self.ws(nested);
            // A `|` after a line break continues the rule.
            if !nested && matches!(self.peek(), Some(b'\r' | b'\n')) {
                let save = self.i;
                self.ws(true);
                if self.peek() == Some(b'|') { continue; }
                self.i = save;
                return Ok(seq);
            }
            let start_len = seq.len();
            match self.peek() {
                None | Some(b'|') | Some(b')') => return Ok(seq),
                Some(b'"') => { self.i += 1; self.literal(&mut seq)?; }
                Some(b'[') => { self.i += 1; seq.push(self.class()?); }
                Some(b'.') => { self.i += 1; seq.push(Elem::Chars { ranges: vec![], negated: true }); }
                Some(b'(') => {
                    self.i += 1;
                    let alts = self.alternatives(true)?;
                    self.ws(true);
                    if self.peek() != Some(b')') { bail!("unclosed `(` at byte {}", self.i); }
                    self.i += 1;
                    let r = self.b.fresh(alts);
                    seq.push(Elem::Rule(r));
                }
                Some(c) if c.is_ascii_alphanumeric() || c == b'-' || c == b'_' => {
                    // Stop at the next rule's `name ::=`.
                    let save = self.i;
                    let name = self.name()?;
                    let after = self.i;
                    self.ws(false);
                    if self.s[self.i..].starts_with(b"::=") { self.i = save; return Ok(seq); }
                    self.i = after;
                    let id = self.b.id(&name);
                    seq.push(Elem::Rule(id));
                }
                Some(c) => bail!("unexpected `{}` at byte {}", c as char, self.i),
            }
            self.repetition(&mut seq, start_len)?;
        }
    }

    /// Apply a postfix operator to the item(s) just pushed at `seq[start..]`.
    fn repetition(&mut self, seq: &mut Vec<Elem>, start: usize) -> Result<()> {
        // An operator may follow its item after spaces, as in `[a-z] {2,3}`.
        let save = self.i;
        self.ws(false);
        if !matches!(self.peek(), Some(b'*' | b'+' | b'?' | b'{')) { self.i = save; }
        let (min, max): (u32, Option<u32>) = match self.peek() {
            Some(b'*') => { self.i += 1; (0, None) }
            Some(b'+') => { self.i += 1; (1, None) }
            Some(b'?') => { self.i += 1; (0, Some(1)) }
            Some(b'{') => {
                self.i += 1;
                let num = |p: &mut Self| -> Option<u32> {
                    let st = p.i;
                    while p.peek().is_some_and(|c| c.is_ascii_digit()) { p.i += 1; }
                    std::str::from_utf8(&p.s[st..p.i]).ok()?.parse().ok()
                };
                self.ws(false);
                let lo = num(self).context("expected a count after `{`")?;
                self.ws(false);
                let hi = if self.peek() == Some(b',') {
                    self.i += 1;
                    self.ws(false);
                    num(self)
                } else { Some(lo) };
                self.ws(false);
                if self.peek() != Some(b'}') { bail!("expected `}}` at byte {}", self.i); }
                self.i += 1;
                if hi.is_some_and(|h| h < lo) { bail!("repetition {{{lo},{}}} has max below min", hi.unwrap()); }
                (lo, hi)
            }
            _ => return Ok(()),
        };
        // The repeated unit: one element (a literal of several characters is grouped).
        let item: Vec<Elem> = seq.drain(start..).collect();
        let unit = if item.len() == 1 { item.into_iter().next().unwrap() } else { Elem::Rule(self.b.fresh(vec![item])) };
        for _ in 0..min { seq.push(unit.clone()); }
        match max {
            None => {
                // R ::= unit R |
                let r = self.b.fresh(vec![]);
                self.b.rules[r as usize] = Some(vec![vec![unit, Elem::Rule(r)], vec![]]);
                seq.push(Elem::Rule(r));
            }
            Some(hi) => {
                // Optional tail of hi-min units, nested so each is only allowed after
                // the previous: R_k ::= unit R_{k-1} |
                let mut tail: Option<u32> = None;
                for _ in min..hi {
                    let mut first = vec![unit.clone()];
                    if let Some(t) = tail { first.push(Elem::Rule(t)); }
                    tail = Some(self.b.fresh(vec![first, vec![]]));
                }
                if let Some(t) = tail { seq.push(Elem::Rule(t)); }
            }
        }
        Ok(())
    }

    fn escape(&mut self) -> Result<u32> {
        let c = self.peek().context("unfinished escape")?;
        self.i += 1;
        let hex = |p: &mut Self, n: usize| -> Result<u32> {
            let st = p.i;
            p.i += n;
            let t = std::str::from_utf8(p.s.get(st..p.i).context("unfinished escape")?)?;
            u32::from_str_radix(t, 16).with_context(|| format!("bad hex escape `{t}`"))
        };
        Ok(match c {
            b'n' => '\n' as u32, b'r' => '\r' as u32, b't' => '\t' as u32,
            b'x' => hex(self, 2)?, b'u' => hex(self, 4)?, b'U' => hex(self, 8)?,
            b'\\' | b'"' | b'[' | b']' | b'-' | b'^' | b'/' => c as u32,
            other => bail!("unknown escape `\\{}`", other as char),
        })
    }

    /// One code point of the source (UTF-8), unescaped.
    fn char(&mut self) -> Result<u32> {
        if self.peek() == Some(b'\\') { self.i += 1; return self.escape(); }
        // The parser only stops on character boundaries, so `i` indexes `src` safely.
        let ch = self.src[self.i..].chars().next().context("unexpected end of grammar")?;
        self.i += ch.len_utf8();
        Ok(ch as u32)
    }

    fn literal(&mut self, seq: &mut Vec<Elem>) -> Result<()> {
        loop {
            match self.peek() {
                None => bail!("unterminated string literal"),
                Some(b'"') => { self.i += 1; return Ok(()); }
                _ => { let c = self.char()?; seq.push(Elem::Chars { ranges: vec![(c, c)], negated: false }); }
            }
        }
    }

    fn class(&mut self) -> Result<Elem> {
        let negated = self.peek() == Some(b'^');
        if negated { self.i += 1; }
        let mut ranges = Vec::new();
        loop {
            match self.peek() {
                None => bail!("unterminated character class"),
                Some(b']') => { self.i += 1; return Ok(Elem::Chars { ranges, negated }); }
                _ => {
                    let lo = self.char()?;
                    let hi = if self.peek() == Some(b'-') && self.s.get(self.i + 1) != Some(&b']') {
                        self.i += 1;
                        self.char()?
                    } else { lo };
                    if hi < lo { bail!("character range is reversed"); }
                    ranges.push((lo, hi));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_rules_classes_and_repetition() {
        let g = Grammar::parse("root ::= \"a\" [0-9]+ ( x | \"y\" )? # tail\nx ::= [^\"\\\\] {2,3}\n").unwrap();
        assert_eq!(g.names[g.root as usize], "root");
        // root: "a", one digit, the digit loop, the optional group.
        let root = &g.rules[g.root as usize];
        assert_eq!(root.len(), 1);
        assert_eq!(root[0].len(), 4);
        assert_eq!(root[0][0], Elem::Chars { ranges: vec![('a' as u32, 'a' as u32)], negated: false });
        assert_eq!(root[0][1], Elem::Chars { ranges: vec![('0' as u32, '9' as u32)], negated: false });
        assert!(matches!(root[0][2], Elem::Rule(_)) && matches!(root[0][3], Elem::Rule(_)));
    }

    #[test]
    fn continues_alternatives_on_the_next_line() {
        let g = Grammar::parse("root ::= \"a\"\n  | \"b\"\n").unwrap();
        assert_eq!(g.rules[g.root as usize].len(), 2);
    }

    #[test]
    fn rejects_undefined_and_left_recursive_rules() {
        assert!(Grammar::parse("root ::= missing").is_err());
        assert!(Grammar::parse("root ::= root \"a\" | \"b\"").is_err());
        assert!(Grammar::parse("root ::= x\nx ::= y \"a\"\ny ::= x | \"b\"").is_err());
        assert!(Grammar::parse("root ::= \"a\" root | \"b\"").is_ok());
    }
}
