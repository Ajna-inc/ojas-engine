//! Who may call a server: peers on a trusted network are served without a token,
//! everyone else needs `Authorization: Bearer <token>`. The defaults are narrow:
//! only loopback is trusted, and a server bound beyond loopback must be given a
//! token.

use crate::flags::RunOpts;
use anyhow::{bail, Context, Result};
use std::net::IpAddr;

/// An address range that may call without a token.
#[derive(Debug, Clone)]
pub(crate) struct Trusted {
    pub(crate) net: IpAddr,
    pub(crate) bits: u32,
}

impl Trusted {
    pub(crate) fn parse(s: &str) -> Result<Trusted> {
        let (addr, bits) = match s.split_once('/') {
            Some((a, b)) => (a, b.parse::<u32>().context("CIDR prefix")?),
            None => (s, if s.contains(':') { 128 } else { 32 }),
        };
        let net: IpAddr = addr.parse().with_context(|| format!("{s:?}: not an IP address or CIDR range"))?;
        let max = if net.is_ipv6() { 128 } else { 32 };
        anyhow::ensure!(bits <= max, "{s:?}: /{bits} is too long for this address family");
        Ok(Trusted { net, bits })
    }

    pub(crate) fn contains(&self, ip: IpAddr) -> bool {
        // An IPv4 peer arriving on a dual-stack socket looks like ::ffff:a.b.c.d.
        let ip = match ip {
            IpAddr::V6(v6) => v6.to_ipv4_mapped().map(IpAddr::V4).unwrap_or(IpAddr::V6(v6)),
            v4 => v4,
        };
        match (self.net, ip) {
            (IpAddr::V4(a), IpAddr::V4(b)) => prefix_eq(&a.octets(), &b.octets(), self.bits),
            (IpAddr::V6(a), IpAddr::V6(b)) => prefix_eq(&a.octets(), &b.octets(), self.bits),
            _ => false,
        }
    }
}

fn prefix_eq(a: &[u8], b: &[u8], bits: u32) -> bool {
    let (whole, rest) = ((bits / 8) as usize, bits % 8);
    if a[..whole] != b[..whole] {
        return false;
    }
    if rest == 0 {
        return true;
    }
    let mask = 0xffu8 << (8 - rest);
    a[whole] & mask == b[whole] & mask
}

/// Token comparison with no early exit on the first differing byte.
pub(crate) fn token_ok(want: &str, got: &str) -> bool {
    if want.len() != got.len() {
        return false;
    }
    want.bytes().zip(got.bytes()).fold(0u8, |acc, (a, b)| acc | (a ^ b)) == 0
}

pub(crate) struct Auth {
    pub(crate) token: Option<String>,
    pub(crate) trusted: Vec<Trusted>,
}

impl Auth {
    /// The rules `--token` / `--token-file`, `--trust` and `--host` describe.
    /// `what` names what is being served, for the refusal to bind beyond loopback
    /// without a token.
    pub(crate) fn from_opts(opts: &RunOpts, what: &str) -> Result<Auth> {
        let token = match (&opts.token, &opts.token_file) {
            (Some(_), Some(_)) => bail!("--token and --token-file are alternatives"),
            (Some(t), None) => Some(t.trim().to_string()),
            (None, Some(f)) => Some(std::fs::read_to_string(f).with_context(|| format!("reading {f}"))?.trim().to_string()),
            (None, None) => None,
        };
        if let Some(t) = &token {
            anyhow::ensure!(t.len() >= 16, "the token is {} characters; use at least 16", t.len());
        }
        let mut trusted = vec![Trusted::parse("127.0.0.1/32")?, Trusted::parse("::1/128")?];
        if let Some(list) = &opts.trust {
            for part in list.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                trusted.push(Trusted::parse(part)?);
            }
        }
        let host = opts.host.as_str();
        let loopback_only = host == "127.0.0.1" || host == "::1" || host == "localhost";
        if !loopback_only && token.is_none() {
            bail!(
                "binding {host} without a token would serve {what} to the network.\n\
                 Pass --token-file <file> (a caller on a --trust range still needs no token), \
                 or bind 127.0.0.1."
            );
        }
        Ok(Auth { token, trusted })
    }

    /// Why a request is allowed, or the reason it is not.
    pub(crate) fn check(&self, peer: Option<IpAddr>, bearer: Option<&str>) -> Result<&'static str, &'static str> {
        if let Some(ip) = peer {
            if self.trusted.iter().any(|t| t.contains(ip)) {
                return Ok("trusted network");
            }
        }
        match (&self.token, bearer) {
            (Some(want), Some(got)) if token_ok(want, got) => Ok("token"),
            (Some(_), Some(_)) => Err("bad token"),
            (Some(_), None) => Err("this address is not trusted: send Authorization: Bearer <token>"),
            (None, _) => Err("this address is not trusted and no token is configured"),
        }
    }

    /// One line for a server's banner.
    pub(crate) fn describe(&self) -> String {
        format!("auth: {}, trusted: {}",
            if self.token.is_some() { "token" } else { "trusted networks only" },
            self.trusted.iter().map(|t| format!("{}/{}", t.net, t.bits)).collect::<Vec<_>>().join(" "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cidr_membership() {
        let lan = Trusted::parse("192.168.29.0/24").unwrap();
        assert!(lan.contains("192.168.29.103".parse().unwrap()));
        assert!(!lan.contains("192.168.30.1".parse().unwrap()));
        let host = Trusted::parse("10.1.2.3").unwrap();
        assert!(host.contains("10.1.2.3".parse().unwrap()));
        assert!(!host.contains("10.1.2.4".parse().unwrap()));
        // An odd prefix length.
        let odd = Trusted::parse("10.0.0.0/12").unwrap();
        assert!(odd.contains("10.15.255.1".parse().unwrap()));
        assert!(!odd.contains("10.16.0.1".parse().unwrap()));
        // A v4 peer on a dual-stack socket.
        assert!(lan.contains("::ffff:192.168.29.7".parse().unwrap()));
        assert!(Trusted::parse("127.0.0.1/32").unwrap().contains("127.0.0.1".parse().unwrap()));
    }

    #[test]
    fn auth_rules() {
        let lan: IpAddr = "192.168.29.5".parse().unwrap();
        let far: IpAddr = "8.8.8.8".parse().unwrap();
        let a = Auth { token: Some("0123456789abcdef".into()), trusted: vec![Trusted::parse("192.168.29.0/24").unwrap()] };
        assert!(a.check(Some(lan), None).is_ok(), "the trusted LAN needs no token");
        assert!(a.check(Some(far), Some("0123456789abcdef")).is_ok());
        assert!(a.check(Some(far), Some("wrong")).is_err());
        assert!(a.check(Some(far), None).is_err());
        assert!(a.check(None, None).is_err(), "an unknown peer is not trusted");
        // No token configured: only trusted ranges get in.
        let b = Auth { token: None, trusted: vec![Trusted::parse("127.0.0.1/32").unwrap()] };
        assert!(b.check(Some("127.0.0.1".parse().unwrap()), None).is_ok());
        assert!(b.check(Some(far), Some("anything")).is_err());
    }

    #[test]
    fn a_server_beyond_loopback_needs_a_token() {
        let open = RunOpts { host: "0.0.0.0".into(), ..RunOpts::default() };
        assert!(Auth::from_opts(&open, "this model").is_err());
        let short = RunOpts { host: "0.0.0.0".into(), token: Some("short".into()), ..RunOpts::default() };
        assert!(Auth::from_opts(&short, "this model").is_err());
        let ok = RunOpts { host: "0.0.0.0".into(), token: Some("0123456789abcdef".into()), ..RunOpts::default() };
        assert!(Auth::from_opts(&ok, "this model").is_ok());
        assert!(Auth::from_opts(&RunOpts::default(), "this model").is_ok(), "loopback needs none");
    }

    #[test]
    fn token_compare_is_length_safe() {
        assert!(token_ok("abcd", "abcd"));
        assert!(!token_ok("abcd", "abce"));
        assert!(!token_ok("abcd", "abcde"));
        assert!(!token_ok("", "x"));
    }
}
