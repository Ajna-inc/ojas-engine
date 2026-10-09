//! Invite-only pools.
//!
//! A pool is a name and an admin key. The admin signs one invite per member:
//! Ed25519 over `(pool name, member PeerId, expiry)`. A node is a member if it is
//! the admin or holds an unexpired invite for its own PeerId.
//!
//! **Why not libp2p `pnet`:** a pre-shared-key transport only wraps TCP; QUIC has
//! its own TLS handshake and cannot carry it, and NAT'd members need QUIC and relay
//! circuits. Membership is therefore checked above the transport, on every
//! connection, whatever carried it.
//!
//! **Why identify's `agent_version`:** identify already runs on every connection,
//! in both directions, before anything else of ours, and its contents are bound to
//! the noise-authenticated PeerId. The invite names that PeerId, so it is not a
//! secret: someone who copies it cannot use it from another key. Carrying it there
//! costs no extra round trip or protocol, and a peer that never identifies is
//! timed out and dropped the same way as one that fails the check. The node holds
//! back RPCs, streams and gossip from a peer until its credential has verified.

use anyhow::{bail, Context, Result};
use base64::Engine;
use libp2p::identity::{Keypair, PublicKey};
use libp2p::PeerId;
use serde::{Deserialize, Serialize};
use std::path::Path;

const DOMAIN: &[u8] = b"ojas-pool-invite/v1";
const TOKEN_PREFIX: &str = "ojasinv1.";
const B64: base64::engine::GeneralPurpose = base64::engine::general_purpose::URL_SAFE_NO_PAD;

/// The pool file every member holds: `{"name": ..., "admin": "<PeerId>"}`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Pool {
    pub name: String,
    /// The admin's PeerId. An Ed25519 PeerId inlines the public key, so this is the
    /// verification key as well as a name.
    #[serde(with = "peer_str")]
    pub admin: PeerId,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Invite {
    pub pool: String,
    #[serde(with = "peer_str")]
    pub member: PeerId,
    /// Unix seconds.
    pub expires: u64,
    /// base64url Ed25519 signature by the pool admin.
    pub sig: String,
}

/// What a node presents to prove it belongs.
#[derive(Clone, Debug, PartialEq)]
pub enum Credential {
    /// This node's key is the pool's admin key.
    Admin,
    Invite(Invite),
}

impl Pool {
    pub fn new(name: &str, admin: &PublicKey) -> Result<Pool> {
        validate_name(name)?;
        admin.clone().try_into_ed25519().map_err(|_| anyhow::anyhow!("pool admin key must be Ed25519"))?;
        Ok(Pool { name: name.to_string(), admin: admin.to_peer_id() })
    }

    pub fn load(path: &Path) -> Result<Pool> {
        let s = std::fs::read_to_string(path).with_context(|| format!("reading pool file {}", path.display()))?;
        let p: Pool = serde_json::from_str(&s).with_context(|| format!("parsing pool file {}", path.display()))?;
        validate_name(&p.name)?;
        crate::key::public_key_of(&p.admin).context("pool admin is not an Ed25519 PeerId")?;
        Ok(p)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        std::fs::write(path, serde_json::to_string_pretty(self)? + "\n")
            .with_context(|| format!("writing pool file {}", path.display()))
    }

    /// Sign an invite for `member`, valid until `expires` (unix seconds).
    pub fn invite(&self, admin: &Keypair, member: PeerId, expires: u64) -> Result<Invite> {
        if admin.public().to_peer_id() != self.admin {
            bail!("this key is not the admin of pool {:?} (admin is {})", self.name, self.admin);
        }
        let sig = admin.sign(&signed_bytes(&self.name, &member, expires)).context("signing invite")?;
        Ok(Invite { pool: self.name.clone(), member, expires, sig: B64.encode(sig) })
    }

    /// Whether `peer`, presenting `cred`, is a member at time `now`.
    pub fn admits(&self, peer: &PeerId, cred: &Credential, now: u64) -> Result<()> {
        match cred {
            Credential::Admin if *peer == self.admin => Ok(()),
            Credential::Admin => bail!("claims to be the admin but is not"),
            Credential::Invite(inv) => self.verify(peer, inv, now),
        }
    }

    pub fn verify(&self, peer: &PeerId, inv: &Invite, now: u64) -> Result<()> {
        if inv.pool != self.name {
            bail!("invite is for pool {:?}, not {:?}", inv.pool, self.name);
        }
        if inv.member != *peer {
            bail!("invite was issued to {}, not to this peer", inv.member);
        }
        if inv.expires <= now {
            bail!("invite expired at {}", inv.expires);
        }
        let admin = crate::key::public_key_of(&self.admin).context("pool admin is not an Ed25519 PeerId")?;
        let sig = B64.decode(&inv.sig).context("invite signature is not base64url")?;
        if !admin.verify(&signed_bytes(&inv.pool, &inv.member, inv.expires), &sig) {
            bail!("invite signature does not verify against the pool admin");
        }
        Ok(())
    }
}

impl Invite {
    pub fn to_token(&self) -> String {
        format!("{TOKEN_PREFIX}{}", B64.encode(serde_json::to_vec(self).expect("invite serialises")))
    }

    pub fn from_token(s: &str) -> Result<Invite> {
        let body = s.trim().strip_prefix(TOKEN_PREFIX).context("not an ojas invite token")?;
        let json = B64.decode(body).context("invite token is not base64url")?;
        serde_json::from_slice(&json).context("invite token does not decode")
    }
}

impl Credential {
    /// The form carried in identify's agent_version.
    pub fn to_wire(&self) -> String {
        match self {
            Credential::Admin => "admin".into(),
            Credential::Invite(i) => i.to_token(),
        }
    }

    pub fn from_wire(s: &str) -> Result<Credential> {
        if s == "admin" {
            return Ok(Credential::Admin);
        }
        Invite::from_token(s).map(Credential::Invite)
    }

    pub fn expires(&self) -> Option<u64> {
        match self {
            Credential::Admin => None,
            Credential::Invite(i) => Some(i.expires),
        }
    }
}

/// `ojas-node/<version> pool=<name> cred=<credential>`. Space separated: neither
/// a pool name nor a token may contain whitespace.
pub fn agent_version(engine_version: &str, pool: &Pool, cred: &Credential) -> String {
    format!("ojas-node/{engine_version} pool={} cred={}", pool.name, cred.to_wire())
}

/// The pool name and credential in a peer's agent_version.
pub fn parse_agent_version(s: &str) -> Result<(String, Credential)> {
    let field = |k: &str| s.split_whitespace().find_map(|w| w.strip_prefix(k)).map(str::to_string);
    let pool = field("pool=").context("agent_version carries no pool")?;
    let cred = field("cred=").context("agent_version carries no credential")?;
    Ok((pool, Credential::from_wire(&cred)?))
}

fn signed_bytes(pool: &str, member: &PeerId, expires: u64) -> Vec<u8> {
    let m = member.to_bytes();
    let mut b = Vec::with_capacity(DOMAIN.len() + pool.len() + m.len() + 24);
    // Length-prefixed fields, so no two different tuples share a byte string.
    for part in [DOMAIN, pool.as_bytes(), &m] {
        b.extend_from_slice(&(part.len() as u64).to_le_bytes());
        b.extend_from_slice(part);
    }
    b.extend_from_slice(&expires.to_le_bytes());
    b
}

fn validate_name(name: &str) -> Result<()> {
    if name.is_empty() || name.len() > 64 || !name.chars().all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c)) {
        bail!("pool name must be 1-64 characters of [A-Za-z0-9-_.], got {name:?}");
    }
    Ok(())
}

mod peer_str {
    use libp2p::PeerId;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(p: &PeerId, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&p.to_string())
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<PeerId, D::Error> {
        String::deserialize(d)?.parse().map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (Keypair, Pool, Keypair) {
        let admin = Keypair::generate_ed25519();
        let pool = Pool::new("lab", &admin.public()).unwrap();
        (admin, pool, Keypair::generate_ed25519())
    }

    #[test]
    fn an_invite_admits_its_member_until_it_expires() {
        let (admin, pool, member) = setup();
        let me = member.public().to_peer_id();
        let inv = pool.invite(&admin, me, 1_000).unwrap();
        let tok = Invite::from_token(&inv.to_token()).unwrap();
        assert_eq!(tok, inv);
        pool.verify(&me, &tok, 999).unwrap();
        assert!(pool.verify(&me, &tok, 1_000).unwrap_err().to_string().contains("expired"));
    }

    #[test]
    fn an_invite_is_useless_to_any_other_key() {
        let (admin, pool, member) = setup();
        let inv = pool.invite(&admin, member.public().to_peer_id(), u64::MAX).unwrap();
        let thief = Keypair::generate_ed25519().public().to_peer_id();
        assert!(pool.verify(&thief, &inv, 0).is_err());
        assert!(pool.admits(&thief, &Credential::Admin, 0).is_err());
        pool.admits(&admin.public().to_peer_id(), &Credential::Admin, 0).unwrap();
    }

    #[test]
    fn tampering_or_another_admin_or_pool_fails() {
        let (admin, pool, member) = setup();
        let me = member.public().to_peer_id();
        let inv = pool.invite(&admin, me, 5_000).unwrap();
        let mut longer = inv.clone();
        longer.expires = 9_000;
        assert!(pool.verify(&me, &longer, 0).unwrap_err().to_string().contains("signature"));
        let mut renamed = inv.clone();
        renamed.pool = "other".into();
        assert!(pool.verify(&me, &renamed, 0).is_err());
        let other = Pool::new("lab", &Keypair::generate_ed25519().public()).unwrap();
        assert!(other.verify(&me, &inv, 0).is_err(), "same name, different admin");
        assert!(pool.invite(&member, me, 1).is_err(), "only the admin signs");
    }

    #[test]
    fn credentials_survive_agent_version() {
        let (admin, pool, member) = setup();
        let inv = pool.invite(&admin, member.public().to_peer_id(), 77).unwrap();
        for cred in [Credential::Admin, Credential::Invite(inv)] {
            let av = agent_version("0.1.0", &pool, &cred);
            assert_eq!(parse_agent_version(&av).unwrap(), ("lab".to_string(), cred));
        }
        assert!(parse_agent_version("rust-libp2p/0.47").is_err());
        assert!(Pool::new("has space", &admin.public()).is_err());
    }
}
