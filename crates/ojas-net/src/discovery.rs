//! Bootstrap dialling.
//!
//! Adapted from SwarmLLM (MIT OR Apache-2.0), `src/network/discovery.rs` @ b14482d:
//! one dial per *peer*, never one per address. A bare-address dial carries no peer
//! condition, so N addresses for one peer become N connections, up to the per-peer
//! cap, and request_response then spreads requests across connections that may have
//! quietly died. A dial by PeerId with all its addresses races them and keeps one.

use crate::relay::target_peer;
use libp2p::{Multiaddr, PeerId};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dial {
    pub peer: Option<PeerId>,
    pub addrs: Vec<Multiaddr>,
}

pub fn plan(addrs: &[Multiaddr]) -> Vec<Dial> {
    let mut plan: Vec<Dial> = Vec::new();
    for a in addrs {
        let peer = target_peer(a);
        match plan.iter_mut().find(|d| d.peer.is_some() && d.peer == peer) {
            Some(d) => d.addrs.push(a.clone()),
            None => plan.push(Dial { peer, addrs: vec![a.clone()] }),
        }
    }
    plan
}

/// mDNS finds both sides at once. Only the smaller PeerId dials, so a pair never
/// races two connections against each other and keeps a half-open loser.
pub fn mdns_should_dial(me: &PeerId, other: &PeerId) -> bool {
    me < other
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn addresses_of_one_peer_become_one_dial() {
        let p = PeerId::random();
        let a: Multiaddr = format!("/ip4/127.0.0.1/tcp/1/p2p/{p}").parse().unwrap();
        let b: Multiaddr = format!("/ip4/127.0.0.1/udp/1/quic-v1/p2p/{p}").parse().unwrap();
        let bare: Multiaddr = "/ip4/127.0.0.1/tcp/2".parse().unwrap();
        let plan = plan(&[a.clone(), bare.clone(), b.clone()]);
        assert_eq!(plan, vec![Dial { peer: Some(p), addrs: vec![a, b] }, Dial { peer: None, addrs: vec![bare] }]);
        let q = PeerId::random();
        assert_ne!(mdns_should_dial(&p, &q), mdns_should_dial(&q, &p));
    }
}
