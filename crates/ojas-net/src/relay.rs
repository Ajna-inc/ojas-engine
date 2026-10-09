//! Relay server limits and circuit-address helpers.
//!
//! Adapted from SwarmLLM (MIT OR Apache-2.0), `src/network/relay.rs` @ b14482d.
//! The relay is a *data path* here, not a bootstrap aid: DCUtR does not succeed
//! across symmetric NAT or CGNAT, which is most home links, so two NAT'd members
//! may stream every token through the coordinator's relay for a whole
//! generation. libp2p's defaults (2 minutes, 128 KiB per circuit, 16 circuits) are
//! sized for a brief hand-off before a hole punch and cut those streams off.

use libp2p::multiaddr::Protocol;
use libp2p::{Multiaddr, PeerId};
use std::time::Duration;

pub fn server_config() -> libp2p::relay::Config {
    libp2p::relay::Config {
        max_reservations: 128,
        max_reservations_per_peer: 4,
        reservation_duration: Duration::from_secs(3600),
        // Matches the reservations: a peer granted one can open a circuit.
        max_circuits: 128,
        max_circuits_per_peer: 8,
        max_circuit_duration: Duration::from_secs(3600),
        max_circuit_bytes: 1 << 30,
        ..Default::default()
    }
}

/// The address to listen on to be reachable through `relay` (`.../p2p/<relay>`).
pub fn circuit_listen_addr(relay: &Multiaddr) -> Multiaddr {
    relay.clone().with(Protocol::P2pCircuit)
}

pub fn is_circuit(addr: &Multiaddr) -> bool {
    addr.iter().any(|p| matches!(p, Protocol::P2pCircuit))
}

/// The peer an address leads to: the *last* `/p2p/` component. A circuit address is
/// `.../p2p/<relay>/p2p-circuit/p2p/<target>`; the first one is the relay.
pub fn target_peer(addr: &Multiaddr) -> Option<PeerId> {
    addr.iter().filter_map(|p| if let Protocol::P2p(id) = p { Some(id) } else { None }).last()
}

/// Whether an address can be dialled directly (has a network hop, no circuit).
/// Filters what we put into kademlia: a relay-carried inbound connection reports a
/// bare `/p2p/<peer>` send-back address that is no path at all.
pub fn is_direct(addr: &Multiaddr) -> bool {
    let mut hop = false;
    for p in addr.iter() {
        match p {
            Protocol::P2pCircuit => return false,
            Protocol::Ip4(_) | Protocol::Ip6(_) | Protocol::Dns(_) | Protocol::Dns4(_) | Protocol::Dns6(_) | Protocol::Dnsaddr(_) => hop = true,
            _ => {}
        }
    }
    hop
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_target_of_a_circuit_is_the_last_peer() {
        let relay = PeerId::random();
        let target = PeerId::random();
        let a: Multiaddr = format!("/ip4/1.2.3.4/tcp/1/p2p/{relay}/p2p-circuit/p2p/{target}").parse().unwrap();
        assert_eq!(target_peer(&a), Some(target));
        assert!(is_circuit(&a) && !is_direct(&a));
        let d: Multiaddr = format!("/ip4/1.2.3.4/udp/1/quic-v1/p2p/{relay}").parse().unwrap();
        assert!(is_direct(&d) && target_peer(&d) == Some(relay));
        assert!(!is_direct(&format!("/p2p/{relay}").parse().unwrap()));
    }
}
