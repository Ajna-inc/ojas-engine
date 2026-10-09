//! The combined libp2p behaviour.
//!
//! Adapted from SwarmLLM (MIT OR Apache-2.0), `src/network/behaviour.rs` @ b14482d,
//! keeping the lessons that cost them field time:
//!
//! * `request_response` is field 0. The derive polls fields in order, and a
//!   connection returns from its poll on every behaviour event: kademlia and
//!   gossipsub polled first starve its outbound substream requests.
//! * kademlia is forced to `Mode::Server`. Two client-mode peers reject each
//!   other's queries, and the rejected negotiations flood the per-connection event
//!   channel until `SendRequest` commands stop reaching the handler.
//! * gossipsub message ids are BLAKE3 of the data (and source), never
//!   `DefaultHasher`, which is seeded per process and breaks deduplication.
//! * `max_established_per_peer` is 3, not 1: DCUtR dials the direct connection while
//!   the relayed one is still open, so a cap of 1 silently disables hole punching.

use crate::codec::RpcCodec;
use libp2p::kad::store::MemoryStore;
use libp2p::swarm::behaviour::toggle::Toggle;
use libp2p::swarm::NetworkBehaviour;
use libp2p::{
    allow_block_list, autonat, connection_limits, dcutr, gossipsub, identify, identity::Keypair, kad, mdns, relay,
    request_response, StreamProtocol,
};
use ojas_swarm_proto::peer::{AGENT_PROTOCOL, PROTO_RPC};
use std::time::Duration;

pub const KAD_PROTOCOL: &str = "/ojas/kad/1.0.0";
/// Training rounds and blob chunks can take minutes on a CPU member; the token
/// stream is not on this path.
pub const RPC_TIMEOUT: Duration = Duration::from_secs(600);
pub const MAX_PER_PEER: u32 = 3;
/// Gossip carries Announce only: small.
const MAX_GOSSIP: usize = 256 << 10;

#[derive(NetworkBehaviour)]
pub struct Behaviour {
    pub rpc: request_response::Behaviour<RpcCodec>,
    pub kademlia: kad::Behaviour<MemoryStore>,
    pub gossipsub: gossipsub::Behaviour,
    pub identify: identify::Behaviour,
    pub stream: libp2p_stream::Behaviour,
    pub autonat_client: Toggle<autonat::v2::client::Behaviour>,
    pub autonat_server: Toggle<autonat::v2::server::Behaviour>,
    pub dcutr: Toggle<dcutr::Behaviour>,
    pub relay_client: relay::client::Behaviour,
    pub relay_server: Toggle<relay::Behaviour>,
    pub limits: connection_limits::Behaviour,
    /// Peers that failed the membership check. Blocking refuses both directions at
    /// the swarm, so an internal redial (kademlia, mDNS) cannot bring them back.
    pub blocked: allow_block_list::Behaviour<allow_block_list::BlockedPeers>,
    pub mdns: Toggle<mdns::tokio::Behaviour>,
}

pub struct Options {
    pub agent_version: String,
    pub relay_server: bool,
    pub mdns: bool,
    pub autonat: bool,
    pub dcutr: bool,
    pub max_connections: u32,
    pub heartbeat: Duration,
}

pub fn build(key: &Keypair, relay_client: relay::client::Behaviour, o: &Options) -> anyhow::Result<Behaviour> {
    let me = key.public().to_peer_id();

    let mut kad_cfg = kad::Config::new(StreamProtocol::new(KAD_PROTOCOL));
    kad_cfg.set_query_timeout(Duration::from_secs(30));
    let mut kademlia = kad::Behaviour::with_config(me, MemoryStore::new(me), kad_cfg);
    kademlia.set_mode(Some(kad::Mode::Server));

    let msg_id = |m: &gossipsub::Message| {
        let mut h = blake3::Hasher::new();
        h.update(&m.data);
        if let Some(src) = &m.source {
            h.update(&src.to_bytes());
        }
        gossipsub::MessageId::from(h.finalize().as_bytes()[..16].to_vec())
    };
    let gs_cfg = gossipsub::ConfigBuilder::default()
        .heartbeat_interval(o.heartbeat)
        .validation_mode(gossipsub::ValidationMode::Strict)
        // Hold every message until the node has checked its sender is a member.
        .validate_messages()
        .message_id_fn(msg_id)
        .max_transmit_size(MAX_GOSSIP)
        // Pools are small: a mesh of a handful of peers, like SwarmLLM's tiny tier.
        .mesh_n(3)
        .mesh_n_low(1)
        .mesh_n_high(6)
        .mesh_outbound_min(1)
        .build()
        .map_err(|e| anyhow::anyhow!("gossipsub config: {e}"))?;
    let gossipsub = gossipsub::Behaviour::new(gossipsub::MessageAuthenticity::Signed(key.clone()), gs_cfg)
        .map_err(|e| anyhow::anyhow!("gossipsub: {e}"))?;

    let rpc = request_response::Behaviour::with_codec(
        RpcCodec,
        [(StreamProtocol::new(PROTO_RPC), request_response::ProtocolSupport::Full)],
        request_response::Config::default().with_request_timeout(RPC_TIMEOUT),
    );

    let identify = identify::Behaviour::new(
        identify::Config::new(AGENT_PROTOCOL.to_string(), key.public())
            .with_agent_version(o.agent_version.clone())
            .with_push_listen_addr_updates(true),
    );

    let (ac, asrv) = if o.autonat {
        (Some(autonat::v2::client::Behaviour::default()), Some(autonat::v2::server::Behaviour::default()))
    } else {
        (None, None)
    };

    let relay_server = o.relay_server.then(|| relay::Behaviour::new(me, crate::relay::server_config()));

    let limits = connection_limits::Behaviour::new(
        connection_limits::ConnectionLimits::default()
            .with_max_established_per_peer(Some(MAX_PER_PEER))
            .with_max_established(Some(o.max_connections))
            .with_max_pending_incoming(Some(64)),
    );

    let mdns = if o.mdns {
        let cfg = mdns::Config { ttl: Duration::from_secs(300), query_interval: Duration::from_secs(10), enable_ipv6: false };
        Some(mdns::tokio::Behaviour::new(cfg, me)?)
    } else {
        None
    };

    Ok(Behaviour {
        rpc,
        kademlia,
        gossipsub,
        identify,
        stream: libp2p_stream::Behaviour::new(),
        autonat_client: ac.into(),
        autonat_server: asrv.into(),
        dcutr: o.dcutr.then(|| dcutr::Behaviour::new(me)).into(),
        relay_client,
        relay_server: relay_server.into(),
        limits,
        blocked: allow_block_list::Behaviour::default(),
        mdns: mdns.into(),
    })
}
