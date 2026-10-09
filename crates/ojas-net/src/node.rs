//! The node: a libp2p Swarm in one task, a cloneable [`Node`] handle, and
//! [`Event`]s for the application.
//!
//! The swarm builder (TCP + QUIC, DNS, relay client, a large per-connection event
//! buffer) is adapted from SwarmLLM (MIT OR Apache-2.0),
//! `src/network/manager/mod.rs` @ b14482d; peer gating on identify's protocol name
//! from their `src/network/manager/identify.rs` (`peer_speaks_swarmllm`). The state
//! here is small on purpose: who is a member, which requests are in flight.
//!
//! Membership gates everything that reaches the application. An inbound RPC or
//! stream from a peer whose credential has not verified yet waits (up to the
//! identify deadline) rather than being refused, because the first request on a new
//! connection routinely arrives before that peer's identify does.

use crate::behaviour::{self, Behaviour, BehaviourEvent};
use crate::codec::{Request, Response};
use crate::discovery;
use crate::pool::{self, Credential, Pool};
use crate::relay;
use crate::stream::PeerStream;
use anyhow::{anyhow, bail, Context, Result};
use futures::StreamExt;
use libp2p::gossipsub::{self, IdentTopic, MessageAcceptance};
use libp2p::identity::Keypair;
use libp2p::request_response::{self, OutboundRequestId, ResponseChannel};
use libp2p::swarm::dial_opts::{DialOpts, PeerCondition};
use libp2p::swarm::SwarmEvent;
use libp2p::{identify, kad, mdns, noise, tcp, yamux, Multiaddr, PeerId, StreamProtocol, Swarm};
use ojas_swarm_proto::peer::{Announce, PeerReq, PeerResp, AGENT_PROTOCOL, PROTO_STREAM, TOPIC_NODES};
use serde::Serialize;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot, Notify};

pub struct NetConfig {
    pub key: Keypair,
    pub pool: Pool,
    pub credential: Credential,
    pub listen: Vec<Multiaddr>,
    pub bootstrap: Vec<Multiaddr>,
    /// Relays to listen through (`.../p2p/<relay>`), for a node without inbound
    /// reachability. The coordinator is the natural one.
    pub relays: Vec<Multiaddr>,
    /// Addresses to advertise as reachable, e.g. a port-forwarded public address.
    pub external: Vec<Multiaddr>,
    pub relay_server: bool,
    pub mdns: bool,
    pub autonat: bool,
    pub dcutr: bool,
    pub max_connections: u32,
    pub engine_version: String,
    /// How long a new connection may go without a verified credential.
    pub identify_deadline: Duration,
    pub gossip_heartbeat: Duration,
}

impl NetConfig {
    pub fn new(key: Keypair, pool: Pool, credential: Credential) -> NetConfig {
        NetConfig {
            key,
            pool,
            credential,
            listen: vec!["/ip4/0.0.0.0/tcp/0".parse().unwrap(), "/ip4/0.0.0.0/udp/0/quic-v1".parse().unwrap()],
            bootstrap: vec![],
            relays: vec![],
            external: vec![],
            relay_server: false,
            mdns: true,
            autonat: true,
            dcutr: true,
            max_connections: 256,
            engine_version: env!("CARGO_PKG_VERSION").into(),
            identify_deadline: Duration::from_secs(10),
            gossip_heartbeat: Duration::from_secs(1),
        }
    }
}

pub enum Event {
    /// A request from a verified member. Answer through `reply`; dropping it answers
    /// with an error.
    Rpc { peer: PeerId, req: PeerReq, payload: Vec<u8>, reply: Responder },
    /// A [`PROTO_STREAM`] opened by a verified member.
    Stream(PeerStream),
    Announce { peer: PeerId, announce: Announce },
    MemberUp(PeerId),
    MemberDown(PeerId),
    Rejected { peer: PeerId, reason: String },
}

pub struct Responder {
    cmd: mpsc::UnboundedSender<Command>,
    ch: Option<ResponseChannel<Response>>,
}

impl Responder {
    pub fn send(mut self, resp: PeerResp, payload: Vec<u8>) {
        if let Some(ch) = self.ch.take() {
            let _ = self.cmd.send(Command::Respond(ch, (resp, payload)));
        }
    }
}

impl Drop for Responder {
    fn drop(&mut self) {
        if let Some(ch) = self.ch.take() {
            let r = PeerResp::Error { message: "request was not handled".into(), retry: true };
            let _ = self.cmd.send(Command::Respond(ch, (r, Vec::new())));
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct PeerView {
    pub peer: String,
    pub member: bool,
    pub agent: String,
    pub addrs: Vec<String>,
}

enum Command {
    Rpc(PeerId, Request, oneshot::Sender<Result<Response>>),
    Respond(ResponseChannel<Response>, Response),
    Publish(Vec<u8>),
    Dial(Multiaddr),
    ListenAddrs(oneshot::Sender<Vec<Multiaddr>>),
    Peers(oneshot::Sender<Vec<PeerView>>),
}

#[derive(Clone, Copy, PartialEq)]
enum Status {
    Member { expires: Option<u64> },
    Rejected,
}

/// Who has proven membership. Shared with the tasks that hold inbound work until
/// its sender verifies.
#[derive(Default)]
struct Members {
    state: Mutex<HashMap<PeerId, Status>>,
    changed: Notify,
}

impl Members {
    fn get(&self, p: &PeerId) -> Option<Status> {
        self.state.lock().unwrap().get(p).copied()
    }
    fn is(&self, p: &PeerId) -> bool {
        matches!(self.get(p), Some(Status::Member { .. }))
    }
    fn set(&self, p: PeerId, s: Status) -> Option<Status> {
        let old = self.state.lock().unwrap().insert(p, s);
        self.changed.notify_waiters();
        old
    }
    fn forget(&self, p: &PeerId) -> Option<Status> {
        let mut st = self.state.lock().unwrap();
        // A rejection outlives the connection; membership must be shown again.
        let old = match st.get(p) {
            Some(Status::Member { .. }) => st.remove(p),
            other => other.copied(),
        };
        drop(st);
        self.changed.notify_waiters();
        old
    }
    fn list(&self) -> Vec<PeerId> {
        self.state.lock().unwrap().iter().filter(|(_, s)| matches!(s, Status::Member { .. })).map(|(p, _)| *p).collect()
    }

    async fn wait(&self, p: &PeerId, timeout: Duration) -> bool {
        let until = tokio::time::Instant::now() + timeout;
        loop {
            let notified = self.changed.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            match self.get(p) {
                Some(Status::Member { .. }) => return true,
                Some(Status::Rejected) => return false,
                None => {}
            }
            if tokio::time::timeout_at(until, notified).await.is_err() {
                return false;
            }
        }
    }
}

#[derive(Clone)]
pub struct Node {
    me: PeerId,
    cmd: mpsc::UnboundedSender<Command>,
    control: libp2p_stream::Control,
    members: Arc<Members>,
    deadline: Duration,
}

impl Node {
    /// Build the swarm, start listening and dialling, and return the handle and the
    /// application's event stream.
    pub async fn start(cfg: NetConfig) -> Result<(Node, mpsc::Receiver<Event>)> {
        let me = cfg.key.public().to_peer_id();
        cfg.pool.admits(&me, &cfg.credential, crate::unix_now()).context("this node's own credential does not admit it to the pool")?;
        let opts = behaviour::Options {
            agent_version: pool::agent_version(&cfg.engine_version, &cfg.pool, &cfg.credential),
            relay_server: cfg.relay_server,
            mdns: cfg.mdns,
            autonat: cfg.autonat,
            dcutr: cfg.dcutr,
            max_connections: cfg.max_connections,
            heartbeat: cfg.gossip_heartbeat,
        };
        let mut swarm = libp2p::SwarmBuilder::with_existing_identity(cfg.key.clone())
            .with_tokio()
            // yamux defaults on purpose: the deprecated window setters fall back to
            // yamux 0.12, which SwarmLLM measured stalling substream opens for ~30 s.
            .with_tcp(tcp::Config::default().nodelay(true), noise::Config::new, yamux::Config::default)?
            .with_quic()
            .with_dns()?
            .with_relay_client(noise::Config::new, yamux::Config::default)?
            .with_behaviour(|key, relay| behaviour::build(key, relay, &opts).map_err(|e| e.into()))
            .map_err(|e| anyhow!("behaviour: {e}"))?
            .with_swarm_config(|c| {
                c.with_idle_connection_timeout(Duration::from_secs(3600))
                    // The default of 7 fills in the post-connect burst of identify,
                    // kademlia and gossipsub events and blocks the connection task.
                    .with_per_connection_event_buffer_size(64)
                    .with_notify_handler_buffer_size(std::num::NonZeroUsize::new(256).unwrap())
            })
            .build();

        let topic = IdentTopic::new(TOPIC_NODES);
        swarm.behaviour_mut().gossipsub.subscribe(&topic)?;
        for a in &cfg.listen {
            swarm.listen_on(a.clone()).with_context(|| format!("listening on {a}"))?;
        }
        for a in &cfg.external {
            swarm.add_external_address(a.clone());
        }

        let mut control = swarm.behaviour().stream.new_control();
        let incoming = control.accept(StreamProtocol::new(PROTO_STREAM)).map_err(|e| anyhow!("{e}"))?;
        let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();
        let (ev_tx, ev_rx) = mpsc::channel(1024);
        let members = Arc::new(Members::default());
        let node = Node { me, cmd: cmd_tx.clone(), control, members: members.clone(), deadline: cfg.identify_deadline };

        tokio::spawn(accept_streams(incoming, members.clone(), ev_tx.clone(), cfg.identify_deadline));
        let driver = Driver {
            swarm,
            me,
            pool: cfg.pool,
            topic,
            cmd_tx,
            events: ev_tx,
            members,
            pending: HashMap::new(),
            unverified: HashMap::new(),
            agents: HashMap::new(),
            bootstrap: discovery::plan(&cfg.bootstrap),
            relays: cfg.relays.into_iter().map(|r| (r, None)).collect(),
            deadline: cfg.identify_deadline,
        };
        tokio::spawn(driver.run(cmd_rx));
        Ok((node, ev_rx))
    }

    pub fn peer_id(&self) -> PeerId {
        self.me
    }

    pub fn is_member(&self, p: &PeerId) -> bool {
        self.members.is(p)
    }

    pub fn members(&self) -> Vec<PeerId> {
        self.members.list()
    }

    /// Wait until `p` has verified, or the identify deadline passes.
    pub async fn wait_member(&self, p: &PeerId) -> bool {
        self.members.wait(p, self.deadline).await
    }

    pub async fn rpc(&self, peer: PeerId, req: PeerReq, payload: Vec<u8>) -> Result<(PeerResp, Vec<u8>)> {
        let (tx, rx) = oneshot::channel();
        self.cmd.send(Command::Rpc(peer, (req, payload), tx)).map_err(|_| anyhow!("node stopped"))?;
        rx.await.map_err(|_| anyhow!("node stopped"))?
    }

    /// Open a [`PROTO_STREAM`] to a member.
    pub async fn open_stream(&self, peer: PeerId) -> Result<PeerStream> {
        if !self.members.is(&peer) {
            bail!("{peer} is not a verified member");
        }
        let s = self.control.clone().open_stream(peer, StreamProtocol::new(PROTO_STREAM)).await.map_err(|e| anyhow!("opening stream to {peer}: {e}"))?;
        Ok(PeerStream::new(peer, s))
    }

    pub fn publish(&self, a: &Announce) -> Result<()> {
        self.cmd.send(Command::Publish(serde_json::to_vec(a)?)).map_err(|_| anyhow!("node stopped"))
    }

    pub fn dial(&self, addr: Multiaddr) -> Result<()> {
        self.cmd.send(Command::Dial(addr)).map_err(|_| anyhow!("node stopped"))
    }

    pub async fn listen_addrs(&self) -> Vec<Multiaddr> {
        let (tx, rx) = oneshot::channel();
        let _ = self.cmd.send(Command::ListenAddrs(tx));
        rx.await.unwrap_or_default()
    }

    pub async fn peers(&self) -> Vec<PeerView> {
        let (tx, rx) = oneshot::channel();
        let _ = self.cmd.send(Command::Peers(tx));
        rx.await.unwrap_or_default()
    }
}

async fn accept_streams(mut incoming: libp2p_stream::IncomingStreams, members: Arc<Members>, ev: mpsc::Sender<Event>, deadline: Duration) {
    while let Some((peer, s)) = incoming.next().await {
        let (members, ev) = (members.clone(), ev.clone());
        tokio::spawn(async move {
            if members.wait(&peer, deadline).await {
                let _ = ev.send(Event::Stream(PeerStream::new(peer, s))).await;
            } else {
                tracing::debug!(%peer, "dropping stream from a non-member");
            }
        });
    }
}

struct Driver {
    swarm: Swarm<Behaviour>,
    me: PeerId,
    pool: Pool,
    topic: IdentTopic,
    cmd_tx: mpsc::UnboundedSender<Command>,
    events: mpsc::Sender<Event>,
    members: Arc<Members>,
    pending: HashMap<OutboundRequestId, oneshot::Sender<Result<Response>>>,
    /// Connected peers whose credential has not verified, and since when.
    unverified: HashMap<PeerId, Instant>,
    agents: HashMap<PeerId, (String, Vec<Multiaddr>)>,
    bootstrap: Vec<discovery::Dial>,
    /// Relay -> the circuit listener we hold through it.
    relays: Vec<(Multiaddr, Option<libp2p::core::transport::ListenerId>)>,
    deadline: Duration,
}

impl Driver {
    async fn run(mut self, mut cmds: mpsc::UnboundedReceiver<Command>) {
        self.dial_bootstrap();
        let mut tick = tokio::time::interval(Duration::from_secs(1));
        let mut ticks: u64 = 0;
        loop {
            tokio::select! {
                ev = self.swarm.select_next_some() => self.on_swarm(ev),
                c = cmds.recv() => match c {
                    Some(c) => self.on_command(c),
                    None => return,
                },
                _ = tick.tick() => {
                    ticks += 1;
                    self.on_tick(ticks);
                }
            }
        }
    }

    fn emit(&self, e: Event) {
        if self.events.try_send(e).is_err() {
            tracing::warn!("application event queue full; dropping a network event");
        }
    }

    fn dial_bootstrap(&mut self) {
        for d in &self.bootstrap {
            let opts = match d.peer {
                Some(p) if p == self.me => continue,
                Some(p) if self.swarm.is_connected(&p) => continue,
                Some(p) => DialOpts::peer_id(p).addresses(d.addrs.clone()).condition(PeerCondition::DisconnectedAndNotDialing).build(),
                None => DialOpts::unknown_peer_id().address(d.addrs[0].clone()).build(),
            };
            if let Err(e) = self.swarm.dial(opts) {
                tracing::debug!("bootstrap dial {:?}: {e}", d.addrs);
            }
        }
    }

    fn on_tick(&mut self, ticks: u64) {
        let now = Instant::now();
        let late: Vec<PeerId> = self.unverified.iter().filter(|(_, t)| now.duration_since(**t) > self.deadline).map(|(p, _)| *p).collect();
        for p in late {
            self.unverified.remove(&p);
            if !self.members.is(&p) {
                tracing::info!(peer = %p, "no verified credential within {:?}; disconnecting", self.deadline);
                let _ = self.swarm.disconnect_peer_id(p);
            }
        }
        let unix = crate::unix_now();
        let expired: Vec<PeerId> = self.members.state.lock().unwrap().iter()
            .filter(|(_, s)| matches!(s, Status::Member { expires: Some(e) } if *e <= unix)).map(|(p, _)| *p).collect();
        for p in expired {
            self.reject(p, "invite expired".into());
        }
        // Bootstrap peers that dropped are redialled: a member that lost its only
        // link (coordinator restart) must find its way back by itself.
        if ticks.is_multiple_of(10) {
            self.dial_bootstrap();
        }
        if ticks % 30 == 1 {
            self.relisten_relays();
        }
        if ticks % 60 == 5 && !self.members.list().is_empty() {
            let _ = self.swarm.behaviour_mut().kademlia.bootstrap();
        }
    }

    fn relisten_relays(&mut self) {
        for i in 0..self.relays.len() {
            if self.relays[i].1.is_some() {
                continue;
            }
            let addr = relay::circuit_listen_addr(&self.relays[i].0);
            match self.swarm.listen_on(addr.clone()) {
                Ok(id) => self.relays[i].1 = Some(id),
                Err(e) => tracing::warn!("listening through relay {addr}: {e}"),
            }
        }
    }

    fn on_command(&mut self, c: Command) {
        match c {
            Command::Rpc(peer, req, tx) => {
                if !self.members.is(&peer) {
                    let _ = tx.send(Err(anyhow!("{peer} is not a verified member")));
                    return;
                }
                let id = self.swarm.behaviour_mut().rpc.send_request(&peer, req);
                self.pending.insert(id, tx);
            }
            Command::Respond(ch, r) => {
                if self.swarm.behaviour_mut().rpc.send_response(ch, r).is_err() {
                    tracing::debug!("response not sent: the requester went away");
                }
            }
            Command::Publish(b) => {
                if let Err(e) = self.swarm.behaviour_mut().gossipsub.publish(self.topic.clone(), b) {
                    tracing::debug!("announce not published: {e}");
                }
            }
            Command::Dial(a) => {
                if let Err(e) = self.swarm.dial(a.clone()) {
                    tracing::warn!("dial {a}: {e}");
                }
            }
            Command::ListenAddrs(tx) => {
                let mut v: Vec<Multiaddr> = self.swarm.listeners().cloned().collect();
                v.extend(self.swarm.external_addresses().cloned());
                let _ = tx.send(v);
            }
            Command::Peers(tx) => {
                let v = self
                    .swarm
                    .connected_peers()
                    .map(|p| {
                        let (agent, addrs) = self.agents.get(p).cloned().unwrap_or_default();
                        PeerView {
                            peer: p.to_string(),
                            member: self.members.is(p),
                            agent: agent.split_whitespace().next().unwrap_or("").to_string(),
                            addrs: addrs.iter().map(|a| a.to_string()).collect(),
                        }
                    })
                    .collect();
                let _ = tx.send(v);
            }
        }
    }

    fn reject(&mut self, peer: PeerId, reason: String) {
        tracing::warn!(%peer, "refusing peer: {reason}");
        self.members.set(peer, Status::Rejected);
        self.unverified.remove(&peer);
        self.swarm.behaviour_mut().blocked.block_peer(peer);
        self.swarm.behaviour_mut().kademlia.remove_peer(&peer);
        let _ = self.swarm.disconnect_peer_id(peer);
        self.emit(Event::Rejected { peer, reason });
    }

    fn on_identify(&mut self, peer: PeerId, info: identify::Info) {
        // A foreign libp2p node is not an ojas node however healthy its connection;
        // completing identify proves nothing.
        if info.protocol_version != AGENT_PROTOCOL {
            return self.reject(peer, format!("speaks {:?}, not {AGENT_PROTOCOL}", info.protocol_version));
        }
        let verdict = pool::parse_agent_version(&info.agent_version).and_then(|(name, cred)| {
            if name != self.pool.name {
                bail!("belongs to pool {name:?}");
            }
            self.pool.admits(&peer, &cred, crate::unix_now())?;
            Ok(cred.expires())
        });
        let expires = match verdict {
            Ok(e) => e,
            Err(e) => return self.reject(peer, e.to_string()),
        };
        self.unverified.remove(&peer);
        let fresh = !matches!(self.members.set(peer, Status::Member { expires }), Some(Status::Member { .. }));
        for a in info.listen_addrs.iter().filter(|a| relay::is_direct(a)) {
            self.swarm.behaviour_mut().kademlia.add_address(&peer, a.clone());
        }
        self.agents.insert(peer, (info.agent_version, info.listen_addrs));
        if fresh {
            tracing::info!(%peer, "member verified");
            self.emit(Event::MemberUp(peer));
            // Small pools: learn the rest of the membership from this one.
            self.swarm.behaviour_mut().kademlia.get_closest_peers(self.me);
        }
    }

    fn on_swarm(&mut self, ev: SwarmEvent<BehaviourEvent>) {
        match ev {
            SwarmEvent::NewListenAddr { address, .. } => tracing::info!("listening on {address}/p2p/{}", self.me),
            SwarmEvent::ListenerClosed { listener_id, reason, .. } => {
                if let Some(r) = self.relays.iter_mut().find(|r| r.1 == Some(listener_id)) {
                    tracing::warn!("relay listener via {} closed: {reason:?}", r.0);
                    r.1 = None;
                }
            }
            SwarmEvent::ConnectionEstablished { peer_id, num_established, .. } => {
                if num_established.get() == 1 && self.members.get(&peer_id).is_none() {
                    self.unverified.insert(peer_id, Instant::now());
                }
            }
            SwarmEvent::ConnectionClosed { peer_id, num_established: 0, .. } => {
                self.unverified.remove(&peer_id);
                self.agents.remove(&peer_id);
                if matches!(self.members.forget(&peer_id), Some(Status::Member { .. })) {
                    self.emit(Event::MemberDown(peer_id));
                }
            }
            SwarmEvent::Behaviour(b) => self.on_behaviour(b),
            _ => {}
        }
    }

    fn on_behaviour(&mut self, ev: BehaviourEvent) {
        match ev {
            BehaviourEvent::Rpc(e) => self.on_rpc(e),
            BehaviourEvent::Identify(identify::Event::Received { peer_id, info, .. }) => self.on_identify(peer_id, info),
            BehaviourEvent::Gossipsub(gossipsub::Event::Message { propagation_source, message_id, message }) => {
                // Members validate before they forward, so a message a member hands
                // us came from a member even when we have no link to its origin.
                let ok = self.members.is(&propagation_source);
                let announce = ok.then(|| serde_json::from_slice::<Announce>(&message.data).ok()).flatten();
                let verdict = match (&announce, ok) {
                    (Some(_), _) => MessageAcceptance::Accept,
                    (None, true) => MessageAcceptance::Reject,
                    // Not verified *yet* is the common case on a new connection.
                    (None, false) => MessageAcceptance::Ignore,
                };
                self.swarm.behaviour_mut().gossipsub.report_message_validation_result(&message_id, &propagation_source, verdict);
                if let (Some(announce), Some(src)) = (announce, message.source) {
                    if src != self.me {
                        self.emit(Event::Announce { peer: src, announce });
                    }
                }
            }
            BehaviourEvent::Kademlia(kad::Event::OutboundQueryProgressed { result: kad::QueryResult::GetClosestPeers(Ok(r)), .. }) => {
                for p in r.peers {
                    if p.peer_id == self.me || self.swarm.is_connected(&p.peer_id) || self.members.get(&p.peer_id) == Some(Status::Rejected) {
                        continue;
                    }
                    let addrs: Vec<Multiaddr> = p.addrs.into_iter().filter(relay::is_direct).collect();
                    let opts = DialOpts::peer_id(p.peer_id).addresses(addrs).condition(PeerCondition::DisconnectedAndNotDialing).build();
                    let _ = self.swarm.dial(opts);
                }
            }
            BehaviourEvent::Mdns(mdns::Event::Discovered(found)) => {
                for (p, a) in found {
                    if self.members.get(&p) == Some(Status::Rejected) || self.swarm.is_connected(&p) || !discovery::mdns_should_dial(&self.me, &p) {
                        continue;
                    }
                    let _ = self.swarm.dial(DialOpts::peer_id(p).addresses(vec![a]).condition(PeerCondition::DisconnectedAndNotDialing).build());
                }
            }
            BehaviourEvent::RelayClient(e) => tracing::info!("relay client: {e:?}"),
            BehaviourEvent::RelayServer(e) => tracing::debug!("relay server: {e:?}"),
            BehaviourEvent::Dcutr(e) => tracing::info!("dcutr: {e:?}"),
            BehaviourEvent::AutonatClient(e) => tracing::debug!("autonat: {e:?}"),
            _ => {}
        }
    }

    fn on_rpc(&mut self, ev: request_response::Event<Request, Response>) {
        use request_response::{Event as E, Message};
        match ev {
            E::Message { peer, message: Message::Request { request: (req, payload), channel, .. }, .. } => {
                if self.members.get(&peer) == Some(Status::Rejected) {
                    return;
                }
                let (members, events, cmd, deadline) = (self.members.clone(), self.events.clone(), self.cmd_tx.clone(), self.deadline);
                tokio::spawn(async move {
                    let reply = Responder { cmd, ch: Some(channel) };
                    if members.wait(&peer, deadline).await {
                        let _ = events.send(Event::Rpc { peer, req, payload, reply }).await;
                    } else {
                        reply.send(PeerResp::Error { message: "not a member of this pool".into(), retry: false }, Vec::new());
                    }
                });
            }
            E::Message { message: Message::Response { request_id, response }, .. } => {
                if let Some(tx) = self.pending.remove(&request_id) {
                    let _ = tx.send(Ok(response));
                }
            }
            E::OutboundFailure { peer, request_id, error, .. } => {
                if let Some(tx) = self.pending.remove(&request_id) {
                    let _ = tx.send(Err(anyhow!("rpc to {peer}: {error}")));
                }
            }
            E::InboundFailure { peer, error, .. } => tracing::debug!(%peer, "inbound rpc failed: {error}"),
            E::ResponseSent { .. } => {}
        }
    }
}
