//! Nodes on loopback: members verify each other and talk; an impostor is cut off.

use libp2p::identity::Keypair;
use ojas_net::{blob, Credential, Event, Multiaddr, NetConfig, Node, Pool};
use ojas_swarm_proto::peer::{Announce, PeerReq, PeerResp};
use ojas_swarm_proto::{BlobId, Finish, GenerateReq, ModelId, ReqId, Sampling};
use std::time::Duration;
use tokio::sync::mpsc;

fn cfg(key: Keypair, pool: &Pool, cred: Credential, boot: Vec<Multiaddr>) -> NetConfig {
    let mut c = NetConfig::new(key, pool.clone(), cred);
    c.listen = vec!["/ip4/127.0.0.1/tcp/0".parse().unwrap()];
    c.bootstrap = boot;
    c.mdns = false;
    c.autonat = false;
    c.identify_deadline = Duration::from_secs(3);
    c
}

async fn addr_of(n: &Node) -> Multiaddr {
    for _ in 0..100 {
        if let Some(a) = n.listen_addrs().await.into_iter().next() {
            return a.with(libp2p::multiaddr::Protocol::P2p(n.peer_id()));
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("no listen address")
}

async fn until<T>(rx: &mut mpsc::Receiver<Event>, mut f: impl FnMut(Event) -> Option<T>) -> T {
    tokio::time::timeout(Duration::from_secs(15), async {
        loop {
            if let Some(t) = f(rx.recv().await.expect("node stopped")) {
                return t;
            }
        }
    })
    .await
    .expect("timed out")
}

/// Serve one node's events: echo RPCs, stream three tokens per Generate.
fn serve(mut rx: mpsc::Receiver<Event>, announces: mpsc::UnboundedSender<Announce>) {
    tokio::spawn(async move {
        while let Some(ev) = rx.recv().await {
            match ev {
                Event::Rpc { req, payload, reply, .. } => match req {
                    PeerReq::BlobGet { .. } | PeerReq::BlobPut { .. } => reply.send(PeerResp::Error { message: "no blobs".into(), retry: false }, vec![]),
                    _ => reply.send(PeerResp::Ok, payload),
                },
                Event::Stream(mut s) => {
                    tokio::spawn(async move {
                        let Some((PeerReq::Generate(g), _)) = s.recv::<PeerReq>().await.unwrap() else { return };
                        for t in 0..3 {
                            s.send(&PeerResp::Token { req: g.req, token: t, text: None }, &[]).await.unwrap();
                        }
                        s.send(&PeerResp::Done { req: g.req, prompt_tokens: 1, completion_tokens: 3, finish: Finish::Length, backend: ojas_swarm_proto::Backend::Cpu }, &[]).await.unwrap();
                    });
                }
                Event::Announce { announce, .. } => {
                    let _ = announces.send(announce);
                }
                _ => {}
            }
        }
    });
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn members_talk_and_an_impostor_is_refused() {
    let _ = tracing_subscriber::fmt().with_env_filter(tracing_subscriber::EnvFilter::from_default_env()).with_test_writer().try_init();
    let admin = Keypair::generate_ed25519();
    let pool = Pool::new("t", &admin.public()).unwrap();
    let (a, mut a_rx) = Node::start(cfg(admin.clone(), &pool, Credential::Admin, vec![])).await.unwrap();
    let a_addr = addr_of(&a).await;

    let bk = Keypair::generate_ed25519();
    let inv = pool.invite(&admin, bk.public().to_peer_id(), u64::MAX).unwrap();
    let (b, mut b_rx) = Node::start(cfg(bk, &pool, Credential::Invite(inv), vec![a_addr.clone()])).await.unwrap();
    until(&mut a_rx, |e| matches!(e, Event::MemberUp(p) if p == b.peer_id()).then_some(())).await;
    until(&mut b_rx, |e| matches!(e, Event::MemberUp(p) if p == a.peer_id()).then_some(())).await;

    // An impostor pool with the same name: its admin signed its own invite.
    let fake_admin = Keypair::generate_ed25519();
    let fake = Pool::new("t", &fake_admin.public()).unwrap();
    let dk = Keypair::generate_ed25519();
    let finv = fake.invite(&fake_admin, dk.public().to_peer_id(), u64::MAX).unwrap();
    let (d, _d_rx) = Node::start(cfg(dk, &fake, Credential::Invite(finv), vec![a_addr])).await.unwrap();
    let d_id = d.peer_id();
    // Each side refuses the other; whichever identify lands first ends the
    // connection, so A may see a rejection or only a closed link. Either way D
    // never becomes a member.
    tokio::time::sleep(Duration::from_secs(1)).await;
    assert!(!a.is_member(&d_id) && !d.is_member(&a.peer_id()));
    assert!(d.rpc(a.peer_id(), PeerReq::TrainLeave { run: "r".into(), member: 0 }, vec![]).await.is_err());

    let (atx, mut arx) = mpsc::unbounded_channel();
    serve(a_rx, atx);
    let (btx, _brx) = mpsc::unbounded_channel();
    serve(b_rx, btx);

    let (r, p) = b.rpc(a.peer_id(), PeerReq::TrainLeave { run: "r".into(), member: 0 }, b"echo".to_vec()).await.unwrap();
    assert_eq!((r, p), (PeerResp::Ok, b"echo".to_vec()));

    let mut s = b.open_stream(a.peer_id()).await.unwrap();
    let g = GenerateReq { req: ReqId(9), model: ModelId([1; 32]), prompt: vec![1], max_tokens: 3, sampling: Sampling::greedy(), want_text: false };
    s.send(&PeerReq::Generate(g), &[]).await.unwrap();
    let mut toks = vec![];
    while let Some((m, _)) = s.recv::<PeerResp>().await.unwrap() {
        match m {
            PeerResp::Token { token, .. } => toks.push(token),
            PeerResp::Done { .. } => break,
            other => panic!("{other:?}"),
        }
    }
    assert_eq!(toks, [0, 1, 2]);

    let ann = Announce { proto: 1, features: 1, engine_version: "x".into(), workers: vec![], models: vec![], busy: 2, unix_ms: 1, relays: vec![] };
    let got = tokio::time::timeout(Duration::from_secs(15), async {
        loop {
            b.publish(&ann).unwrap();
            if let Ok(Some(a)) = tokio::time::timeout(Duration::from_millis(500), arx.recv()).await {
                return a;
            }
        }
    })
    .await
    .unwrap();
    assert_eq!(got.busy, 2);

    let err = blob::get(&b, a.peer_id(), BlobId([0; 32]), 10).await.unwrap_err();
    assert!(err.to_string().contains("no blobs"), "{err}");
}
