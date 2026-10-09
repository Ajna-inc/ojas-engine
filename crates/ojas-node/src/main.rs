//! `ojas-node` — the ojas swarm daemon. See README.md.

mod api;
mod app;
mod config;
mod coord;
mod ipc;
mod mock;
mod route;
mod text;
mod train;
mod worker;

use anyhow::{bail, Context, Result};
use app::{App, ModelEntry, Slot};
use clap::{Args, Parser, Subcommand};
use config::{Config, ModelCfg, TrainCfg};
use ojas_net::blob::BlobStore;
use ojas_net::{key, Credential, Invite, Multiaddr, NetConfig, Node, PeerId, Pool};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, AtomicU64};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

#[derive(Parser)]
#[command(name = "ojas-node", version, about = "ojas swarm node: decentralised inference and DiLoCo training")]
struct Cli {
    #[command(subcommand)]
    cmd: Option<Cmd>,
    #[command(flatten)]
    run: RunArgs,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run the daemon (the default).
    Run(Box<RunArgs>),
    /// Print this node's PeerId, creating its key if there is none.
    Id {
        #[arg(long)]
        key: Option<PathBuf>,
    },
    #[command(subcommand)]
    Pool(PoolCmd),
    /// The engine-worker IPC with no engine, for tests.
    #[command(hide = true)]
    MockWorker {
        #[arg(long)]
        socket: String,
        #[arg(long, default_value = "cpu")]
        device: String,
    },
}

#[derive(Subcommand)]
enum PoolCmd {
    /// Create a pool administered by this key (created if absent).
    Init {
        #[arg(long)]
        name: String,
        #[arg(long)]
        key: Option<PathBuf>,
        #[arg(long, default_value = "pool.json")]
        out: PathBuf,
    },
    /// Sign an invite for a member's PeerId and print the token.
    Invite {
        peer: String,
        #[arg(long, default_value_t = 30)]
        days: u64,
        #[arg(long)]
        key: Option<PathBuf>,
        #[arg(long, default_value = "pool.json")]
        pool: PathBuf,
    },
}

#[derive(Args, Default)]
struct RunArgs {
    /// TOML or JSON config; flags override it.
    #[arg(long, short)]
    config: Option<PathBuf>,
    #[arg(long)]
    key: Option<PathBuf>,
    #[arg(long)]
    pool: Option<PathBuf>,
    /// Invite token, or @file.
    #[arg(long)]
    invite: Option<String>,
    #[arg(long)]
    listen: Vec<String>,
    #[arg(long)]
    bootstrap: Vec<String>,
    /// Listen through this relay (multiaddr with /p2p/<relay>).
    #[arg(long)]
    relay: Vec<String>,
    #[arg(long)]
    device: Vec<String>,
    /// GGUF to load (and tokenise with).
    #[arg(long)]
    model: Vec<PathBuf>,
    #[arg(long)]
    api_port: Option<u16>,
    #[arg(long)]
    api_token_file: Option<PathBuf>,
    #[arg(long)]
    data_dir: Option<PathBuf>,
    #[arg(long)]
    worker_bin: Option<PathBuf>,
    /// Coordinate the DiLoCo run described by this run config.
    #[arg(long)]
    coordinator: Option<PathBuf>,
    /// Train as a member of a run: the coordinator's multiaddr incl. /p2p/<PeerId>.
    #[arg(long, requires = "run")]
    train: Option<String>,
    #[arg(long)]
    run: Option<String>,
    #[arg(long)]
    relay_server: bool,
    #[arg(long)]
    no_mdns: bool,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info,libp2p=warn".into()))
        .with_writer(std::io::stderr)
        .init();
    let cli = Cli::parse();
    let rt = tokio::runtime::Runtime::new()?;
    // `ojas-node --key k id` and `ojas-node id --key k` must name the same key: falling
    // back to the default here would silently print (or make an admin of) another one.
    let global_key = cli.run.key.clone();
    let key_or = |k: Option<PathBuf>| k.or_else(|| global_key.clone()).unwrap_or_else(default_key);
    match cli.cmd {
        None => rt.block_on(run(cli.run)),
        Some(Cmd::Run(a)) => rt.block_on(run(*a)),
        Some(Cmd::Id { key }) => {
            let kp = key::load_or_create(&key_or(key))?;
            println!("{}", kp.public().to_peer_id());
            Ok(())
        }
        Some(Cmd::Pool(PoolCmd::Init { name, key, out })) => {
            if out.exists() {
                bail!("{} exists; refusing to replace a pool", out.display());
            }
            let kp = key::load_or_create(&key_or(key))?;
            let pool = Pool::new(&name, &kp.public())?;
            pool.save(&out)?;
            eprintln!("pool {name:?} written to {}; this key is its admin", out.display());
            println!("{}", pool.admin);
            Ok(())
        }
        Some(Cmd::Pool(PoolCmd::Invite { peer, days, key, pool })) => {
            let kp = key::load(&key_or(key))?;
            let pool = Pool::load(&pool)?;
            let member: PeerId = peer.parse().context("not a PeerId")?;
            let inv = pool.invite(&kp, member, ojas_net::unix_now() + days * 86_400)?;
            println!("{}", inv.to_token());
            Ok(())
        }
        Some(Cmd::MockWorker { socket, device }) => rt.block_on(mock::run(&socket, &device)),
    }
}

fn home() -> PathBuf {
    std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")).map(PathBuf::from).unwrap_or_else(|| ".".into())
}

fn default_key() -> PathBuf {
    home().join(".ojas").join("node.key")
}

fn merge(a: RunArgs) -> Result<Config> {
    let mut c = match &a.config {
        Some(p) => Config::load(p)?,
        None => Config::default(),
    };
    macro_rules! set {
        ($f:ident) => {
            if a.$f.is_some() {
                c.$f = a.$f;
            }
        };
    }
    set!(key);
    set!(pool);
    set!(invite);
    set!(api_port);
    set!(api_token_file);
    set!(data_dir);
    set!(worker_bin);
    set!(coordinator);
    c.listen.extend(a.listen);
    c.bootstrap.extend(a.bootstrap);
    c.relays.extend(a.relay);
    c.devices.extend(a.device);
    c.models.extend(a.model.into_iter().map(|path| ModelCfg { path, ..Default::default() }));
    if let (Some(coordinator), Some(run)) = (a.train, a.run) {
        c.train = Some(TrainCfg { coordinator, run });
    }
    c.relay_server |= a.relay_server || c.coordinator.is_some();
    if a.no_mdns {
        c.mdns = Some(false);
    }
    if c.data_dir.is_none() {
        c.data_dir = Some(home().join(".ojas").join("node"));
    }
    if c.key.is_none() {
        c.key = Some(default_key());
    }
    Ok(c)
}

fn addrs(v: &[String], what: &str) -> Result<Vec<Multiaddr>> {
    v.iter().map(|s| s.parse().with_context(|| format!("{what} {s:?} is not a multiaddr"))).collect()
}

fn worker_bin(c: &Config) -> PathBuf {
    if let Some(b) = &c.worker_bin {
        return b.clone();
    }
    let name = format!("ojas{}", std::env::consts::EXE_SUFFIX);
    std::env::current_exe().ok().and_then(|e| e.parent().map(|d| d.join(&name))).filter(|p| p.exists()).unwrap_or_else(|| PathBuf::from(name))
}

/// The bearer token for the local API: read, or created 0600 on first run.
fn api_token(path: &Path) -> Result<String> {
    if let Ok(t) = std::fs::read_to_string(path) {
        if !t.trim().is_empty() {
            return Ok(t.trim().to_string());
        }
    }
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d)?;
    }
    let t: String = (0..32).map(|_| format!("{:02x}", rand::random::<u8>())).collect();
    let mut o = std::fs::OpenOptions::new();
    o.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        o.mode(0o600);
    }
    use std::io::Write;
    o.open(path)?.write_all(t.as_bytes())?;
    tracing::info!("api token written to {}", path.display());
    Ok(t)
}

async fn run(a: RunArgs) -> Result<()> {
    let cfg = merge(a)?;
    let data_dir = cfg.data_dir();
    std::fs::create_dir_all(&data_dir)?;
    let kp = key::load_or_create(&cfg.key_path())?;
    let me = kp.public().to_peer_id();
    let pool = Pool::load(cfg.pool.as_deref().context("no pool file: give --pool (create one with `ojas-node pool init`)")?)?;
    let cred = if me == pool.admin {
        Credential::Admin
    } else {
        let tok = cfg.invite_token()?.context("this node is not the pool admin: give --invite <token> (from `ojas-node pool invite`)")?;
        Credential::Invite(Invite::from_token(&tok)?)
    };

    let mut net = NetConfig::new(kp, pool.clone(), cred);
    if !cfg.listen.is_empty() {
        net.listen = addrs(&cfg.listen, "listen address")?;
    } else {
        net.listen = addrs(&["/ip4/0.0.0.0/tcp/4801".into(), "/ip4/0.0.0.0/udp/4801/quic-v1".into()], "listen")?;
    }
    net.bootstrap = addrs(&cfg.bootstrap, "bootstrap address")?;
    net.relays = addrs(&cfg.relays, "relay address")?;
    net.external = addrs(&cfg.external, "external address")?;
    net.relay_server = cfg.relay_server;
    net.mdns = cfg.mdns.unwrap_or(true);
    net.autonat = cfg.autonat.unwrap_or(true);
    let coordinator = match &cfg.coordinator {
        Some(p) => Some(Arc::new(Mutex::new(coord::open(p).with_context(|| format!("opening run config {}", p.display()))?))),
        None => None,
    };
    let (node, events) = Node::start(net).await?;
    tracing::info!("peer id {me}, pool {:?}", pool.name);

    let models: Vec<Arc<ModelEntry>> = cfg
        .models
        .iter()
        .map(|m| {
            let mut id = m.id.as_deref().map(|h| ojas_swarm_proto::ModelId::from_hex(h).with_context(|| format!("model id {h:?} is not 64 hex digits"))).transpose()?;
            // A model routed but not loaded here still needs its identity to be asked for
            // by id. Hash the file when it is present: one read, then cached.
            if id.is_none() && (!m.load.unwrap_or(true) || cfg.devices.is_empty()) && m.path.is_file() {
                let p = m.path.to_string_lossy();
                id = Some(ojas_formats::swarm_id::identity(&p, &mut |_, _| {}).with_context(|| format!("hashing {p}"))?.0.model);
            }
            Ok(Arc::new(ModelEntry { name: m.name(), path: m.path.clone(), load: m.load.unwrap_or(true), context: m.context.unwrap_or(4096), id: RwLock::new(id), tok: RwLock::new(None) }))
        })
        .collect::<Result<_>>()?;
    for m in &models {
        let m = m.clone();
        tokio::task::spawn_blocking(move || match text::Tok::open(&m.path) {
            Ok(t) => *m.tok.write().unwrap() = Some(Arc::new(t)),
            Err(e) => tracing::error!("tokenizer for {}: {e:#}", m.name),
        });
    }
    let slots: Vec<Arc<Slot>> = cfg.devices.iter().map(|d| Arc::new(Slot { device: d.clone(), cur: RwLock::new(None), restarts: AtomicU32::new(0) })).collect();
    let app = Arc::new(App {
        node,
        pool: pool.name.clone(),
        models,
        slots: slots.clone(),
        table: Mutex::new(Default::default()),
        tok_s: Mutex::new(Default::default()),
        next_req: AtomicU64::new(1),
        blobs: Mutex::new(BlobStore::new(1 << 30)),
        coordinator,
        train_status: Mutex::new(serde_json::Value::Null),
        announce_now: Default::default(),
        announce_every: Duration::from_secs(cfg.announce_secs.unwrap_or(5).max(1)),
        worker_up: Default::default(),
        cfg: cfg.clone(),
    });

    let bin = worker_bin(&cfg);
    let args = cfg.worker_args.clone().unwrap_or_else(|| vec!["engine-worker".into()]);
    for (i, s) in slots.iter().enumerate() {
        let spec = worker::Spec {
            bin: bin.clone(),
            args: args.clone(),
            env: cfg.worker_env.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
            device: s.device.clone(),
            socket: data_dir.join(format!("w{i}.sock")),
        };
        tokio::spawn(app::supervise(app.clone(), s.clone(), spec));
    }
    tokio::spawn(app::events(app.clone(), events));
    tokio::spawn(app::announce_loop(app.clone()));
    if let Some(t) = &cfg.train {
        let coord: Multiaddr = t.coordinator.parse().context("--train must be a multiaddr")?;
        tokio::spawn(train::member(app.clone(), coord, t.run.clone()));
    }
    let port = cfg.api_port.unwrap_or(8780);
    if port != 0 {
        let token = api_token(&cfg.api_token_file.clone().unwrap_or_else(|| data_dir.join("api.token")))?;
        let app = app.clone();
        tokio::spawn(async move {
            if let Err(e) = api::serve(app, port, token).await {
                tracing::error!("api: {e:#}");
                std::process::exit(1);
            }
        });
    }
    tokio::signal::ctrl_c().await?;
    tracing::info!("shutting down");
    for w in app.live_workers() {
        w.shutdown();
    }
    tokio::time::sleep(Duration::from_millis(300)).await;
    Ok(())
}
