//! The node's Ed25519 identity, kept in a key file.
//!
//! Adapted from SwarmLLM (MIT OR Apache-2.0), `src/network/transport.rs` @ b14482d:
//! the PeerId of an Ed25519 key inlines the public key, so a PeerId alone is enough
//! to verify a signature (the pool admin is named by PeerId for that reason).
//!
//! The file holds libp2p's protobuf encoding of the keypair. It is created with
//! mode 0600 on unix, atomically (`create_new`), so two processes racing to create
//! it cannot both win and end up with different identities.

use anyhow::{bail, Context, Result};
use libp2p::identity::{Keypair, PublicKey};
use libp2p::PeerId;
use std::io::Write;
use std::path::Path;

pub fn load(path: &Path) -> Result<Keypair> {
    let bytes = std::fs::read(path).with_context(|| format!("reading key {}", path.display()))?;
    let kp = Keypair::from_protobuf_encoding(&bytes).with_context(|| format!("decoding key {}", path.display()))?;
    if kp.clone().try_into_ed25519().is_err() {
        bail!("{}: not an Ed25519 key", path.display());
    }
    warn_if_readable(path);
    Ok(kp)
}

/// Load the key at `path`, or generate one there if the file does not exist.
pub fn load_or_create(path: &Path) -> Result<Keypair> {
    if path.exists() {
        return load(path);
    }
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    let kp = Keypair::generate_ed25519();
    let bytes = kp.to_protobuf_encoding().context("encoding key")?;
    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    match opts.open(path) {
        Ok(mut f) => {
            f.write_all(&bytes)?;
            f.sync_all()?;
            Ok(kp)
        }
        // Lost a race with another process: theirs is the identity.
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => load(path),
        Err(e) => Err(e).with_context(|| format!("creating key {}", path.display())),
    }
}

/// The public key inlined in an Ed25519 PeerId.
pub fn public_key_of(peer: &PeerId) -> Option<PublicKey> {
    let mh = peer.as_ref();
    // 0x00 = identity multihash: the digest is the protobuf-encoded public key.
    if mh.code() != 0x00 {
        return None;
    }
    PublicKey::try_decode_protobuf(mh.digest()).ok()
}

fn warn_if_readable(path: &Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(m) = std::fs::metadata(path) {
            if m.permissions().mode() & 0o077 != 0 {
                tracing::warn!("{} is readable by other users; chmod 600 it", path.display());
            }
        }
    }
    #[cfg(not(unix))]
    let _ = path;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_key_file_is_created_once_and_reloads_to_the_same_peer() {
        let dir = std::env::temp_dir().join(format!("ojas-net-key-{}", std::process::id()));
        let path = dir.join("node.key");
        let _ = std::fs::remove_file(&path);
        let a = load_or_create(&path).unwrap();
        let b = load_or_create(&path).unwrap();
        assert_eq!(a.public().to_peer_id(), b.public().to_peer_id());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(std::fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o600);
        }
        let pk = public_key_of(&a.public().to_peer_id()).unwrap();
        assert_eq!(pk, a.public());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
