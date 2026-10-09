//! `--coordinator`: the run's coordinator answers every training and blob request
//! this node receives. It is transport agnostic; all this does is hand it the
//! requesting PeerId.

use crate::app::Coordinate;
use anyhow::Result;
use std::path::Path;

#[cfg(feature = "coordinator")]
impl Coordinate for ojas_swarm::coord::Coordinator {
    fn handle(&mut self, from: &str, req: ojas_swarm_proto::peer::PeerReq, payload: Vec<u8>) -> (ojas_swarm_proto::peer::PeerResp, Vec<u8>) {
        ojas_swarm::coord::Coordinator::handle(self, from, req, payload)
    }
}

#[cfg(feature = "coordinator")]
pub fn open(path: &Path) -> Result<Box<dyn Coordinate>> {
    Ok(Box::new(ojas_swarm::coord::Coordinator::from_config_file(path)?))
}

#[cfg(not(feature = "coordinator"))]
pub fn open(path: &Path) -> Result<Box<dyn Coordinate>> {
    anyhow::bail!("{}: this ojas-node was built without the `coordinator` feature", path.display())
}
