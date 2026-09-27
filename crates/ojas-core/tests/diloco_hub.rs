//! Hub protocol end to end with stand-in workers: no model, no GPU, so the
//! aggregation contract is testable on every platform and in CI.
use ojas_core::{diloco, wire};
use std::net::TcpStream;

/// Each worker "trains" by walking a fixed step toward zero, so the expected
/// trajectory is arithmetic: theta_start 10, deltas 2 and 4, outer_lr 0.5
/// -> 8.5 after round 0, and so on.
fn fake_worker(port: u16, sig: u32, start: Vec<f32>, step: f32) -> std::thread::JoinHandle<Vec<f32>> {
    std::thread::spawn(move || {
        let mut s = loop {
            match TcpStream::connect(("127.0.0.1", port)) {
                Ok(s) => break s,
                Err(_) => std::thread::sleep(std::time::Duration::from_millis(20)),
            }
        };
        wire::send_vec(&mut s, sig, &start).unwrap();
        let mut last = start;
        loop {
            let (round, theta) = wire::recv_vec(&mut s).unwrap();
            if round == diloco::DONE { return last; }
            let delta: Vec<f32> = theta.iter().map(|_| step).collect();
            last = theta;
            wire::send_vec(&mut s, round, &delta).unwrap();
        }
    })
}

#[test]
fn two_workers_agree_and_converge() {
    let port = 47821;
    let hub = std::thread::spawn(move || diloco::hub(port, 2, 2, 0.5, None));
    let a = fake_worker(port, 0xabcd, vec![10.0, 10.0], 2.0);
    let b = fake_worker(port, 0xabcd, vec![10.0, 10.0], 4.0);
    hub.join().unwrap().unwrap();
    // Round 0 broadcast the base 10; round 1 broadcast 10 - 0.5*mean(2,4) = 8.5.
    assert_eq!(a.join().unwrap(), vec![8.5, 8.5]);
    assert_eq!(b.join().unwrap(), vec![8.5, 8.5]);
}

#[test]
fn mismatched_architecture_is_refused() {
    let port = 47822;
    let hub = std::thread::spawn(move || diloco::hub(port, 2, 1, 0.5, None));
    let _a = fake_worker(port, 0xabcd, vec![1.0, 1.0], 0.1);
    let _b = fake_worker(port, 0x1234, vec![1.0, 1.0], 0.1);   // different signature
    let err = hub.join().unwrap().unwrap_err().to_string();
    assert!(err.contains("run is"), "expected a signature refusal, got: {err}");
}
