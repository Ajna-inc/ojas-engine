# ojas-node

The ojas swarm daemon: a libp2p node (`ojas-net`) that supervises one
`ojas engine-worker` per device, serves an OpenAI-compatible API on 127.0.0.1,
routes generations to whichever pool member holds the model, and runs DiLoCo
training as a member or as the coordinator.

## A pool

Pools are invite-only. The admin key signs one invite per member PeerId; peers
that cannot show one are disconnected and nothing they send is acted on.

```sh
# on the coordinator / admin machine
ojas-node pool init --name lab            # writes pool.json; this node's key is the admin
# on a new member
ojas-node id                              # prints its PeerId (creates ~/.ojas/node.key)
# back on the admin
ojas-node pool invite 12D3KooW... --days 30   # prints an ojasinv1.… token
```

Copy `pool.json` and the token to the member.

## Run a node

```sh
ojas-node --pool pool.json --invite ojasinv1.… \
  --bootstrap /ip4/203.0.113.7/tcp/4801/p2p/12D3KooW... \
  --device metal --model ~/models/qwen3-4b.gguf
```

or put the same in a `node.toml` / `node.json` and pass `--config node.toml`:

```toml
pool = "pool.json"
invite = "@invite.txt"
listen = ["/ip4/0.0.0.0/tcp/4801", "/ip4/0.0.0.0/udp/4801/quic-v1"]
bootstrap = ["/ip4/203.0.113.7/tcp/4801/p2p/12D3KooW..."]
relays = ["/ip4/203.0.113.7/tcp/4801/p2p/12D3KooW..."]   # behind NAT: be reachable via the coordinator
devices = ["cuda"]
api_port = 8780

[[models]]
path = "/models/qwen3-4b.gguf"

[[models]]                 # routed to other members, tokenised here, not loaded
path = "/models/qwen3-32b.gguf"
load = false
id = "<ModelId hex from a member's /status>"
```

The API listens on `127.0.0.1:8780` with a bearer token from
`~/.ojas/node/api.token` (created on first run):

```sh
curl -H "Authorization: Bearer $(cat ~/.ojas/node/api.token)" localhost:8780/v1/chat/completions \
  -d '{"model": "qwen3-4b", "messages": [{"role": "user", "content": "hi"}], "stream": true}'
```

`GET /status` shows members, the announce table, workers and training state.

## Training

Coordinator (also a relay and the natural bootstrap node):

```sh
ojas-node --pool pool.json --coordinator run.json --listen /ip4/0.0.0.0/tcp/4801
```

Member (dials out only, so NAT is fine):

```sh
ojas-node --pool pool.json --invite ojasinv1.… --device cuda \
  --train /ip4/203.0.113.7/tcp/4801/p2p/12D3KooW... --run bytes-demo
```

## Notes

- Workers default to the `ojas` binary next to `ojas-node`, then `ojas` on PATH
  (`worker_bin`, `worker_args` override). A worker that exits or reports a fatal
  error is restarted and its models reloaded.
- `ojas-node mock-worker` speaks the worker IPC with no engine; the tests use it.
