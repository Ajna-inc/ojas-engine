#!/usr/bin/env python3
"""Estimate useful one-token traffic from the existing Flash GGUF inventory.

Not a hardware-counter measurement: assumes each active weight read once,
FP32 recurrent state read+write once, and unique FP32 KV entries read once.
"""
import argparse
import collections
import json

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('inventory')
p.add_argument('--bandwidth-gbs', type=float, default=371.47269376454386)
p.add_argument('--sequence-length', type=int, default=154)
a = p.parse_args()
assert a.bandwidth_gbs > 0 and a.sequence_length > 0
with open(a.inventory) as f:
    inventory = json.load(f)
m = inventory['metadata']
assert m['general.architecture'] == 'qwen4exp'
experts = m['qwen4exp.expert_count']
selected = m['qwen4exp.expert_used_count']
traffic = collections.defaultdict(float)
excluded = 0
for t in inventory['tensors']:
    n, b = t['name'], t['bytes']
    if n == 'per_layer_token_embd.weight':
        traffic['ple'] += 1440  # This checkpoint: 2560 IQ4_NL values.
    elif n == 'token_embd.weight':
        traffic['embedding'] += b / t['dims'][1]
    elif '.indexer.' in n:
        excluded += b  # Current decoder uses dense gated attention.
    elif '_exps.' in n:
        traffic['routed_experts'] += b * selected / experts
    elif n.startswith('output'):
        traffic['head'] += b
    elif '.ple_' in n:
        traffic['ple'] += b
    elif '.hc_' in n:
        traffic['hc'] += b
    elif '.ffn_' in n:
        traffic['shared_router'] += b
    else:
        traffic['token_mixers_weights'] += b
ssm_layers = sum(t['name'].endswith('.ssm_a') for t in inventory['tensors'])
attn_layers = m['qwen4exp.block_count'] - ssm_layers
traffic['recurrent_state_read_write'] = ssm_layers * m['qwen4exp.ssm.state_size'] * m['qwen4exp.ssm.inner_size'] * 4 * 2
traffic['unique_kv_reads'] = attn_layers * a.sequence_length * m['qwen4exp.attention.head_count_kv'] * (m['qwen4exp.attention.key_length'] + m['qwen4exp.attention.value_length']) * 4
b = sum(traffic.values())
print(json.dumps({
    'assumptions': 'same packed weights, batch 1, no speculation; optimistic useful bytes, not measured DRAM traffic; excludes scratch and redundant cache misses',
    'sequence_length': a.sequence_length,
    'bytes_by_category': dict(traffic),
    'unused_indexer_storage_bytes': excluded,
    'total_useful_GB_per_token': b / 1e9,
    'read_probe_GB_s': a.bandwidth_gbs,
    'ideal_read_only_roof_tps': a.bandwidth_gbs * 1e9 / b,
    'advertised_400_GB_s_roof_tps': 400e9 / b,
    'scenarios': [{
        'tps': tps, 'ms_per_token': 1000 / tps,
        'required_useful_GB_s': b * tps / 1e9,
        'fraction_of_read_probe': b * tps / (a.bandwidth_gbs * 1e9),
    } for tps in [23.16, 30, 35, 40, 50, 60]],
}, indent=2))
