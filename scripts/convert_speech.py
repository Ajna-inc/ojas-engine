"""Convert a speech checkpoint (safetensors, with config.json and tokenizer.json beside it) to the
GGUF layout the CPU speech decoder loads, tokenizer included so a server can turn tokens into text.

usage: convert_speech.py <checkpoint-dir> <out.gguf> <name>
"""
# HF speech checkpoint (safetensors) -> the GGUF layout CpuWhisper::load reads.
# Tensor copying only: no model is run.
import sys, json, os, numpy as np, gguf
from safetensors.numpy import load_file
src, dst, name = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = json.load(open(os.path.join(src, "config.json")))
sd = load_file(os.path.join(src, "model.safetensors"))
sd = {k.removeprefix("model."): v.astype(np.float32) for k, v in sd.items()}
tok = json.load(open(os.path.join(src, "tokenizer.json")))
vocab = tok["model"]["vocab"]; merges = tok["model"]["merges"]
merges = [m if isinstance(m, str) else " ".join(m) for m in merges]
n = cfg["vocab_size"]
tokens = [None] * n; types = [1] * n
for t, i in vocab.items(): tokens[i] = t
for a in tok["added_tokens"]:
    tokens[a["id"]] = a["content"]; types[a["id"]] = 3
assert all(t is not None for t in tokens)
tid = {t: i for i, t in enumerate(tokens)}

def mel_filters(n_mels, sr=16000, n_fft=400):
    # Slaney-scale, slaney-normalised triangular filters, [n_mels, n_fft/2+1].
    def hz_to_mel(f):
        f = np.asarray(f, dtype=np.float64); mel = f / (200.0 / 3)
        min_log_hz = 1000.0; min_log_mel = min_log_hz / (200.0 / 3); logstep = np.log(6.4) / 27.0
        return np.where(f >= min_log_hz, min_log_mel + np.log(np.maximum(f, 1e-10) / min_log_hz) / logstep, mel)
    def mel_to_hz(m):
        m = np.asarray(m, dtype=np.float64); f = m * (200.0 / 3)
        min_log_hz = 1000.0; min_log_mel = min_log_hz / (200.0 / 3); logstep = np.log(6.4) / 27.0
        return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), f)
    fft = np.linspace(0, sr / 2, n_fft // 2 + 1)
    pts = mel_to_hz(np.linspace(hz_to_mel(0.0), hz_to_mel(sr / 2), n_mels + 2))
    fdiff = np.diff(pts); ramps = pts[:, None] - fft[None, :]
    w = np.zeros((n_mels, len(fft)))
    for i in range(n_mels):
        w[i] = np.maximum(0, np.minimum(-ramps[i] / fdiff[i], ramps[i + 2] / fdiff[i + 1]))
    w *= (2.0 / (pts[2:n_mels + 2] - pts[:n_mels]))[:, None]
    return w.astype(np.float32)

w = gguf.GGUFWriter(dst, "whisper")
w.add_name(name)
for k, v in [("d_model", cfg["d_model"]), ("encoder_layers", cfg["encoder_layers"]), ("decoder_layers", cfg["decoder_layers"]),
             ("attention.head_count", cfg["encoder_attention_heads"]), ("num_mel_bins", cfg["num_mel_bins"]),
             ("max_source_positions", cfg["max_source_positions"]), ("max_target_positions", cfg["max_target_positions"]),
             ("vocab_size", n)]:
    w.add_uint32(f"whisper.{k}", v)
for k, t in [("sot", "<|startoftranscript|>"), ("eot", "<|endoftext|>"), ("lang_en", "<|en|>"), ("transcribe", "<|transcribe|>"),
             ("no_timestamps", "<|notimestamps|>"), ("ts_begin", "<|0.00|>")]:
    w.add_uint32(f"whisper.token.{k}", tid[t])
w.add_tokenizer_model("gpt2"); w.add_tokenizer_pre("gpt-2")
w.add_token_list(tokens); w.add_token_types(types); w.add_token_merges(merges)
def add(name, a): w.add_tensor(name, np.ascontiguousarray(a, dtype=np.float32))
add("mel_filters", mel_filters(cfg["num_mel_bins"]))
add("enc.conv1.weight", sd["encoder.conv1.weight"]); add("enc.conv1.bias", sd["encoder.conv1.bias"])
add("enc.conv2.weight", sd["encoder.conv2.weight"]); add("enc.conv2.bias", sd["encoder.conv2.bias"])
add("enc.pos", sd["encoder.embed_positions.weight"]); add("dec.pos", sd["decoder.embed_positions.weight"])
add("dec.tok_embd", sd["decoder.embed_tokens.weight"])
def layers(pre, hf, count, cross):
    for i in range(count):
        p = f"{hf}.layers.{i}."
        def ln(d, s):
            add(f"{pre}.{i}.{d}.weight", sd[p + s + ".weight"]); add(f"{pre}.{i}.{d}.bias", sd[p + s + ".bias"])
        def proj(d, s, bias=True):
            add(f"{pre}.{i}.{d}.weight", sd[p + s + ".weight"])
            if bias: add(f"{pre}.{i}.{d}.bias", sd[p + s + ".bias"])
        ln("attn_ln", "self_attn_layer_norm")
        proj("attn_q", "self_attn.q_proj"); proj("attn_k", "self_attn.k_proj", False)
        proj("attn_v", "self_attn.v_proj"); proj("attn_o", "self_attn.out_proj")
        if cross:
            ln("xattn_ln", "encoder_attn_layer_norm")
            proj("xattn_q", "encoder_attn.q_proj"); proj("xattn_k", "encoder_attn.k_proj", False)
            proj("xattn_v", "encoder_attn.v_proj"); proj("xattn_o", "encoder_attn.out_proj")
        ln("mlp_ln", "final_layer_norm"); proj("fc1", "fc1"); proj("fc2", "fc2")
layers("enc", "encoder", cfg["encoder_layers"], False); layers("dec", "decoder", cfg["decoder_layers"], True)
add("enc_ln.weight", sd["encoder.layer_norm.weight"]); add("enc_ln.bias", sd["encoder.layer_norm.bias"])
add("dec_ln.weight", sd["decoder.layer_norm.weight"]); add("dec_ln.bias", sd["decoder.layer_norm.bias"])
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print("wrote", dst)
