//! Gate: a vision projector (`mmproj`) loads as a second part of the decoder's
//! GGUF, with its metadata, and cannot change anything the decoder declared.
//!
//! The failure this guards against is silent: `Gguf::attach` drops the incoming
//! file's KV entirely, and for an mmproj the KV is the configuration — ~20
//! `clip.*` keys that exist nowhere else. Attaching one with `attach` yields a
//! model with 154 extra tensors and no declared shape for them. So this test
//! asserts both halves: `attach_with_meta` brings the keys across, and plain
//! `attach` still does not, which proves the delegate did not change the MTP
//! path's behaviour.
//!
//! Models come from OJAS_TEST_MMPROJ; the test skips when unset rather than
//! hard-coding one machine's absolute paths. OJAS_REQUIRE_MODEL_TESTS turns the
//! skip into a failure, for the release matrix.
//!
//!   OJAS_TEST_MMPROJ=/path/to/model.gguf              (sidecar auto-discovered)
//!   OJAS_TEST_MMPROJ=/path/to/model.gguf:/path/to/mmproj.gguf

use ojas_formats::gguf::{Gguf, Meta};
use ojas_formats::mmproj;
use std::collections::HashMap;

/// (decoder, projector), or None when the test should skip.
fn paths() -> Option<(String, String)> {
    let Ok(spec) = std::env::var("OJAS_TEST_MMPROJ") else {
        assert!(std::env::var_os("OJAS_REQUIRE_MODEL_TESTS").is_none(),
            "required model tests need OJAS_TEST_MMPROJ");
        eprintln!("skip: set OJAS_TEST_MMPROJ=<model.gguf>[:<mmproj.gguf>] to run the mmproj gate");
        return None;
    };
    let parts: Vec<&str> = spec.split(':').filter(|p| !p.is_empty()).collect();
    assert!(matches!(parts.len(), 1 | 2), "OJAS_TEST_MMPROJ takes <model.gguf>[:<mmproj.gguf>], got {spec:?}");
    let model = parts[0].to_string();
    assert!(std::path::Path::new(&model).exists(), "configured model is missing: {model}");
    // One path means "find the sidecar the way the loader will" — so the default
    // discovery path is covered by the same run, not only the explicit one.
    let proj = mmproj::discover(std::path::Path::new(&model), parts.get(1).copied())
        .unwrap()
        .unwrap_or_else(|| panic!("no mmproj discovered next to {model}; pass it as OJAS_TEST_MMPROJ={model}:<mmproj.gguf>"));
    Some((model, proj.to_string_lossy().into_owned()))
}

fn snapshot(g: &Gguf) -> HashMap<String, String> {
    g.meta.iter().map(|(k, v)| (k.clone(), format!("{v:?}"))).collect()
}

#[test]
fn attach_with_meta_brings_the_clip_config_across() {
    let Some((model, proj)) = paths() else { return };

    let mut g = Gguf::open(&model).unwrap();
    let side = Gguf::open(&proj).unwrap();

    // Everything the loader must prove before it allocates.
    mmproj::validate(&g, &side).unwrap();

    let want: Vec<String> = side.meta.keys().filter(|k| mmproj::keep_kv(k)).cloned().collect();
    let want_tensors: Vec<String> = side.tensors.keys().filter(|n| mmproj::keep_tensor(n)).cloned().collect();
    assert!(!want.is_empty(), "{proj}: no clip.* keys — wrong file?");
    let before = snapshot(&g);
    assert!(!before.keys().any(|k| k.starts_with("clip.")), "decoder already has clip.* keys");
    let n_before = g.tensors.len();

    let added = g.attach_with_meta(&proj, mmproj::keep_tensor, mmproj::keep_kv).unwrap();
    assert_eq!(added, want_tensors.len(), "every v.*/mm.* tensor must be adopted");
    assert_eq!(g.tensors.len(), n_before + added);
    eprintln!("ok: {proj} -> {added} tensors, {} clip.* keys", want.len());

    // 1. The KV crossed over, value for value.
    let after = snapshot(&g);
    for k in &want {
        assert_eq!(after.get(k), Some(&format!("{:?}", side.meta[k])), "clip key {k} did not cross intact");
    }
    // 2. Nothing the decoder declared moved. This is the assertion that makes
    //    `attach_with_meta` safe to call on a model that is already loaded.
    for (k, v) in &before {
        assert_eq!(after.get(k), Some(v), "decoder key {k} was overwritten by the sidecar");
    }
    assert_eq!(after.len(), before.len() + want.len(), "unexpected extra keys merged");

    // 3. Every adopted tensor resolves — same path the uploader takes.
    let proj_len = std::fs::metadata(&proj).unwrap().len();
    for name in &want_tensors {
        let (part, off, bytes, ty) = g.tensor_meta(name)
            .unwrap_or_else(|| panic!("{name} does not resolve after attach"));
        assert_ne!(part, 0, "{name} must resolve into the attached part, not the decoder's file");
        assert!(bytes > 0 && off + bytes <= proj_len, "{name}: [{off},{bytes}) outside {proj} ({proj_len} bytes)");
        assert!(matches!(ty, 0 | 1), "{name}: unexpected ggml type {ty}");
        assert_eq!(g.tensors[name].dims, side.tensors[name].dims);
    }
    // ...and at least one actually reads back through the attached file handle.
    let probe = want_tensors.iter().find(|n| n.starts_with("mm.")).expect("no mm.* projector tensor");
    let (dims, _, raw) = g.read_tensor(probe).unwrap();
    assert!(!raw.is_empty() && dims == side.tensors[probe].dims, "{probe} read back wrong");

    // 4. The typed accessors read the merged keys — including the two array
    //    shapes an mmproj needs. GGUF BOOL arrays decode to IntArr, so
    //    is_deepstack_layers goes through int_arr, not a bool accessor.
    let nb = g.meta_u32("clip.vision.block_count").unwrap();
    let key = "clip.vision.is_deepstack_layers";
    let deep = g.int_arr(key).expect("is_deepstack_layers must read as an int array");
    assert_eq!(deep.len(), nb as usize, "one deepstack flag per vision block");
    assert!(deep.iter().all(|&b| b == 0 || b == 1), "bool array decoded to non-boolean values: {deep:?}");
    assert_eq!(Some(deep), side.int_arr(key), "{key} did not survive the merge intact");
    eprintln!("ok: {key} = {deep:?} ({nb} blocks)");
    for key in ["clip.vision.image_mean", "clip.vision.image_std"] {
        let v = g.float_arr(key).unwrap_or_else(|| panic!("{key} must read as a float array"));
        assert_eq!(v.len(), 3, "{key} is per-channel: {v:?}");
        assert!(v.iter().all(|x| x.is_finite()), "{key}: {v:?}");
        assert_eq!(Some(v), side.float_arr(key), "{key} did not survive the merge intact");
        eprintln!("ok: {key} = {v:?}");
    }
    assert_eq!(g.meta_u32("clip.vision.projection_dim"),
        g.meta_u32(&format!("{}.embedding_length", g.arch())),
        "the projector writes into the decoder's embedding stream");
    assert!(g.meta_f32("clip.vision.attention.layer_norm_epsilon").is_some_and(|e| e > 0.0));
    assert_eq!(g.meta_bool("clip.has_vision_encoder"), Some(true));
}

/// The delegate proof: `attach` is `attach_with_meta(.., |_| false)`, so it must
/// still take the tensors and still drop every key. If this ever passes clip.*
/// through, the MTP sidecar path at `decoder/load.rs` changed behaviour too.
#[test]
fn plain_attach_still_drops_the_metadata() {
    let Some((model, proj)) = paths() else { return };
    let mut g = Gguf::open(&model).unwrap();
    let side = Gguf::open(&proj).unwrap();
    let before = snapshot(&g);

    let added = g.attach(&proj, mmproj::keep_tensor).unwrap();
    assert_eq!(added, side.tensors.keys().filter(|n| mmproj::keep_tensor(n)).count());
    assert_eq!(snapshot(&g), before, "attach must not change metadata at all");
    assert!(g.meta_u32("clip.vision.projection_dim").is_none(), "attach leaked a clip.* key");
}

/// "Main model wins" must hold because of `or_insert`, not because the filter
/// happened to exclude the colliding keys. Merge everything and check the
/// decoder's identity survives: an mmproj declares general.architecture=clip and
/// general.type=mmproj, so a last-write-wins merge would turn the decoder into a
/// clip model and every `arch()`-keyed lookup after it would miss.
#[test]
fn a_permissive_filter_still_cannot_overwrite_the_decoder() {
    let Some((model, proj)) = paths() else { return };
    let mut g = Gguf::open(&model).unwrap();
    let side = Gguf::open(&proj).unwrap();
    let arch = g.arch();
    let before = snapshot(&g);
    let collisions: Vec<&String> = side.meta.keys().filter(|k| before.contains_key(*k)).collect();
    assert!(collisions.len() >= 2, "expected general.* collisions to exercise the rule, got {collisions:?}");
    assert!(matches!(side.meta.get("general.architecture"), Some(Meta::Str(s)) if s != &arch),
        "the sidecar must declare a different architecture for this test to mean anything");

    g.attach_with_meta(&proj, mmproj::keep_tensor, |_| true).unwrap();
    assert_eq!(g.arch(), arch, "the sidecar overwrote general.architecture");
    let after = snapshot(&g);
    for (k, v) in &before {
        assert_eq!(after.get(k), Some(v), "decoder key {k} was overwritten");
    }
    // ...and the non-colliding keys did arrive, so the filter is what selects,
    // not what protects.
    assert!(g.meta_u32("clip.vision.projection_dim").is_some());
}
