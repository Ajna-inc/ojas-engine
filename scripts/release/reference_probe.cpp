// Reference-only test helper. Links to a separately installed llama.cpp;
// it is not a dependency of the engine or its normal build.
//
// Build the text-only helper exactly as before. Define OJAS_REFERENCE_MMPROJ and
// add -lmtmd to also get the vision encoder / projector oracle; nothing about the
// text path changes when the flag is off.
#include <llama.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <stdexcept>
#include <cstdlib>
#include <cmath>
#include <filesystem>
#include <sstream>
#ifdef OJAS_REFERENCE_MMPROJ
#include <mtmd.h>
#include <mtmd-helper.h>
#include <mtmd-debug.h>
#endif
struct trace_state {
    const char * dir;
    int pos = 0;
    bool vision = false;              // vision graphs use their own name filter
    std::map<std::string, int> seen;  // vision only: occurrence count per tensor name
};
static bool text_selected(const std::string & name) {
    return name.find("l_last") == 0 || name == "hc_init" || name == "result_norm"
        || name.find("ffn_moe_logits-") == 0 || name.find("ffn_moe_topk-") == 0
        || name.find("Qcur-") == 0 || name.find("Kcur-") == 0 || name.find("Vcur-") == 0
        || name.find("attn_gated-") == 0 || name.find("attn_output-") == 0;
}
#ifdef OJAS_REFERENCE_MMPROJ
// clip_graph::cb() formats "<stage>-<layer>" for per-layer nodes and a bare name
// for the whole-graph ones (build_norm's post-layer-norm lands on "norm_b-<n_layer>",
// the projector's build_ffn runs with il == -1 so its stages are bare). "inp_raw" is
// the preprocessed pixel plane and is a graph input rather than a node; it is reached
// through the first node that consumes it, see vision_source_pending below.
static const char * const vision_defaults[] = {
    "inp_raw", "inp_pos_emb", "positions", "patch_bias", "pos_embed",
    "pre_ln", "post_ln", "norm_w", "norm_b", "ln1", "ln2",
    "layer_inp_normed", "Qcur", "Kcur", "Vcur", "attn_", "kqv_out",
    "ffn_", "layer_out", "pixel_shuffle", "deepstack", "embeddings",
    "result_embd", "mm",
};
// A comma separated OJAS_REFERENCE_VISION_FILTER replaces the list above; a "*"
// entry keeps every eligible node.
struct vision_filter_state { std::vector<std::string> prefixes; bool custom = false; };
static const vision_filter_state & vision_filter() {
    static const vision_filter_state state = [] {
        vision_filter_state s;
        if (auto env = std::getenv("OJAS_REFERENCE_VISION_FILTER")) {
            std::stringstream split(env);
            for (std::string item; std::getline(split, item, ',');) if (!item.empty()) s.prefixes.push_back(item);
            s.custom = !s.prefixes.empty();
        }
        if (!s.custom) s.prefixes.assign(std::begin(vision_defaults), std::end(vision_defaults));
        return s;
    }();
    return state;
}
// ggml_backend_sched names a cross-backend copy "<backend>#<tensor>#<slot>"
// (ggml-backend.cpp, ggml_format_name(tensor_copy, "%s#%s#%d", ...)). It holds the
// same values, so both the filter and the file name key on the tensor; without this
// a Metal reference silently loses every graph input, "inp_raw" included, because
// there it arrives as "MTL0#inp_raw#0".
static std::string vision_name(const char * raw) {
    const std::string name = raw;
    const auto first = name.find('#'), last = name.rfind('#');
    if (first == std::string::npos) return name;
    if (first == last) return name.substr(first + 1);
    return name.substr(first + 1, last - first - 1);
}
static bool vision_selected(const ggml_tensor * t) {
    // Quantized and f16 leaves are never comparable output, so they are filtered
    // here instead of tripping the type assert below.
    if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_I32) return false;
    const std::string name = vision_name(t->name);
    if (name.empty()) return false;
    const auto & filter = vision_filter();
    // ggml auto-names a view of a traced tensor "<name> (permuted)" and friends.
    // Those hold the same values in a different layout and double the dump, so the
    // default filter drops them; an explicit filter is honoured verbatim.
    if (!filter.custom && name.find(" (") != std::string::npos) return false;
    for (const auto & prefix : filter.prefixes) if (prefix == "*" || name.rfind(prefix, 0) == 0) return true;
    return false;
}
#endif
static void write_tensor(trace_state & s, const ggml_tensor * t, const std::string & name) {
    std::vector<char> storage(ggml_nbytes(t));
    ggml_backend_tensor_get(t, storage.data(), 0, storage.size());
    // A contiguous 4-byte tensor already is the row-major buffer the de-striding
    // loop would rebuild element by element; vision activations are large enough
    // that the difference is minutes.
    const bool contiguous = ggml_is_contiguous(t) && ggml_nbytes(t) == (size_t) ggml_nelements(t)*4;
    std::vector<char> bytes(contiguous ? 0 : ggml_nelements(t)*4);
    for (size_t i=0; i<bytes.size()/4; ++i) {
        size_t index=i, offset=0;
        for (int dim=0; dim<4; ++dim) { offset+=(index % t->ne[dim])*t->nb[dim]; index/=t->ne[dim]; }
        std::copy_n(storage.data()+offset,4,bytes.data()+i*4);
    }
    const std::vector<char> & payload = contiguous ? storage : bytes;
    std::ofstream out(std::string(s.dir) + "/" + std::to_string(s.pos) + "-" + name + (t->type == GGML_TYPE_I32 ? ".i32" : ".f32"), std::ios::binary);
    out.write(payload.data(), payload.size());
    if (!out) std::abort();
}
#ifdef OJAS_REFERENCE_MMPROJ
// A graph input is a leaf, so the scheduler never offers it to this callback on its
// own. "inp_raw" -- the preprocessed pixel plane, and the only place preprocessing
// output is observable, since mtmd_debug_preprocess_image only logs geometry -- is
// exactly such a leaf. Reaching it through the first consuming node costs one extra
// predicate and is the difference between having a preprocessing oracle and not.
static bool vision_source_pending(const trace_state & s, const ggml_tensor * t) {
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        const ggml_tensor * src = t->src[i];
        if (src && src->buffer && vision_selected(src) && !s.seen.count(vision_name(src->name))) return true;
    }
    return false;
}
#endif
static bool trace_tensor(ggml_tensor * t, bool ask, void * data) {
    auto & s = *static_cast<trace_state *>(data);
#ifdef OJAS_REFERENCE_MMPROJ
    const std::string name = s.vision ? vision_name(t->name) : t->name;
    const bool selected = s.vision ? vision_selected(t) : text_selected(name);
    if (ask) return selected || (s.vision && vision_source_pending(s, t));
    if (s.vision) {
        // Graph inputs do not repeat, so the first node that consumes one records it.
        for (int i = 0; i < GGML_MAX_SRC; ++i) {
            ggml_tensor * src = t->src[i];
            if (!src || !src->buffer || !vision_selected(src)) continue;
            const std::string leaf = vision_name(src->name);
            // Do not touch the count of a name already recorded as a node; a tensor
            // is a source of everything downstream of it and would inflate it.
            if (s.seen.count(leaf)) continue;
            s.seen[leaf] = 1;
            write_tensor(s, src, leaf);
        }
    }
#else
    const std::string name = t->name;
    const bool selected = text_selected(name);
    if (ask) return selected;
#endif
    if (!selected) return true;
    if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_I32) std::abort();
#ifdef OJAS_REFERENCE_MMPROJ
    // Names repeat: clip_graph::build_norm is called twice per layer with the same
    // index, so "norm_w-3" is two distinct tensors, and one encode can run the graph
    // more than once for tiles. The k-th occurrence in evaluation order is recorded
    // as "<name>~k" so nothing is overwritten and the pairing stays deterministic.
    if (s.vision) {
        const int k = ++s.seen[name];
        write_tensor(s, t, k == 1 ? name : name + "~" + std::to_string(k));
        return true;
    }
#endif
    write_tensor(s, t, name);
    return true;
}
#ifdef OJAS_REFERENCE_MMPROJ
struct vision_options {
    const char * mmproj = nullptr;
    const char * image = nullptr;
    std::string mode = "chunk";      // chunk | encode | preproc
    std::string synthetic;           // white|black|gray|cb|rainbow|red|green|blue
    int size = 768;                  // edge length for --synthetic
    int max_tokens = 0;              // 0 keeps the mmproj metadata default
    bool only = false;               // skip the text logits stage
};
// Same patterns the upstream mtmd-debug tool generates, so its numbers and these
// are directly comparable. Rows are nx*3 interleaved RGB.
static bool synthetic_f32(const std::string & kind, int n, std::vector<std::vector<float>> & out) {
    out.assign(n, std::vector<float>(n * 3, 0.0f));
    if (kind == "black") return true;
    if (kind == "white" || kind == "gray") {
        const float v = kind == "white" ? 1.0f : 0.5f;
        for (auto & row : out) std::fill(row.begin(), row.end(), v);
        return true;
    }
    if (kind == "red" || kind == "green" || kind == "blue") {
        const int c = kind == "red" ? 0 : kind == "green" ? 1 : 2;
        for (auto & row : out) for (int x = 0; x < n; ++x) row[x*3 + c] = 1.0f;
        return true;
    }
    if (kind == "cb") {
        for (int y = 0; y < n; ++y) for (int x = 0; x < n; ++x) {
            const float v = ((x + y) % 2) ? 0.0f : 1.0f;
            out[y][x*3+0] = v; out[y][x*3+1] = v; out[y][x*3+2] = v;
        }
        return true;
    }
    if (kind == "rainbow") {
        const float cx = n / 2.0f, cy = n / 2.0f;
        const float max_dist = std::sqrt(cx*cx + cy*cy);
        for (int y = 0; y < n; ++y) for (int x = 0; x < n; ++x) {
            const float dx = x - cx, dy = y - cy;
            float hue = std::atan2(dy, dx) / (2.0f * 3.14159265f);
            if (hue < 0) hue += 1.0f;
            float sat = std::sqrt(dx*dx + dy*dy) / max_dist;
            if (sat > 1.0f) sat = 1.0f;
            const float h6 = hue * 6.0f;
            const int i6 = (int) h6;
            const float f = h6 - i6, p = 1.0f - sat, q = 1.0f - sat*f, t = 1.0f - sat*(1.0f - f);
            float r, g, b;
            switch (i6 % 6) {
                case 0: r=1; g=t; b=p; break;
                case 1: r=q; g=1; b=p; break;
                case 2: r=p; g=1; b=t; break;
                case 3: r=p; g=q; b=1; break;
                case 4: r=t; g=p; b=1; break;
                default: r=1; g=p; b=q; break;
            }
            out[y][x*3+0] = r; out[y][x*3+1] = g; out[y][x*3+2] = b;
        }
        return true;
    }
    return false;
}
static bool synthetic_u8(const std::string & kind, int n, std::vector<uint8_t> & out) {
    std::vector<std::vector<float>> rows;
    if (!synthetic_f32(kind, n, rows)) return false;
    out.resize((size_t) n * n * 3);
    for (int y = 0; y < n; ++y) for (int i = 0; i < n*3; ++i) {
        const float v = rows[y][i] * 255.0f;
        out[(size_t) y * n * 3 + i] = (uint8_t) std::min(255.0f, std::max(0.0f, std::round(v)));
    }
    return true;
}
static void report(const char * stage, const std::string & path, const float * v, size_t count,
                   int n_tokens, int n_embd) {
    double sum = 0, sq = 0;
    float lo = count ? v[0] : 0.0f, hi = count ? v[0] : 0.0f;
    bool finite = true;
    for (size_t i = 0; i < count; ++i) {
        sum += v[i]; sq += (double) v[i] * v[i];
        lo = std::min(lo, v[i]); hi = std::max(hi, v[i]);
        if (!std::isfinite(v[i])) finite = false;
    }
    const double mean = count ? sum / count : 0.0;
    const double var  = count ? sq / count - mean * mean : 0.0;
    std::cout << "{\"stage\":\"" << stage << "\",\"path\":\"" << path << "\",\"n_tokens\":" << n_tokens
              << ",\"n_embd\":" << n_embd << ",\"count\":" << count
              << ",\"mean\":" << mean << ",\"std\":" << std::sqrt(std::max(0.0, var))
              << ",\"min\":" << lo << ",\"max\":" << hi
              << ",\"finite\":" << (finite ? "true" : "false") << "}\n";
}
// Returns 0 on success. The mtmd context is created against the already loaded
// text model, so this costs one extra mmproj load and nothing else.
static int run_vision(llama_model * model, const vision_options & opt,
                      const std::string & prefix, const std::string & prompt, bool gpu) {
    trace_state trace{std::getenv("OJAS_REFERENCE_VISION_TRACE")};
    trace.vision = true;
    auto mp = mtmd_context_params_default();
    mp.use_gpu       = gpu;
    mp.print_timings = true;
    mp.n_threads     = 4;
    mp.warmup        = false; // a warmup encode would otherwise land in the trace
    if (opt.max_tokens > 0) mp.image_max_tokens = opt.max_tokens;
    if (trace.dir) {
        std::filesystem::create_directories(trace.dir);
        mp.cb_eval = trace_tensor; mp.cb_eval_user_data = &trace;
    }
    mtmd_context * mctx = mtmd_init_from_file(opt.mmproj, model, mp);
    if (!mctx) { std::cerr << "mmproj load failed: " << opt.mmproj << "\n"; return 1; }
    if (!mtmd_support_vision(mctx)) { std::cerr << "mmproj has no vision encoder\n"; mtmd_free(mctx); return 1; }
    int status = 0;
    if (opt.mode == "encode") {
        // Pre-processed f32 planes go straight into the ViT, so the encoder is
        // validated without any dependence on resize/normalize.
        std::vector<std::vector<float>> image;
        if (!opt.synthetic.empty()) {
            if (!synthetic_f32(opt.synthetic, opt.size, image)) { std::cerr << "unknown --synthetic kind\n"; status = 2; }
        } else { std::cerr << "--vision-mode encode needs --synthetic\n"; status = 2; }
        if (!status) mtmd_debug_encode_image(mctx, image);
    } else if (opt.mode == "preproc") {
        // Preprocessing alone. Note the installed mtmd only logs the resulting
        // entry geometry; the pixel values themselves come from the "inp_raw"
        // node of a chunk-mode trace.
        std::vector<uint8_t> rgb; int nx = opt.size, ny = opt.size;
        if (!opt.synthetic.empty()) {
            if (!synthetic_u8(opt.synthetic, opt.size, rgb)) { std::cerr << "unknown --synthetic kind\n"; status = 2; }
        } else if (opt.image) {
            auto wrapper = mtmd_helper_bitmap_init_from_file(mctx, opt.image, false);
            if (!wrapper.bitmap) { std::cerr << "image load failed: " << opt.image << "\n"; status = 2; }
            else {
                nx = (int) mtmd_bitmap_get_nx(wrapper.bitmap);
                ny = (int) mtmd_bitmap_get_ny(wrapper.bitmap);
                const unsigned char * data = mtmd_bitmap_get_data(wrapper.bitmap);
                rgb.assign(data, data + mtmd_bitmap_get_n_bytes(wrapper.bitmap));
                mtmd_bitmap_free(wrapper.bitmap);
            }
        } else { std::cerr << "--vision-mode preproc needs --image or --synthetic\n"; status = 2; }
        if (!status) mtmd_debug_preprocess_image(mctx, rgb, nx, ny);
    } else if (opt.mode == "chunk") {
        mtmd_bitmap * bitmap = nullptr;
        if (!opt.synthetic.empty()) {
            std::vector<uint8_t> rgb;
            if (!synthetic_u8(opt.synthetic, opt.size, rgb)) std::cerr << "unknown --synthetic kind\n";
            else bitmap = mtmd_bitmap_init(opt.size, opt.size, rgb.data());
        } else if (opt.image) {
            auto wrapper = mtmd_helper_bitmap_init_from_file(mctx, opt.image, false);
            bitmap = wrapper.bitmap;
        }
        if (!bitmap) { std::cerr << "no image to encode\n"; mtmd_free(mctx); return 2; }
        // Only the image chunk is encoded here, so a prompt without the media
        // marker is replaced by a bare marker rather than silently dropping it.
        const std::string marker = mtmd_get_marker(mctx);
        const std::string text = prompt.find(marker) != std::string::npos ? prompt : marker;
        mtmd_input_text it{text.c_str(), text.size(), true, true};
        mtmd_input_chunks * chunks = mtmd_input_chunks_init();
        const mtmd_bitmap * bitmaps[1] = {bitmap};
        if (mtmd_tokenize(mctx, chunks, &it, bitmaps, 1)) { std::cerr << "mtmd_tokenize failed\n"; status = 3; }
        const mtmd_input_chunk * image_chunk = nullptr;
        for (size_t i = 0; !status && i < mtmd_input_chunks_size(chunks); ++i) {
            const mtmd_input_chunk * c = mtmd_input_chunks_get(chunks, i);
            if (mtmd_input_chunk_get_type(c) == MTMD_INPUT_CHUNK_TYPE_IMAGE) { image_chunk = c; break; }
        }
        if (!status && !image_chunk) { std::cerr << "no image chunk produced\n"; status = 3; }
        if (!status && mtmd_encode_chunk(mctx, image_chunk)) { std::cerr << "mtmd_encode_chunk failed\n"; status = 4; }
        if (!status) {
            const int n_tokens = (int) mtmd_input_chunk_get_n_tokens(image_chunk);
            const int n_embd   = llama_model_n_embd_inp(model);
            const float * embd = mtmd_get_output_embd(mctx);
            if (!embd) { std::cerr << "mtmd_get_output_embd returned null\n"; status = 4; }
            else {
                const size_t count = (size_t) n_tokens * n_embd;
                const std::string path = prefix + ".mmproj.f32";
                std::ofstream out(path, std::ios::binary);
                out.write(reinterpret_cast<const char *>(embd), count * sizeof(float));
                if (!out) { std::cerr << "cannot write " << path << "\n"; status = 5; }
                std::ofstream meta(prefix + ".mmproj.json");
                meta << "{\"n_tokens\":" << n_tokens << ",\"n_embd\":" << n_embd
                     << ",\"rows\":" << n_tokens << ",\"mmproj\":\"" << opt.mmproj << "\"}\n";
                if (!status) report("mmproj", path, embd, count, n_tokens, n_embd);
            }
        }
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bitmap);
    } else { std::cerr << "unknown --vision-mode: " << opt.mode << "\n"; status = 2; }
    mtmd_free(mctx);
    return status;
}
#endif
int main(int argc, char ** argv) {
    const char * usage =
        "usage: reference_probe model prompt-file output-prefix [gpu]"
#ifdef OJAS_REFERENCE_MMPROJ
        "\n                       [--mmproj path] [--image path] [--vision-mode chunk|encode|preproc]"
        "\n                       [--synthetic white|black|gray|cb|rainbow|red|green|blue]"
        "\n                       [--image-size N] [--image-max-tokens N] [--vision-only]"
#endif
        "\n";
    if (argc < 4) { std::cerr << usage; return 2; }
    bool gpu = false;
#ifdef OJAS_REFERENCE_MMPROJ
    vision_options vision;
#endif
    // Everything after the three required positionals is optional and trailing,
    // so the historical [model, prompt, prefix, "gpu"] call stays byte-compatible.
    for (int i = 4; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag.rfind("--", 0) != 0) { gpu = true; continue; } // historically the bare "gpu" token
        const bool has_value = i + 1 < argc;
#ifdef OJAS_REFERENCE_MMPROJ
        if (flag == "--mmproj"           && has_value) { vision.mmproj     = argv[++i]; continue; }
        if (flag == "--image"            && has_value) { vision.image      = argv[++i]; continue; }
        if (flag == "--vision-mode"      && has_value) { vision.mode       = argv[++i]; continue; }
        if (flag == "--synthetic"        && has_value) { vision.synthetic  = argv[++i]; continue; }
        if (flag == "--image-size"       && has_value) { vision.size       = std::stoi(argv[++i]); continue; }
        if (flag == "--image-max-tokens" && has_value) { vision.max_tokens = std::stoi(argv[++i]); continue; }
        if (flag == "--vision-only") { vision.only = true; continue; }
#endif
        std::cerr << "unrecognized argument: " << flag << "\n" << usage;
        return 2;
    }
    llama_backend_init();
    auto mp = llama_model_default_params(); mp.n_gpu_layers = gpu ? 99 : 0;
    // Large CPU references must be able to retain mmap weights instead of
    // eagerly materializing a second, repacked copy of the checkpoint.
    if (std::getenv("OJAS_REFERENCE_NO_REPACK")) mp.use_extra_bufts = false;
    llama_model_tensor_buft_override expert_overrides[3] = {};
    int override_count = 0;
    if (std::getenv("OJAS_REFERENCE_CPU_EXPERTS")) {
        auto cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (!cpu) return 9;
        expert_overrides[override_count++] = {"\\.ffn_.*_exps\\.weight", ggml_backend_dev_buffer_type(cpu)};
        mp.tensor_buft_overrides = expert_overrides;
    }
    if (std::getenv("OJAS_REFERENCE_CPU_PLE")) {
        auto cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (!cpu) return 9;
        expert_overrides[override_count++] = {"^per_layer_token_embd\\.weight$", ggml_backend_dev_buffer_type(cpu)};
        mp.tensor_buft_overrides = expert_overrides;
    }
    auto model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 3;
    const auto vocab = llama_model_get_vocab(model);
    auto cp = llama_context_default_params();
    cp.offload_kqv = gpu && !std::getenv("OJAS_REFERENCE_CPU_KQV");
    cp.op_offload = cp.offload_kqv;
    if (std::getenv("OJAS_REFERENCE_NO_FLASH")) cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.n_ctx = 512; cp.n_batch = 512; cp.n_threads = 4; cp.n_threads_batch = 4;
    trace_state trace{std::getenv("OJAS_REFERENCE_TRACE")};
    if (trace.dir) { std::filesystem::create_directories(trace.dir); cp.cb_eval = trace_tensor; cp.cb_eval_user_data = &trace; }
    int continuation = 8;
    if (auto value = std::getenv("OJAS_REFERENCE_OUTPUT")) continuation = std::stoi(value);
    if (continuation < 1 || continuation > 64) return 6;
    // A page prompt is thousands of tokens. The cap stays explicit and recorded,
    // it is simply no longer set below any realistic OCR prompt.
    int max_prompt = 8192;
    if (auto value = std::getenv("OJAS_REFERENCE_MAX_PROMPT")) max_prompt = std::stoi(value);
    if (max_prompt < 1) return 6;
#ifdef OJAS_REFERENCE_MMPROJ
    if (vision.mmproj) {
        std::ifstream input(argv[2]);
        std::string prompt = input ? std::string((std::istreambuf_iterator<char>(input)), {}) : std::string();
        if (int status = run_vision(model, vision, argv[3], prompt, gpu)) {
            llama_model_free(model); llama_backend_free();
            return 10 + status;
        }
    } else if (vision.image || vision.only) { std::cerr << "--image/--vision-only need --mmproj\n"; return 2; }
    if (vision.only) { llama_model_free(model); llama_backend_free(); return 0; }
#endif
    std::vector<std::pair<std::string, std::string>> cases = {{argv[2], argv[3]}};
    // Optional additional prompt-file<TAB>output-prefix pairs reuse loaded weights
    // while each prompt gets a fresh context. Trace mode is single-case only.
    if (auto file = std::getenv("OJAS_REFERENCE_CASES")) {
        if (trace.dir) return 6;
        std::ifstream manifest(file); if (!manifest) return 5;
        for (std::string line; std::getline(manifest, line);) {
            const auto tab = line.find('\t');
            if (tab == std::string::npos || tab == 0 || tab+1 == line.size()) return 6;
            cases.emplace_back(line.substr(0,tab), line.substr(tab+1));
        }
    }
    for (const auto & fixture : cases) {
    std::ifstream input(fixture.first); if (!input) return 5;
    std::string prompt((std::istreambuf_iterator<char>(input)), {});
    std::vector<llama_token> tokens(prompt.size()*2 + 32);
    int n = llama_tokenize(vocab, prompt.data(), prompt.size(), tokens.data(), tokens.size(), false, true);
    if (n <= 0 || n > max_prompt) return 6;
    tokens.resize(n);
    // The context only grows when the prompt needs it, so short fixtures keep the
    // historical 512-token reference geometry unchanged.
    auto fp = cp;
    while (fp.n_ctx < (uint32_t)(n + continuation + 1)) fp.n_ctx *= 2;
    fp.n_batch = fp.n_ctx;
    auto ctx = llama_init_from_model(model, fp); if (!ctx) return 4;
    std::ofstream ids(fixture.second+".ids");
    std::ofstream logits(fixture.second+".f32", std::ios::binary);
    ids << n << '\n';
    // Record every prompt token and the reference's continuation. Ojas replays
    // this identical sequence, so a mismatch cannot change later inputs.
    for (int i=0; i<n+continuation; ++i) {
        trace.pos = i;
        llama_token t=tokens[i]; ids << t << '\n';
        auto batch=llama_batch_get_one(&t,1);
        if(llama_decode(ctx,batch)) return 7;
        float * row=llama_get_logits_ith(ctx,-1);
        int v=llama_vocab_n_tokens(vocab);
        if(i>=n-1) logits.write(reinterpret_cast<char *>(row),v*sizeof(float));
        if(i>=n-1) tokens.push_back(std::max_element(row,row+v)-row);
    }
    if(!ids || !logits) return 8;
    llama_free(ctx);
    }
    llama_model_free(model); llama_backend_free();
}
