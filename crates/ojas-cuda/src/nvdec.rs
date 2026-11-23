//! NVDEC hardware video decode (libnvcuvid, loaded at runtime like the rest
//! of the driver stack): Annex-B H.264 / HEVC bitstream in, NV12 frames in
//! device memory out — camera streams enter the vision pipeline without a
//! host decode or a raw-frame upload.
//!
//! Struct layouts follow NVIDIA's MIT-licensed Video Codec SDK headers as
//! shipped in FFmpeg's nv-codec-headers (`dynlink_cuviddec.h`,
//! `dynlink_nvcuvid.h`, revision eddcea9e); `tcu_ulong` is `unsigned long`
//! (64-bit) on Linux, enums are `int`.
//!
//! Flow: the NVIDIA parser (`cuvidParseVideoData`) calls back on this thread
//! to (1) create the decoder when it sees the sequence header, (2) submit each
//! picture to the decode engine (asynchronous), (3) hand out display-order
//! pictures, which are only queued. Collecting maps the queued pictures and
//! copies them into a ring of NV12 slots owned by the decoder. A returned
//! frame stays valid until the ring wraps (`slots` more frames).
//!
//! Many cameras: `feed` every decoder first (all their pictures decode on the
//! engine concurrently), then `collect_all` maps them together behind one
//! stream sync — mapping right in the display callback would wait for each
//! camera's decode in turn (the FFmpeg cuviddec frame-queue scheme).
//! `keep_every` drops pictures before the map/copy (analysis fps below the
//! stream fps: the engine still decodes every picture, P-frames need them).
//! `Mode::Keyframes` submits only intra pictures to the engine — roughly
//! 1/GOP of the decode work, for idle or low-tier cameras; switching back to
//! `Mode::All` takes effect at the next intra picture (nothing may reference
//! a skipped one).
//!
//! Output size: by default frames have the stream's displayed size. With
//! `DecoderOptions::output` the engine scales each picture itself (the
//! create-info's target size and rectangle), stretched or letterboxed as
//! asked; frames, ring slots and copies then have that size, and
//! `NvDecoder::geom` keeps the mapping back to stream pixels.
//!
//! Threads: every `CudaGpu` of a device holds the device's primary context,
//! bound to the calling thread (`bind_thread`), never floating — so driver
//! calls made here need no `cuvidCtxLock` section (the SDK asks for one only
//! around a floating context). Each decoder still owns a context lock and
//! passes it as `vidLock`, as the SDK's own decoder class does: the codec
//! library then enters the context itself for its internal driver calls
//! rather than relying on the caller's thread state. The lock is per decoder,
//! so it is never contended and decoders on different threads do not wait on
//! each other through it. What decoders do share is the device's decode
//! engine and, per `CudaGpu`, the stream their surface copies are queued on.

use std::collections::VecDeque;
use std::ffi::c_void;
use std::sync::Arc;

use anyhow::{anyhow, bail, ensure, Result};
use cudarc::driver::sys;

use crate::CudaGpu;

/// Which pictures go to the decode engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    All,
    /// intra (key) pictures only
    Keyframes,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Codec {
    H264,
    Hevc,
}

impl Codec {
    fn id(self) -> i32 {
        match self {
            Codec::H264 => 4, // cudaVideoCodec_H264
            Codec::Hevc => 8, // cudaVideoCodec_HEVC
        }
    }
}

/// How the source picture is placed in a scaled output surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fit {
    /// fills the whole surface; the aspect changes unless the sizes agree
    Stretch,
    /// aspect kept, centred; the rest of the surface is bars whose content is
    /// the decoder's (read only `OutputGeom::rect`)
    Letterbox,
}

/// Decoder output size: the decode engine scales each picture to `width` ×
/// `height` (rounded down to even, NV12) instead of a resize kernel later.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutputSize {
    pub width: usize,
    pub height: usize,
    pub fit: Fit,
}

/// Where the source picture is in the decoded surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutputGeom {
    /// the stream's displayed size
    pub src_w: usize,
    pub src_h: usize,
    /// the decoded surface (`DecodedFrame::{w, h}`)
    pub w: usize,
    pub h: usize,
    /// the picture inside the surface: x0, y0, x1, y1 (the whole surface
    /// unless letterboxed)
    pub rect: [usize; 4],
}

impl OutputGeom {
    /// Surface pixels per source pixel, per axis.
    pub fn scale(&self) -> (f32, f32) {
        let [x0, y0, x1, y1] = self.rect;
        ((x1 - x0) as f32 / self.src_w as f32, (y1 - y0) as f32 / self.src_h as f32)
    }

    /// Surface coordinates → source pixels (unclipped).
    pub fn to_source(&self, x: f32, y: f32) -> (f32, f32) {
        let (sx, sy) = self.scale();
        ((x - self.rect[0] as f32) / sx, (y - self.rect[1] as f32) / sy)
    }

    /// Source pixels → surface coordinates.
    pub fn to_surface(&self, x: f32, y: f32) -> (f32, f32) {
        let (sx, sy) = self.scale();
        (x * sx + self.rect[0] as f32, y * sy + self.rect[1] as f32)
    }
}

/// The surface a `src_w` × `src_h` stream decodes to: its own size without
/// `out`, else the asked size with the picture stretched or letterboxed
/// (every edge even, so chroma samples stay aligned).
pub fn output_geom(src_w: usize, src_h: usize, out: Option<OutputSize>) -> OutputGeom {
    let Some(o) = out else { return OutputGeom { src_w, src_h, w: src_w, h: src_h, rect: [0, 0, src_w, src_h] } };
    let (w, h) = ((o.width & !1).max(2), (o.height & !1).max(2));
    let rect = match o.fit {
        Fit::Stretch => [0, 0, w, h],
        Fit::Letterbox => {
            let r = (w as f32 / src_w as f32).min(h as f32 / src_h as f32);
            let cw = (((src_w as f32 * r).round() as usize) & !1).clamp(2, w);
            let ch = (((src_h as f32 * r).round() as usize) & !1).clamp(2, h);
            let (x0, y0) = (((w - cw) / 2) & !1, ((h - ch) / 2) & !1);
            [x0, y0, x0 + cw, y0 + ch]
        }
    };
    OutputGeom { src_w, src_h, w, h, rect }
}

/// How a decoder is created; the default is `NvDecoder::new`'s behaviour
/// with 8 ring slots.
#[derive(Debug, Clone, Copy)]
pub struct DecoderOptions {
    /// decoded frames kept alive in the output ring
    pub slots: usize,
    /// hand out one displayed picture in `keep_every`
    pub keep_every: u32,
    /// scale in the decoder; `None`: frames at the stream's size
    pub output: Option<OutputSize>,
}

impl Default for DecoderOptions {
    fn default() -> Self {
        DecoderOptions { slots: 8, keep_every: 1, output: None }
    }
}

// ---- FFI (see module docs for the source of each layout) ----

#[repr(C)]
#[derive(Clone, Copy)]
struct VideoFormat {
    codec: i32,
    frame_rate: [u32; 2],
    progressive_sequence: u8,
    bit_depth_luma_minus8: u8,
    bit_depth_chroma_minus8: u8,
    min_num_decode_surfaces: u8,
    coded_width: u32,
    coded_height: u32,
    display_area: [i32; 4], // left, top, right, bottom
    chroma_format: i32,
    bitrate: u32,
    display_aspect_ratio: [i32; 2],
    video_signal_description: [u8; 4],
    seqhdr_data_length: u32,
}

#[repr(C)]
struct DispInfo {
    picture_index: i32,
    progressive_frame: i32,
    top_field_first: i32,
    repeat_first_field: i32,
    timestamp: i64,
}

#[repr(C)]
struct SourcePacket {
    flags: u64,
    payload_size: u64,
    payload: *const u8,
    timestamp: i64,
}

type SeqCb = unsafe extern "C" fn(*mut c_void, *mut VideoFormat) -> i32;
type DecCb = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i32;
type DispCb = unsafe extern "C" fn(*mut c_void, *mut DispInfo) -> i32;

#[repr(C)]
struct ParserParams {
    codec_type: i32,
    max_num_decode_surfaces: u32,
    clock_rate: u32,
    error_threshold: u32,
    max_display_delay: u32,
    bitfields: u32, // bAnnexb:1, bMemoryOptimize:1, reserved:30
    reserved1: [u32; 4],
    user_data: *mut c_void,
    pfn_sequence: Option<SeqCb>,
    pfn_decode: Option<DecCb>,
    pfn_display: Option<DispCb>,
    pfn_op_point: *const c_void,
    pfn_sei: *const c_void,
    reserved2: [*mut c_void; 5],
    ext_video_info: *mut c_void,
}

#[repr(C)]
struct DecodeCreateInfo {
    width: u64,
    height: u64,
    num_decode_surfaces: u64,
    codec_type: i32,
    chroma_format: i32,
    creation_flags: u64,
    bit_depth_minus8: u64,
    intra_decode_only: u64,
    max_width: u64,
    max_height: u64,
    reserved1: u64,
    display_area: [i16; 4], // left, top, right, bottom
    output_format: i32,
    deinterlace_mode: i32,
    target_width: u64,
    target_height: u64,
    num_output_surfaces: u64,
    vid_lock: *mut c_void,
    target_rect: [i16; 4],
    enable_histogram: u64,
    enable_decode_features: u64,
    reserved2: [u64; 3],
}

#[repr(C)]
struct ProcParams {
    progressive_frame: i32,
    second_field: i32,
    top_field_first: i32,
    unpaired_field: i32,
    reserved_flags: u32,
    reserved_zero: u32,
    raw_input_dptr: u64,
    raw_input_pitch: u32,
    raw_input_format: u32,
    raw_output_dptr: u64,
    raw_output_pitch: u32,
    reserved1: u32,
    output_stream: *mut c_void,
    reserved: [u32; 46],
    histogram_dptr: *mut u64,
    ext: *mut c_void,
}

const CUVID_PKT_ENDOFSTREAM: u64 = 0x01;
const CUVID_PKT_TIMESTAMP: u64 = 0x02;

struct Api {
    _lib: libloading::Library,
    create_parser: unsafe extern "C" fn(*mut *mut c_void, *mut ParserParams) -> i32,
    parse: unsafe extern "C" fn(*mut c_void, *mut SourcePacket) -> i32,
    destroy_parser: unsafe extern "C" fn(*mut c_void) -> i32,
    create_decoder: unsafe extern "C" fn(*mut *mut c_void, *mut DecodeCreateInfo) -> i32,
    destroy_decoder: unsafe extern "C" fn(*mut c_void) -> i32,
    decode_picture: unsafe extern "C" fn(*mut c_void, *mut c_void) -> i32,
    map: unsafe extern "C" fn(*mut c_void, i32, *mut u64, *mut u32, *mut ProcParams) -> i32,
    unmap: unsafe extern "C" fn(*mut c_void, u64) -> i32,
    ctx_lock_create: unsafe extern "C" fn(*mut *mut c_void, sys::CUcontext) -> i32,
    ctx_lock_destroy: unsafe extern "C" fn(*mut c_void) -> i32,
}

impl Api {
    fn load() -> Result<Api> {
        unsafe {
            let lib = libloading::Library::new("libnvcuvid.so.1")
                .or_else(|_| libloading::Library::new("libnvcuvid.so"))
                .map_err(|e| anyhow!("NVDEC unavailable (libnvcuvid): {e}"))?;
            macro_rules! sym {
                ($n:literal) => {
                    *lib.get($n).map_err(|e| anyhow!("libnvcuvid: {}: {e}", String::from_utf8_lossy($n)))?
                };
            }
            Ok(Api {
                create_parser: sym!(b"cuvidCreateVideoParser\0"),
                parse: sym!(b"cuvidParseVideoData\0"),
                destroy_parser: sym!(b"cuvidDestroyVideoParser\0"),
                create_decoder: sym!(b"cuvidCreateDecoder\0"),
                destroy_decoder: sym!(b"cuvidDestroyDecoder\0"),
                decode_picture: sym!(b"cuvidDecodePicture\0"),
                map: sym!(b"cuvidMapVideoFrame64\0"),
                unmap: sym!(b"cuvidUnmapVideoFrame64\0"),
                ctx_lock_create: sym!(b"cuvidCtxLockCreate\0"),
                ctx_lock_destroy: sym!(b"cuvidCtxLockDestroy\0"),
                _lib: lib,
            })
        }
    }
}

/// Hardware decode is usable here: the driver's libnvcuvid loads.
pub fn available() -> bool {
    Api::load().is_ok()
}

/// One decoded frame, NV12 in device memory: Y plane (`h` rows of `pitch`
/// bytes) then the interleaved UV plane at `uv_off`.
#[derive(Debug, Clone, Copy)]
pub struct DecodedFrame {
    pub ptr: u64,
    pub pitch: usize,
    pub w: usize,
    pub h: usize,
    pub uv_off: usize,
    /// the packet timestamp the frame was decoded from
    pub timestamp: i64,
}

struct Inner {
    api: Arc<Api>,
    stream: sys::CUstream,
    /// context lock of this decoder (`vidLock`), destroyed after the decoder
    lock: *mut c_void,
    decoder: *mut c_void,
    /// asked output size (None: the stream's)
    output: Option<OutputSize>,
    /// stream size and where it is in the surface, once the decoder exists
    geom: Option<OutputGeom>,
    /// decoded surface size (the output size, not the stream's)
    w: usize,
    h: usize,
    ring: u64,
    slot_bytes: usize,
    slots: usize,
    next: usize,
    /// display-order pictures not yet mapped: (index, progressive, top field first, timestamp)
    pending: VecDeque<(i32, i32, i32, i64)>,
    /// decode surfaces beyond the parser's minimum: bound on `pending`
    extra: usize,
    keep_every: u32,
    shown: u64,
    /// pictures parsed in display order, decoded or skipped
    pictures: u64,
    /// frames already copied (queue overflow: mapped synchronously)
    ready: Vec<DecodedFrame>,
    mode: Mode,
    /// skipping inter pictures (Keyframes mode, or All until the next intra)
    skipping: bool,
    /// picture indices whose decode was skipped (not displayed)
    skipped: Vec<i32>,
    err: Option<String>,
}

impl Inner {
    fn fail(&mut self, e: String) -> i32 {
        if self.err.is_none() {
            self.err = Some(e);
        }
        0
    }
}

unsafe extern "C" fn on_sequence(user: *mut c_void, fmt: *mut VideoFormat) -> i32 {
    let me = &mut *(user as *mut Inner);
    let f = *fmt;
    if f.chroma_format != 1 || f.bit_depth_luma_minus8 != 0 {
        return me.fail(format!("NVDEC: only 8-bit 4:2:0 streams (chroma {}, depth-8 {})", f.chroma_format, f.bit_depth_luma_minus8));
    }
    let [l, t, r, b] = f.display_area;
    let (sw, sh) = ((r - l) as usize, (b - t) as usize);
    if let Some(g) = me.geom.filter(|_| !me.decoder.is_null()) {
        if (sw, sh) == (g.src_w, g.src_h) {
            return (f.min_num_decode_surfaces as usize + me.extra) as i32;
        }
        return me.fail(format!("NVDEC: resolution change {}x{} -> {sw}x{sh} (recreate the decoder)", g.src_w, g.src_h));
    }
    // the surface: the stream's size, or the asked one (scaled by the engine)
    let geom = output_geom(sw, sh, me.output);
    let (w, h) = (geom.w, geom.h);
    let target_rect = match me.output {
        Some(OutputSize { fit: Fit::Letterbox, .. }) => geom.rect.map(|v| v as i16),
        _ => [0; 4], // null rectangle: the whole surface
    };
    // queued pictures keep their surfaces: min + extra, and `pending <= extra`
    let surfaces = f.min_num_decode_surfaces as u64 + me.extra as u64;
    let mut ci = DecodeCreateInfo {
        width: f.coded_width as u64,
        height: f.coded_height as u64,
        num_decode_surfaces: surfaces,
        codec_type: f.codec,
        chroma_format: 1,
        creation_flags: 4, // cudaVideoCreate_PreferCUVID
        bit_depth_minus8: 0,
        intra_decode_only: 0,
        max_width: f.coded_width as u64,
        max_height: f.coded_height as u64,
        reserved1: 0,
        display_area: [l as i16, t as i16, r as i16, b as i16],
        output_format: 0,    // NV12
        deinterlace_mode: 0, // weave
        target_width: w as u64,
        target_height: h as u64,
        num_output_surfaces: 2, // mapped at once per decoder (collect maps one at a time)
        vid_lock: me.lock,
        target_rect,
        enable_histogram: 0,
        enable_decode_features: 0,
        reserved2: [0; 3],
    };
    let rc = (me.api.create_decoder)(&mut me.decoder, &mut ci);
    if rc != 0 {
        return me.fail(format!("cuvidCreateDecoder failed ({rc}) for {sw}x{sh} -> {w}x{h}"));
    }
    me.geom = Some(geom);
    me.w = w;
    me.h = h;
    me.slot_bytes = w * h * 3 / 2;
    let mut p: sys::CUdeviceptr = 0;
    let rc = sys::cuMemAlloc_v2(&mut p, me.slot_bytes * me.slots);
    if rc != sys::CUresult::CUDA_SUCCESS {
        return me.fail(format!("NVDEC ring alloc: {rc:?}"));
    }
    me.ring = p;
    surfaces as i32
}

unsafe extern "C" fn on_decode(user: *mut c_void, pic: *mut c_void) -> i32 {
    let me = &mut *(user as *mut Inner);
    if me.decoder.is_null() {
        return me.fail("NVDEC: picture before sequence header".into());
    }
    // CUVIDPICPARAMS: CurrPicIdx at byte 8, intra_pic_flag at byte 60
    let idx = *(pic as *const i32).add(2);
    let intra = *((pic as *const u8).add(60) as *const i32) != 0;
    me.skipped.retain(|&i| i != idx);
    if intra {
        me.skipping = me.mode == Mode::Keyframes;
    } else if me.skipping || me.mode == Mode::Keyframes {
        me.skipping = true;
        me.skipped.push(idx);
        return 1;
    }
    let rc = (me.api.decode_picture)(me.decoder, pic);
    if rc != 0 {
        return me.fail(format!("cuvidDecodePicture failed ({rc})"));
    }
    1
}

unsafe extern "C" fn on_display(user: *mut c_void, info: *mut DispInfo) -> i32 {
    let me = &mut *(user as *mut Inner);
    if info.is_null() {
        return 1; // end of stream
    }
    let d = &*info;
    me.pictures += 1;
    if let Some(k) = me.skipped.iter().position(|&i| i == d.picture_index) {
        me.skipped.swap_remove(k);
        return 1;
    }
    me.shown += 1;
    if (me.shown - 1) % me.keep_every as u64 != 0 {
        return 1;
    }
    // a full queue would let the parser reuse a queued surface: copy the
    // oldest picture out now (synchronously) to make room
    if me.pending.len() >= me.extra {
        match me.map_one() {
            Ok(Some((f, surf))) => {
                let rc = sys::cuStreamSynchronize(me.stream);
                (me.api.unmap)(me.decoder, surf);
                if rc != sys::CUresult::CUDA_SUCCESS {
                    return me.fail(format!("NVDEC copy sync: {rc:?}"));
                }
                if me.ready.len() >= me.slots {
                    return me.fail(format!("NVDEC: more than {} frames from one feed (ring slots): feed smaller pieces", me.slots));
                }
                me.ready.push(f);
            }
            Ok(None) => {}
            Err(e) => return me.fail(e.to_string()),
        }
    }
    me.pending.push_back((d.picture_index, d.progressive_frame, d.top_field_first, d.timestamp));
    1
}

impl Inner {
    /// Map the oldest queued picture and enqueue its copy into the next ring
    /// slot; returns the frame and the mapped surface (unmap after the copy ran).
    unsafe fn map_one(&mut self) -> Result<Option<(DecodedFrame, u64)>> {
        let Some((idx, prog, tff, ts)) = self.pending.pop_front() else { return Ok(None) };
        let mut pp: ProcParams = std::mem::zeroed();
        pp.progressive_frame = prog;
        pp.top_field_first = tff;
        pp.output_stream = self.stream as *mut c_void;
        let (mut src, mut pitch) = (0u64, 0u32);
        let rc = (self.api.map)(self.decoder, idx, &mut src, &mut pitch, &mut pp);
        if rc != 0 {
            bail!("cuvidMapVideoFrame failed ({rc})");
        }
        let dst = self.ring + (self.next * self.slot_bytes) as u64;
        self.next = (self.next + 1) % self.slots;
        // the mapped surface has the output size (the create-info's target), whatever the stream's
        let (w, h) = (self.w, self.h);
        // luma, then chroma (source chroma plane follows the target height, even-aligned)
        let planes = [(0usize, 0usize, h), ((h + 1) & !1, h, h / 2)];
        for (src_row, dst_row, rows) in planes {
            let m = sys::CUDA_MEMCPY2D {
                srcXInBytes: 0,
                srcY: 0,
                srcMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
                srcHost: std::ptr::null(),
                srcDevice: src + (src_row * pitch as usize) as u64,
                srcArray: std::ptr::null_mut(),
                srcPitch: pitch as usize,
                dstXInBytes: 0,
                dstY: 0,
                dstMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
                dstHost: std::ptr::null_mut(),
                dstDevice: dst + (dst_row * w) as u64,
                dstArray: std::ptr::null_mut(),
                dstPitch: w,
                WidthInBytes: w,
                Height: rows,
            };
            let rc = sys::cuMemcpy2DAsync_v2(&m, self.stream);
            if rc != sys::CUresult::CUDA_SUCCESS {
                (self.api.unmap)(self.decoder, src);
                bail!("NVDEC surface copy: {rc:?}");
            }
        }
        Ok(Some((DecodedFrame { ptr: dst, pitch: w, w, h, uv_off: w * h, timestamp: ts }, src)))
    }
}

/// A hardware decoder for one Annex-B H.264/HEVC stream (one camera).
pub struct NvDecoder {
    parser: *mut c_void,
    inner: Box<Inner>,
    gpu: Arc<CudaGpu>,
}

// The decoder is used from one thread at a time (callbacks run inside
// `decode` on the calling thread); moving it between threads is fine.
unsafe impl Send for NvDecoder {}

impl NvDecoder {
    /// `slots`: decoded frames kept alive in the output ring (>= the number
    /// of frames the caller holds before consuming them).
    pub fn new(gpu: Arc<CudaGpu>, codec: Codec, slots: usize) -> Result<NvDecoder> {
        Self::with_options(gpu, codec, slots, 1)
    }

    /// `keep_every`: hand out one displayed picture in `keep_every` (the rest
    /// are dropped before the map/copy; the engine still decodes them).
    pub fn with_options(gpu: Arc<CudaGpu>, codec: Codec, slots: usize, keep_every: u32) -> Result<NvDecoder> {
        Self::with(gpu, codec, DecoderOptions { slots, keep_every, output: None })
    }

    /// A decoder with every option. With `output`, frames come out at that
    /// size (`DecodedFrame::{w, h, pitch, uv_off}` describe the scaled
    /// surface); `geom()` maps their pixels back to the stream's.
    pub fn with(gpu: Arc<CudaGpu>, codec: Codec, opts: DecoderOptions) -> Result<NvDecoder> {
        let DecoderOptions { slots, keep_every, output } = opts;
        if let Some(o) = output {
            ensure!(o.width >= 2 && o.height >= 2 && o.width <= i16::MAX as usize && o.height <= i16::MAX as usize, "NVDEC output size {}x{}", o.width, o.height);
        }
        let api = Arc::new(Api::load()?);
        gpu.bind_thread()?;
        let mut lock = std::ptr::null_mut();
        let rc = unsafe { (api.ctx_lock_create)(&mut lock, gpu.cu_ctx()) };
        if rc != 0 {
            bail!("cuvidCtxLockCreate failed ({rc})");
        }
        let mut inner = Box::new(Inner {
            api: api.clone(),
            stream: gpu.cu_stream(),
            lock,
            decoder: std::ptr::null_mut(),
            output,
            geom: None,
            w: 0,
            h: 0,
            ring: 0,
            slot_bytes: 0,
            slots: slots.max(2),
            next: 0,
            pending: VecDeque::new(),
            extra: 4,
            keep_every: keep_every.max(1),
            shown: 0,
            pictures: 0,
            ready: vec![],
            mode: Mode::All,
            skipping: false,
            skipped: vec![],
            err: None,
        });
        let mut pp = ParserParams {
            codec_type: codec.id(),
            max_num_decode_surfaces: 1, // the sequence callback returns the real count
            clock_rate: 0,
            error_threshold: 0,
            max_display_delay: 0, // lowest latency: frames come out as soon as decodable
            bitfields: 0,
            reserved1: [0; 4],
            user_data: &mut *inner as *mut Inner as *mut c_void,
            pfn_sequence: Some(on_sequence),
            pfn_decode: Some(on_decode),
            pfn_display: Some(on_display),
            pfn_op_point: std::ptr::null(),
            pfn_sei: std::ptr::null(),
            reserved2: [std::ptr::null_mut(); 5],
            ext_video_info: std::ptr::null_mut(),
        };
        let mut parser = std::ptr::null_mut();
        let rc = unsafe { (api.create_parser)(&mut parser, &mut pp) };
        if rc != 0 {
            unsafe { (api.ctx_lock_destroy)(lock) };
            bail!("cuvidCreateVideoParser failed ({rc})");
        }
        Ok(NvDecoder { parser, inner, gpu })
    }

    /// Feed bitstream bytes (any split: whole access units or arbitrary
    /// chunks); pictures are submitted to the decode engine and displayable
    /// ones queued (past `max_pending()` queued, the oldest is copied out
    /// synchronously to free its surface). Returns the number of frames
    /// waiting for `collect`. `eos` flushes the pictures held for reordering.
    pub fn feed(&mut self, data: &[u8], timestamp: i64, eos: bool) -> Result<usize> {
        self.gpu.bind_thread()?;
        let mut pkt = SourcePacket {
            flags: CUVID_PKT_TIMESTAMP | if eos { CUVID_PKT_ENDOFSTREAM } else { 0 },
            payload_size: data.len() as u64,
            payload: data.as_ptr(),
            timestamp,
        };
        let rc = unsafe { (self.inner.api.parse)(self.parser, &mut pkt) };
        if let Some(e) = self.inner.err.take() {
            bail!(e);
        }
        if rc != 0 {
            bail!("cuvidParseVideoData failed ({rc})");
        }
        Ok(self.pending())
    }

    /// Pictures of the stream passed so far (decoded or skipped): stream time
    /// covered, in frames.
    pub fn pictures(&self) -> u64 {
        self.inner.pictures
    }

    /// Choose which pictures are decoded from here on (see `Mode`).
    pub fn set_mode(&mut self, mode: Mode) {
        self.inner.mode = mode;
    }

    /// Queued pictures the caller may let accumulate between collects.
    pub fn max_pending(&self) -> usize {
        self.inner.extra
    }

    /// Frames waiting for `collect` (queued pictures + overflow copies).
    pub fn pending(&self) -> usize {
        self.inner.pending.len() + self.inner.ready.len()
    }

    /// Map and copy up to `max` queued pictures (one stream sync).
    pub fn collect(&mut self, max: usize) -> Result<Vec<DecodedFrame>> {
        Ok(collect_all(std::slice::from_mut(self), max)?.remove(0))
    }

    /// `feed` then `collect` everything: the single-camera convenience.
    pub fn decode(&mut self, data: &[u8], timestamp: i64, eos: bool) -> Result<Vec<DecodedFrame>> {
        self.feed(data, timestamp, eos)?;
        self.collect(usize::MAX)
    }

    /// Decoded frame size (0 before the first sequence header): the output
    /// size when one was asked, else the stream's.
    pub fn size(&self) -> (usize, usize) {
        (self.inner.w, self.inner.h)
    }

    /// The stream's size and where its picture is in the decoded frames
    /// (None before the first sequence header).
    pub fn geom(&self) -> Option<OutputGeom> {
        self.inner.geom
    }
}

/// Collect up to `max` queued pictures from each decoder (all on the same
/// `CudaGpu`): map, enqueue every copy, sync the stream once, unmap. Per
/// decoder, one mapped surface per pass (`num_output_surfaces` bounds it).
pub fn collect_all(decs: &mut [NvDecoder], max: usize) -> Result<Vec<Vec<DecodedFrame>>> {
    // overflow copies first (already complete), oldest first
    let mut out: Vec<Vec<DecodedFrame>> = decs
        .iter_mut()
        .map(|d| {
            let n = d.inner.ready.len().min(max);
            d.inner.ready.drain(..n).collect()
        })
        .collect();
    let Some(first) = decs.first() else { return Ok(out) };
    let gpu = first.gpu.clone();
    ensure!(decs.iter().all(|d| Arc::ptr_eq(&d.gpu, &gpu)), "collect_all: decoders on different CudaGpu handles");
    gpu.bind_thread()?;
    loop {
        let mut mapped: Vec<(usize, u64)> = vec![];
        let mut res = Ok(());
        for (i, d) in decs.iter_mut().enumerate() {
            if out[i].len() >= max {
                continue;
            }
            match unsafe { d.inner.map_one() } {
                Ok(Some((f, surf))) => {
                    out[i].push(f);
                    mapped.push((i, surf));
                }
                Ok(None) => {}
                Err(e) => {
                    res = Err(e);
                    break;
                }
            }
        }
        if mapped.is_empty() {
            res?;
            return Ok(out);
        }
        // surfaces must not be released before their copies have run
        let rc = unsafe { sys::cuStreamSynchronize(gpu.cu_stream()) };
        for (i, surf) in mapped {
            unsafe { (decs[i].inner.api.unmap)(decs[i].inner.decoder, surf) };
        }
        res?;
        ensure!(rc == sys::CUresult::CUDA_SUCCESS, "NVDEC copy sync: {rc:?}");
    }
}

impl Drop for NvDecoder {
    fn drop(&mut self) {
        let _ = self.gpu.bind_thread();
        unsafe {
            (self.inner.api.destroy_parser)(self.parser);
            if !self.inner.decoder.is_null() {
                (self.inner.api.destroy_decoder)(self.inner.decoder);
            }
            if self.inner.ring != 0 {
                sys::cuMemFree_v2(self.inner.ring);
            }
            (self.inner.api.ctx_lock_destroy)(self.inner.lock);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_output_is_the_stream_size() {
        // odd sizes pass through untouched: nothing is rounded without an output size
        let g = output_geom(1919, 1081, None);
        assert_eq!((g.w, g.h, g.rect), (1919, 1081, [0, 0, 1919, 1081]));
        assert_eq!(g.scale(), (1.0, 1.0));
        assert_eq!(g.to_source(100.0, 50.0), (100.0, 50.0));
    }

    #[test]
    fn stretch_fills_the_surface() {
        let g = output_geom(1920, 1080, Some(OutputSize { width: 641, height: 641, fit: Fit::Stretch }));
        assert_eq!((g.w, g.h, g.rect), (640, 640, [0, 0, 640, 640]));
        let (sx, sy) = g.scale();
        assert!((sx - 1.0 / 3.0).abs() < 1e-6 && (sy - 640.0 / 1080.0).abs() < 1e-6);
        let (x, y) = g.to_source(320.0, 320.0);
        assert!((x - 960.0).abs() < 1e-2 && (y - 540.0).abs() < 1e-2);
    }

    #[test]
    fn letterbox_keeps_aspect_on_even_edges() {
        let g = output_geom(1920, 1080, Some(OutputSize { width: 640, height: 640, fit: Fit::Letterbox }));
        assert_eq!((g.w, g.h, g.rect), (640, 640, [0, 140, 640, 500]));
        // surface -> source -> surface round trip, and the picture's corners
        assert_eq!(g.to_source(0.0, 140.0), (0.0, 0.0));
        let (x, y) = g.to_source(640.0, 500.0);
        assert!((x - 1920.0).abs() < 1e-2 && (y - 1080.0).abs() < 1e-2);
        let (x, y) = g.to_surface(960.0, 540.0);
        assert!((x - 320.0).abs() < 1e-3 && (y - 320.0).abs() < 1e-3);
        // portrait and awkward sizes: rect inside the surface, every edge even, aspect within a pixel pair
        for (sw, sh, w, h) in [(1080usize, 1920usize, 640usize, 640usize), (1280, 720, 416, 416), (704, 576, 320, 192), (3840, 2160, 2, 2), (100, 4000, 64, 64)] {
            let g = output_geom(sw, sh, Some(OutputSize { width: w, height: h, fit: Fit::Letterbox }));
            let [x0, y0, x1, y1] = g.rect;
            assert!(x0 < x1 && y0 < y1 && x1 <= g.w && y1 <= g.h, "{g:?}");
            assert!([x0, y0, x1, y1, g.w, g.h].iter().all(|v| v % 2 == 0), "{g:?}");
            let r = (w as f32 / sw as f32).min(h as f32 / sh as f32);
            assert!(((x1 - x0) as f32 - sw as f32 * r).abs() <= 2.0 && ((y1 - y0) as f32 - sh as f32 * r).abs() <= 2.0, "{g:?}");
        }
    }
}
