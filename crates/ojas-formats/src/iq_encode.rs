//! IQ2_XXS / IQ3_XXS *encoders* (quantizers), ported byte-exact from
//! the reference quantizers (`quantize_row_iq2_xxs_impl` @3222,
//! `quantize_row_iq3_xxs_impl` @3866). A mechanical port; output is
//! byte-identical to the reference chunk quantizer.
//!
//! The kmap (inverse grid) and kneighbours (per-missing-point neighbour
//! lists) are built once at runtime from the shared grid tables in
//! `iq_tables.rs` (matching `iq2xs_init_impl` / `iq3xs_init_impl`) and
//! cached in a `OnceLock`.
//!
//! Public API:
//!   `encode_iq2_xxs(x, k, imatrix) -> Vec<u8>`  (66 B / 256)
//!   `encode_iq3_xxs(x, k, imatrix) -> Vec<u8>`  (98 B / 256)
//! where `k = n_per_row` (multiple of 256), `x.len() == k` (one row),
//! `imatrix.len() == k` (per-column weights). `imatrix` must be non-null
//! and positive (the reference asserts `quant_weights`).

use std::sync::OnceLock;

const QK_K: usize = 256;

// The encoder builds its search grid from these *seed* tables (packed
// 2-bit / 3-bit lattice coordinates), exactly as the reference `iq2xs_init_impl`
// / `iq3xs_init_impl` do. This is a different grid from the shipped
// `IQ2XXS_GRID` / `IQ3XXS_GRID` decode tables in `iq_tables.rs` — the
// encoder grid uses magnitudes {1,3,5,7} (2*l+1, l in 0..3) while the
// decode grid uses {8,25,43}. Both share the same 256 grid *indices*
// (the output stores indices, which the decoder later maps to {8,25,43}),
// so they must not be interchanged. See the reference @3039 / @3691.
static KGRID_2BIT_256: [u16; 256] = [
    0, 2, 5, 8, 10, 17, 20, 32, 34, 40, 42, 65, 68, 80, 88, 97,
    100, 128, 130, 138, 162, 257, 260, 272, 277, 320, 388, 408, 512, 514, 546, 642,
    1025, 1028, 1040, 1057, 1060, 1088, 1090, 1096, 1120, 1153, 1156, 1168, 1188, 1280, 1282, 1288,
    1312, 1350, 1385, 1408, 1425, 1545, 1552, 1600, 1668, 1700, 2048, 2053, 2056, 2068, 2088, 2113,
    2116, 2128, 2130, 2184, 2308, 2368, 2562, 2580, 4097, 4100, 4112, 4129, 4160, 4192, 4228, 4240,
    4245, 4352, 4360, 4384, 4432, 4442, 4480, 4644, 4677, 5120, 5128, 5152, 5157, 5193, 5248, 5400,
    5474, 5632, 5654, 6145, 6148, 6160, 6208, 6273, 6400, 6405, 6560, 6737, 8192, 8194, 8202, 8260,
    8289, 8320, 8322, 8489, 8520, 8704, 8706, 9217, 9220, 9232, 9280, 9302, 9472, 9537, 9572, 9872,
    10248, 10272, 10388, 10820, 16385, 16388, 16400, 16408, 16417, 16420, 16448, 16456, 16470, 16480, 16513, 16516,
    16528, 16640, 16672, 16737, 16768, 16773, 16897, 16912, 16968, 16982, 17000, 17408, 17416, 17440, 17536, 17561,
    17682, 17700, 17920, 18433, 18436, 18448, 18496, 18501, 18688, 18776, 18785, 18818, 19013, 19088, 20480, 20488,
    20497, 20505, 20512, 20608, 20616, 20740, 20802, 20900, 21137, 21648, 21650, 21770, 22017, 22100, 22528, 22545,
    22553, 22628, 22848, 23048, 24580, 24592, 24640, 24680, 24832, 24917, 25112, 25184, 25600, 25605, 25872, 25874,
    25988, 26690, 32768, 32770, 32778, 32833, 32898, 33028, 33048, 33088, 33297, 33793, 33796, 33808, 33813, 33856,
    33888, 34048, 34118, 34196, 34313, 34368, 34400, 34818, 35076, 35345, 36868, 36880, 36900, 36928, 37025, 37142,
    37248, 37445, 37888, 37922, 37956, 38225, 39041, 39200, 40962, 41040, 41093, 41225, 41472, 42008, 43088, 43268,
];

static KGRID_256: [u16; 256] = [
    0, 2, 4, 9, 11, 15, 16, 18, 25, 34, 59, 61, 65, 67, 72, 74,
    81, 85, 88, 90, 97, 108, 120, 128, 130, 132, 137, 144, 146, 153, 155, 159,
    169, 175, 189, 193, 199, 200, 202, 213, 248, 267, 287, 292, 303, 315, 317, 321,
    327, 346, 362, 413, 436, 456, 460, 462, 483, 497, 513, 515, 520, 522, 529, 531,
    536, 538, 540, 551, 552, 576, 578, 585, 592, 594, 641, 643, 648, 650, 657, 664,
    698, 704, 706, 720, 729, 742, 758, 769, 773, 808, 848, 852, 870, 889, 901, 978,
    992, 1024, 1026, 1033, 1035, 1040, 1042, 1046, 1049, 1058, 1089, 1091, 1093, 1096, 1098, 1105,
    1112, 1139, 1143, 1144, 1152, 1154, 1161, 1167, 1168, 1170, 1183, 1184, 1197, 1217, 1224, 1228,
    1272, 1276, 1309, 1323, 1347, 1367, 1377, 1404, 1473, 1475, 1486, 1509, 1537, 1544, 1546, 1553,
    1555, 1576, 1589, 1594, 1600, 1602, 1616, 1625, 1636, 1638, 1665, 1667, 1672, 1685, 1706, 1722,
    1737, 1755, 1816, 1831, 1850, 1856, 1862, 1874, 1901, 1932, 1950, 1971, 2011, 2032, 2052, 2063,
    2077, 2079, 2091, 2095, 2172, 2192, 2207, 2208, 2224, 2230, 2247, 2277, 2308, 2345, 2356, 2389,
    2403, 2424, 2501, 2504, 2506, 2520, 2570, 2593, 2616, 2624, 2630, 2646, 2669, 2700, 2714, 2746,
    2754, 2795, 2824, 2835, 2839, 2874, 2882, 2905, 2984, 3028, 3042, 3092, 3108, 3110, 3124, 3153,
    3185, 3215, 3252, 3288, 3294, 3364, 3397, 3434, 3483, 3523, 3537, 3587, 3589, 3591, 3592, 3610,
    3626, 3670, 3680, 3722, 3749, 3754, 3776, 3789, 3803, 3824, 3857, 3873, 3904, 3906, 3924, 3992,
];
const GROUP_MAX_EPS: f32 = 1e-15;
const GROUP_MAX_EPS_IQ3_XXS: f32 = 1e-8;

/// Exact port of the reference `nearest_int` (round-half-to-even via the
/// 12582912.f magic-add trick). Matches the C bit-manipulation.
#[inline(always)]
fn nearest_int(fval: f32) -> i32 {
    debug_assert!(fval.abs() <= 4194303.0);
    let val = fval + 12582912.0f32;
    let i = val.to_bits() as i32;
    (i & 0x007fffff) - 0x00400000
}

#[inline(always)]
fn fp32_to_fp16_bits(f: f32) -> u16 {
    half::f16::from_f32(f).to_bits()
}

// ============================ IQ2_XXS grid data ============================

struct Iq2Data {
    /// the_grid: 256 u64, each holds 8 magnitude bytes (2*l+1). Same as
    /// IQ2XXS_GRID (which already stores magnitudes).
    grid: Vec<u64>,
    /// kmap: size 43692, index = packed 8x2bit of (mag-1)/2. >=0 => grid
    /// index; <0 => -(offset+1) into `neighbours`.
    map: Vec<i32>,
    /// kneighbours: [count, idx0, idx1, ...] slices.
    neighbours: Vec<u16>,
}

struct Iq3Data {
    /// the_grid: 256 u32, each holds 4 magnitude bytes (2*l+1).
    grid: Vec<u32>,
    /// kmap: size 4096, index = packed 4x3bit of (mag-1)/2.
    map: Vec<i32>,
    neighbours: Vec<u16>,
}

static IQ2: OnceLock<Iq2Data> = OnceLock::new();
static IQ3: OnceLock<Iq3Data> = OnceLock::new();

fn iq2_data() -> &'static Iq2Data {
    IQ2.get_or_init(build_iq2_data)
}
fn iq3_data() -> &'static Iq3Data {
    IQ3.get_or_init(build_iq3_data)
}

/// Comparison used by the reference `qsort` (iq2/iq3_compare_func):
/// sort by d2 ascending, tie-break by grid index j ascending.
/// C `qsort` is not stable, but the total order here is strict on (d2, j),
/// so the result is deterministic regardless of stability.
#[inline]
fn cmp_dist(a: &(i32, i32), b: &(i32, i32)) -> std::cmp::Ordering {
    a.0.cmp(&b.0).then(a.1.cmp(&b.1))
}

fn build_iq2_data() -> Iq2Data {
    let grid_size = 256usize;
    let kmap_size = 43692usize;
    let nwant = 2usize;

    // Build the_grid from the seed lattice coords: pos[i] = 2*l+1.
    let mut grid = vec![0u64; grid_size];
    for k in 0..grid_size {
        let seed = KGRID_2BIT_256[k];
        let mut g: u64 = 0;
        for i in 0..8 {
            let l = (seed >> (2 * i)) & 0x3;
            let pos = (2 * l + 1) as u64; // magnitude byte in {1,3,5,7}
            g |= pos << (8 * i);
        }
        grid[k] = g;
    }

    let mut map = vec![-1i32; kmap_size];
    for i in 0..grid_size {
        let g = grid[i];
        let mut index: u16 = 0;
        for k in 0..8 {
            let byte = ((g >> (8 * k)) & 0xff) as i32;
            let q = ((byte - 1) / 2) as u16;
            index |= q << (2 * k);
        }
        map[index as usize] = i as i32;
    }

    // Pass 1: count neighbours per missing point.
    let mut n_per_i = vec![0i32; kmap_size];
    for i in 0..kmap_size {
        if map[i] >= 0 {
            continue;
        }
        let mut pos = [0i8; 8];
        for k in 0..8 {
            let l = (i >> (2 * k)) & 0x3;
            pos[k] = (2 * l + 1) as i8;
        }
        let mut dist2: Vec<(i32, i32)> = Vec::with_capacity(grid_size);
        for j in 0..grid_size {
            let pg = grid[j];
            let mut d2 = 0i32;
            for k in 0..8 {
                let g = ((pg >> (8 * k)) & 0xff) as i8 as i32;
                let diff = g - pos[k] as i32;
                d2 += diff * diff;
            }
            dist2.push((d2, j as i32));
        }
        dist2.sort_by(cmp_dist);
        let mut n = 0i32;
        let mut d2 = dist2[0].0;
        let mut nhave = 1;
        for j in 0..grid_size {
            if dist2[j].0 > d2 {
                if nhave == nwant {
                    break;
                }
                d2 = dist2[j].0;
                nhave += 1;
            }
            n += 1;
        }
        n_per_i[i] = n;
    }

    // Prefix offsets.
    let mut offsets = vec![-1i32; kmap_size];
    let mut counter = 0i32;
    for i in 0..kmap_size {
        if map[i] >= 0 {
            continue;
        }
        offsets[i] = counter;
        counter += 1 + n_per_i[i];
    }
    let total = counter as usize;
    let mut neighbours = vec![0u16; total];

    // Pass 2: fill neighbour lists.
    for i in 0..kmap_size {
        if map[i] >= 0 {
            continue;
        }
        let mut pos = [0i8; 8];
        for k in 0..8 {
            let l = (i >> (2 * k)) & 0x3;
            pos[k] = (2 * l + 1) as i8;
        }
        let mut dist2: Vec<(i32, i32)> = Vec::with_capacity(grid_size);
        for j in 0..grid_size {
            let pg = grid[j];
            let mut d2 = 0i32;
            for k in 0..8 {
                let g = ((pg >> (8 * k)) & 0xff) as i8 as i32;
                let diff = g - pos[k] as i32;
                d2 += diff * diff;
            }
            dist2.push((d2, j as i32));
        }
        dist2.sort_by(cmp_dist);
        let mut local_counter = offsets[i] as usize;
        map[i] = -(local_counter as i32 + 1);
        let start = local_counter;
        local_counter += 1;
        let mut d2 = dist2[0].0;
        let mut n = 0u16;
        let mut nhave = 1;
        for j in 0..grid_size {
            if dist2[j].0 > d2 {
                if nhave == nwant {
                    break;
                }
                d2 = dist2[j].0;
                nhave += 1;
            }
            neighbours[local_counter] = dist2[j].1 as u16;
            local_counter += 1;
            n += 1;
        }
        neighbours[start] = n;
    }

    Iq2Data { grid, map, neighbours }
}

fn build_iq3_data() -> Iq3Data {
    let grid_size = 256usize;
    let kmap_size = 4096usize;
    let nwant = 2usize;

    // Build the_grid from seed lattice coords: pos[i] = 2*l+1, l in 0..7.
    let mut grid = vec![0u32; grid_size];
    for k in 0..grid_size {
        let seed = KGRID_256[k];
        let mut g: u32 = 0;
        for i in 0..4 {
            let l = (seed >> (3 * i)) & 0x7;
            let pos = (2 * l + 1) as u32; // magnitude byte in {1,3,..,15}
            g |= pos << (8 * i);
        }
        grid[k] = g;
    }

    let mut map = vec![-1i32; kmap_size];
    for i in 0..grid_size {
        let g = grid[i];
        let mut index: u16 = 0;
        for k in 0..4 {
            let byte = ((g >> (8 * k)) & 0xff) as i32;
            let q = ((byte - 1) / 2) as u16;
            index |= q << (3 * k);
        }
        map[index as usize] = i as i32;
    }

    let mut n_per_i = vec![0i32; kmap_size];
    for i in 0..kmap_size {
        if map[i] >= 0 {
            continue;
        }
        let mut pos = [0i8; 4];
        for k in 0..4 {
            let l = (i >> (3 * k)) & 0x7;
            pos[k] = (2 * l + 1) as i8;
        }
        let mut dist2: Vec<(i32, i32)> = Vec::with_capacity(grid_size);
        for j in 0..grid_size {
            let pg = grid[j];
            let mut d2 = 0i32;
            for k in 0..4 {
                let g = ((pg >> (8 * k)) & 0xff) as i8 as i32;
                let diff = g - pos[k] as i32;
                d2 += diff * diff;
            }
            dist2.push((d2, j as i32));
        }
        dist2.sort_by(cmp_dist);
        let mut n = 0i32;
        let mut d2 = dist2[0].0;
        let mut nhave = 1;
        for j in 0..grid_size {
            if dist2[j].0 > d2 {
                if nhave == nwant {
                    break;
                }
                d2 = dist2[j].0;
                nhave += 1;
            }
            n += 1;
        }
        n_per_i[i] = n;
    }

    let mut offsets = vec![-1i32; kmap_size];
    let mut counter = 0i32;
    for i in 0..kmap_size {
        if map[i] >= 0 {
            continue;
        }
        offsets[i] = counter;
        counter += 1 + n_per_i[i];
    }
    let total = counter as usize;
    let mut neighbours = vec![0u16; total];

    for i in 0..kmap_size {
        if map[i] >= 0 {
            continue;
        }
        let mut pos = [0i8; 4];
        for k in 0..4 {
            let l = (i >> (3 * k)) & 0x7;
            pos[k] = (2 * l + 1) as i8;
        }
        let mut dist2: Vec<(i32, i32)> = Vec::with_capacity(grid_size);
        for j in 0..grid_size {
            let pg = grid[j];
            let mut d2 = 0i32;
            for k in 0..4 {
                let g = ((pg >> (8 * k)) & 0xff) as i8 as i32;
                let diff = g - pos[k] as i32;
                d2 += diff * diff;
            }
            dist2.push((d2, j as i32));
        }
        dist2.sort_by(cmp_dist);
        let mut local_counter = offsets[i] as usize;
        map[i] = -(local_counter as i32 + 1);
        let start = local_counter;
        local_counter += 1;
        let mut d2 = dist2[0].0;
        let mut n = 0u16;
        let mut nhave = 1;
        for j in 0..grid_size {
            if dist2[j].0 > d2 {
                if nhave == nwant {
                    break;
                }
                d2 = dist2[j].0;
                nhave += 1;
            }
            neighbours[local_counter] = dist2[j].1 as u16;
            local_counter += 1;
            n += 1;
        }
        neighbours[start] = n;
    }

    Iq3Data { grid, map, neighbours }
}

// ======================= make_qp_quants (reference @1018) =======================

fn make_qp_quants(n: usize, nmax: i32, x: &[f32], l: &mut [u8], quant_weights: &[f32]) -> f32 {
    let mut max = 0.0f32;
    for i in 0..n {
        max = max.max(x[i]);
    }
    if max < GROUP_MAX_EPS {
        for i in 0..n {
            l[i] = 0;
        }
        return 0.0;
    }
    let mut iscale = nmax as f32 / max;
    for i in 0..n {
        l[i] = nearest_int(iscale * x[i]) as u8;
    }
    let scale = 1.0 / iscale;
    let mut best_mse = 0.0f32;
    for i in 0..n {
        let diff = x[i] - scale * l[i] as f32;
        let w = quant_weights[i];
        best_mse += w * diff * diff;
    }
    for is in -4..=4 {
        if is == 0 {
            continue;
        }
        let iscale_is = (0.1f32 * is as f32 + nmax as f32) / max;
        let scale_is = 1.0 / iscale_is;
        let mut mse = 0.0f32;
        for i in 0..n {
            let mut li = nearest_int(iscale_is * x[i]);
            li = li.min(nmax);
            let diff = x[i] - scale_is * li as f32;
            let w = quant_weights[i];
            mse += w * diff * diff;
        }
        if mse < best_mse {
            best_mse = mse;
            iscale = iscale_is;
        }
    }
    let mut sumlx = 0.0f32;
    let mut suml2 = 0.0f32;
    for i in 0..n {
        let mut li = nearest_int(iscale * x[i]);
        li = li.min(nmax);
        l[i] = li as u8;
        let w = quant_weights[i];
        sumlx += w * x[i] * li as f32;
        suml2 += w * li as f32 * li as f32;
    }
    for _itry in 0..5 {
        let mut n_changed = 0;
        for i in 0..n {
            let w = quant_weights[i];
            let slx = sumlx - w * x[i] * l[i] as f32;
            let sl2 = suml2 - w * l[i] as f32 * l[i] as f32;
            if slx > 0.0 && sl2 > 0.0 {
                let mut new_l = nearest_int(x[i] * sl2 / slx);
                new_l = new_l.min(nmax);
                if new_l != l[i] as i32 {
                    let slx2 = slx + w * x[i] * new_l as f32;
                    let sl2b = sl2 + w * new_l as f32 * new_l as f32;
                    if slx2 * slx2 * suml2 > sumlx * sumlx * sl2b {
                        l[i] = new_l as u8;
                        sumlx = slx2;
                        suml2 = sl2b;
                        n_changed += 1;
                    }
                }
            }
        }
        if n_changed == 0 {
            break;
        }
    }
    if suml2 > 0.0 {
        sumlx / suml2
    } else {
        0.0
    }
}

// ===================== iq2_find_best_neighbour (@3198) =====================

fn iq2_find_best_neighbour(
    data: &Iq2Data,
    neigh_off: usize,
    xval: &[f32],
    weight: &[f32],
    scale: f32,
    l: &mut [i8],
) -> i32 {
    let num_neighbors = data.neighbours[neigh_off] as usize;
    debug_assert!(num_neighbors > 0);
    let mut best_d2 = f32::MAX;
    let mut grid_index: i32 = -1;
    for j in 1..=num_neighbors {
        let gi = data.neighbours[neigh_off + j] as usize;
        let pg = data.grid[gi];
        let mut d2 = 0.0f32;
        for i in 0..8 {
            let q = ((pg >> (8 * i)) & 0xff) as i8 as f32;
            let diff = scale * q - xval[i];
            d2 += weight[i] * diff * diff;
        }
        if d2 < best_d2 {
            best_d2 = d2;
            grid_index = gi as i32;
        }
    }
    debug_assert!(grid_index >= 0);
    let pg = data.grid[grid_index as usize];
    for i in 0..8 {
        let byte = ((pg >> (8 * i)) & 0xff) as i8 as i32;
        l[i] = ((byte - 1) / 2) as i8;
    }
    grid_index
}

fn iq3_find_best_neighbour(
    data: &Iq3Data,
    neigh_off: usize,
    xval: &[f32],
    weight: &[f32],
    scale: f32,
    l: &mut [i8],
) -> i32 {
    let num_neighbors = data.neighbours[neigh_off] as usize;
    debug_assert!(num_neighbors > 0);
    let mut best_d2 = f32::MAX;
    let mut grid_index: i32 = -1;
    for j in 1..=num_neighbors {
        let gi = data.neighbours[neigh_off + j] as usize;
        let pg = data.grid[gi];
        let mut d2 = 0.0f32;
        for i in 0..4 {
            let q = ((pg >> (8 * i)) & 0xff) as i8 as f32;
            let diff = scale * q - xval[i];
            d2 += weight[i] * diff * diff;
        }
        if d2 < best_d2 {
            best_d2 = d2;
            grid_index = gi as i32;
        }
    }
    debug_assert!(grid_index >= 0);
    let pg = data.grid[grid_index as usize];
    for i in 0..4 {
        let byte = ((pg >> (8 * i)) & 0xff) as i8 as i32;
        l[i] = ((byte - 1) / 2) as i8;
    }
    grid_index
}

// ============================ IQ2_XXS encoder =============================

/// Encode one row (`k` floats, `k` multiple of 256) into IQ2_XXS blocks
/// (66 bytes each). `imatrix` (length `k`) are per-column quant weights.
pub fn encode_iq2_xxs(x: &[f32], k: usize, imatrix: &[f32]) -> Vec<u8> {
    assert!(k % QK_K == 0, "k must be a multiple of 256");
    assert_eq!(x.len(), k);
    assert_eq!(imatrix.len(), k);
    let data = iq2_data();

    let k_max_q = 3i32;
    let nbl = k / QK_K;
    let mut out = vec![0u8; nbl * 66];

    let mut scales = [0.0f32; QK_K / 32];
    let mut weight = [0.0f32; 32];
    let mut xval = [0.0f32; 32];
    let mut l = [0i8; 32];
    let mut laux = [0i8; 32];
    let mut waux = [0.0f32; 32];
    let mut block_signs = [0u8; 4];
    let mut q2 = [0u32; 2 * (QK_K / 32)];

    for ibl in 0..nbl {
        let blk = &mut out[ibl * 66..(ibl + 1) * 66];
        // d = 0 initially.
        for v in q2.iter_mut() {
            *v = 0;
        }

        let mut max_scale = 0.0f32;

        let xbl = &x[QK_K * ibl..QK_K * ibl + QK_K];
        let mut sumx2 = 0.0f32;
        for i in 0..QK_K {
            sumx2 += xbl[i] * xbl[i];
        }
        let sigma2 = sumx2 / QK_K as f32;

        for ib in 0..(QK_K / 32) {
            let xb = &xbl[32 * ib..32 * ib + 32];
            let qw = &imatrix[QK_K * ibl + 32 * ib..QK_K * ibl + 32 * ib + 32];
            for i in 0..32 {
                weight[i] = qw[i] * (sigma2 + xb[i] * xb[i]).sqrt();
            }
            for i in 0..32 {
                waux[i] = weight[i].sqrt();
            }
            for kk in 0..4 {
                let mut nflip = 0;
                let mut s: u8 = 0;
                for i in 0..8 {
                    if xb[8 * kk + i] >= 0.0 {
                        xval[8 * kk + i] = xb[8 * kk + i];
                    } else {
                        xval[8 * kk + i] = -xb[8 * kk + i];
                        nflip += 1;
                        s |= 1 << i;
                    }
                }
                if nflip % 2 != 0 {
                    let mut imin = 0;
                    let mut min = weight[8 * kk] * xb[8 * kk] * xb[8 * kk];
                    for i in 1..8 {
                        let ax = weight[8 * kk + i] * xb[8 * kk + i] * xb[8 * kk + i];
                        if ax < min {
                            min = ax;
                            imin = i;
                        }
                    }
                    xval[8 * kk + imin] = -xval[8 * kk + imin];
                    s ^= 1 << imin;
                }
                block_signs[kk] = s & 127;
            }
            let mut max = xval[0];
            for i in 1..32 {
                max = max.max(xval[i]);
            }
            if max < GROUP_MAX_EPS {
                scales[ib] = 0.0;
                for v in l.iter_mut() {
                    *v = 0;
                }
                continue;
            }
            // make_qp_quants writes into L as u8.
            let mut l_u8 = [0u8; 32];
            let mut scale = make_qp_quants(32, k_max_q + 1, &xval, &mut l_u8, &weight);
            for i in 0..32 {
                l[i] = l_u8[i] as i8;
            }
            let eff_max = scale * k_max_q as f32;
            if eff_max <= 0.0 {
                scales[ib] = 0.0;
                for v in l.iter_mut() {
                    *v = 0;
                }
                continue;
            }
            let mut best = 0.0f32;
            for is in -6..=6 {
                let id = (2 * k_max_q - 1) as f32 + is as f32 * 0.1f32;
                let id = id / eff_max;
                let this_scale = 1.0 / id;
                for kk in 0..4 {
                    for i in 0..8 {
                        let li = nearest_int(0.5f32 * (id * xval[8 * kk + i] - 1.0));
                        laux[8 * kk + i] = li.max(0).min(k_max_q - 1) as i8;
                    }
                    let mut u: u16 = 0;
                    for i in 0..8 {
                        u |= (laux[8 * kk + i] as u16) << (2 * i);
                    }
                    let grid_index = data.map[u as usize];
                    if grid_index < 0 {
                        let neigh_off = (-data.map[u as usize] - 1) as usize;
                        iq2_find_best_neighbour(
                            data,
                            neigh_off,
                            &xval[8 * kk..],
                            &waux[8 * kk..],
                            this_scale,
                            &mut laux[8 * kk..],
                        );
                    }
                }
                let mut sumqx = 0.0f32;
                let mut sumq2 = 0.0f32;
                for i in 0..32 {
                    let w = weight[i];
                    let q = 2.0 * laux[i] as f32 + 1.0;
                    sumqx += w * xval[i] * q;
                    sumq2 += w * q * q;
                }
                if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
                    scale = sumqx / sumq2;
                    best = scale * sumqx;
                    l.copy_from_slice(&laux);
                }
            }
            if scale > 0.0 {
                let id = 1.0 / scale;
                for kk in 0..4 {
                    let mut u: u16 = 0;
                    for i in 0..8 {
                        let mut li = nearest_int(0.5f32 * (id * xval[8 * kk + i] - 1.0));
                        li = li.max(0).min(k_max_q - 1);
                        u |= (li as u16) << (2 * i);
                    }
                    let grid_index = data.map[u as usize];
                    let grid_index = if grid_index < 0 {
                        let neigh_off = (-data.map[u as usize] - 1) as usize;
                        iq2_find_best_neighbour(
                            data,
                            neigh_off,
                            &xval[8 * kk..],
                            &waux[8 * kk..],
                            scale,
                            &mut l[8 * kk..],
                        )
                    } else {
                        grid_index
                    };
                    let pg = data.grid[grid_index as usize];
                    for i in 0..8 {
                        let byte = ((pg >> (8 * i)) & 0xff) as i8 as i32;
                        l[8 * kk + i] = ((byte - 1) / 2) as i8;
                    }
                }
                let mut sumqx = 0.0f32;
                let mut sumq2 = 0.0f32;
                for i in 0..32 {
                    let w = weight[i];
                    let q = 2.0 * l[i] as f32 + 1.0;
                    sumqx += w * xval[i] * q;
                    sumq2 += w * q * q;
                }
                if sumq2 > 0.0 {
                    scale = sumqx / sumq2;
                }
            }
            if scale < 0.0 {
                scale = -scale;
                for kk in 0..4 {
                    block_signs[kk] = (!block_signs[kk]) & 127;
                }
            }
            for kk in 0..4 {
                let mut u: u16 = 0;
                for i in 0..8 {
                    u |= (l[8 * kk + i] as u16) << (2 * i);
                }
                let grid_index = data.map[u as usize];
                if grid_index < 0 {
                    panic!("Oops: found point {} not on grid", u);
                }
                q2[2 * ib] |= (grid_index as u32) << (8 * kk);
                q2[2 * ib + 1] |= (block_signs[kk] as u32) << (7 * kk);
            }
            debug_assert!(scale >= 0.0);
            scales[ib] = scale;
            max_scale = max_scale.max(scale);
        }

        if max_scale == 0.0 {
            // qs already zeroed; d already 0.
            let dbits = fp32_to_fp16_bits(0.0);
            blk[0..2].copy_from_slice(&dbits.to_le_bytes());
            for i in 0..64 {
                blk[2 + i] = 0;
            }
            continue;
        }

        let d = max_scale / 31.0;
        let dbits = fp32_to_fp16_bits(d);
        let id = 1.0 / d;
        for ib in 0..(QK_K / 32) {
            let mut li = nearest_int(0.5f32 * (id * scales[ib] - 1.0));
            li = li.max(0).min(15);
            q2[2 * ib + 1] |= (li as u32) << 28;
        }
        blk[0..2].copy_from_slice(&dbits.to_le_bytes());
        for i in 0..16 {
            blk[2 + 4 * i..2 + 4 * i + 4].copy_from_slice(&q2[i].to_le_bytes());
        }
    }

    out
}

// ============================ IQ3_XXS encoder =============================

/// Encode one row (`k` floats, `k` multiple of 256) into IQ3_XXS blocks
/// (98 bytes each). `imatrix` (length `k`) are per-column quant weights.
pub fn encode_iq3_xxs(x: &[f32], k: usize, imatrix: &[f32]) -> Vec<u8> {
    assert!(k % QK_K == 0, "k must be a multiple of 256");
    assert_eq!(x.len(), k);
    assert_eq!(imatrix.len(), k);
    let data = iq3_data();

    let k_max_q = 8i32;
    let nbl = k / QK_K;
    let mut out = vec![0u8; nbl * 98];

    let mut scales = [0.0f32; QK_K / 32];
    let mut weight = [0.0f32; 32];
    let mut xval = [0.0f32; 32];
    let mut l = [0i8; 32];
    let mut laux = [0i8; 32];
    let mut waux = [0.0f32; 32];
    let mut is_on_grid = [true; 8];
    let mut is_on_grid_aux = [true; 8];
    let mut block_signs = [0u8; 8];
    // Reference scratch is one array, uint8_t q3[3*(QK_K/8)+QK_K/32], with
    // scales_and_signs aliased at q3+QK_K/4 (=64). The two regions are kept as
    // separate arrays here; the block payload after the f16 d is 96 bytes:
    // grid-index bytes [0..64), then scales_and_signs (8 u32 LE) [64..96).
    let mut q3 = [0u8; 3 * (QK_K / 8)]; // 96 grid-index bytes
    let mut scales_and_signs = [0u32; QK_K / 32]; // 8 u32

    for ibl in 0..nbl {
        let blk = &mut out[ibl * 98..(ibl + 1) * 98];
        for v in q3.iter_mut() {
            *v = 0;
        }
        for v in scales_and_signs.iter_mut() {
            *v = 0;
        }

        let mut max_scale = 0.0f32;

        let xbl = &x[QK_K * ibl..QK_K * ibl + QK_K];
        let mut sumx2 = 0.0f32;
        for i in 0..QK_K {
            sumx2 += xbl[i] * xbl[i];
        }
        let sigma2 = 2.0 * sumx2 / QK_K as f32;

        for ib in 0..(QK_K / 32) {
            let xb = &xbl[32 * ib..32 * ib + 32];
            let qw = &imatrix[QK_K * ibl + 32 * ib..QK_K * ibl + 32 * ib + 32];
            for i in 0..32 {
                weight[i] = qw[i] * (sigma2 + xb[i] * xb[i]).sqrt();
            }
            for i in 0..32 {
                waux[i] = weight[i].sqrt();
            }
            for kk in 0..4 {
                let mut nflip = 0;
                let mut s: u8 = 0;
                for i in 0..8 {
                    if xb[8 * kk + i] >= 0.0 {
                        xval[8 * kk + i] = xb[8 * kk + i];
                    } else {
                        xval[8 * kk + i] = -xb[8 * kk + i];
                        nflip += 1;
                        s |= 1 << i;
                    }
                }
                if nflip % 2 != 0 {
                    let mut imin = 0;
                    let mut min = weight[8 * kk] * xb[8 * kk] * xb[8 * kk];
                    for i in 1..8 {
                        let ax = weight[8 * kk + i] * xb[8 * kk + i] * xb[8 * kk + i];
                        if ax < min {
                            min = ax;
                            imin = i;
                        }
                    }
                    xval[8 * kk + imin] = -xval[8 * kk + imin];
                    s ^= 1 << imin;
                }
                block_signs[kk] = s & 127;
            }
            let mut max = xval[0];
            for i in 1..32 {
                max = max.max(xval[i]);
            }
            for v in l.iter_mut() {
                *v = 0;
            }
            if max < GROUP_MAX_EPS_IQ3_XXS {
                scales[ib] = 0.0;
                continue;
            }
            let mut best = 0.0f32;
            let mut scale = max / (2 * k_max_q - 1) as f32;
            for kk in 0..8 {
                is_on_grid[kk] = true;
            }
            for is in -15..=15 {
                let id = (2 * k_max_q - 1) as f32 + is as f32 * 0.2f32;
                let id = id / max;
                let this_scale = 1.0 / id;
                for kk in 0..8 {
                    for i in 0..4 {
                        let li = nearest_int(0.5f32 * (id * xval[4 * kk + i] - 1.0));
                        laux[4 * kk + i] = li.max(0).min(k_max_q - 1) as i8;
                    }
                    let mut u: u16 = 0;
                    for i in 0..4 {
                        u |= (laux[4 * kk + i] as u16) << (3 * i);
                    }
                    let grid_index = data.map[u as usize];
                    is_on_grid_aux[kk] = true;
                    if grid_index < 0 {
                        is_on_grid_aux[kk] = false;
                        let neigh_off = (-data.map[u as usize] - 1) as usize;
                        iq3_find_best_neighbour(
                            data,
                            neigh_off,
                            &xval[4 * kk..],
                            &waux[4 * kk..],
                            this_scale,
                            &mut laux[4 * kk..],
                        );
                    }
                }
                let mut sumqx = 0.0f32;
                let mut sumq2 = 0.0f32;
                for i in 0..32 {
                    let w = weight[i];
                    let q = 2.0 * laux[i] as f32 + 1.0;
                    sumqx += w * xval[i] * q;
                    sumq2 += w * q * q;
                }
                if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
                    scale = sumqx / sumq2;
                    best = scale * sumqx;
                    for i in 0..32 {
                        l[i] = laux[i];
                    }
                    for kk in 0..8 {
                        is_on_grid[kk] = is_on_grid_aux[kk];
                    }
                }
            }
            let mut n_not_ongrid = 0;
            for kk in 0..8 {
                if !is_on_grid[kk] {
                    n_not_ongrid += 1;
                }
            }
            if n_not_ongrid > 0 && scale > 0.0 {
                let id = 1.0 / scale;
                for kk in 0..8 {
                    if is_on_grid[kk] {
                        continue;
                    }
                    let mut u: u16 = 0;
                    for i in 0..4 {
                        let mut li = nearest_int(0.5f32 * (id * xval[4 * kk + i] - 1.0));
                        li = li.max(0).min(k_max_q - 1);
                        u |= (li as u16) << (3 * i);
                    }
                    let grid_index = data.map[u as usize];
                    let grid_index = if grid_index < 0 {
                        let neigh_off = (-data.map[u as usize] - 1) as usize;
                        iq3_find_best_neighbour(
                            data,
                            neigh_off,
                            &xval[4 * kk..],
                            &waux[4 * kk..],
                            scale,
                            &mut l[4 * kk..],
                        )
                    } else {
                        grid_index
                    };
                    let pg = data.grid[grid_index as usize];
                    for i in 0..4 {
                        let byte = ((pg >> (8 * i)) & 0xff) as i8 as i32;
                        l[4 * kk + i] = ((byte - 1) / 2) as i8;
                    }
                }
                let mut sumqx = 0.0f32;
                let mut sumq2 = 0.0f32;
                for i in 0..32 {
                    let w = weight[i];
                    let q = 2.0 * l[i] as f32 + 1.0;
                    sumqx += w * xval[i] * q;
                    sumq2 += w * q * q;
                }
                if sumq2 > 0.0 {
                    scale = sumqx / sumq2;
                }
            }
            if scale < 0.0 {
                scale = -scale;
                for kk in 0..4 {
                    block_signs[kk] = (!block_signs[kk]) & 127;
                }
            }
            for kk in 0..8 {
                let mut u: u16 = 0;
                for i in 0..4 {
                    u |= (l[4 * kk + i] as u16) << (3 * i);
                }
                let grid_index = data.map[u as usize];
                if grid_index < 0 {
                    panic!("Oops: found point {} not on grid", u);
                }
                // grid_size == 256: q3[8*ib+k] = grid_index (fits in a byte).
                q3[8 * ib + kk] = grid_index as u8;
            }
            scales_and_signs[ib] = block_signs[0] as u32
                | ((block_signs[1] as u32) << 7)
                | ((block_signs[2] as u32) << 14)
                | ((block_signs[3] as u32) << 21);
            debug_assert!(scale >= 0.0);
            scales[ib] = scale;
            max_scale = max_scale.max(scale);
        }

        if max_scale == 0.0 {
            let dbits = fp32_to_fp16_bits(0.0);
            blk[0..2].copy_from_slice(&dbits.to_le_bytes());
            for i in 0..96 {
                blk[2 + i] = 0;
            }
            continue;
        }

        let d = max_scale / 31.0;
        let dbits = fp32_to_fp16_bits(d * 1.0125f32);
        let id = 1.0 / d;
        for ib in 0..(QK_K / 32) {
            let mut li = nearest_int(0.5f32 * (id * scales[ib] - 1.0));
            li = li.max(0).min(15);
            scales_and_signs[ib] |= (li as u32) << 28;
        }
        blk[0..2].copy_from_slice(&dbits.to_le_bytes());
        // quant_size = 98 - 2 = 96 bytes (the reference memcpys q3, whose
        // scales_and_signs region starts at q3+QK_K/4 = 64):
        // bytes [0..64) = grid indices, [64..96) = scales/signs.
        blk[2..2 + 64].copy_from_slice(&q3[0..64]);
        for i in 0..8 {
            blk[2 + 64 + 4 * i..2 + 64 + 4 * i + 4].copy_from_slice(&scales_and_signs[i].to_le_bytes());
        }
    }

    out
}
