//! Projective warps composited straight into a destination (the layer transform step).

use craft_resample::downsample2_clamped_rgba;
use effectcraft_color::{BlendMode, blend_pixel};
use effectcraft_geom::{Mat3, Rect, vec2};
use rayon::prelude::*;
use std::borrow::Cow;

use crate::{Image, Px, hash_noise};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Sampling {
    Nearest,
    #[default]
    Bilinear,
    Bicubic,
}

#[derive(Clone, Copy, Debug)]
pub struct WarpOpts {
    pub sampling: Sampling,
    pub opacity: f32,
    pub mode: BlendMode,
    pub seed: u32,
    /// Only touch destination pixels inside this rect (dst pixel coords).
    pub clip: Option<Rect>,
}

#[derive(Clone, Copy)]
struct WarpOperation {
    allow_fast: bool,
    accumulate: Option<f32>,
    quantize: Option<f32>,
    source_opaque: bool,
}

impl Default for WarpOpts {
    fn default() -> Self {
        WarpOpts { sampling: Sampling::Bilinear, opacity: 1.0, mode: BlendMode::Normal, seed: 0, clip: None }
    }
}

/// Box-filter halving (for minification).
fn half(src: &Image) -> Image {
    if let Some((width, height, data)) = downsample2_clamped_rgba(&src.data, src.width as usize, src.height as usize) {
        return Image { width: width as u32, height: height as u32, data };
    }
    Image::new(src.width.div_ceil(2).max(1), src.height.div_ceil(2).max(1))
}

/// Draw `src` transformed by `m` (src pixel space → dst pixel space) onto `dst` with blending.
///
/// Affine transforms with non-stencil modes take a fast path (per-row span clipping, interior
/// bilinear without bounds checks, integer-translation copies, inlined Normal blending); it
/// matches [`composite_warp_reference`] to float rounding.
pub fn composite_warp(dst: &mut Image, src: &Image, m: &Mat3, opts: &WarpOpts) {
    composite_warp_impl::<false>(dst, src, m, opts, WarpOperation { allow_fast: true, accumulate: None, quantize: None, source_opaque: false });
}

/// [`composite_warp`], also returning the destination pixels it changed when it knows them:
/// `Some([x0, y0, x1, y1])` (half-open, empty when nothing was drawn) on the fast affine path,
/// `None` otherwise. With `quantize` (an 8/16 bpc destination's levels), the fast path also
/// quantises each pixel it blends ([`crate::quantize_px`]): exactly when this returns `Some`.
pub fn composite_warp_changed(dst: &mut Image, src: &Image, m: &Mat3, opts: &WarpOpts, quantize: Option<f32>) -> Option<[u32; 4]> {
    composite_warp_impl::<true>(dst, src, m, opts, WarpOperation { allow_fast: true, accumulate: None, quantize, source_opaque: false })
}

/// [`composite_warp_changed`] for a source whose pixels are known to be fully opaque. This skips
/// building a transparency-tile map while retaining the same bilinear edge handling.
pub fn composite_warp_changed_opaque(dst: &mut Image, src: &Image, m: &Mat3, opts: &WarpOpts, quantize: Option<f32>) -> Option<[u32; 4]> {
    composite_warp_impl::<true>(dst, src, m, opts, WarpOperation { allow_fast: true, accumulate: None, quantize, source_opaque: true })
}

/// [`composite_warp`] with inline quantisation on the affine fast path. The caller can skip a
/// later quantisation pass when this returns `true`; it does not return changed bounds because
/// that reduction is unnecessary for this use. A no-op also returns `true`, since no later
/// quantisation is needed.
pub fn composite_warp_quantized(dst: &mut Image, src: &Image, m: &Mat3, opts: &WarpOpts, levels: f32) -> bool {
    composite_warp_impl::<false>(dst, src, m, opts, WarpOperation { allow_fast: true, accumulate: None, quantize: Some(levels), source_opaque: false })
        .is_some()
}

/// [`composite_warp_quantized`] for a source whose pixels are known to be fully opaque.
pub fn composite_warp_quantized_opaque(dst: &mut Image, src: &Image, m: &Mat3, opts: &WarpOpts, levels: f32) -> bool {
    composite_warp_impl::<false>(dst, src, m, opts, WarpOperation { allow_fast: true, accumulate: None, quantize: Some(levels), source_opaque: true }).is_some()
}

/// `dst += warp(src) · weight` (plain sum, no blending): motion-blur sub-samples accumulate
/// straight into one buffer instead of one full-size image per sample.
pub fn accumulate_warp(dst: &mut Image, src: &Image, m: &Mat3, sampling: Sampling, weight: f32) {
    let opts = WarpOpts { sampling, ..Default::default() };
    if m.is_affine() && m.inverse().is_some_and(|i| i.is_affine()) {
        composite_warp_impl::<false>(dst, src, m, &opts, WarpOperation { allow_fast: true, accumulate: Some(weight), quantize: None, source_opaque: false });
        return;
    }
    let mut tmp = Image::new(dst.width, dst.height);
    composite_warp(&mut tmp, src, m, &opts);
    dst.data.par_iter_mut().zip(tmp.data.par_iter()).for_each(|(d, s)| {
        for c in 0..4 {
            d[c] += s[c] * weight;
        }
    });
}

/// The straightforward per-pixel implementation (reference for the fast path's tests).
pub fn composite_warp_reference(dst: &mut Image, src: &Image, m: &Mat3, opts: &WarpOpts) {
    composite_warp_impl::<false>(dst, src, m, opts, WarpOperation { allow_fast: false, accumulate: None, quantize: None, source_opaque: false });
}

/// Bilinear blend of four taps (same operation order as [`Image::sample_bilinear`]).
#[inline(always)]
fn bilerp(a: Px, b: Px, c: Px, d: Px, tx: f32, ty: f32) -> Px {
    let mut o = [0.0; 4];
    for i in 0..4 {
        let top = a[i] + (b[i] - a[i]) * tx;
        let bot = c[i] + (d[i] - c[i]) * tx;
        o[i] = top + (bot - top) * ty;
    }
    o
}

/// Side of the square source tiles of [`ClearTiles`].
const TILE: usize = 16;

/// Which `TILE`×`TILE` source tiles are fully transparent, the one-pixel apron to their right
/// and below included: the four taps of a bilinear sample whose top-left tap lies in such a
/// tile all have zero alpha, so the sample is transparent and draws nothing.
struct ClearTiles {
    cols: usize,
    clear: Vec<bool>,
}

impl ClearTiles {
    fn new(img: &Image) -> ClearTiles {
        let (w, h) = (img.width as usize, img.height as usize);
        let (cols, rows) = (w.div_ceil(TILE), h.div_ceil(TILE));
        if w.saturating_mul(h) <= 64 * 1024 {
            let mut clear = vec![true; cols.saturating_mul(rows)];
            for ty in 0..rows {
                for tx in 0..cols {
                    let x1 = (tx * TILE + TILE + 1).min(w);
                    let y1 = (ty * TILE + TILE + 1).min(h);
                    clear[ty * cols + tx] = (ty * TILE..y1).all(|y| img.data[y * w + tx * TILE..y * w + x1].iter().all(|p| p[3] == 0.0));
                }
            }
            return ClearTiles { cols, clear };
        }
        let clear = (0..rows)
            .into_par_iter()
            .flat_map_iter(|ty| {
                let mut row_clear = vec![true; cols];
                for y in ty * TILE..(ty * TILE + TILE + 1).min(h) {
                    let row = &img.data[y * w..(y + 1) * w];
                    for (tx, c) in row_clear.iter_mut().enumerate() {
                        if *c {
                            *c = row[tx * TILE..(tx * TILE + TILE + 1).min(w)].iter().all(|p| p[3] == 0.0);
                        }
                    }
                }
                row_clear
            })
            .collect();
        ClearTiles { cols, clear }
    }

    #[inline(always)]
    fn is_clear(&self, x: usize, y: usize) -> bool {
        self.clear.get((y / TILE) * self.cols + x / TILE).copied().unwrap_or(false)
    }
}

/// Bilinear sample with an unchecked interior fast path; `None` when every interior tap is in a
/// [`ClearTiles`] tile (the sample would have zero alpha).
#[inline(always)]
fn sample_bilinear_fast(img: &Image, clear: Option<&ClearTiles>, x: f64, y: f64) -> Option<Px> {
    let fx = x - 0.5;
    let fy = y - 0.5;
    let x0 = fx.floor();
    let y0 = fy.floor();
    let (xi, yi) = (x0 as i64, y0 as i64);
    let (w, h) = (img.width as i64, img.height as i64);
    if xi >= 0 && yi >= 0 && xi + 1 < w && yi + 1 < h {
        if clear.is_some_and(|tiles| tiles.is_clear(xi as usize, yi as usize)) {
            return None;
        }
        let tx = (fx - x0) as f32;
        let ty = (fy - y0) as f32;
        let i = yi as usize * w as usize + xi as usize;
        let row0 = &img.data[i..i + 2];
        let row1 = &img.data[i + w as usize..i + w as usize + 2];
        return Some(bilerp(row0[0], row0[1], row1[0], row1[1], tx, ty));
    }
    Some(img.sample_bilinear(x, y))
}

/// One affine warp prepared once and applied to independent output rows. This lets a caller
/// preserve layer order within each row while Rayon owns disjoint rows for the whole layer batch.
pub struct PreparedWarpRows<'a> {
    img: Cow<'a, Image>,
    inv: Mat3,
    bounds: [usize; 4],
    sampling: Sampling,
    mode: BlendMode,
    opacity: f32,
    seed: u32,
    levels: Option<(f32, f32)>,
    clear: Option<ClearTiles>,
}

/// Prepare the affine fast path for a row-batched compositor. `None` means the caller should use
/// the general warp path (projective transforms and stencil modes require full-frame semantics).
pub fn prepare_warp_rows<'a>(
    src: &'a Image,
    m: &Mat3,
    opts: &WarpOpts,
    dst_width: u32,
    dst_height: u32,
    quantize: Option<f32>,
    source_opaque: bool,
) -> Option<PreparedWarpRows<'a>> {
    if opts.mode.is_stencil() || !m.is_affine() {
        return None;
    }
    if src.is_empty() || dst_width == 0 || dst_height == 0 || opts.opacity <= 0.0 {
        return Some(PreparedWarpRows::noop(src, opts, quantize));
    }
    let Some(mut inv) = m.inverse() else { return Some(PreparedWarpRows::noop(src, opts, quantize)) };
    if !inv.is_affine() {
        return None;
    }

    // Match composite_warp_impl's box-filter levels and inverse-matrix adjustment for
    // minification. The destination bounds remain based on the original source dimensions.
    let bounds = m.map_rect(&Rect::from_size(src.width as f64, src.height as f64)).inflate(1.0).round_out();
    let full = Rect::from_size(dst_width as f64, dst_height as f64);
    let mut area = bounds.intersect(&full);
    if let Some(clip) = opts.clip {
        area = area.intersect(&clip);
    }
    if area.is_empty() {
        return Some(PreparedWarpRows::noop(src, opts, quantize));
    }
    let bounds = [area.x0.max(0.0) as usize, area.x1.min(dst_width as f64) as usize, area.y0.max(0.0) as usize, area.y1.min(dst_height as f64) as usize];
    let mut img = Cow::Borrowed(src);
    let scale = m.mean_scale();
    if scale < 0.5 {
        let mut s = scale;
        let mut f = 1.0;
        let mut level = half(src);
        f *= 0.5;
        s *= 2.0;
        while s < 0.5 && level.width > 1 && level.height > 1 {
            level = half(&level);
            f *= 0.5;
            s *= 2.0;
        }
        img = Cow::Owned(level);
        inv = Mat3::scale(vec2(f, f)) * inv;
    }

    let int_tx =
        inv.0[0][0] == 1.0 && inv.0[1][1] == 1.0 && inv.0[0][1] == 0.0 && inv.0[1][0] == 0.0 && inv.0[0][2].fract() == 0.0 && inv.0[1][2].fract() == 0.0;
    let clear = (opts.sampling == Sampling::Bilinear && !int_tx && !source_opaque).then(|| ClearTiles::new(&img));
    Some(PreparedWarpRows {
        img,
        inv,
        bounds,
        sampling: opts.sampling,
        mode: opts.mode,
        opacity: opts.opacity,
        seed: opts.seed,
        levels: quantize.map(|levels| (levels, 1.0 / levels)),
        clear,
    })
}

impl PreparedWarpRows<'_> {
    fn noop<'a>(src: &'a Image, opts: &WarpOpts, quantize: Option<f32>) -> PreparedWarpRows<'a> {
        PreparedWarpRows {
            img: Cow::Borrowed(src),
            inv: Mat3::IDENTITY,
            bounds: [0; 4],
            sampling: opts.sampling,
            mode: opts.mode,
            opacity: opts.opacity,
            seed: opts.seed,
            levels: quantize.map(|levels| (levels, 1.0 / levels)),
            clear: None,
        }
    }

    /// Destination rows that can receive samples, as a half-open interval.
    pub fn y_bounds(&self) -> (usize, usize) {
        (self.bounds[2], self.bounds[3])
    }

    /// Composite this warp into one destination row. `row` spans the full output width.
    #[inline]
    pub fn blend_row(&self, y: usize, row: &mut [Px]) {
        let [x0, x1, y0, y1] = self.bounds;
        if y < y0 || y >= y1 || x0 >= x1 || row.len() < x1 {
            return;
        }
        let m = self.inv.0;
        let (iw, ih) = (self.img.width as f64, self.img.height as f64);
        let step = vec2(m[0][0], m[1][0]);
        let p0 = vec2(m[0][0] * (x0 as f64 + 0.5) + m[0][1] * (y as f64 + 0.5) + m[0][2], m[1][0] * (x0 as f64 + 0.5) + m[1][1] * (y as f64 + 0.5) + m[1][2]);
        let n = x1 - x0;
        let (ax, bx) = span(p0.x, step.x, -1.0, iw + 1.0, n);
        let (ay, by) = span(p0.y, step.y, -1.0, ih + 1.0, n);
        let (lo, hi) = (ax.max(ay), bx.min(by));
        if lo >= hi {
            return;
        }
        let int_tx = m[0][0] == 1.0 && m[1][1] == 1.0 && m[0][1] == 0.0 && m[1][0] == 0.0 && m[0][2].fract() == 0.0 && m[1][2].fract() == 0.0;
        if int_tx {
            let sy = y as i64 + m[1][2] as i64;
            if sy < 0 || sy >= self.img.height as i64 {
                return;
            }
            let dx = m[0][2] as i64;
            let w = self.img.width as usize;
            let srow = &self.img.data[sy as usize * w..(sy as usize + 1) * w];
            let xa = (x0 as i64 + lo as i64).max(-dx).max(0) as usize;
            let xb = ((x0 + hi) as i64).min(self.img.width as i64 - dx).max(0) as usize;
            for x in xa..xb.max(xa) {
                let mut s = srow[(x as i64 + dx) as usize];
                if self.sampling == Sampling::Bicubic {
                    s = [s[0].max(0.0), s[1].max(0.0), s[2].max(0.0), s[3].max(0.0)];
                }
                self.blend_pixel(row, x, y, s);
            }
            return;
        }

        for i in lo..hi {
            let x = x0 + i;
            let p = vec2(p0.x + step.x * i as f64, p0.y + step.y * i as f64);
            if p.x < -1.0 || p.y < -1.0 || p.x > iw + 1.0 || p.y > ih + 1.0 {
                continue;
            }
            let sample = match self.sampling {
                Sampling::Nearest => Some(self.img.get(p.x.floor() as i64, p.y.floor() as i64)),
                Sampling::Bilinear => sample_bilinear_fast(&self.img, self.clear.as_ref(), p.x, p.y),
                Sampling::Bicubic => Some(self.img.sample_bicubic(p.x, p.y)),
            };
            let Some(s) = sample else { continue };
            self.blend_pixel(row, x, y, s);
        }
    }

    #[inline(always)]
    fn blend_pixel(&self, row: &mut [Px], x: usize, y: usize, mut s: Px) {
        if s[3] <= 0.0 {
            return;
        }
        if self.opacity < 1.0 {
            for channel in &mut s {
                *channel *= self.opacity;
            }
        }
        let d = row[x];
        row[x] = if self.mode == BlendMode::Normal {
            let k = 1.0 - s[3];
            [s[0] + d[0] * k, s[1] + d[1] * k, s[2] + d[2] * k, s[3] + d[3] * k]
        } else {
            let noise = if matches!(self.mode, BlendMode::Dissolve | BlendMode::DancingDissolve) { hash_noise(x as u32, y as u32, self.seed) } else { 0.5 };
            blend_pixel(self.mode, d, s, noise)
        };
        if let Some((levels, inv)) = self.levels {
            crate::quantize_px(&mut row[x], levels, inv);
        }
    }
}

/// Index range `[lo, hi)` of `i` in `0..n` for which `p0 + i·s` may lie in `[min, max]`
/// (conservative by one step on each side; callers still check per pixel).
fn span(p0: f64, s: f64, min: f64, max: f64, n: usize) -> (usize, usize) {
    if s.abs() < 1e-12 {
        return if p0 >= min - 1e-9 && p0 <= max + 1e-9 { (0, n) } else { (0, 0) };
    }
    let (a, b) = ((min - p0) / s, (max - p0) / s);
    let (a, b) = if a < b { (a, b) } else { (b, a) };
    let lo = (a.floor() - 1.0).max(0.0);
    let hi = (b.ceil() + 2.0).min(n as f64);
    if hi <= lo { (0, 0) } else { (lo as usize, hi as usize) }
}

/// Columns `[a, b)` of one destination row that a warp changed (`a == b`: none).
#[derive(Clone, Copy)]
struct RowSpan {
    a: usize,
    b: usize,
}

impl RowSpan {
    const NONE: RowSpan = RowSpan { a: usize::MAX, b: 0 };
    #[inline(always)]
    fn add(&mut self, x: usize) {
        self.a = self.a.min(x);
        self.b = x + 1;
    }
}

/// Union of changed pixels as `[x0, y0, x1, y1]` (half-open; `x0 >= x1` when empty).
fn union(r: [usize; 4], q: [usize; 4]) -> [usize; 4] {
    match (r[0] < r[2], q[0] < q[2]) {
        (false, _) => q,
        (_, false) => r,
        _ => [r[0].min(q[0]), r[1].min(q[1]), r[2].max(q[2]), r[3].max(q[3])],
    }
}

/// Sampling, blending and destination bounds for an affine row operation.
struct AffineRows {
    sampling: Sampling,
    opacity: f32,
    quantize: Option<f32>,
    bounds: [usize; 4],
}

/// Blend visible source samples, optionally returning the rectangle of changed pixels.
fn affine_rows<const TRACK_CHANGES: bool>(
    dst: &mut Image,
    img: &Image,
    inv: &Mat3,
    rows: AffineRows,
    source_opaque: bool,
    blend: impl Fn(Px, Px, u32, u32) -> Px + Sync,
) -> [usize; 4] {
    let AffineRows { sampling, opacity: op, quantize, bounds: [x0, x1, y0, y1] } = rows;
    let levels = quantize.map(|l| (l, 1.0 / l));
    let w = dst.width as usize;
    let m = inv.0;
    let (iw, ih) = (img.width as f64, img.height as f64);
    let step = vec2(m[0][0], m[1][0]);
    // Pure integer translation: every sample lands on a pixel centre, so all filters return the
    // source pixel itself (bicubic still clamps negatives, as in the reference).
    let int_tx = m[0][0] == 1.0 && m[1][1] == 1.0 && m[0][1] == 0.0 && m[1][0] == 0.0 && m[0][2].fract() == 0.0 && m[1][2].fract() == 0.0;
    // Bilinear samples skip the transparent parts of the source (most of a thin shape's or a
    // glow-padded layer's buffer) without reading them.
    let clear = (sampling == Sampling::Bilinear && !int_tx && !source_opaque).then(|| ClearTiles::new(img));
    let empty = [usize::MAX, usize::MAX, 0, 0];
    let draw_row = |y: usize, row: &mut [Px]| {
        let mut changed = RowSpan::NONE;
        let fy = y as f64 + 0.5;
        let p0 = vec2(m[0][0] * (x0 as f64 + 0.5) + m[0][1] * fy + m[0][2], m[1][0] * (x0 as f64 + 0.5) + m[1][1] * fy + m[1][2]);
        let n = x1 - x0;
        let (ax, bx) = span(p0.x, step.x, -1.0, iw + 1.0, n);
        let (ay, by) = span(p0.y, step.y, -1.0, ih + 1.0, n);
        let (lo, hi) = (ax.max(ay), bx.min(by));
        if lo >= hi {
            return empty;
        }
        if int_tx {
            let sy = y as i64 + m[1][2] as i64;
            if sy < 0 || sy >= img.height as i64 {
                return empty;
            }
            let dx = m[0][2] as i64;
            let srow = &img.data[sy as usize * img.width as usize..(sy as usize + 1) * img.width as usize];
            let xa = (x0 as i64 + lo as i64).max(-dx).max(0) as usize;
            let xb = ((x0 + hi) as i64).min(img.width as i64 - dx).max(0) as usize;
            for x in xa..xb.max(xa) {
                let mut s = srow[(x as i64 + dx) as usize];
                if sampling == Sampling::Bicubic {
                    s = [s[0].max(0.0), s[1].max(0.0), s[2].max(0.0), s[3].max(0.0)];
                }
                if s[3] <= 0.0 {
                    continue;
                }
                if op < 1.0 {
                    for c in s.iter_mut() {
                        *c *= op;
                    }
                }
                row[x] = blend(row[x], s, x as u32, y as u32);
                if let Some((l, inv)) = levels {
                    crate::quantize_px(&mut row[x], l, inv);
                }
                if TRACK_CHANGES {
                    changed.add(x);
                }
            }
        } else {
            for i in lo..hi {
                let x = x0 + i;
                let p = vec2(p0.x + step.x * i as f64, p0.y + step.y * i as f64);
                if p.x < -1.0 || p.y < -1.0 || p.x > iw + 1.0 || p.y > ih + 1.0 {
                    continue;
                }
                let mut s = match sampling {
                    Sampling::Nearest => img.get(p.x.floor() as i64, p.y.floor() as i64),
                    Sampling::Bilinear => match sample_bilinear_fast(img, clear.as_ref(), p.x, p.y) {
                        Some(s) => s,
                        None => continue,
                    },
                    Sampling::Bicubic => img.sample_bicubic(p.x, p.y),
                };
                if s[3] <= 0.0 {
                    continue;
                }
                if op < 1.0 {
                    for c in s.iter_mut() {
                        *c *= op;
                    }
                }
                row[x] = blend(row[x], s, x as u32, y as u32);
                if let Some((l, inv)) = levels {
                    crate::quantize_px(&mut row[x], l, inv);
                }
                if TRACK_CHANGES {
                    changed.add(x);
                }
            }
        }
        if changed.a < changed.b { [changed.a, y, changed.b, y + 1] } else { empty }
    };
    let rows = dst.data.par_chunks_mut(w).enumerate().skip(y0).take(y1 - y0);
    if TRACK_CHANGES {
        rows.map(|(y, row)| draw_row(y, row)).reduce(|| empty, union)
    } else {
        rows.for_each(|(y, row)| {
            draw_row(y, row);
        });
        empty
    }
}

fn composite_affine<const TRACK_CHANGES: bool>(
    dst: &mut Image,
    img: &Image,
    inv: &Mat3,
    opts: &WarpOpts,
    bounds: [usize; 4],
    quantize: Option<f32>,
    source_opaque: bool,
) -> [usize; 4] {
    match opts.mode {
        BlendMode::Normal => affine_rows::<TRACK_CHANGES>(
            dst,
            img,
            inv,
            AffineRows { sampling: opts.sampling, opacity: opts.opacity, quantize, bounds },
            source_opaque,
            |d, s, _, _| {
                let k = 1.0 - s[3];
                [s[0] + d[0] * k, s[1] + d[1] * k, s[2] + d[2] * k, s[3] + d[3] * k]
            },
        ),
        BlendMode::Dissolve | BlendMode::DancingDissolve => {
            let seed = opts.seed;
            affine_rows::<TRACK_CHANGES>(
                dst,
                img,
                inv,
                AffineRows { sampling: opts.sampling, opacity: opts.opacity, quantize, bounds },
                source_opaque,
                |d, s, x, y| blend_pixel(opts.mode, d, s, hash_noise(x, y, seed)),
            )
        }
        _ => affine_rows::<TRACK_CHANGES>(
            dst,
            img,
            inv,
            AffineRows { sampling: opts.sampling, opacity: opts.opacity, quantize, bounds },
            source_opaque,
            |d, s, _, _| blend_pixel(opts.mode, d, s, 0.5),
        ),
    }
}

fn composite_warp_impl<const TRACK_CHANGES: bool>(dst: &mut Image, src: &Image, m: &Mat3, opts: &WarpOpts, operation: WarpOperation) -> Option<[u32; 4]> {
    let WarpOperation { allow_fast, accumulate, quantize, source_opaque } = operation;
    const NOTHING: Option<[u32; 4]> = Some([0; 4]);
    if src.is_empty() || dst.is_empty() || opts.opacity <= 0.0 {
        return NOTHING;
    }
    let Some(inv) = m.inverse() else { return NOTHING };
    // Minification: pick a pre-filtered level so sampling doesn't alias.
    let scale = m.mean_scale();
    let mut level_img;
    let mut img = src;
    let mut inv = inv;
    if scale < 0.5 && m.is_affine() {
        let mut s = scale;
        let mut f = 1.0;
        level_img = half(src);
        f *= 0.5;
        s *= 2.0;
        while s < 0.5 && level_img.width > 1 && level_img.height > 1 {
            level_img = half(&level_img);
            f *= 0.5;
            s *= 2.0;
        }
        img = &level_img;
        inv = Mat3::scale(vec2(f, f)) * inv;
    }
    let bounds = m.map_rect(&Rect::from_size(src.width as f64, src.height as f64)).inflate(1.0).round_out();
    let full = Rect::from_size(dst.width as f64, dst.height as f64);
    let mut area = bounds.intersect(&full);
    if let Some(c) = opts.clip {
        area = area.intersect(&c);
    }
    if area.is_empty() {
        return NOTHING;
    }
    let (x0, x1) = (area.x0.max(0.0) as usize, area.x1.min(dst.width as f64) as usize);
    let (y0, y1) = (area.y0.max(0.0) as usize, area.y1.min(dst.height as f64) as usize);
    let w = dst.width as usize;
    let affine = inv.is_affine();
    let mode = opts.mode;
    let op = opts.opacity;
    let changed = |r: [usize; 4]| Some(r.map(|v| v.min(u32::MAX as usize) as u32));
    if let Some(k) = accumulate
        && affine
    {
        let r = affine_rows::<TRACK_CHANGES>(
            dst,
            img,
            &inv,
            AffineRows { sampling: opts.sampling, opacity: 1.0, quantize: None, bounds: [x0, x1, y0, y1] },
            source_opaque,
            |d, s, _, _| [d[0] + s[0] * k, d[1] + s[1] * k, d[2] + s[2] * k, d[3] + s[3] * k],
        );
        return if TRACK_CHANGES { changed(r) } else { Some([0; 4]) };
    }
    if allow_fast && affine && !mode.is_stencil() {
        let r = composite_affine::<TRACK_CHANGES>(dst, img, &inv, opts, [x0, x1, y0, y1], quantize, source_opaque);
        return if TRACK_CHANGES { changed(r) } else { Some([0; 4]) };
    }
    dst.data.par_chunks_mut(w).enumerate().skip(y0).take(y1 - y0).for_each(|(y, row)| {
        let fy = y as f64 + 0.5;
        let mut sp = inv.apply(vec2(x0 as f64 + 0.5, fy));
        let step = inv.apply_vec(vec2(1.0, 0.0));
        for x in x0..x1 {
            let p = if affine {
                let p = sp;
                sp += step;
                p
            } else {
                inv.apply(vec2(x as f64 + 0.5, fy))
            };
            if p.x < -1.0 || p.y < -1.0 || p.x > img.width as f64 + 1.0 || p.y > img.height as f64 + 1.0 {
                if mode.is_stencil() {
                    row[x] = blend_pixel(mode, row[x], [0.0; 4], 0.5);
                }
                continue;
            }
            let mut s = match opts.sampling {
                Sampling::Nearest => img.get(p.x.floor() as i64, p.y.floor() as i64),
                Sampling::Bilinear => img.sample_bilinear(p.x, p.y),
                Sampling::Bicubic => img.sample_bicubic(p.x, p.y),
            };
            if s[3] <= 0.0 && !mode.is_stencil() {
                continue;
            }
            if op < 1.0 {
                for c in s.iter_mut() {
                    *c *= op;
                }
            }
            let n = if matches!(mode, BlendMode::Dissolve | BlendMode::DancingDissolve) { hash_noise(x as u32, y as u32, opts.seed) } else { 0.5 };
            row[x] = blend_pixel(mode, row[x], s, n);
        }
    });
    // Stencil modes also clear everything outside the layer bounds.
    if mode.is_stencil() && matches!(mode, BlendMode::StencilAlpha | BlendMode::StencilLuma) {
        for (y, row) in dst.data.chunks_mut(w).enumerate() {
            for (x, px) in row.iter_mut().enumerate() {
                if x < x0 || x >= x1 || y < y0 || y >= y1 {
                    *px = [0.0; 4];
                }
            }
        }
    }
    None
}

/// Resize with bilinear filtering (box pre-filter when shrinking).
pub fn resample(src: &Image, w: u32, h: u32) -> Image {
    let mut out = Image::new(w, h);
    if src.is_empty() || w == 0 || h == 0 {
        return out;
    }
    let m = Mat3::scale(vec2(w as f64 / src.width as f64, h as f64 / src.height as f64));
    composite_warp(&mut out, src, &m, &WarpOpts::default());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_warp_copies() {
        let mut src = Image::new(4, 4);
        src.set(1, 2, [0.5, 0.5, 0.5, 1.0]);
        let mut dst = Image::new(4, 4);
        composite_warp(&mut dst, &src, &Mat3::IDENTITY, &WarpOpts::default());
        assert_eq!(dst.get(1, 2), [0.5, 0.5, 0.5, 1.0]);
        assert_eq!(dst.get(0, 0), [0.0; 4]);
    }

    #[test]
    fn translate_by_integer() {
        let mut src = Image::new(4, 4);
        src.set(0, 0, [1.0; 4]);
        let mut dst = Image::new(8, 8);
        composite_warp(&mut dst, &src, &Mat3::translate(vec2(3.0, 2.0)), &WarpOpts::default());
        assert_eq!(dst.get(3, 2), [1.0; 4]);
    }

    fn test_image(w: u32, h: u32, seed: u32) -> Image {
        let mut img = Image::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let a = if (x + y) % 7 == 0 { 0.0 } else { hash_noise(x, y, seed) };
                img.set(x, y, [hash_noise(x, y, seed + 1) * a, hash_noise(x, y, seed + 2) * a, hash_noise(x, y, seed + 3) * a, a]);
            }
        }
        img
    }

    /// The fast affine path equals the reference per-pixel path (modes, filters, transforms).
    #[test]
    fn fast_path_matches_reference() {
        let src = test_image(37, 23, 5);
        let mats = [
            Mat3::IDENTITY,
            Mat3::translate(vec2(5.0, -3.0)),
            Mat3::translate(vec2(-40.0, 2.0)),
            Mat3::translate(vec2(3.25, 7.5)),
            Mat3::translate(vec2(20.0, 10.0)) * Mat3::rotate_deg(23.0) * Mat3::scale(vec2(1.3, 0.8)),
            Mat3::translate(vec2(30.0, 5.0)) * Mat3::rotate_deg(-120.0) * Mat3::scale(vec2(0.7, 0.7)),
            Mat3::translate(vec2(10.0, 10.0)) * Mat3::scale(vec2(-1.0, 1.0)),
            Mat3::scale(vec2(0.3, 0.3)),
        ];
        let modes = [BlendMode::Normal, BlendMode::Add, BlendMode::Multiply, BlendMode::Screen, BlendMode::Dissolve, BlendMode::Overlay];
        for (mi, m) in mats.iter().enumerate() {
            for mode in modes {
                for sampling in [Sampling::Nearest, Sampling::Bilinear, Sampling::Bicubic] {
                    for opacity in [1.0, 0.6] {
                        let opts = WarpOpts { sampling, opacity, mode, seed: 9, clip: None };
                        let mut a = test_image(48, 40, 77);
                        let mut b = a.clone();
                        composite_warp(&mut a, &src, m, &opts);
                        composite_warp_reference(&mut b, &src, m, &opts);
                        for (i, (p, q)) in a.data.iter().zip(&b.data).enumerate() {
                            for c in 0..4 {
                                assert!((p[c] - q[c]).abs() < 1e-4, "mat {mi} {mode:?} {sampling:?} op {opacity} px {i}: {p:?} vs {q:?}");
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn opaque_source_path_matches_generic_path() {
        let mut src = Image::new(37, 23);
        let width = src.width as usize;
        for (index, pixel) in src.data.iter_mut().enumerate() {
            let x = (index % width) as u32;
            let y = (index / width) as u32;
            *pixel = [hash_noise(x, y, 1), hash_noise(x, y, 2), hash_noise(x, y, 3), 1.0];
        }
        let mats = [
            Mat3::translate(vec2(5.0, -3.0)),
            Mat3::translate(vec2(3.25, 7.5)),
            Mat3::translate(vec2(20.0, 10.0)) * Mat3::rotate_deg(23.0) * Mat3::scale(vec2(1.3, 0.8)),
        ];
        for (mi, m) in mats.iter().enumerate() {
            for mode in [BlendMode::Normal, BlendMode::Add, BlendMode::Multiply, BlendMode::Screen, BlendMode::Overlay, BlendMode::Darken, BlendMode::Dissolve]
            {
                for sampling in [Sampling::Nearest, Sampling::Bilinear, Sampling::Bicubic] {
                    for quantize in [None, Some(255.0)] {
                        let opts = WarpOpts { sampling, mode, opacity: 0.7, seed: 9, clip: None };
                        let mut before = test_image(48, 40, 77);
                        if let Some(levels) = quantize {
                            for pixel in &mut before.data {
                                crate::quantize_px(pixel, levels, 1.0 / levels);
                            }
                        }
                        let mut generic = before.clone();
                        let mut opaque = before;
                        let generic_rect = composite_warp_changed(&mut generic, &src, m, &opts, quantize);
                        let opaque_rect = composite_warp_changed_opaque(&mut opaque, &src, m, &opts, quantize);
                        assert_eq!(opaque_rect, generic_rect, "mat {mi} {mode:?} {sampling:?} {quantize:?}");
                        assert_eq!(opaque.data, generic.data, "mat {mi} {mode:?} {sampling:?} {quantize:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn row_batched_warp_matches_full_frame_path() {
        let transparent = test_image(37, 23, 13);
        let mut opaque = transparent.clone();
        for pixel in &mut opaque.data {
            let alpha = pixel[3];
            if alpha > 0.0 {
                for channel in &mut pixel[..3] {
                    *channel /= alpha;
                }
            }
            pixel[3] = 1.0;
        }
        let mats = [
            Mat3::translate(vec2(5.0, -3.0)),
            Mat3::translate(vec2(3.25, 7.5)),
            Mat3::translate(vec2(20.0, 10.0)) * Mat3::rotate_deg(23.0) * Mat3::scale(vec2(1.3, 0.8)),
            Mat3::translate(vec2(12.0, 9.0)) * Mat3::scale(vec2(0.3, 0.4)),
        ];
        for (mi, m) in mats.iter().enumerate() {
            for mode in [BlendMode::Normal, BlendMode::Add, BlendMode::Multiply, BlendMode::Screen, BlendMode::Overlay, BlendMode::Darken, BlendMode::Dissolve]
            {
                for sampling in [Sampling::Nearest, Sampling::Bilinear, Sampling::Bicubic] {
                    for quantize in [None, Some(255.0)] {
                        for source_opaque in [false, true] {
                            let src = if source_opaque { &opaque } else { &transparent };
                            let opts = WarpOpts { sampling, mode, opacity: 0.7, seed: 9, clip: None };
                            let mut expected = test_image(48, 40, 77);
                            let mut actual = expected.clone();
                            if let Some(levels) = quantize {
                                composite_warp_quantized(&mut expected, src, m, &opts, levels);
                            } else {
                                composite_warp(&mut expected, src, m, &opts);
                            }
                            let prepared =
                                prepare_warp_rows(src, m, &opts, actual.width, actual.height, quantize, source_opaque).expect("affine row warp is supported");
                            let width = actual.width as usize;
                            for (y, row) in actual.data.chunks_mut(width).enumerate() {
                                prepared.blend_row(y, row);
                            }
                            assert_eq!(actual.data, expected.data, "mat {mi} {mode:?} {sampling:?} {quantize:?}, opaque={source_opaque}");
                        }
                    }
                }
            }
        }
    }

    /// A mostly transparent source: a thin ring with a few stray pixels (one with negative zero alpha).
    fn sparse_image(w: u32, h: u32) -> Image {
        let mut img = Image::new(w, h);
        let (cx, cy, r) = (w as f32 / 2.0, h as f32 / 2.0, w.min(h) as f32 * 0.4);
        for y in 0..h {
            for x in 0..w {
                let d = ((x as f32 + 0.5 - cx).hypot(y as f32 + 0.5 - cy) - r).abs();
                if d < 1.5 {
                    let a = 1.0 - d / 1.5;
                    img.set(x, y, [0.2 * a, 0.6 * a, a, a]);
                }
            }
        }
        img.set(3, 2, [0.5, 0.5, 0.5, 0.5]);
        img.set(w - 1, h - 1, [1.0; 4]);
        img.set(w / 2, 1, [0.0, 0.0, 0.0, -0.0]);
        img
    }

    #[test]
    fn clear_tiles_hold_only_transparent_taps() {
        let img = sparse_image(150, 97);
        let clear = ClearTiles::new(&img);
        let mut skipped = 0;
        for y in 0..img.height as usize - 1 {
            for x in 0..img.width as usize - 1 {
                if clear.is_clear(x, y) {
                    skipped += 1;
                    for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                        assert_eq!(img.get(x as i64 + dx, y as i64 + dy)[3], 0.0, "tap ({}, {}) of a clear tile", x as i64 + dx, y as i64 + dy);
                    }
                }
            }
        }
        assert!(skipped > 100 * 70, "the ring's empty middle is skipped: {skipped}");
    }

    /// The changed rectangle holds every pixel a warp changed, and the skipped transparent
    /// samples leave the result as the reference path draws it.
    #[test]
    fn changed_rect_covers_the_changed_pixels() {
        let src = sparse_image(90, 70);
        let mats = [
            Mat3::translate(vec2(7.0, 3.0)),
            Mat3::translate(vec2(10.25, 4.5)),
            Mat3::translate(vec2(60.0, 50.0)) * Mat3::rotate_deg(33.0) * Mat3::scale(vec2(0.9, 1.2)) * Mat3::translate(vec2(-45.0, -35.0)),
            Mat3::translate(vec2(-30.0, -20.0)),
            Mat3::scale(vec2(0.3, 0.3)),
        ];
        for (mi, m) in mats.iter().enumerate() {
            for mode in [BlendMode::Normal, BlendMode::Add, BlendMode::Dissolve] {
                let opts = WarpOpts { mode, seed: 3, ..Default::default() };
                let before = test_image(120, 100, 4);
                let mut a = before.clone();
                let r = composite_warp_changed(&mut a, &src, m, &opts, None);
                let Some([x0, y0, x1, y1]) = r else { panic!("mat {mi}: the fast path tracks what it changed") };
                for (i, (p, q)) in a.data.iter().zip(&before.data).enumerate() {
                    let (x, y) = (i as u32 % a.width, i as u32 / a.width);
                    if p != q {
                        assert!(x >= x0 && x < x1 && y >= y0 && y < y1, "mat {mi} {mode:?}: ({x}, {y}) changed outside {r:?}");
                    }
                }
                let mut b = before.clone();
                composite_warp_reference(&mut b, &src, m, &opts);
                for (i, (p, q)) in a.data.iter().zip(&b.data).enumerate() {
                    for c in 0..4 {
                        assert!((p[c] - q[c]).abs() < 1e-4, "mat {mi} {mode:?} px {i}: {p:?} vs {q:?}");
                    }
                }
            }
        }
        let mut dst = Image::new(10, 10);
        assert_eq!(composite_warp_changed(&mut dst, &Image::new(4, 4), &Mat3::IDENTITY, &WarpOpts::default(), None), Some([u32::MAX, u32::MAX, 0, 0]));
        let stencil = WarpOpts { mode: BlendMode::StencilAlpha, ..Default::default() };
        assert_eq!(composite_warp_changed(&mut dst, &src, &Mat3::IDENTITY, &stencil, Some(255.0)), None, "stencils change the whole destination");
    }

    /// Quantising while blending equals blending, then quantising the changed rectangle of an
    /// already quantised destination (what an 8/16 bpc comp did after every layer), bit for bit.
    #[test]
    fn quantize_while_blending_matches_quantizing_after() {
        let src = sparse_image(90, 70);
        let mats = [Mat3::translate(vec2(7.0, 3.0)), Mat3::translate(vec2(60.0, 50.0)) * Mat3::rotate_deg(33.0) * Mat3::translate(vec2(-45.0, -35.0))];
        for m in &mats {
            for mode in [BlendMode::Normal, BlendMode::Screen, BlendMode::Dissolve] {
                for (levels, opacity) in [(255.0f32, 1.0), (32_768.0, 0.7)] {
                    let mut before = test_image(120, 100, 8);
                    for p in &mut before.data {
                        crate::quantize_px(p, levels, 1.0 / levels);
                    }
                    let opts = WarpOpts { mode, opacity, seed: 5, ..Default::default() };
                    let mut after = before.clone();
                    let Some([x0, y0, x1, y1]) = composite_warp_changed(&mut after, &src, m, &opts, None) else { panic!("fast path") };
                    for y in y0..y1 {
                        for x in x0..x1 {
                            let i = after.idx(x, y);
                            crate::quantize_px(&mut after.data[i], levels, 1.0 / levels);
                        }
                    }
                    let mut fused = before.clone();
                    let r = composite_warp_changed(&mut fused, &src, m, &opts, Some(levels));
                    assert_eq!(r, Some([x0, y0, x1, y1]));
                    let mut untracked = before.clone();
                    assert!(composite_warp_quantized(&mut untracked, &src, m, &opts, levels));
                    for (i, (p, q)) in fused.data.iter().zip(&after.data).enumerate() {
                        assert_eq!(p.map(f32::to_bits), q.map(f32::to_bits), "{mode:?} {levels} px {i}");
                    }
                    for (i, (p, q)) in untracked.data.iter().zip(&fused.data).enumerate() {
                        assert_eq!(p.map(f32::to_bits), q.map(f32::to_bits), "untracked {mode:?} {levels} px {i}");
                    }
                }
            }
        }
    }

    /// Accumulating sub-samples equals warping each into its own image and summing.
    #[test]
    fn accumulate_matches_per_sample_images() {
        let src = test_image(30, 20, 11);
        let persp = Mat3::square_to_quad([vec2(5.0, 4.0), vec2(40.0, 2.0), vec2(44.0, 30.0), vec2(2.0, 35.0)]) * Mat3::scale(vec2(1.0 / 30.0, 1.0 / 20.0));
        let mats = [Mat3::translate(vec2(3.0, 4.0)), Mat3::translate(vec2(5.5, 4.25)) * Mat3::rotate_deg(12.0), persp];
        let k = 1.0 / mats.len() as f32;
        let mut fast = Image::new(48, 40);
        let mut slow = Image::new(48, 40);
        for m in &mats {
            accumulate_warp(&mut fast, &src, m, Sampling::Bilinear, k);
            let mut img = Image::new(48, 40);
            composite_warp_reference(&mut img, &src, m, &WarpOpts::default());
            for (a, b) in slow.data.iter_mut().zip(&img.data) {
                for c in 0..4 {
                    a[c] += b[c] * k;
                }
            }
        }
        for (p, q) in fast.data.iter().zip(&slow.data) {
            for c in 0..4 {
                assert!((p[c] - q[c]).abs() < 1e-4, "{p:?} vs {q:?}");
            }
        }
    }

    #[test]
    fn downscale_preserves_average() {
        let src = Image::filled(64, 64, [0.25, 0.5, 0.75, 1.0]);
        let out = resample(&src, 8, 8);
        let p = out.get(4, 4);
        assert!((p[1] - 0.5).abs() < 1e-4 && (p[3] - 1.0).abs() < 1e-4, "{p:?}");
    }
}
