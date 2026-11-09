//! Compositing primitives: each mirrors a CPU operation in `effectcraft-raster` /
//! `effectcraft-render` and returns a new image (GPU images are immutable).

use effectcraft_color::{BlendMode, Conversion};
use effectcraft_geom::{Mat3, Rect, vec2};
use effectcraft_project::MatteKind;
use effectcraft_raster::Sampling;

use crate::context::{Enc, GpuImage, Params};

/// Index of a blend mode in `BlendMode::ALL` (the kernels' mode numbers).
pub fn mode_id(m: BlendMode) -> u32 {
    BlendMode::ALL.iter().position(|x| *x == m).unwrap_or(0) as u32
}

fn sampling_id(s: Sampling) -> u32 {
    match s {
        Sampling::Nearest => 0,
        Sampling::Bilinear => 1,
        Sampling::Bicubic => 2,
    }
}

fn rows(m: &Mat3) -> [[f32; 4]; 3] {
    m.0.map(|r| [r[0] as f32, r[1] as f32, r[2] as f32, 0.0])
}

/// Box-filter halving (minification pre-filter).
pub fn half(e: &mut Enc, src: &GpuImage) -> GpuImage {
    let out = e.scratch(src.width.div_ceil(2).max(1), src.height.div_ceil(2).max(1));
    e.pixels("half", &Params::default(), src, None, &out, None);
    out
}

fn half_quantized(e: &mut Enc, src: &GpuImage, levels: f32) -> GpuImage {
    let out = e.scratch(src.width.div_ceil(2).max(1), src.height.div_ceil(2).max(1));
    let mut p = Params::default();
    p.f[0] = [levels, 1.0 / levels, 0.0, 0.0];
    e.pixels("half", &p, src, None, &out, None);
    out
}

/// Sampling and blending policy for a transformed GPU image.
pub struct WarpOptions {
    pub sampling: Sampling,
    pub mode: BlendMode,
    pub opacity: f32,
    pub seed: u32,
}

/// Result of applying one bounded warp into the next of two ping-pong canvases.
pub enum PingPongWarp {
    /// This warp needs the regular full-copy path (for example stencil blending).
    Unsupported,
    /// The warp did not touch the canvas.
    Unchanged,
    /// The updated canvas and its dispatch footprint, including workgroup-edge quantization.
    Drawn((u32, u32, u32, u32)),
}

/// Draw `src` transformed by `m` onto `dst`, or accumulate it with a sample weight.
pub fn warp(e: &mut Enc, dst: &GpuImage, src: &GpuImage, m: &Mat3, opts: WarpOptions, accumulate: Option<f32>) -> GpuImage {
    warp_q(e, dst, src, m, opts, accumulate, None)
}

/// [`warp`], clamping and quantising the result to `levels` (8/16 bpc: the comp after a layer,
/// in the same pass).
pub fn warp_q(e: &mut Enc, dst: &GpuImage, src: &GpuImage, m: &Mat3, opts: WarpOptions, accumulate: Option<f32>, levels: Option<f32>) -> GpuImage {
    warp_q_with_source_quantization(e, dst, src, m, opts, accumulate, levels, None)
}

/// [`warp_q`] for a source whose bit-depth quantization is deferred until GPU sampling.
pub fn warp_q_source_quantized(e: &mut Enc, dst: &GpuImage, src: &GpuImage, m: &Mat3, opts: WarpOptions, levels: Option<f32>) -> GpuImage {
    warp_q_with_source_quantization(e, dst, src, m, opts, None, levels, levels)
}

fn warp_q_with_source_quantization(
    e: &mut Enc,
    dst: &GpuImage,
    src: &GpuImage,
    m: &Mat3,
    opts: WarpOptions,
    accumulate: Option<f32>,
    levels: Option<f32>,
    source_levels: Option<f32>,
) -> GpuImage {
    let Some((img, p, region, clear)) = warp_params(e, dst, src, m, opts, accumulate, levels, source_levels, None) else { return dst.clone() };
    let mut out = e.scratch(dst.width, dst.height);
    if clear {
        // Stencil modes clear pixels outside the source footprint too.
        e.pixels("warp_blend", &p, &img, Some(dst), &out, None);
    } else {
        // Other blend modes leave the canvas outside this layer's transformed bounds unchanged.
        // Copy it with the GPU's texture-copy path, then run the shader only over the layer area.
        e.copy_image(dst, &out);
        e.pixels_region("warp_blend", &p, &img, Some(dst), &out, region, None);
    }
    out.levels = levels;
    out
}

/// Apply an ordinary bounded warp while alternating between two full-size canvases. `next`
/// must already match `current` outside `previous`; this copies only that prior dirty region
/// before writing the new layer. The returned region includes workgroup-edge quantization.
pub fn warp_q_pingpong(
    e: &mut Enc,
    current: &GpuImage,
    next: &mut GpuImage,
    src: &GpuImage,
    m: &Mat3,
    opts: WarpOptions,
    levels: Option<f32>,
    previous: Option<(u32, u32, u32, u32)>,
    quantize_source: bool,
) -> PingPongWarp {
    warp_q_pingpong_inner(e, current, next, src, m, opts, levels, previous, quantize_source, None)
}

/// [`warp_q_pingpong`] for an opaque rectangle whose uniform pixels are passed as parameters.
pub fn warp_q_solid_pingpong(
    e: &mut Enc,
    current: &GpuImage,
    next: &mut GpuImage,
    color: [f32; 4],
    size: (u32, u32),
    m: &Mat3,
    opts: WarpOptions,
    levels: Option<f32>,
    previous: Option<(u32, u32, u32, u32)>,
    quantize_source: bool,
) -> PingPongWarp {
    let src = e.g.dummy_image();
    warp_q_pingpong_inner(e, current, next, &src, m, opts, levels, previous, quantize_source, Some((color, size)))
}

fn warp_q_pingpong_inner(
    e: &mut Enc,
    current: &GpuImage,
    next: &mut GpuImage,
    src: &GpuImage,
    m: &Mat3,
    opts: WarpOptions,
    levels: Option<f32>,
    previous: Option<(u32, u32, u32, u32)>,
    quantize_source: bool,
    solid: Option<([f32; 4], (u32, u32))>,
) -> PingPongWarp {
    if current.width != next.width || current.height != next.height {
        return PingPongWarp::Unsupported;
    }
    let source_levels = quantize_source.then_some(levels).flatten();
    let Some((img, p, region, clear)) = warp_params(e, current, src, m, opts, None, levels, source_levels, solid) else { return PingPongWarp::Unchanged };
    if clear {
        return PingPongWarp::Unsupported;
    }
    let draw_region = dispatch_region(region, current.width, current.height);
    let mut work_region = draw_region;
    if let Some(previous) = previous {
        let union = (previous.0.min(draw_region.0), previous.1.min(draw_region.1), previous.2.max(draw_region.2), previous.3.max(draw_region.3));
        let area = |(x0, y0, x1, y1): (u32, u32, u32, u32)| u64::from(x1 - x0) * u64::from(y1 - y0);
        // `warp_blend` copies `current` to `next` outside the layer bounds. When the two
        // dispatch footprints overlap or are close, it can also copy the previous dirty region,
        // replacing a separate texture-copy command with the same shader dispatch. Bound the
        // extra work to four times the two footprints; for distant layers, keep the direct copy.
        if area(union) <= (area(previous) + area(draw_region)).saturating_mul(4) {
            work_region = union;
        } else {
            e.copy_region(current, next, previous);
        }
    }
    e.pixels_region("warp_blend", &p, &img, Some(current), next, work_region, None);
    next.levels = levels;
    PingPongWarp::Drawn(draw_region)
}

fn warp_params(
    e: &mut Enc,
    dst: &GpuImage,
    src: &GpuImage,
    m: &Mat3,
    opts: WarpOptions,
    accumulate: Option<f32>,
    levels: Option<f32>,
    source_levels: Option<f32>,
    solid: Option<([f32; 4], (u32, u32))>,
) -> Option<(GpuImage, Params, (u32, u32, u32, u32), bool)> {
    let WarpOptions { sampling, mode, opacity, seed } = opts;
    if src.width == 0 || src.height == 0 || opacity <= 0.0 {
        return None;
    }
    let Some(inv0) = m.inverse() else { return None };
    // Minification: pick a pre-filtered level so sampling doesn't alias.
    let scale = m.mean_scale();
    let mut img = src.clone();
    let mut inv = inv0;
    let (mut source_width, mut source_height) = solid.map_or((src.width, src.height), |(_, size)| size);
    let (bounds_width, bounds_height) = (source_width, source_height);
    if scale < 0.5 && m.is_affine() {
        let mut s = scale * 2.0;
        let mut f = 0.5;
        if solid.is_some() {
            (source_width, source_height) = (source_width.div_ceil(2).max(1), source_height.div_ceil(2).max(1));
            while s < 0.5 && source_width > 1 && source_height > 1 {
                (source_width, source_height) = (source_width.div_ceil(2).max(1), source_height.div_ceil(2).max(1));
                f *= 0.5;
                s *= 2.0;
            }
        } else {
            img = if let Some(source_levels) = source_levels { half_quantized(e, src, source_levels) } else { half(e, src) };
            while s < 0.5 && img.width > 1 && img.height > 1 {
                img = half(e, &img);
                f *= 0.5;
                s *= 2.0;
            }
            source_width = img.width;
            source_height = img.height;
        }
        inv = Mat3::scale(vec2(f, f)) * inv;
    }
    let bounds = m.map_rect(&Rect::from_size(bounds_width as f64, bounds_height as f64)).inflate(1.0).round_out();
    let area = bounds.intersect(&Rect::from_size(dst.width as f64, dst.height as f64));
    if area.is_empty() {
        return None;
    }
    let (x0, x1) = (area.x0.max(0.0) as u32, area.x1.min(dst.width as f64) as u32);
    let (y0, y1) = (area.y0.max(0.0) as u32, area.y1.min(dst.height as f64) as u32);
    let mut p = Params::default();
    let clear = matches!(mode, BlendMode::StencilAlpha | BlendMode::StencilLuma);
    let flags = u32::from(accumulate.is_some()) | if clear && accumulate.is_none() { 2 } else { 0 } | if solid.is_some() { 4 } else { 0 };
    p.u[0] = [mode_id(mode), sampling_id(sampling), seed, flags];
    let r = rows(&inv);
    p.f[0] = r[0];
    p.f[1] = r[1];
    p.f[2] = r[2];
    p.f[3] = [opacity, accumulate.unwrap_or(1.0), levels.unwrap_or(0.0), levels.map_or(0.0, |v| 1.0 / v)];
    p.f[4] = [x0 as f32, y0 as f32, x1 as f32, y1 as f32];
    p.f[5] = [source_levels.unwrap_or(0.0), source_levels.map_or(0.0, |v| 1.0 / v), 0.0, 0.0];
    if let Some((color, _)) = solid {
        p.f[6] = color;
        p.f[7] = [source_width as f32, source_height as f32, 0.0, 0.0];
    }
    Some((img, p, (x0, y0, x1, y1), clear))
}

fn dispatch_region(region: (u32, u32, u32, u32), width: u32, height: u32) -> (u32, u32, u32, u32) {
    let (x0, y0, x1, y1) = region;
    (x0 / 16 * 16, y0 / 16 * 16, x1.div_ceil(16).saturating_mul(16).min(width), y1.div_ceil(16).saturating_mul(16).min(height))
}

/// `Image::blend_from`: blend `src` (same size) onto `dst`, clamping and quantising the result
/// to `levels` when set (see [`warp_q`]).
pub fn blend_full(e: &mut Enc, dst: &GpuImage, src: &GpuImage, mode: BlendMode, opacity: f32, seed: u32, levels: Option<f32>) -> GpuImage {
    let mut p = Params::default();
    p.u[0] = [mode_id(mode), 0, seed, 0];
    p.f[0] = [opacity, levels.unwrap_or(0.0), levels.map_or(0.0, |v| 1.0 / v), 0.0];
    let mut out = e.scratch(dst.width, dst.height);
    e.pixels("blend_full", &p, src, Some(dst), &out, None);
    out.levels = levels;
    out
}

/// Multiply an isolated layer by its track matte.
pub fn matte(e: &mut Enc, iso: &GpuImage, matte: &GpuImage, kind: MatteKind) -> GpuImage {
    let mut p = Params::default();
    p.u[0][0] = match kind {
        MatteKind::Alpha => 0,
        MatteKind::AlphaInverted => 1,
        MatteKind::Luma => 2,
        MatteKind::LumaInverted => 3,
    };
    let out = e.scratch(iso.width, iso.height);
    e.pixels("matte", &p, iso, Some(matte), &out, None);
    out
}

/// Preserve Transparency: multiply by the alpha below.
pub fn preserve(e: &mut Enc, iso: &GpuImage, canvas: &GpuImage) -> GpuImage {
    let out = e.scratch(iso.width, iso.height);
    e.pixels("preserve", &Params::default(), iso, Some(canvas), &out, None);
    out
}

/// Knockout: clear what lies below the layer's content.
pub fn knockout(e: &mut Enc, canvas: &GpuImage, shape: &GpuImage) -> GpuImage {
    let out = e.scratch(canvas.width, canvas.height);
    e.pixels("knockout", &Params::default(), canvas, Some(shape), &out, None);
    out
}

/// Styled-layer finish: move `canvas` toward `tmp` by `opacity` on the enabled channels.
pub fn channel_mix(e: &mut Enc, canvas: &GpuImage, tmp: &GpuImage, channels: [bool; 3], opacity: f32) -> GpuImage {
    let mut p = Params::default();
    p.u[0] = [channels[0] as u32, channels[1] as u32, channels[2] as u32, 0];
    p.f[0][0] = opacity;
    let out = e.scratch(canvas.width, canvas.height);
    e.pixels("channel_mix", &p, canvas, Some(tmp), &out, None);
    out
}

/// Adjustment layer finish: move `canvas` toward `adjusted` by `matte`'s alpha × `opacity`.
pub fn adjust_mix(e: &mut Enc, canvas: &GpuImage, adjusted: &GpuImage, matte: &GpuImage, opacity: f32) -> GpuImage {
    let (rows, stride) = e.image_rows(matte);
    let mut p = Params::default();
    p.u[0][0] = stride;
    p.f[0][0] = opacity;
    let out = e.scratch(canvas.width, canvas.height);
    e.pixels("adjust_mix", &p, canvas, Some(adjusted), &out, Some(&rows));
    out
}

/// Clamp and quantise to `levels` steps (8/16 bpc).
pub fn quantize(e: &mut Enc, img: &GpuImage, levels: f32) -> GpuImage {
    // Quantising is idempotent: an image a fused kernel already quantised stays as it is.
    if img.levels == Some(levels) {
        return img.clone();
    }
    let mut p = Params::default();
    p.f[0][0] = levels;
    p.f[0][1] = 1.0 / levels;
    let mut out = e.scratch(img.width, img.height);
    e.pixels("quantize", &p, img, None, &out, None);
    out.levels = Some(levels);
    out
}

fn curve_id(s: Option<effectcraft_color::ColorSpace>) -> u32 {
    use effectcraft_color::space::Curve;
    match s.map(|s| s.curve()) {
        None | Some(Curve::Linear) => 0,
        Some(Curve::Srgb) => 1,
        Some(Curve::Gamma24) => 2,
        Some(Curve::Pq) => 3,
        Some(Curve::Hlg) => 4,
    }
}

/// Colour conversion on straight colour (`render::color::convert`).
pub fn convert(e: &mut Enc, img: &GpuImage, c: &Conversion) -> GpuImage {
    let mut p = Params::default();
    p.u[0] = [curve_id(c.decode), c.matrix.is_some() as u32, curve_id(c.encode), 0];
    if let Some(m) = c.matrix {
        for i in 0..3 {
            p.f[i] = [m[i][0], m[i][1], m[i][2], 0.0];
        }
    }
    let out = e.scratch(img.width, img.height);
    e.pixels("convert", &p, img, None, &out, None);
    out
}
