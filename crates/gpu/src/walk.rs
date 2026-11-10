//! The per-frame compositing walk on the GPU: the same steps as `Renderer::draw_comp` (layer
//! order, 3D runs, collapsed precomps, track mattes, Preserve Transparency, layer styles, motion
//! blur, bit-depth quantisation, colour conversions), with layer pixels taken from the CPU
//! renderer and its layer cache. Classic and Advanced 3D runs (environment skies included) and
//! adjustment layers run on the GPU; steps without a GPU implementation (anything a kernel
//! declines) read the canvas back, run on the CPU and upload the result.

use std::sync::Arc;

use effectcraft_color::BlendMode;
use effectcraft_effects::Buf;
use effectcraft_project::{ItemId, Layer, Quality};
use effectcraft_raster::Image;
use effectcraft_render::{EvalCtx, Renderer, StyledLayer, styles};
use effectcraft_time::Tick;

use crate::context::{Enc, GpuImage};
use crate::ops;

/// Two canvases whose pixels differ only in `dirty`; ordinary bounded layers can copy that
/// rectangle forward, then update their own dispatch area in the next canvas.
struct PingPongCanvas {
    images: [GpuImage; 2],
    front: usize,
    dirty: Option<(u32, u32, u32, u32)>,
}

impl PingPongCanvas {
    fn new(e: &mut Enc, canvas: &GpuImage) -> Self {
        let mut a = e.scratch(canvas.width, canvas.height);
        let mut b = e.scratch(canvas.width, canvas.height);
        e.copy_image(canvas, &a);
        e.copy_image(canvas, &b);
        a.levels = canvas.levels;
        b.levels = canvas.levels;
        Self { images: [a, b], front: 0, dirty: None }
    }

    fn draw(
        &mut self,
        e: &mut Enc,
        src: &GpuImage,
        matrix: &effectcraft_geom::Mat3,
        opts: ops::WarpOptions,
        levels: Option<f32>,
        quantize_source: bool,
    ) -> bool {
        let next = 1 - self.front;
        let current = self.images[self.front].clone();
        match ops::warp_q_pingpong(e, &current, &mut self.images[next], src, matrix, opts, levels, self.dirty, quantize_source) {
            ops::PingPongWarp::Unsupported => false,
            ops::PingPongWarp::Unchanged => true,
            ops::PingPongWarp::Drawn(region) => {
                self.front = next;
                self.dirty = Some(region);
                true
            }
        }
    }

    fn draw_solid(
        &mut self,
        e: &mut Enc,
        color: [f32; 4],
        size: (u32, u32),
        matrix: &effectcraft_geom::Mat3,
        opts: ops::WarpOptions,
        levels: Option<f32>,
        quantize_source: bool,
    ) -> bool {
        let next = 1 - self.front;
        let current = self.images[self.front].clone();
        match ops::warp_q_solid_pingpong(e, &current, &mut self.images[next], color, size, matrix, opts, levels, self.dirty, quantize_source) {
            ops::PingPongWarp::Unsupported => false,
            ops::PingPongWarp::Unchanged => true,
            ops::PingPongWarp::Drawn(region) => {
                self.front = next;
                self.dirty = Some(region);
                true
            }
        }
    }

    fn finish(self, canvas: &mut GpuImage) {
        *canvas = self.images[self.front].clone();
    }
}

/// The size [`render`] renders `comp_id` at, without rendering it.
pub(crate) fn frame_size(r: &Renderer, comp_id: ItemId, t: Tick) -> Option<(u32, u32)> {
    let ctx = r.eval_ctx(comp_id, t)?;
    Some(effectcraft_render::frame_size(ctx.comp.width, ctx.comp.height, r.opts.scale))
}

/// Render a comp frame into a GPU image (the canvas in the output space, as
/// `Renderer::comp_frame_cpu` returns it). `None` when something cannot run here (the caller
/// renders on the CPU instead).
pub(crate) fn render<'g>(e: &mut Enc<'g>, r: &Renderer, comp_id: ItemId, t: Tick) -> Option<GpuImage> {
    // Region-of-interest frames are cropped on the CPU path.
    if r.opts.roi.is_some() {
        return None;
    }
    let ctx = r.eval_ctx(comp_id, t)?;
    let (w, h) = effectcraft_render::frame_size(ctx.comp.width, ctx.comp.height, r.opts.scale);
    if r.depth() > 16 || !e.g.fits(w, h) {
        return None;
    }
    let mut canvas = e.zeros(w, h);
    draw_comp(e, r, &ctx, &mut canvas)?;
    let pipe = r.pipe();
    if let Some(c) = pipe.from_blend() {
        canvas = ops::convert(e, &canvas, &c);
    }
    if let Some(l) = pipe.levels {
        canvas = ops::quantize(e, &canvas, l);
    }
    if r.depth() == 0
        && let Some(c) = pipe.output()
    {
        canvas = ops::convert(e, &canvas, &c);
        if let Some(l) = pipe.levels {
            canvas = ops::quantize(e, &canvas, l);
        }
    }
    if r.depth() == 0
        && let Some((lin, mode, enc)) = pipe.output_hdr()
    {
        on_cpu(e, &mut canvas, |img| {
            effectcraft_render::color::output_hdr(img, lin, mode, enc);
            pipe.quantize(img);
        })?;
    }
    Some(canvas)
}

/// Run a CPU drawing step on the canvas (read back, draw, upload).
fn on_cpu(e: &mut Enc, canvas: &mut GpuImage, f: impl FnOnce(&mut Image)) -> Option<()> {
    let mut img = e.download(canvas)?;
    f(&mut img);
    *canvas = e.g.upload_image(&img)?;
    Some(())
}

fn draw_comp<'a>(e: &mut Enc, r: &Renderer<'a>, ctx: &EvalCtx<'a>, canvas: &mut GpuImage) -> Option<()> {
    let visible = r.visible_layers(ctx);
    let occluded = r.occluded_bounded_layers(ctx, &visible, canvas.width, canvas.height);
    let levels = r.pipe().levels;
    // Quantizing uploaded texels is equivalent to quantizing CPU-rendered sources only when
    // the renderer is already in the blending space. A color-space conversion must keep its
    // full-precision input until after conversion.
    // Quantized source layers can use compact uploads without changing the frame hash. High-bit
    // sources still use RGBA32Float. Keep duplicate footage hits in the encoder's format-specific
    // source cache and avoid staging a redundant float copy first.
    let compact_8bit_sources = true;
    let mut i = 0;
    while i < visible.len() {
        // A device that ran out of memory (or was lost) on an earlier layer fails every later
        // one: give the frame back to the CPU now instead of drawing the rest of the comp.
        e.g.check_health().ok()?;
        if occluded[i] {
            i += 1;
            continue;
        }
        if r.can_prepare_gpu_layer(visible[i]) {
            let mut end = i + 1;
            while end < visible.len() && r.can_prepare_gpu_layer(visible[end]) {
                end += 1;
            }
            if end - i >= 16 {
                const BATCH: usize = 32;
                let mut ping_pong: Option<PingPongCanvas> = None;
                for start in (i..end).step_by(BATCH) {
                    let stop = (start + BATCH).min(end);
                    let prepared = r.prepare_gpu_layer_batch(ctx, &visible[start..stop], &occluded[start..stop], canvas.width, canvas.height)?;
                    let mut uploaded = vec![None; prepared.len()];
                    for (offset, st) in prepared.iter().enumerate() {
                        let Some(st) = st.as_ref() else { continue };
                        let layer = visible[start + offset];
                        if !st.plain || r.layer_opacity(ctx, layer) <= 0.0 || st.solid_color.is_some() {
                            continue;
                        }
                        let placement = r.placement(ctx, layer, &st.body);
                        if !matches!(placement.matrices.as_slice(), [_]) {
                            continue;
                        }
                        let key = r.gpu_footage_source_key(ctx, layer, st.quantize_source());
                        uploaded[offset] = Some(if compact_8bit_sources && st.quantize_source() {
                            if let Some(key) = key { e.stage_footage_source_8bit(key, &st.body, true)? } else { e.stage_upload_buf_8bit(&st.body, true)? }
                        } else if let Some(key) = key {
                            e.stage_footage_source(key, &st.body)?
                        } else {
                            e.stage_upload_buf(&st.body)?
                        });
                    }
                    e.flush_staged_uploads();
                    for (offset, st) in prepared.into_iter().enumerate() {
                        e.g.check_health().ok()?;
                        let Some(st) = st else { continue };
                        let layer = visible[start + offset];
                        let opacity = r.layer_opacity(ctx, layer);
                        let quantize_source = st.quantize_source() || compact_8bit_sources;
                        let prepared_by_ping_pong = if st.plain && opacity > 0.0 {
                            let placement = r.placement(ctx, layer, &st.body);
                            if let [matrix] = placement.matrices.as_slice() {
                                let ping_pong = ping_pong.get_or_insert_with(|| PingPongCanvas::new(e, canvas));
                                let opts = ops::WarpOptions { sampling: placement.sampling, mode: layer.blend_mode, opacity, seed: placement.seed };
                                if let (Some(color), Some(size)) = (st.solid_color, st.solid_size) {
                                    ping_pong.draw_solid(e, color, size, matrix, opts, levels, quantize_source)
                                } else {
                                    let src = if compact_8bit_sources && st.quantize_source() {
                                        if let Some(src) = uploaded[offset].take() {
                                            src
                                        } else if let Some(key) = r.gpu_footage_source_key(ctx, layer, true) {
                                            e.upload_footage_source_8bit(key, &st.body, true)?
                                        } else {
                                            let image = e.g.upload_buf_8bit(&st.body, true)?;
                                            e.uses(&image);
                                            image
                                        }
                                    } else if let Some(src) = uploaded[offset].take() {
                                        src
                                    } else if let Some(key) = r.gpu_footage_source_key(ctx, layer, st.quantize_source()) {
                                        e.upload_footage_source(key, &st.body)?
                                    } else {
                                        e.g.upload_buf(&st.body)?
                                    };
                                    ping_pong.draw(e, &src, matrix, opts, levels, quantize_source)
                                }
                            } else {
                                false
                            }
                        } else {
                            false
                        };
                        if !prepared_by_ping_pong && let Some(ping_pong) = ping_pong.take() {
                            ping_pong.finish(canvas);
                        }
                        let full_quantize = if prepared_by_ping_pong {
                            e.mem_gate();
                            false
                        } else {
                            !draw_prepared_layer(e, r, ctx, layer, canvas, st)?
                        };
                        if full_quantize && let Some(l) = levels {
                            *canvas = ops::quantize(e, canvas, l);
                        }
                    }
                }
                if let Some(ping_pong) = ping_pong {
                    ping_pong.finish(canvas);
                }
                i = end;
                continue;
            }
        }
        let full_quantize = if visible[i].is_3d() {
            let mut j = i;
            while j < visible.len() && visible[j].is_3d() {
                j += 1;
            }
            draw_3d(e, r, ctx, &visible[i..j], canvas)?;
            i = j;
            true
        } else {
            let full_quantize = match r.collapsed(ctx, visible[i]) {
                Some(collapsed) => {
                    draw_collapsed(e, r, ctx, visible[i], &collapsed, canvas)?;
                    true
                }
                None => !draw_layer(e, r, ctx, visible[i], canvas)?,
            };
            i += 1;
            // 8/16 bpc: the comp is an integer buffer after every layer.
            full_quantize
        };
        if full_quantize && let Some(l) = levels {
            *canvas = ops::quantize(e, canvas, l);
        }
    }
    Some(())
}

/// A run of consecutive 3D layers: Classic 3D on the GPU (`classic3d`), with 3D adjustment
/// layers splitting the run as on the CPU; Advanced 3D, wireframes and runs the kernel cannot
/// hold draw on the CPU.
fn draw_3d<'a>(e: &mut Enc, r: &Renderer<'a>, ctx: &EvalCtx<'a>, run: &[&'a Layer], canvas: &mut GpuImage) -> Option<()> {
    // Environment Light Background layers: the sky behind the rest of the run (drawn first, as
    // `three_d::compose::draw_run` does).
    if run.iter().any(|l| l.environment_background) {
        for l in run.iter().filter(|l| l.environment_background) {
            if let Some(sky) = r.sky_draw(ctx, l) {
                *canvas = crate::adv3d::sky(e, &sky, canvas)?;
            }
        }
        let rest: Vec<&'a Layer> = run.iter().copied().filter(|l| !l.environment_background).collect();
        return draw_3d(e, r, ctx, &rest, canvas);
    }
    if let Some(k) = run.iter().position(|l| l.switches.adjustment) {
        draw_3d(e, r, ctx, &run[..k], canvas)?;
        draw_layer(e, r, ctx, run[k], canvas)?;
        return draw_3d(e, r, ctx, &run[k + 1..], canvas);
    }
    if run.is_empty() {
        return Some(());
    }
    // Advanced 3D: rendered and composited on the GPU (`adv3d`).
    if let Some(prep) = r.prepare_adv_run(ctx, run, (canvas.width, canvas.height)) {
        match crate::adv3d::draw_run(e, &prep, canvas) {
            Some(img) => *canvas = img,
            None => on_cpu(e, canvas, |img| r.draw_3d_run(ctx, run, img))?,
        }
        return Some(());
    }
    // Advanced 3D with layers on the 2D compositing path (blend modes, track mattes, Preserve
    // Transparency).
    if let Some(split) = r.split_adv_run(ctx, run, (canvas.width, canvas.height)) {
        match draw_split(e, r, ctx, &split, canvas) {
            Some(img) => *canvas = img,
            None => on_cpu(e, canvas, |img| r.draw_3d_run(ctx, run, img))?,
        }
        return Some(());
    }
    match r.prepare_3d_run(ctx, run, (canvas.width, canvas.height)).and_then(|prep| crate::classic3d::draw_run(e, &prep, canvas)) {
        Some(img) => *canvas = img,
        None => on_cpu(e, canvas, |img| r.draw_3d_run(ctx, run, img))?,
    }
    Some(())
}

/// `three_d::adv::draw_run`'s split path on the GPU canvas: the plain layers as one scene over
/// the canvas, then each special layer (far to near) rendered alone, hidden where the main
/// scene is nearer, matted (a 3D matte through the camera, else the 2D placement), Preserve
/// Transparency and blended. `None` = render the run on the CPU.
fn draw_split(e: &mut Enc, r: &Renderer, ctx: &EvalCtx, split: &effectcraft_render::three_d::adv::SplitRun, canvas: &GpuImage) -> Option<GpuImage> {
    let (w, h) = (canvas.width, canvas.height);
    let main_prep = split.main();
    if main_prep.out != (w, h) {
        return None;
    }
    let encode = !main_prep.linear;
    let main = crate::adv3d::resolve_run(e, &main_prep)?;
    let mut cv = match &main {
        Some(m) => crate::adv3d::finish(e, m, encode, Some(canvas)),
        None => canvas.clone(),
    };
    for i in 0..split.len() {
        let Some(iso) = crate::adv3d::resolve_run(e, &split.special(i))? else { continue };
        let iso = match &main {
            Some(m) => crate::adv3d::occluded(e, &iso, m, encode),
            None => crate::adv3d::finish(e, &iso, encode, None),
        };
        let matte3d = split.with_matte3d(i, |p| match p {
            None => Some(e.zeros(w, h)),
            Some(p) => Some(match crate::adv3d::resolve_run(e, p)? {
                Some(m) => crate::adv3d::finish(e, &m, encode, None),
                None => e.zeros(w, h),
            }),
        });
        let matte3d = match matte3d {
            Some(m) => Some(m?),
            None => None,
        };
        composite_iso_with(e, r, ctx, split.layer(i), iso, &mut cv, 1.0, matte3d, None)?;
    }
    Some(cv)
}

fn draw_collapsed<'a>(
    e: &mut Enc,
    r: &Renderer<'a>,
    ctx: &EvalCtx<'a>,
    layer: &'a Layer,
    collapsed: &effectcraft_render::Collapsed,
    canvas: &mut GpuImage,
) -> Option<()> {
    let opacity = r.layer_opacity(ctx, layer);
    if opacity <= 0.0 {
        return Some(());
    }
    let matte = r.track_matte(ctx, layer).is_some();
    if layer.blend_mode == BlendMode::Normal && !matte && !layer.preserve_transparency {
        if let Some((sub, nctx)) = r.collapse_into(ctx, layer, collapsed, opacity) {
            draw_comp(e, &sub, &nctx, canvas)?;
        }
        return Some(());
    }
    let Some((sub, nctx)) = r.collapse_into(ctx, layer, collapsed, 1.0) else { return Some(()) };
    let mut iso = e.zeros(canvas.width, canvas.height);
    draw_comp(e, &sub, &nctx, &mut iso)?;
    composite_iso(e, r, ctx, layer, iso, canvas, opacity)
}

fn draw_layer(e: &mut Enc, r: &Renderer, ctx: &EvalCtx, layer: &Layer, canvas: &mut GpuImage) -> Option<bool> {
    let out = draw_layer_on(e, r, ctx, layer, canvas, None);
    e.mem_gate();
    out
}

fn draw_prepared_layer(e: &mut Enc, r: &Renderer, ctx: &EvalCtx, layer: &Layer, canvas: &mut GpuImage, st: StyledLayer) -> Option<bool> {
    let out = draw_layer_on(e, r, ctx, layer, canvas, Some(st));
    e.mem_gate();
    out
}

/// `true` means placement or a no-op has already left the canvas quantised; more complex paths
/// need the compositor's full-frame quantisation after this layer.
fn draw_layer_on(e: &mut Enc, r: &Renderer, ctx: &EvalCtx, layer: &Layer, canvas: &mut GpuImage, prepared: Option<StyledLayer>) -> Option<bool> {
    if layer.switches.adjustment {
        if draw_adjustment(e, r, ctx, layer, canvas).is_some() {
            return Some(false);
        }
        return on_cpu(e, canvas, |img| r.draw_layer_cpu(ctx, layer, img)).map(|()| false);
    }
    if r.quality(layer) == Quality::Wireframe {
        // The layer's bounds as a one-pixel outline (the CPU's pixels).
        *canvas = crate::adv3d::wireframe(e, canvas, &r.wireframe_pixels(ctx, layer, (canvas.width, canvas.height)));
        return Some(false);
    }
    let opacity = r.layer_opacity(ctx, layer);
    if opacity <= 0.0 {
        return Some(true);
    }
    if let Some(done) = draw_precomp(e, r, ctx, layer, canvas, opacity) {
        return done.map(|()| true);
    }
    if let Some(done) = draw_resident_fx(e, r, ctx, layer, canvas, opacity) {
        return done.map(|()| true);
    }
    let Some(st) = prepared.or_else(|| r.styled_layer(ctx, layer)) else { return Some(true) };
    if st.plain {
        return composite_layer(e, r, ctx, layer, &st.body, canvas, opacity, st.quantize_source()).map(|()| true);
    }
    // Layer styles: exterior passes with their own modes, then the body; the layer's opacity
    // fades the whole stack; knockout and channel switches as on the CPU.
    let bl = styles::blending(ctx, layer);
    let (w, h) = (canvas.width, canvas.height);
    if r.track_matte(ctx, layer).is_some() || layer.preserve_transparency {
        let mut iso = e.zeros(w, h);
        for (b, m) in &st.passes {
            iso = place(e, r, ctx, layer, b, &iso, *m, 1.0, None)?;
        }
        iso = place(e, r, ctx, layer, &st.body, &iso, BlendMode::Normal, 1.0, None)?;
        return composite_iso(e, r, ctx, layer, iso, canvas, opacity).map(|()| false);
    }
    let mut tmp = canvas.clone();
    if bl.knockout > 0 {
        let k = place(e, r, ctx, layer, &st.content, &e.zeros(w, h), BlendMode::Normal, 1.0, None)?;
        tmp = ops::knockout(e, &tmp, &k);
    }
    for (b, m) in &st.passes {
        tmp = place(e, r, ctx, layer, b, &tmp, *m, 1.0, None)?;
    }
    tmp = place(e, r, ctx, layer, &st.body, &tmp, layer.blend_mode, 1.0, None)?;
    *canvas = ops::channel_mix(e, canvas, &tmp, bl.channels, opacity);
    Some(false)
}

/// A plain layer with effects, its effect stack run on the GPU: the source (+ masks, from the
/// layer cache) uploads once while unchanged, the effects stay resident and the result is
/// placed directly; the CPU path instead runs each GPU-capable effect run through an
/// upload → kernels → readback, then uploads the result again to place it. `None`: not this
/// kind of layer (draw it the usual way); `Some(None)`: the GPU failed.
fn draw_resident_fx(e: &mut Enc, r: &Renderer, ctx: &EvalCtx, layer: &Layer, canvas: &mut GpuImage, opacity: f32) -> Option<Option<()>> {
    if !layer.switches.effects
        || layer.effects().is_none_or(|f| f.children.is_empty())
        || styles::active(ctx, layer)
        || layer.preserve_transparency
        || r.track_matte(ctx, layer).is_some()
        // Unchanged since the last frame: the cached result uploads once (the usual path).
        || r.layer_is_static(ctx, layer)
    {
        return None;
    }
    let input = r.layer_input(ctx, layer, 0)?;
    if input.img.is_empty() {
        return Some(Some(()));
    }
    let Some(src) = e.g.upload_buf(&input) else { return Some(None) };
    let pipe = r.pipe();
    // `layer_input` is in effect space: run the stack there, then back to layer space.
    let mut fx = crate::effects::GpuFx::with_cpu_shadow(e, src, input, pipe.levels);
    let origin = r.run_layer_effects_on(ctx, layer, &mut fx);
    let Some((mut img, mut offset, scale)) = fx.finish() else { return Some(None) };
    offset = [offset[0] - origin[0] * scale, offset[1] - origin[1] * scale];
    if img.width == 0 || img.height == 0 {
        return Some(Some(()));
    }
    if let Some(c) = pipe.to_blend() {
        img = ops::convert(e, &img, &c);
    }
    // The placement only reads the buffer's size, offset and scale.
    let shape = Buf { img: Image { width: img.width, height: img.height, data: Vec::new() }, offset, scale };
    Some(place_src(e, r, ctx, layer, &shape, img, canvas, layer.blend_mode, opacity, pipe.levels, false).map(|c| *canvas = c))
}

/// A 2D precomp layer, its nested comp rendered resident on the GPU (`walk::render` one level
/// down) instead of by the CPU compositor and uploaded every frame (`Renderer::layer_source`
/// renders nested comps on the CPU: depth > 0). Then the layer's effects (resident), its
/// placement, track matte / Preserve Transparency and blend, as for any layer. `None`: not this
/// kind of layer, or the nested comp needs the CPU path (masks, layer styles, proxies,
/// overrides, frame blending, nested motion-blur sampling; see `Renderer::precomp_source`);
/// `Some(None)`: the GPU failed.
fn draw_precomp(e: &mut Enc, r: &Renderer, ctx: &EvalCtx, layer: &Layer, canvas: &mut GpuImage, opacity: f32) -> Option<Option<()>> {
    if layer.masks().is_some_and(|m| !m.children.is_empty()) || styles::active(ctx, layer) {
        return None;
    }
    let (sub, item, lt, bs) = r.precomp_source(ctx, layer)?;
    // A nested comp the GPU can't take (nested too deep, over the texture limit, content it
    // declines) goes back to the CPU path for this layer only; only a device failure fails the
    // frame.
    let key = sub.gpu_static_comp_frame_key(item, lt);
    let mut src = match key {
        Some(key) => match e.static_comp(&key) {
            Some(src) => src,
            None => {
                let Some(src) = render(e, &sub, item, lt) else {
                    return if e.g.check_health().is_ok() { None } else { Some(None) };
                };
                e.cache_static_comp(key, &src);
                src
            }
        },
        None => {
            let Some(src) = render(e, &sub, item, lt) else {
                return if e.g.check_health().is_ok() { None } else { Some(None) };
            };
            src
        }
    };
    let pipe = r.pipe();
    // `masked_source` quantises a non-solid source to the project's depth.
    if let Some(l) = pipe.levels {
        src = ops::quantize(e, &src, l);
    }
    let (mut img, offset, scale) = if layer.switches.effects && layer.effects().is_some_and(|f| !f.children.is_empty()) {
        // Effects run in effect space (0.7.0): from the layer bounds' corner.
        let o = ctx.effect_bounds(layer).1;
        let mut fx = crate::effects::GpuFx::new(e, src, [o[0] * bs, o[1] * bs], bs, pipe.levels);
        r.run_layer_effects_on(ctx, layer, &mut fx);
        let Some((img, off, sc)) = fx.finish() else { return Some(None) };
        (img, [off[0] - o[0] * sc, off[1] - o[1] * sc], sc)
    } else {
        (src, [0.0; 2], bs)
    };
    if img.width == 0 || img.height == 0 {
        return Some(Some(()));
    }
    if let Some(c) = pipe.to_blend() {
        img = ops::convert(e, &img, &c);
    }
    let shape = Buf { img: Image { width: img.width, height: img.height, data: Vec::new() }, offset, scale };
    if r.track_matte(ctx, layer).is_some() || layer.preserve_transparency {
        let Some(iso) = place_src(e, r, ctx, layer, &shape, img, &e.zeros(canvas.width, canvas.height), BlendMode::Normal, 1.0, None, false) else {
            return Some(None);
        };
        return Some(composite_iso(e, r, ctx, layer, iso, canvas, opacity));
    }
    Some(place_src(e, r, ctx, layer, &shape, img, canvas, layer.blend_mode, opacity, pipe.levels, false).map(|c| *canvas = c))
}

/// Adjustment layer (`Renderer::draw_adjustment`): the effect stack runs on the comp below,
/// resident on the GPU (non-GPU effects read back, run on the CPU and upload), and the result
/// replaces the canvas inside the layer's footprint × track matte × opacity, or blends onto it in
/// the layer's blend mode when that isn't Normal. `None` = draw it on the CPU.
fn draw_adjustment(e: &mut Enc, r: &Renderer, ctx: &EvalCtx, layer: &Layer, canvas: &mut GpuImage) -> Option<()> {
    let opacity = r.layer_opacity(ctx, layer);
    let Some(foot) = r.adjustment_footprint(ctx, layer) else { return Some(()) };
    let (w, h) = (canvas.width, canvas.height);
    let matte = place(e, r, ctx, layer, &foot, &e.zeros(w, h), BlendMode::Normal, 1.0, None)?;
    let matte = track_matte(e, r, ctx, layer, matte, None)?;
    let pipe = r.pipe();
    // Effects run in the working space, not the linear blending space.
    let below = match pipe.from_blend() {
        Some(c) => ops::convert(e, canvas, &c),
        None => canvas.clone(),
    };
    let mut fx = crate::effects::GpuFx::new(e, below, [0.0; 2], r.opts.scale, pipe.levels);
    r.run_effects_on(ctx, layer, true, &mut fx);
    let (mut adjusted, offset, _) = fx.finish()?;
    // An effect that changed the frame (Motion Tile's Output size): back onto the canvas.
    if let Some((dx, dy)) = effectcraft_effects::adjustment_crop(offset, [0.0; 2], (adjusted.width, adjusted.height), (w, h)) {
        adjusted = e.crop(&adjusted, dx, dy, w, h);
    }
    if adjusted.width != w || adjusted.height != h {
        return Some(());
    }
    if let Some(c) = pipe.to_blend() {
        adjusted = ops::convert(e, &adjusted, &c);
    }
    *canvas = if layer.blend_mode == BlendMode::Normal {
        ops::adjust_mix(e, canvas, &adjusted, &matte, opacity)
    } else {
        // The adjusted comp is a layer in that mode over the comp below (#2618).
        let iso = ops::matte(e, &adjusted, &matte, effectcraft_project::MatteKind::Alpha);
        ops::blend_full(e, canvas, &iso, layer.blend_mode, opacity, layer.id.0 as u32, None)
    };
    Some(())
}

fn composite_layer(
    e: &mut Enc,
    r: &Renderer,
    ctx: &EvalCtx,
    layer: &Layer,
    buf: &Arc<Buf>,
    canvas: &mut GpuImage,
    opacity: f32,
    quantize_source: bool,
) -> Option<()> {
    let matte = r.track_matte(ctx, layer).is_some();
    if !matte && !layer.preserve_transparency {
        // The comp is quantised after the layer (8/16 bpc) in the same pass.
        let source_is_8bit = false;
        let src = if source_is_8bit { e.g.upload_buf_8bit(buf, quantize_source)? } else { e.g.upload_buf(buf)? };
        *canvas = place_src(e, r, ctx, layer, buf, src, canvas, layer.blend_mode, opacity, r.pipe().levels, quantize_source || source_is_8bit)?;
        return Some(());
    }
    // Render in isolation, then matte / preserve transparency, then blend.
    let iso = place(e, r, ctx, layer, buf, &e.zeros(canvas.width, canvas.height), BlendMode::Normal, 1.0, None)?;
    composite_iso(e, r, ctx, layer, iso, canvas, opacity)
}

/// Track matte / Preserve Transparency on an isolated layer render, then blend.
fn composite_iso(e: &mut Enc, r: &Renderer, ctx: &EvalCtx, layer: &Layer, iso: GpuImage, canvas: &mut GpuImage, opacity: f32) -> Option<()> {
    composite_iso_with(e, r, ctx, layer, iso, canvas, opacity, None, r.pipe().levels)
}

/// `Renderer::composite_iso_with`: the matte from `matte_img` when given (a 3D matte drawn
/// through the camera), else the matte layer placed in 2D; the blend quantised to `levels`.
#[allow(clippy::too_many_arguments)]
fn composite_iso_with(
    e: &mut Enc,
    r: &Renderer,
    ctx: &EvalCtx,
    layer: &Layer,
    mut iso: GpuImage,
    canvas: &mut GpuImage,
    opacity: f32,
    matte_img: Option<GpuImage>,
    levels: Option<f32>,
) -> Option<()> {
    iso = track_matte(e, r, ctx, layer, iso, matte_img)?;
    if layer.preserve_transparency {
        iso = ops::preserve(e, &iso, canvas);
    }
    *canvas = ops::blend_full(e, canvas, &iso, layer.blend_mode, opacity, layer.id.0 as u32, levels);
    Some(())
}

/// `Renderer::apply_track_matte`: `img` (comp-sized) times the layer's track matte, if it has
/// one; `matte_img` as in [`composite_iso_with`].
fn track_matte(e: &mut Enc, r: &Renderer, ctx: &EvalCtx, layer: &Layer, img: GpuImage, matte_img: Option<GpuImage>) -> Option<GpuImage> {
    let Some((m, kind)) = r.track_matte(ctx, layer) else { return Some(img) };
    let mimg = match matte_img.filter(|i| (i.width, i.height) == (img.width, img.height)) {
        Some(i) => i,
        None => {
            let mut mimg = e.zeros(img.width, img.height);
            if m.is_active_at(ctx.time)
                && let Some(mb) = r.layer_buf(ctx, m)
            {
                let mo = ctx.opacity(m) as f32;
                mimg = place(e, r, ctx, m, &mb, &mimg, BlendMode::Normal, mo, None)?;
            }
            mimg
        }
    };
    Some(ops::matte(e, &img, &mimg, kind))
}

/// Draw a processed layer buffer onto `target` (transform, motion blur sub-samples), quantising
/// the result to `levels` when set.
#[allow(clippy::too_many_arguments)]
fn place(
    e: &mut Enc,
    r: &Renderer,
    ctx: &EvalCtx,
    layer: &Layer,
    buf: &Arc<Buf>,
    target: &GpuImage,
    mode: BlendMode,
    opacity: f32,
    levels: Option<f32>,
) -> Option<GpuImage> {
    if buf.img.is_empty() || opacity <= 0.0 {
        return Some(target.clone());
    }
    let src = e.g.upload_buf(buf)?;
    place_src(e, r, ctx, layer, buf, src, target, mode, opacity, levels, false)
}

/// [`place`] of a GPU image `src` shaped like `buf`.
#[allow(clippy::too_many_arguments)]
fn place_src(
    e: &mut Enc,
    r: &Renderer,
    ctx: &EvalCtx,
    layer: &Layer,
    buf: &Buf,
    src: GpuImage,
    target: &GpuImage,
    mode: BlendMode,
    opacity: f32,
    levels: Option<f32>,
    quantize_source: bool,
) -> Option<GpuImage> {
    let pl = r.placement(ctx, layer, buf);
    if let [m] = pl.matrices.as_slice() {
        let opts = ops::WarpOptions { sampling: pl.sampling, mode, opacity, seed: pl.seed };
        return Some(if quantize_source {
            ops::warp_q_source_quantized(e, target, &src, m, opts, levels)
        } else {
            ops::warp_q(e, target, &src, m, opts, None, levels)
        });
    }
    let mut acc = e.zeros(target.width, target.height);
    let k = 1.0 / pl.matrices.len() as f32;
    for m in &pl.matrices {
        acc = ops::warp(e, &acc, &src, m, ops::WarpOptions { sampling: pl.sampling, mode: BlendMode::Normal, opacity: 1.0, seed: 0 }, Some(k));
    }
    Some(ops::blend_full(e, target, &acc, mode, opacity, pl.seed, levels))
}
