use super::frames::{adaptive_samples, frame_blend_mode};
use super::opts::{LayerTiming, Through};
use super::{
    Arc, BlendMode, Buf, Comp, EvalCtx, Image, IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, Layer, LayerSource, Mat3,
    MatteKind, ParallelIterator, Quality, Region, Renderer, Sampling, Tick, WarpOpts, cache, color, composite_warp_changed, masks, styles, vec2,
};
use effectcraft_geom::Rect;
use effectcraft_project::{ItemId, ItemKind};

const OCCLUSION_TILE: usize = 4;

#[derive(Clone, Copy)]
struct OpaqueBounds {
    /// Draw bounds, conservatively expanded for bilinear sampling and clipped to the output.
    draw: [usize; 4],
    /// Interior that remains opaque with bilinear sampling.
    opaque: [f64; 4],
}

/// Blend policy for placing a layer, including optional output-depth quantization.
struct PlacementBlend {
    mode: BlendMode,
    opacity: f32,
    quantize: Option<f32>,
}

/// A stable source identity that the GPU may reuse within one encoder.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GpuFootageSourceKey {
    project: usize,
    item: ItemId,
    time: Tick,
    scale: u64,
    proxy: bool,
    quantize_source: bool,
}

/// A motion-blur shutter: the comp's, or for a collapsed precomp's layers the outermost
/// containing comp's (they are drawn into its frames).
#[derive(Clone, Copy)]
struct Shutter {
    /// Shutter angle and phase as fractions of a frame.
    angle: f64,
    phase: f64,
    /// Frame duration (seconds).
    frame: f64,
    /// Samples Per Frame and Adaptive Sample Limit.
    samples: u32,
    limit: u32,
}

impl Shutter {
    /// Seconds from the frame time to sub-sample `i` of `n`, spread over the open shutter.
    fn offset(&self, i: usize, n: usize) -> f64 {
        let f = if n > 1 { i as f64 / (n - 1) as f64 } else { 0.0 };
        (self.phase + self.angle * f) * self.frame
    }
}

impl<'a> Renderer<'a> {
    /// Identity for a decoded footage texture that can be shared by duplicate plain layers in
    /// this frame. The GPU cache is encoder-local, so project edits cannot stale the entry.
    pub fn gpu_footage_source_key(&self, ctx: &EvalCtx, layer: &Layer, quantize_source: bool) -> Option<GpuFootageSourceKey> {
        if !self.cache_static_comps
            || self.profile.is_some()
            || layer.switches.collapse
            || self.blurred(ctx, layer)
            || frame_blend_mode(ctx, layer) != super::FrameBlend::Off
            || layer.masks().is_some_and(|m| !m.children.is_empty())
            || layer.switches.effects && layer.effects().is_some_and(|fx| !fx.children.is_empty())
            || styles::active(ctx, layer)
        {
            return None;
        }
        let LayerSource::Footage { item } = &layer.source else { return None };
        let item = ctx.media_item(layer, *item);
        let ItemKind::Footage(f) = &self.project.item(item)?.kind else { return None };
        if !f.has_video || f.missing {
            return None;
        }
        let proxy = self.proxy_for(item);
        let source = proxy.unwrap_or(f);
        let time = if source.kind == super::FootageKind::Still { Tick::ZERO } else { ctx.source_time(layer) };
        Some(GpuFootageSourceKey {
            project: self.project as *const _ as usize,
            item,
            time,
            scale: self.opts.scale.to_bits(),
            proxy: proxy.is_some(),
            quantize_source,
        })
    }

    /// The layer with its styles: exterior passes and body (cached per pass).
    pub(crate) fn styled(&self, ctx: &EvalCtx, layer: &Layer, mut timing: Option<&mut LayerTiming>) -> Option<StyledLayer> {
        let (content, ckey) = self.content_buf_timed(ctx, layer, timing.as_deref_mut())?;
        if !styles::active(ctx, layer) {
            return Some(StyledLayer {
                passes: vec![],
                body: content.clone(),
                content,
                key: ckey,
                ckey,
                plain: true,
                quantize_source: false,
                solid_color: None,
                solid_size: None,
            });
        }
        let modes = styles::pass_modes(ctx, layer);
        let key = ckey.map(|k| cache::styles_key(ctx, layer, k));
        if let (Some(c), Some(k)) = (self.cache, key)
            && let Some(body) = c.get(k)
        {
            let passes: Option<Vec<_>> = modes.iter().enumerate().map(|(i, m)| c.get(cache::derive(k, i as u64 + 1)).map(|b| (b, *m))).collect();
            if let Some(passes) = passes {
                return Some(StyledLayer { passes, body, content, key, ckey, plain: false, quantize_source: false, solid_color: None, solid_size: None });
            }
        }
        let t0 = web_time::Instant::now();
        let size = ctx.source_size(layer);
        let layer_size = if size.0 == 0 { [ctx.comp.width as f64, ctx.comp.height as f64] } else { [size.0 as f64, size.1 as f64] };
        let st = styles::render(ctx, layer, &content, layer_size);
        if let Some(t) = timing {
            t.cached = false;
            t.effects.push(("layerStyles".into(), t0.elapsed().as_secs_f64() * 1e3));
        }
        let mut body = Arc::new(st.body);
        let mut passes: Vec<_> = st.passes.into_iter().map(|(b, m)| (Arc::new(b), m)).collect();
        // The buffers the cache keeps (another worker's, if it kept them first).
        if let (Some(c), Some(k)) = (self.cache_to_keep(), key) {
            body = c.insert(k, body);
            for (i, (b, _)) in passes.iter_mut().enumerate() {
                *b = c.insert(cache::derive(k, i as u64 + 1), b.clone());
            }
        }
        Some(StyledLayer { passes, body, content, key, ckey, plain: false, quantize_source: false, solid_color: None, solid_size: None })
    }

    /// The styled layer's buffers in the blending space.
    pub(crate) fn styled_to_blend(&self, st: StyledLayer) -> StyledLayer {
        if self.pipe.to_blend().is_none() {
            return st;
        }
        let body = self.to_blend(st.body, st.key);
        let content = if st.plain { body.clone() } else { self.to_blend(st.content, st.ckey) };
        let passes = st.passes.into_iter().enumerate().map(|(i, (b, m))| (self.to_blend(b, st.key.map(|k| cache::derive(k, i as u64 + 1))), m)).collect();
        StyledLayer { passes, body, content, ..st }
    }

    pub(crate) fn sampling(&self, layer: &Layer) -> effectcraft_raster::Sampling {
        if self.quality(layer) == Quality::Draft {
            // Draft quality: no interpolation (nearest neighbour), as After Effects' Draft.
            effectcraft_raster::Sampling::Nearest
        } else if self.opts.draft {
            effectcraft_raster::Sampling::Bilinear
        } else if layer.switches.sampling == Sampling::Bicubic {
            effectcraft_raster::Sampling::Bicubic
        } else {
            effectcraft_raster::Sampling::Bilinear
        }
    }

    /// The region of interest of a top-level 2D frame: (x, y) offset of the output in output
    /// pixels and its size. `None` renders the whole comp (no ROI, or a nested comp).
    pub(crate) fn roi_offset(&self) -> Option<(f64, f64, u32, u32)> {
        let r = self.opts.roi.filter(|_| self.top)?;
        let s = self.opts.scale;
        if r[2] <= 0.0 || r[3] <= 0.0 {
            return None;
        }
        Some(((r[0] * s).round(), (r[1] * s).round(), ((r[2] * s).round() as u32).max(1), ((r[3] * s).round() as u32).max(1)))
    }

    /// The region of interest's offset in output pixels for 3D projection (0 without one).
    pub(crate) fn out_offset(&self) -> (f64, f64) {
        self.roi_offset().map_or((0.0, 0.0), |(x, y, _, _)| (x, y))
    }

    /// Scaled comp pixel → output pixel: the region of interest's offset.
    fn out_matrix(&self) -> Mat3 {
        match self.roi_offset() {
            Some((x, y, _, _)) => Mat3::translate(vec2(-x, -y)),
            None => Mat3::IDENTITY,
        }
    }

    /// Buffer pixel → output pixel matrix for the layer at the context time.
    fn buf_matrix(&self, ctx: &EvalCtx, layer: &Layer, buf: &Buf) -> Mat3 {
        let (l2c, _) = ctx.layer_to_comp(layer);
        self.place_matrix(self.outer.map_or(l2c, |o| o * l2c), buf)
    }

    /// The raster mip dimensions a plain solid would use for its current transform. Solid pixels
    /// are uniform, so a caller can construct the final mip directly instead of filling and then
    /// halving the full-resolution source image. The transform depends on the source scale and
    /// offset, not its pixels or dimensions, so use a 1×1 placeholder here.
    pub(crate) fn solid_mip(&self, ctx: &EvalCtx, layer: &Layer, width: u32, height: u32, source_scale: f64) -> Option<(f64, u32, u32)> {
        let buf = Buf { img: Image::new(1, 1), offset: [0.0; 2], scale: source_scale };
        let m = self.buf_matrix(ctx, layer, &buf);
        let mut scale = m.mean_scale();
        if !m.is_affine() || scale >= 0.5 {
            return None;
        }
        let (mut w, mut h) = (width.div_ceil(2).max(1), height.div_ceil(2).max(1));
        let mut factor = 0.5;
        scale *= 2.0;
        while scale < 0.5 && w > 1 && h > 1 {
            w = w.div_ceil(2).max(1);
            h = h.div_ceil(2).max(1);
            factor *= 0.5;
            scale *= 2.0;
        }
        Some((factor, w, h))
    }

    /// Buffer pixel → output pixel matrix for a layer → (top-level) comp matrix.
    fn place_matrix(&self, l2c: Mat3, buf: &Buf) -> Mat3 {
        let s = self.opts.scale;
        self.out_matrix()
            * Mat3::scale(vec2(s, s))
            * l2c
            * Mat3::scale(vec2(1.0 / buf.scale, 1.0 / buf.scale))
            * Mat3::translate(vec2(-buf.offset[0], -buf.offset[1]))
    }

    /// The collapsed precomp layers this comp is drawn through, outermost first.
    fn links(&self) -> impl Iterator<Item = &Through<'a>> {
        self.through.iter().map_while(Option::as_ref)
    }

    /// The shutter of a layer's motion blur: its comp's, or for a collapsed precomp's layers the
    /// outermost containing comp's.
    fn shutter(&self, ctx: &EvalCtx) -> Shutter {
        let c: &Comp = match self.links().next() {
            Some(l) => l.comp,
            None => ctx.comp,
        };
        Shutter {
            angle: c.shutter_angle / 360.0,
            phase: c.shutter_phase / 360.0,
            frame: c.frame_duration().seconds(),
            samples: c.motion_blur_samples,
            limit: c.motion_blur_adaptive_limit,
        }
    }

    /// Whether a layer moves within the shutter: its own motion blur, or that of a collapsed
    /// precomp layer it is drawn through.
    pub(crate) fn blurred(&self, ctx: &EvalCtx, layer: &Layer) -> bool {
        self.mb_on(ctx, layer) || (!self.through_lost && self.links().any(|l| l.blur))
    }

    /// The collapsed precomps' matrix (nested comp → the frame's comp) and the nested comp's time
    /// `dt` seconds into the outermost comp's shutter: each precomp layer moves if it is motion
    /// blurred, and times map down through each one (stretch, time remapping).
    fn outer_at(&self, ctx: &EvalCtx, dt: f64) -> (Option<Mat3>, Tick) {
        let shift = Tick::from_seconds_f64(dt);
        let mut outer: Option<Mat3> = None;
        let mut t: Option<Tick> = None;
        for l in self.links() {
            let at = EvalCtx { comp_id: l.comp_id, comp: l.comp, time: t.unwrap_or(l.time + shift), ..*ctx };
            let (m, _) = if l.blur { at.layer_to_comp(l.layer) } else { EvalCtx { time: l.time, ..at }.layer_to_comp(l.layer) };
            outer = Some(outer.map_or(m, |o| o * m));
            t = Some(at.nested_time(l.layer));
        }
        (outer, t.unwrap_or(ctx.time + shift))
    }

    /// [`Self::buf_matrix`] for the motion-blur sub-sample `dt` seconds into the shutter: the
    /// layer moves if it is motion blurred, the collapsed precomps above it if theirs are.
    fn buf_matrix_at(&self, ctx: &EvalCtx, layer: &Layer, buf: &Buf, dt: f64) -> Mat3 {
        if self.through_lost || self.links().next().is_none() {
            return self.buf_matrix(&ctx.at(ctx.time + Tick::from_seconds_f64(dt)), layer, buf);
        }
        let (outer, t) = self.outer_at(ctx, dt);
        let lctx = if self.mb_on(ctx, layer) { ctx.at(t) } else { *ctx };
        let (l2c, _) = lctx.layer_to_comp(layer);
        self.place_matrix(outer.map_or(l2c, |o| o * l2c), buf)
    }

    /// Shape and text content that animates within the shutter (paths, Trim Paths, text
    /// animators…) of a motion-blurred layer: the times its source is drawn at and averaged,
    /// Samples Per Frame of them, as After Effects does for shape layers. `None` when the content
    /// holds still (compared through its cache key, which leaves the transform out).
    pub(crate) fn content_times(&self, ctx: &EvalCtx, layer: &Layer) -> Option<Vec<Tick>> {
        if !matches!(layer.source, LayerSource::Shape | LayerSource::Text) || !self.mb_on(ctx, layer) {
            return None;
        }
        let sh = self.shutter(ctx);
        if sh.angle <= 0.0 {
            return None;
        }
        let n = if self.opts.draft { 4 } else { (sh.samples as usize).clamp(2, 64) };
        let times: Vec<Tick> = (0..n).map(|i| self.outer_at(ctx, sh.offset(i, n)).1).collect();
        let key = |t: Tick| cache::layer_key(&ctx.at(t), layer, 1.0, false, false);
        let k0 = key(*times.first()?)?;
        let moves = [times.get(n / 2), times.last()].into_iter().flatten().any(|t| key(*t) != Some(k0));
        moves.then_some(times)
    }

    /// The layer's source drawn at each of `times` and averaged (premultiplied) on the union of
    /// their bounds. Content that also changes its raster scale within the shutter, or would need
    /// an outsized buffer, is drawn once.
    pub(crate) fn blurred_source(&self, ctx: &EvalCtx, layer: &Layer, times: &[Tick]) -> Option<Buf> {
        let bufs: Vec<Buf> = times.iter().filter_map(|t| self.source(&ctx.at(*t), layer)).collect();
        let scale = bufs.first()?.scale;
        // Pixel = layer point × scale + offset: with the largest offset every buffer lands at
        // x + (off - its offset) ≥ 0.
        let off = [0, 1].map(|i| bufs.iter().map(|b| b.offset[i]).fold(f64::MIN, f64::max));
        let w = bufs.iter().map(|b| off[0] - b.offset[0] + b.img.width as f64).fold(0.0, f64::max).ceil();
        let h = bufs.iter().map(|b| off[1] - b.offset[1] + b.img.height as f64).fold(0.0, f64::max).ceil();
        if bufs.iter().any(|b| (b.scale - scale).abs() > 1e-9) || !(w * h).is_finite() || w * h > (1u64 << 26) as f64 {
            return self.source(ctx, layer);
        }
        let mut acc = Image::new(w as u32, h as u32);
        let k = 1.0 / bufs.len() as f32;
        for b in &bufs {
            let m = Mat3::translate(vec2(off[0] - b.offset[0], off[1] - b.offset[1]));
            effectcraft_raster::accumulate_warp(&mut acc, &b.img, &m, effectcraft_raster::Sampling::Bilinear, k);
        }
        Some(Buf { img: acc, offset: off, scale })
    }

    /// A processed-layer cache key that also covers content motion blur (the averaged source
    /// depends on the content at every sub-sample).
    pub(crate) fn content_key(&self, ctx: &EvalCtx, layer: &Layer, key: Option<u64>) -> Option<u64> {
        let k = key?;
        let Some(times) = self.content_times(ctx, layer) else { return Some(k) };
        times.iter().try_fold(cache::derive(k, 0x6d62_6c72), |d, t| Some(cache::derive(d, cache::layer_key(&ctx.at(*t), layer, 1.0, false, false)?)))
    }

    /// Whether a layer is motion blurred: the render options, the comp's switch and the layer's
    /// switch (limited by the precomp layers above when switches affect nested comps). Layer
    /// motion, mask motion blur, effects that read the shutter and Advanced 3D all ask this.
    pub(crate) fn mb_on(&self, ctx: &EvalCtx, layer: &Layer) -> bool {
        self.opts.motion_blur && ctx.comp.enable_motion_blur && self.layer_motion_blur(layer)
    }

    /// What a layer's mask motion blur may do in this render (see [`masks::MaskBlur`]).
    pub(crate) fn mask_blur(&self, ctx: &EvalCtx, layer: &Layer) -> masks::MaskBlur {
        masks::MaskBlur {
            allowed: self.opts.motion_blur && ctx.comp.enable_motion_blur && self.inherited.is_none_or(|(_, mb)| mb),
            layer: self.layer_motion_blur(layer),
            samples: if self.opts.draft { 4 } else { ctx.comp.motion_blur_samples as usize },
        }
    }

    /// Motion-blur sub-samples for a layer buffer (1 = no motion blur). As in After Effects, a 2D
    /// layer gets about one sample per output pixel its corners travel during the shutter,
    /// at least Samples Per Frame and at most the comp's Adaptive Sample Limit; a layer that
    /// does not move within the shutter is drawn once.
    fn mb_samples(&self, ctx: &EvalCtx, layer: &Layer, buf: &Buf) -> usize {
        if !self.blurred(ctx, layer) {
            return 1;
        }
        let travel = self.shutter_travel(ctx, layer, buf);
        if travel < 1e-3 {
            return 1;
        }
        if self.opts.draft {
            return 4;
        }
        let sh = self.shutter(ctx);
        adaptive_samples(travel, sh.samples, sh.limit)
    }

    /// Largest distance (output pixels) a corner of the layer buffer moves during the shutter,
    /// measured along the path at a few points.
    fn shutter_travel(&self, ctx: &EvalCtx, layer: &Layer, buf: &Buf) -> f64 {
        let sh = self.shutter(ctx);
        let (w, h) = (buf.img.width as f64, buf.img.height as f64);
        let corners = [(0.0, 0.0), (w, 0.0), (0.0, h), (w, h), (w * 0.5, h * 0.5)];
        const PROBES: usize = 8;
        let mut prev: Option<Vec<effectcraft_geom::Vec2>> = None;
        let mut travel = vec![0.0f64; corners.len()];
        for i in 0..PROBES {
            let m = self.buf_matrix_at(ctx, layer, buf, sh.offset(i, PROBES));
            let pts: Vec<_> = corners.iter().map(|&(x, y)| m.apply(vec2(x, y))).collect();
            if let Some(p) = &prev {
                for (k, (a, b)) in p.iter().zip(&pts).enumerate() {
                    travel[k] += (b.x - a.x).hypot(b.y - a.y);
                }
            }
            prev = Some(pts);
        }
        travel.into_iter().fold(0.0, f64::max)
    }

    /// Canvas pixels that compositing `buf` for `layer` can change: the transformed buffer's
    /// bounds plus the resampling footprint. Stencil/silhouette modes, motion blur and
    /// perspective fall back to the whole canvas.
    ///
    /// A quantised buffer placed pixel-exactly (integer translation, Normal, 100%) over a blank
    /// canvas leaves exactly its own (quantised) pixels: nothing to re-quantise.
    pub(crate) fn footprint(&self, ctx: &EvalCtx, layer: &Layer, buf: &Buf, opacity: f32, blank: bool) -> Region {
        if layer.blend_mode.is_stencil() || self.mb_samples(ctx, layer, buf) > 1 {
            return Region::Full;
        }
        let m = self.buf_matrix(ctx, layer, buf);
        if !m.is_affine() {
            return Region::Full;
        }
        let matte = layer.track_matte.is_some_and(|tm| tm.layer != layer.id && ctx.comp.layer(tm.layer).is_some());
        let k = &m.0;
        let integral = |v: f64| v == v.round();
        if blank
            && layer.blend_mode == BlendMode::Normal
            && opacity == 1.0
            && !matte
            && !layer.preserve_transparency
            && self.pipe.to_blend().is_none()
            && k[0][0] == 1.0
            && k[1][1] == 1.0
            && k[0][1] == 0.0
            && k[1][0] == 0.0
            && integral(k[0][2])
            && integral(k[1][2])
        {
            return Region::Empty;
        }
        let (w, h) = (buf.img.width as f64, buf.img.height as f64);
        let c = [(0.0, 0.0), (w, 0.0), (0.0, h), (w, h)].map(|(x, y)| m.apply(vec2(x, y)));
        let (x0, x1) = (c.iter().map(|p| p.x).fold(f64::INFINITY, f64::min), c.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max));
        let (y0, y1) = (c.iter().map(|p| p.y).fold(f64::INFINITY, f64::min), c.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max));
        color::PxRect::covering(x0, y0, x1, y1, 3.0).map_or(Region::Full, Region::Rect)
    }

    /// How a processed layer buffer lands in the output: one buffer → output matrix, or one per
    /// motion-blur sub-sample (averaged), with the layer's sampling filter and dissolve seed.
    pub fn placement(&self, ctx: &EvalCtx, layer: &Layer, buf: &Buf) -> Placement {
        let samples = self.mb_samples(ctx, layer, buf);
        let (sampling, seed) = (self.sampling(layer), layer.id.0 as u32);
        if samples <= 1 {
            return Placement { matrices: vec![self.buf_matrix(ctx, layer, buf)], sampling, seed };
        }
        let sh = self.shutter(ctx);
        let matrices = (0..samples).map(|i| self.buf_matrix_at(ctx, layer, buf, sh.offset(i, samples))).collect();
        Placement { matrices, sampling, seed }
    }

    /// Draw a processed layer into `target` (transform, motion blur) with `mode`/`opacity`.
    pub(crate) fn place(&self, ctx: &EvalCtx, layer: &Layer, buf: &Buf, target: &mut Image, mode: BlendMode, opacity: f32) {
        self.place_changed(ctx, layer, buf, target, PlacementBlend { mode, opacity, quantize: None });
    }

    /// [`Self::place`], returning the target pixels it changed when the warp tracked them (then
    /// also quantised to `quantize` levels; see [`composite_warp_changed`]).
    fn place_changed(&self, ctx: &EvalCtx, layer: &Layer, buf: &Buf, target: &mut Image, blend: PlacementBlend) -> Option<[u32; 4]> {
        let PlacementBlend { mode, opacity, quantize } = blend;
        let pl = self.placement(ctx, layer, buf);
        let opts = WarpOpts { sampling: pl.sampling, opacity, mode, seed: pl.seed, clip: None };
        if let [m] = pl.matrices.as_slice() {
            let source_opaque = self.source_is_known_opaque(ctx, layer);
            return match quantize {
                Some(levels) if source_opaque => effectcraft_raster::composite_warp_quantized_opaque(target, &buf.img, m, &opts, levels).then_some([0; 4]),
                Some(levels) => effectcraft_raster::composite_warp_quantized(target, &buf.img, m, &opts, levels).then_some([0; 4]),
                None if source_opaque => effectcraft_raster::composite_warp_changed_opaque(target, &buf.img, m, &opts, None),
                None => composite_warp_changed(target, &buf.img, m, &opts, None),
            };
        }
        // Sub-samples sum straight into one buffer (each warp is row-parallel).
        let mut acc = Image::new(target.width, target.height);
        let k = 1.0 / pl.matrices.len() as f32;
        for m in &pl.matrices {
            effectcraft_raster::accumulate_warp(&mut acc, &buf.img, m, opts.sampling, k);
        }
        target.blend_from(&acc, mode, opacity, pl.seed);
        None
    }

    /// Solid and alpha-ignored still sources are fully opaque before transform. Their warped
    /// bilinear path can skip the per-layer transparency-tile scan; masks, effects and styles
    /// may introduce transparent pixels, so those sources keep the general path.
    pub(crate) fn source_is_known_opaque(&self, ctx: &EvalCtx, layer: &Layer) -> bool {
        let (item, solid) = match layer.source {
            LayerSource::Solid { item } => (item, true),
            LayerSource::Footage { item } => (item, false),
            _ => return false,
        };
        if layer.masks().is_some_and(|m| !m.children.is_empty())
            || layer.switches.effects && layer.effects().is_some_and(|fx| fx.groups().any(|group| group.enabled))
            || styles::active(ctx, layer)
        {
            return false;
        }
        if solid {
            self.project.item(item).is_some_and(|it| it.proxy.is_none() && matches!(&it.kind, ItemKind::Solid(_)))
        } else {
            self.project.item(ctx.media_item(layer, item)).is_some_and(|it| {
                it.proxy.is_none()
                    && matches!(&it.kind, ItemKind::Footage(f) if f.kind == effectcraft_project::FootageKind::Still && f.alpha == effectcraft_project::AlphaMode::Ignore && !f.invert_alpha)
            })
        }
    }

    /// Whether a plain precomp layer has nothing to draw at its nested time. Rendering an empty
    /// comp would allocate and warp a full-frame transparent image, so skip that work when it
    /// cannot change the result.
    pub(crate) fn empty_precomp(&self, ctx: &EvalCtx, layer: &Layer) -> bool {
        if !matches!(layer.source, LayerSource::Comp { .. })
            || layer.switches.adjustment
            || layer.blend_mode != BlendMode::Normal
            || layer.track_matte.is_some()
            || layer.preserve_transparency
            || (layer.switches.effects && layer.effects().is_some_and(|fx| fx.groups().any(|g| g.enabled)))
            || styles::active(ctx, layer)
        {
            return false;
        }
        let Some((sub, comp_id, time, _)) = self.precomp_source(ctx, layer) else { return false };
        let Some(nested_ctx) = sub.eval_ctx(comp_id, time) else { return false };
        if nested_ctx.comp.layers.iter().any(|l| l.environment && l.switches.video && l.is_active_at(time)) {
            return false;
        }
        sub.visible_layers(&nested_ctx).is_empty()
    }

    /// The source and style buffers a 2D layer will place. Plain solids can go straight to their
    /// final mip; other layers use the regular source → masks → effects → styles path.
    pub(crate) fn styled_for_draw(&self, ctx: &EvalCtx, layer: &Layer, timing: Option<&mut LayerTiming>) -> Option<StyledLayer> {
        if let Some(buf) = self.plain_solid_mip(ctx, layer) {
            let body = Arc::new(buf);
            return Some(StyledLayer {
                passes: vec![],
                body: body.clone(),
                content: body,
                key: None,
                ckey: None,
                plain: true,
                quantize_source: false,
                solid_color: None,
                solid_size: None,
            });
        }
        self.styled(ctx, layer, timing)
    }

    /// Prepare a plain solid for the GPU without allocating its full uniform raster. The shader
    /// evaluates the solid's rectangular sample directly, including the normal filter edge.
    pub(crate) fn styled_for_gpu_draw(&self, ctx: &EvalCtx, layer: &Layer) -> Option<StyledLayer> {
        if self.pipe.to_blend().is_none()
            && let Some((buf, color, size)) = self.plain_solid_gpu_mip(ctx, layer)
        {
            let body = Arc::new(buf);
            return Some(StyledLayer {
                passes: vec![],
                body: body.clone(),
                content: body,
                key: None,
                ckey: None,
                plain: true,
                quantize_source: false,
                solid_color: Some(color),
                solid_size: Some(size),
            });
        }
        self.styled_for_draw(ctx, layer, None)
    }

    /// Composite precomputed buffers for a plain 2D layer in the original stack order.
    pub(crate) fn draw_prepared_layer(&self, ctx: &EvalCtx, layer: &Layer, st: StyledLayer, canvas: &mut Image, opacity: f32, blank: bool) -> Region {
        if st.plain {
            let footprint = self.footprint(ctx, layer, &st.body, opacity, blank);
            let quantize = if footprint == Region::Empty { None } else { self.pipe.levels };
            match self.composite_layer(ctx, layer, &st.body, canvas, opacity, quantize) {
                Some(_) if quantize.is_some() => Region::Empty,
                drawn => narrow(footprint, drawn),
            }
        } else {
            self.composite_styled(ctx, layer, &st, canvas, opacity);
            Region::Full
        }
    }

    /// Draw a 2D layer into `canvas`; returns the region it may have changed.
    pub(crate) fn draw_layer(&self, ctx: &EvalCtx, layer: &Layer, canvas: &mut Image, blank: bool) -> Region {
        let opacity = ctx.opacity(layer) as f32 * self.opacity_mul;
        if self.quality(layer) == Quality::Wireframe && !layer.switches.adjustment {
            self.draw_wireframe(ctx, layer, canvas);
            return Region::Full;
        }
        if opacity <= 0.0 && !layer.switches.adjustment {
            return Region::Empty;
        }
        if self.empty_precomp(ctx, layer) {
            return Region::Empty;
        }
        if layer.switches.adjustment {
            self.draw_adjustment(ctx, layer, canvas, opacity);
            return Region::Full;
        }
        let mut timing = self.profile.map(|_| LayerTiming { depth: self.depth, layer: layer.name.clone(), ..Default::default() });
        let t0 = web_time::Instant::now();
        let Some(st) = self.styled_for_draw(ctx, layer, timing.as_mut()) else { return Region::Empty };
        let st = self.styled_to_blend(st);
        let t1 = web_time::Instant::now();
        let changed = self.draw_prepared_layer(ctx, layer, st, canvas, opacity, blank);
        if let (Some(prof), Some(mut timing)) = (self.profile, timing) {
            timing.process_ms = (t1 - t0).as_secs_f64() * 1e3;
            timing.composite_ms = t1.elapsed().as_secs_f64() * 1e3;
            if let Ok(mut v) = prof.lock() {
                v.push(timing);
            }
        }
        changed
    }

    /// Skip bounded 2D layers whose complete draw footprint is covered by later opaque solids.
    /// The input is in bottom-to-top order; coverage is accumulated from top to bottom.
    pub fn occluded_bounded_layers(&self, ctx: &EvalCtx, layers: &[&Layer], width: u32, height: u32) -> Vec<bool> {
        let mut culled = vec![false; layers.len()];
        if layers.len() < 64 || width == 0 || height == 0 {
            return culled;
        }
        let cols = (width as usize).div_ceil(OCCLUSION_TILE);
        let rows = (height as usize).div_ceil(OCCLUSION_TILE);
        let Some(cells) = cols.checked_mul(rows).filter(|&n| n <= 1 << 20) else { return culled };
        let mut covered = vec![false; cells];
        let mut uncovered = cells;

        for (index, layer) in layers.iter().enumerate().rev() {
            // Later opaque layers cover every output tile. No earlier 2D layer can then change
            // the final frame, regardless of its source type, effects, matte or blend mode.
            if uncovered == 0 {
                if !layer.is_3d() {
                    culled[index] = true;
                }
                continue;
            }
            let Some(bounds) = self.simple_layer_bounds(ctx, layer, width, height) else { continue };
            if bounds.draw[0] >= bounds.draw[2] || bounds.draw[1] >= bounds.draw[3] {
                culled[index] = true;
                continue;
            }
            let tx0 = bounds.draw[0] / OCCLUSION_TILE;
            let tx1 = bounds.draw[2].div_ceil(OCCLUSION_TILE).min(cols);
            let ty0 = bounds.draw[1] / OCCLUSION_TILE;
            let ty1 = bounds.draw[3].div_ceil(OCCLUSION_TILE).min(rows);
            if (ty0..ty1).all(|y| (tx0..tx1).all(|x| covered[y * cols + x])) {
                culled[index] = true;
                continue;
            }

            if Self::can_occlude_layer(ctx, layer, self) {
                let tile = OCCLUSION_TILE as f64;
                let ox0 = ((bounds.opaque[0].max(0.0) / tile).ceil() as usize).min(cols);
                let ox1 = ((bounds.opaque[2].min(width as f64) / tile).floor() as usize).min(cols);
                let oy0 = ((bounds.opaque[1].max(0.0) / tile).ceil() as usize).min(rows);
                let oy1 = ((bounds.opaque[3].min(height as f64) / tile).floor() as usize).min(rows);
                for y in oy0..oy1 {
                    for x in ox0..ox1 {
                        let cell = y * cols + x;
                        if !covered[cell] {
                            covered[cell] = true;
                            uncovered -= 1;
                        }
                    }
                }
            }
        }
        culled
    }

    /// Conservative output-space bounds for a bounded source with no feature that can extend its
    /// pixels. Masks can only reduce the footprint; effects, styles and collapse can grow it.
    fn simple_layer_bounds(&self, ctx: &EvalCtx, layer: &Layer, width: u32, height: u32) -> Option<OpaqueBounds> {
        if !matches!(layer.source, LayerSource::Solid { .. } | LayerSource::Footage { .. } | LayerSource::Comp { .. })
            || layer.is_3d()
            || layer.switches.collapse
            || layer.switches.adjustment
            || self.quality(layer) == Quality::Wireframe
            || self.blurred(ctx, layer)
            || layer.effects().is_some_and(|g| g.groups().any(|e| e.enabled))
            || styles::active(ctx, layer)
        {
            return None;
        }

        let (source_width, source_height) = ctx.source_size(layer);
        if source_width == 0 || source_height == 0 {
            return None;
        }
        // Transparent pixels still matter to masks and blending, so use the complete source
        // rectangle rather than inspecting pixel alpha.
        let source_bounds = Rect::from_size(source_width as f64, source_height as f64);
        let (layer_to_comp, _) = ctx.layer_to_comp(layer);
        let layer_to_output = self.outer.map_or(layer_to_comp, |outer| outer * layer_to_comp);
        let matrix = self.out_matrix() * Mat3::scale(vec2(self.opts.scale, self.opts.scale)) * layer_to_output;
        if !matrix.is_affine() || ((matrix.0[0][1] != 0.0 || matrix.0[1][0] != 0.0) && Self::can_occlude_layer(ctx, layer, self)) {
            return None;
        }
        let bounds = matrix.map_rect(&source_bounds);
        let [x0, y0, x1, y1] = [bounds.x0, bounds.y0, bounds.x1, bounds.y1];
        if ![x0, y0, x1, y1].into_iter().all(f64::is_finite) {
            return None;
        }
        let to_px = |v: f64, max: u32| (v as i64).clamp(0, max as i64) as usize;
        let draw = [to_px((x0 - 1.0).floor(), width), to_px((y0 - 1.0).floor(), height), to_px((x1 + 1.0).ceil(), width), to_px((y1 + 1.0).ceil(), height)];
        Some(OpaqueBounds { draw, opaque: [x0 + 1.0, y0 + 1.0, x1 - 1.0, y1 - 1.0] })
    }

    /// Whether a bounded, non-expanding 2D layer cannot touch this output canvas.
    pub(crate) fn layer_outside_canvas(&self, ctx: &EvalCtx, layer: &Layer, width: u32, height: u32) -> bool {
        self.simple_layer_bounds(ctx, layer, width, height).is_some_and(|bounds| bounds.draw[0] >= bounds.draw[2] || bounds.draw[1] >= bounds.draw[3])
    }

    /// Simple normal solids and footage explicitly interpreted without alpha can add covered
    /// tiles for later culling. This only uses source metadata, so the culling pass doesn't decode
    /// media it would otherwise never need.
    fn can_occlude_layer(ctx: &EvalCtx, layer: &Layer, r: &Renderer<'_>) -> bool {
        let opaque_source = match &layer.source {
            LayerSource::Solid { item } => r.proxy_for(*item).is_none() && matches!(r.project.item(*item).map(|i| &i.kind), Some(ItemKind::Solid(_))),
            LayerSource::Footage { item } => {
                let item = ctx.media_item(layer, *item);
                r.proxy_for(item).is_none()
                    && r.project.item(item).is_some_and(
                        |i| matches!(&i.kind, ItemKind::Footage(f) if f.has_video && !f.missing && f.alpha == effectcraft_project::AlphaMode::Ignore && !f.invert_alpha),
                    )
            }
            _ => false,
        };
        opaque_source
            && !layer.is_3d()
            && !layer.switches.adjustment
            && !layer.switches.collapse
            && layer.parent.is_none()
            && layer.blend_mode == BlendMode::Normal
            && layer.track_matte.is_none()
            && !layer.preserve_transparency
            && !r.blurred(ctx, layer)
            && ctx.opacity(layer) as f32 * r.opacity_mul == 1.0
            && r.quality(layer) != Quality::Wireframe
            && !layer.masks().is_some_and(|m| !m.children.is_empty())
            && !layer.effects().is_some_and(|g| g.groups().any(|e| e.enabled))
            && !styles::active(ctx, layer)
    }

    /// Composite a styled layer: exterior passes with their own modes, then the body with the
    /// layer's mode; the layer's opacity fades the whole stack. Knockout clears what is below the
    /// layer's content; switched-off R/G/B channels keep the values below.
    fn composite_styled(&self, ctx: &EvalCtx, layer: &Layer, st: &StyledLayer, canvas: &mut Image, opacity: f32) {
        let bl = styles::blending(ctx, layer);
        let matte = layer.track_matte.is_some_and(|tm| tm.layer != layer.id && ctx.comp.layer(tm.layer).is_some());
        if matte || layer.preserve_transparency {
            let mut iso = Image::new(canvas.width, canvas.height);
            for (b, m) in &st.passes {
                self.place(ctx, layer, b, &mut iso, *m, 1.0);
            }
            self.place(ctx, layer, &st.body, &mut iso, BlendMode::Normal, 1.0);
            self.composite_iso(ctx, layer, iso, canvas, opacity);
            return;
        }
        let mut tmp = canvas.clone();
        if bl.knockout > 0 {
            let mut k = Image::new(canvas.width, canvas.height);
            self.place(ctx, layer, &st.content, &mut k, BlendMode::Normal, 1.0);
            tmp.data.par_iter_mut().zip(k.data.par_iter()).for_each(|(p, q)| {
                let keep = 1.0 - q[3].clamp(0.0, 1.0);
                for c in p.iter_mut() {
                    *c *= keep;
                }
            });
        }
        for (b, m) in &st.passes {
            self.place(ctx, layer, b, &mut tmp, *m, 1.0);
        }
        self.place(ctx, layer, &st.body, &mut tmp, layer.blend_mode, 1.0);
        let ch = bl.channels;
        canvas.data.par_iter_mut().zip(tmp.data.par_iter()).for_each(|(c, t)| {
            for i in 0..4 {
                let v = if i < 3 && !ch[i] { c[i] } else { t[i] };
                c[i] += (v - c[i]) * opacity;
            }
        });
    }

    /// Composite a plain layer; returns the canvas pixels it changed when they are known (then
    /// also quantised to `quantize` levels).
    fn composite_layer(&self, ctx: &EvalCtx, layer: &Layer, buf: &Buf, canvas: &mut Image, opacity: f32, quantize: Option<f32>) -> Option<[u32; 4]> {
        let matte = layer.track_matte.is_some_and(|tm| tm.layer != layer.id && ctx.comp.layer(tm.layer).is_some());
        if !matte && !layer.preserve_transparency {
            return self.place_changed(ctx, layer, buf, canvas, PlacementBlend { mode: layer.blend_mode, opacity, quantize });
        }
        // Render in isolation, then matte / preserve transparency, then blend.
        let mut iso = Image::new(canvas.width, canvas.height);
        self.place(ctx, layer, buf, &mut iso, BlendMode::Normal, 1.0);
        self.composite_iso(ctx, layer, iso, canvas, opacity);
        None
    }

    /// Apply the track matte / preserve transparency to an isolated layer render, then blend.
    pub(crate) fn composite_iso(&self, ctx: &EvalCtx, layer: &Layer, iso: Image, canvas: &mut Image, opacity: f32) {
        self.composite_iso_with(ctx, layer, iso, canvas, opacity, None);
    }

    /// [`Self::composite_iso`] with the track matte's pixels already drawn (`matte`, e.g. a 3D
    /// matte seen through the camera); otherwise the matte layer is placed in 2D.
    pub(crate) fn composite_iso_with(&self, ctx: &EvalCtx, layer: &Layer, mut iso: Image, canvas: &mut Image, opacity: f32, matte_img: Option<Image>) {
        self.apply_track_matte(ctx, layer, &mut iso, matte_img);
        if layer.preserve_transparency {
            iso.data.par_iter_mut().zip(canvas.data.par_iter()).for_each(|(p, q)| {
                for c in p.iter_mut() {
                    *c *= q[3];
                }
            });
        }
        canvas.blend_from(&iso, layer.blend_mode, opacity, layer.id.0 as u32);
    }

    /// Multiply `img` (comp-sized) by the layer's track matte, if it has one; `matte_img` as in
    /// [`Self::composite_iso_with`].
    fn apply_track_matte(&self, ctx: &EvalCtx, layer: &Layer, img: &mut Image, matte_img: Option<Image>) {
        let matte = layer.track_matte.and_then(|tm| ctx.comp.layer(tm.layer).filter(|m| m.id != layer.id).map(|m| (m, tm.kind)));
        if let Some((m, kind)) = matte {
            let mimg = match matte_img {
                Some(i) if i.width == img.width && i.height == img.height => i,
                _ => {
                    let mut mimg = Image::new(img.width, img.height);
                    if m.is_active_at(ctx.time)
                        && let Some(mb) = self.layer_buf(ctx, m)
                    {
                        let mo = ctx.opacity(m) as f32;
                        self.place(ctx, m, &mb, &mut mimg, BlendMode::Normal, mo);
                    }
                    mimg
                }
            };
            img.data.par_iter_mut().zip(mimg.data.par_iter()).for_each(|(p, q)| {
                let k = match kind {
                    MatteKind::Alpha => q[3],
                    MatteKind::AlphaInverted => 1.0 - q[3],
                    MatteKind::Luma => effectcraft_color::luminance(q[0], q[1], q[2]),
                    MatteKind::LumaInverted => 1.0 - effectcraft_color::luminance(q[0], q[1], q[2]),
                }
                .clamp(0.0, 1.0);
                for c in p.iter_mut() {
                    *c *= k;
                }
            });
        }
    }

    /// Adjustment layer: effects applied to the comp below, limited to the layer's footprint
    /// and its track matte, and blended back in the layer's blend mode.
    fn draw_adjustment(&self, ctx: &EvalCtx, layer: &Layer, canvas: &mut Image, opacity: f32) {
        let Some(mut foot) = self.source(ctx, layer) else { return };
        let _ = masks::apply(ctx, layer, &mut foot, self.mask_blur(ctx, layer));
        let mut matte = Image::new(canvas.width, canvas.height);
        self.place(ctx, layer, &foot, &mut matte, BlendMode::Normal, 1.0);
        self.apply_track_matte(ctx, layer, &mut matte, None);
        let o = self.roi_offset().map(|(x, y, _, _)| [-x, -y]).unwrap_or([0.0; 2]);
        let mut below = Buf { img: canvas.clone(), offset: o, scale: self.opts.scale };
        // Effects run in the working space, not the linear blending space.
        if let Some(c) = self.pipe.from_blend() {
            color::convert(&mut below.img, &c);
        }
        let mut adjusted = self.apply_effects(ctx, layer, below, true);
        // An effect that changed the frame (Motion Tile's Output size): back onto the canvas.
        if let Some((dx, dy)) =
            effectcraft_effects::adjustment_crop(adjusted.offset, o, (adjusted.img.width, adjusted.img.height), (canvas.width, canvas.height))
        {
            adjusted.img = adjusted.img.crop(dx, dy, canvas.width, canvas.height);
        }
        if let Some(c) = self.pipe.to_blend() {
            color::convert(&mut adjusted.img, &c);
        }
        if adjusted.img.width != canvas.width || adjusted.img.height != canvas.height {
            return;
        }
        if layer.blend_mode != BlendMode::Normal {
            // The adjusted comp is a layer in that mode over the comp below, as in After Effects:
            // an adjustment layer with no effects in Screen screens the comp with itself (#2618).
            let mut iso = adjusted.img;
            iso.data.par_iter_mut().zip(matte.data.par_iter()).for_each(|(p, m)| {
                for c in p.iter_mut() {
                    *c *= m[3];
                }
            });
            canvas.blend_from(&iso, layer.blend_mode, opacity, layer.id.0 as u32);
            return;
        }
        canvas.data.par_iter_mut().zip(adjusted.img.data.par_iter()).zip(matte.data.par_iter()).for_each(|((c, a), m)| {
            let k = m[3] * opacity;
            for i in 0..4 {
                c[i] += (a[i] - c[i]) * k;
            }
        });
    }
}

/// A layer ready to composite (see [`Renderer::styled_layer`]).
pub struct StyledLayer {
    /// Exterior style passes (Drop Shadow, Outer Glow) with their blend modes.
    pub passes: Vec<(Arc<Buf>, BlendMode)>,
    pub body: Arc<Buf>,
    /// Pre-style pixels (knockout shape).
    pub content: Arc<Buf>,
    /// Styled cache key (the content key when plain).
    pub(crate) key: Option<u64>,
    /// Content cache key.
    ckey: Option<u64>,
    /// No layer styles: composite `body` the usual way.
    pub plain: bool,
    /// The GPU will quantize these unquantized source pixels before sampling.
    pub(crate) quantize_source: bool,
    /// A uniform opaque source that the GPU samples without a full source texture.
    pub solid_color: Option<[f32; 4]>,
    /// The logical dimensions of `solid_color` in its final mip.
    pub solid_size: Option<(u32, u32)>,
}

impl StyledLayer {
    /// Whether the accelerator should quantize body texels before filtering and blending them.
    pub fn quantize_source(&self) -> bool {
        self.quantize_source
    }
}

/// Where a layer buffer lands (see [`Renderer::placement`]).
#[derive(Clone, Debug)]
pub struct Placement {
    /// Buffer pixel → output pixel; several = motion-blur sub-samples (equal weights).
    pub matrices: Vec<Mat3>,
    pub sampling: effectcraft_raster::Sampling,
    /// Dissolve noise seed.
    pub seed: u32,
}

/// The part of `footprint` a warp actually changed (`drawn`, see [`composite_warp_changed`]):
/// pixels it left alone are still quantised.
fn narrow(footprint: Region, drawn: Option<[u32; 4]>) -> Region {
    let Some([x0, y0, x1, y1]) = drawn else { return footprint };
    if x0 >= x1 || y0 >= y1 {
        return Region::Empty;
    }
    let r = color::PxRect { x0, y0, x1, y1 };
    match footprint {
        Region::Empty => Region::Empty,
        Region::Full => Region::Rect(r),
        Region::Rect(f) => {
            let q = color::PxRect { x0: r.x0.max(f.x0), y0: r.y0.max(f.y0), x1: r.x1.min(f.x1), y1: r.y1.min(f.y1) };
            if q.x0 >= q.x1 || q.y0 >= q.y1 { Region::Empty } else { Region::Rect(q) }
        }
    }
}
