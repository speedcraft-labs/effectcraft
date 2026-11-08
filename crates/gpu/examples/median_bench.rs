use std::sync::Arc;

use craft_bench::report::Revs;
use craft_bench::{Config, Paired};
use effectcraft_color::Label;
use effectcraft_gpu::Gpu;
use effectcraft_keyframe::Value;
use effectcraft_project::build::{self, Ids};
use effectcraft_project::{AlphaMode, Comp, Footage, FootageKind, ItemKind, LayerSource, Project};
use effectcraft_raster::Image;
use effectcraft_render::{Backend, FootageSource, RenderOpts, Renderer};
use effectcraft_time::{FrameRate, Tick};

struct Input {
    project: Project,
    comp: effectcraft_project::ItemId,
    source: Source,
}

struct Source(Arc<Image>);

impl FootageSource for Source {
    fn frame(&self, _item: effectcraft_project::ItemId, _footage: &Footage, _time: Tick) -> Option<Arc<Image>> {
        Some(Arc::clone(&self.0))
    }
}

fn input(width: u32, height: u32, radius: f64) -> Result<Input, Box<dyn std::error::Error>> {
    let mut pixels = Image::new(width, height);
    let mut state = 0x6d2b_79f5u32;
    for y in 0..height {
        for x in 0..width {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let c = (state & 0x00ff_ffff) as f32 / 16_777_215.0;
            pixels.set(x, y, [c, 1.0 - c, (c * 7.0).fract(), 1.0]);
        }
    }

    let comp = Comp::new(width, height, FrameRate::FPS_30, Tick::from_seconds_f64(1.0));
    let mut project = Project::default();
    project.settings.bit_depth = effectcraft_project::BitDepth::Bpc32;
    let comp_id = project.add_item("Median bench", Label::Lavender, None, ItemKind::Comp(comp.clone().into()));
    let footage = Footage {
        path: "median-bench.png".into(),
        kind: FootageKind::Still,
        width,
        height,
        pixel_aspect: 1.0,
        frame_rate: FrameRate::FPS_30,
        has_video: true,
        alpha: AlphaMode::Premultiplied,
        ..Default::default()
    };
    let footage_id = project.add_item("Input", Label::Sandstone, None, ItemKind::Footage(footage));
    let mut layer = build::layer(&mut project, &comp, "Input", LayerSource::Footage { item: footage_id }, (width, height), None);
    let spec = effectcraft_effects::find("ec.noise.median").ok_or("Median effect is not registered")?;
    let mut next = project.next_id;
    let mut effect = effectcraft_effects::instantiate(spec, &mut Ids(&mut next), spec.name, [width as f64, height as f64]);
    project.next_id = next;
    effect.prop_mut("radius").ok_or("Median has no radius parameter")?.value = Value::Scalar(radius);
    layer.props.sub_mut("effects").ok_or("layer has no effects group")?.children.push(effect.into());
    project.comp_mut(comp_id).ok_or("project has no benchmark comp")?.layers.insert(0, layer);
    Ok(Input { project, comp: comp_id, source: Source(Arc::new(pixels)) })
}

fn cpu(input: &Input) -> Result<Image, String> {
    let renderer = Renderer::new(&input.project, &input.source, RenderOpts { backend: Backend::Cpu, motion_blur: false, ..Default::default() });
    Ok(renderer.comp_frame_cpu(input.comp, Tick::ZERO))
}

fn render_gpu(gpu: &Gpu, input: &Input) -> Result<Image, String> {
    let mut renderer = Renderer::new(&input.project, &input.source, RenderOpts { backend: Backend::Gpu, motion_blur: false, ..Default::default() });
    renderer.accel = Some(gpu);
    gpu.render(&renderer, input.comp, Tick::ZERO).ok_or_else(|| "GPU median render was declined".to_string())
}

fn close(a: &Result<Image, String>, b: &Result<Image, String>) -> bool {
    let (Ok(a), Ok(b)) = (a, b) else { return false };
    a.width == b.width && a.height == b.height && a.data.iter().zip(&b.data).all(|(x, y)| (0..4).all(|c| (x[c] - y[c]).abs() <= 0.001))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Gpu::try_headless().map_err(|error| format!("headless wgpu adapter: {error}"))?;
    let adapter = effectcraft_render::Accelerator::name(&device);
    if !adapter.contains("2080 Ti") {
        return Err(format!("expected the RTX 2080 Ti, selected {adapter}").into());
    }
    eprintln!("adapter: {adapter}");
    let inputs = [
        ("320×240 radius 2 (bisection path)", input(320, 240, 2.0)?),
        ("1280×720 radius 9 (sliding histogram)", input(1280, 720, 9.0)?),
        ("640×480 radius 24 (sliding histogram)", input(640, 480, 24.0)?),
    ];
    let config = Config {
        revs: Revs { old: Some("EffectCraft CPU median renderer".into()), new: Some("EffectCraft wgpu median renderer".into()) },
        ..Config::default()
    };
    let mut report = Paired::new("EffectCraft median render: CPU reference vs wgpu")
        .old(cpu)
        .new(|input| render_gpu(&device, input))
        .check(close)
        .inputs(inputs)
        .run(&config)?;
    report.gpu_adapters.push(adapter);
    let path = std::env::args().nth(1).unwrap_or_else(|| "effectcraft-median-gpu.json".into());
    std::fs::write(&path, report.to_json())?;
    println!("{}", report.to_markdown());
    report.require_publishable()?;
    Ok(())
}
