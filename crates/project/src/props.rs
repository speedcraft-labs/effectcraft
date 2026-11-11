//! The property tree: groups and properties with stable uids and match ids, addressed by paths.

use effectcraft_keyframe::{Keyframe, Value};
use effectcraft_time::Tick;
use serde::de::{self, IntoDeserializer, MapAccess, Visitor, value::MapAccessDeserializer};
use serde::{Deserialize, Deserializer, Serialize};
use std::fmt;

pub type Uid = u64;

/// How a property is presented and edited.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(tag = "kind")]
pub enum ParamUi {
    /// Plain number with an optional valid range and a (narrower) slider range.
    #[default]
    Number,
    Slider {
        min: f64,
        max: f64,
        slider_min: f64,
        slider_max: f64,
        decimals: u8,
    },
    Percent,
    /// Angle in degrees, shown as `revolutions x +degrees`.
    Angle,
    Pixels,
    Point,
    Point3,
    Color,
    Checkbox,
    Popup {
        options: Vec<String>,
    },
    Layer,
    /// A mask of the effect's own layer by 1-based index, 0 = none (a scalar value): an effect's
    /// "Path" popup.
    Mask,
    Path,
    Text,
    /// A file path (a string value) chosen with a "Choose…" button, offering files with these
    /// extensions: Apply Color LUT's LUT, an OCIO config.
    File {
        filters: Vec<String>,
    },
    Gradient,
    /// Not shown in Effect Controls / the timeline (internal data).
    Hidden,
}

#[derive(Deserialize)]
#[serde(tag = "kind")]
enum BufferedParamUi {
    Number,
    Slider { min: f64, max: f64, slider_min: f64, slider_max: f64, decimals: u8 },
    Percent,
    Angle,
    Pixels,
    Point,
    Point3,
    Color,
    Checkbox,
    Popup { options: Vec<String> },
    Layer,
    Mask,
    Path,
    Text,
    File { filters: Vec<String> },
    Gradient,
    Hidden,
}

impl From<BufferedParamUi> for ParamUi {
    fn from(ui: BufferedParamUi) -> Self {
        match ui {
            BufferedParamUi::Number => ParamUi::Number,
            BufferedParamUi::Slider { min, max, slider_min, slider_max, decimals } => ParamUi::Slider { min, max, slider_min, slider_max, decimals },
            BufferedParamUi::Percent => ParamUi::Percent,
            BufferedParamUi::Angle => ParamUi::Angle,
            BufferedParamUi::Pixels => ParamUi::Pixels,
            BufferedParamUi::Point => ParamUi::Point,
            BufferedParamUi::Point3 => ParamUi::Point3,
            BufferedParamUi::Color => ParamUi::Color,
            BufferedParamUi::Checkbox => ParamUi::Checkbox,
            BufferedParamUi::Popup { options } => ParamUi::Popup { options },
            BufferedParamUi::Layer => ParamUi::Layer,
            BufferedParamUi::Mask => ParamUi::Mask,
            BufferedParamUi::Path => ParamUi::Path,
            BufferedParamUi::Text => ParamUi::Text,
            BufferedParamUi::File { filters } => ParamUi::File { filters },
            BufferedParamUi::Gradient => ParamUi::Gradient,
            BufferedParamUi::Hidden => ParamUi::Hidden,
        }
    }
}

#[derive(Clone, Copy, Deserialize)]
enum ParamUiTag {
    Number,
    Slider,
    Percent,
    Angle,
    Pixels,
    Point,
    Point3,
    Color,
    Checkbox,
    Popup,
    Layer,
    Mask,
    Path,
    Text,
    File,
    Gradient,
    Hidden,
}

#[derive(Deserialize)]
struct ParamUiSliderFields {
    min: f64,
    max: f64,
    slider_min: f64,
    slider_max: f64,
    decimals: u8,
}

#[derive(Deserialize)]
struct ParamUiPopupFields {
    options: Vec<String>,
}

#[derive(Deserialize)]
struct ParamUiFileFields {
    filters: Vec<String>,
}

enum TaggedKindField {
    Kind,
    Other(String),
}

impl<'de> Deserialize<'de> for TaggedKindField {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct FieldVisitor;

        impl<'de> Visitor<'de> for FieldVisitor {
            type Value = TaggedKindField;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a tagged property field")
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E> {
                Ok(if value == "kind" { TaggedKindField::Kind } else { TaggedKindField::Other(value.to_owned()) })
            }

            fn visit_borrowed_str<E>(self, value: &'de str) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                self.visit_str(value)
            }

            fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
                Ok(if value == "kind" { TaggedKindField::Kind } else { TaggedKindField::Other(value) })
            }
        }

        deserializer.deserialize_identifier(FieldVisitor)
    }
}

struct ParamUiVisitor;

impl<'de> Visitor<'de> for ParamUiVisitor {
    type Value = ParamUi;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a tagged property UI")
    }

    fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
    where
        M: MapAccess<'de>,
    {
        let Some(first) = map.next_key::<TaggedKindField>()? else {
            return Err(de::Error::custom("empty property UI"));
        };
        match first {
            TaggedKindField::Kind => {}
            TaggedKindField::Other(name) => {
                let mut fields = serde_json::Map::new();
                fields.insert(name, map.next_value()?);
                while let Some(key) = map.next_key::<String>()? {
                    let value = map.next_value()?;
                    if fields.insert(key.clone(), value).is_some() {
                        return Err(de::Error::custom(format!("duplicate field `{key}`")));
                    }
                }
                return BufferedParamUi::deserialize(serde_json::Value::Object(fields)).map(Into::into).map_err(de::Error::custom);
            }
        }

        match map.next_value::<ParamUiTag>()? {
            ParamUiTag::Number => skip_tagged_unit_fields(map).map(|()| ParamUi::Number),
            ParamUiTag::Percent => skip_tagged_unit_fields(map).map(|()| ParamUi::Percent),
            ParamUiTag::Angle => skip_tagged_unit_fields(map).map(|()| ParamUi::Angle),
            ParamUiTag::Pixels => skip_tagged_unit_fields(map).map(|()| ParamUi::Pixels),
            ParamUiTag::Point => skip_tagged_unit_fields(map).map(|()| ParamUi::Point),
            ParamUiTag::Point3 => skip_tagged_unit_fields(map).map(|()| ParamUi::Point3),
            ParamUiTag::Color => skip_tagged_unit_fields(map).map(|()| ParamUi::Color),
            ParamUiTag::Checkbox => skip_tagged_unit_fields(map).map(|()| ParamUi::Checkbox),
            ParamUiTag::Layer => skip_tagged_unit_fields(map).map(|()| ParamUi::Layer),
            ParamUiTag::Mask => skip_tagged_unit_fields(map).map(|()| ParamUi::Mask),
            ParamUiTag::Path => skip_tagged_unit_fields(map).map(|()| ParamUi::Path),
            ParamUiTag::Text => skip_tagged_unit_fields(map).map(|()| ParamUi::Text),
            ParamUiTag::Gradient => skip_tagged_unit_fields(map).map(|()| ParamUi::Gradient),
            ParamUiTag::Hidden => skip_tagged_unit_fields(map).map(|()| ParamUi::Hidden),
            ParamUiTag::Slider => ParamUiSliderFields::deserialize(MapAccessDeserializer::new(TaggedPayloadFields(map))).map(|fields| ParamUi::Slider {
                min: fields.min,
                max: fields.max,
                slider_min: fields.slider_min,
                slider_max: fields.slider_max,
                decimals: fields.decimals,
            }),
            ParamUiTag::Popup => {
                ParamUiPopupFields::deserialize(MapAccessDeserializer::new(TaggedPayloadFields(map))).map(|fields| ParamUi::Popup { options: fields.options })
            }
            ParamUiTag::File => {
                ParamUiFileFields::deserialize(MapAccessDeserializer::new(TaggedPayloadFields(map))).map(|fields| ParamUi::File { filters: fields.filters })
            }
        }
    }
}

fn skip_tagged_unit_fields<'de, M>(mut map: M) -> Result<(), M::Error>
where
    M: MapAccess<'de>,
{
    while let Some(key) = map.next_key::<String>()? {
        if key == "kind" {
            return Err(de::Error::duplicate_field("kind"));
        }
        let _: de::IgnoredAny = map.next_value()?;
    }
    Ok(())
}

struct TaggedPayloadFields<M>(M);

impl<'de, M> MapAccess<'de> for TaggedPayloadFields<M>
where
    M: MapAccess<'de>,
{
    type Error = M::Error;

    fn next_key_seed<K>(&mut self, seed: K) -> Result<Option<K::Value>, Self::Error>
    where
        K: serde::de::DeserializeSeed<'de>,
    {
        let Some(key) = self.0.next_key::<String>()? else { return Ok(None) };
        if key == "kind" {
            return Err(de::Error::duplicate_field("kind"));
        }
        seed.deserialize(key.into_deserializer()).map(Some)
    }

    fn next_value_seed<V>(&mut self, seed: V) -> Result<V::Value, Self::Error>
    where
        V: serde::de::DeserializeSeed<'de>,
    {
        self.0.next_value_seed(seed)
    }

    fn size_hint(&self) -> Option<usize> {
        self.0.size_hint()
    }
}

impl<'de> Deserialize<'de> for ParamUi {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(ParamUiVisitor)
    }
}

/// Expression attached to a property.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Expression {
    pub text: String,
    pub enabled: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Property {
    pub uid: Uid,
    /// Stable id within its parent kind (`position`, `blurriness`…).
    #[serde(rename = "match")]
    pub match_id: String,
    pub name: String,
    /// Static value (used when there are no keyframes).
    pub value: Value,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub keys: Vec<Keyframe>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expr: Option<Expression>,
    #[serde(default)]
    pub ui: ParamUi,
    /// Motion-path semantics (Position, Anchor Point, points).
    #[serde(default)]
    pub spatial: bool,
    /// Only hold keyframes (Source Text, checkboxes, popups).
    #[serde(default)]
    pub hold_only: bool,
    /// Valid only on 3D layers (Orientation, X/Y Rotation…).
    #[serde(default)]
    pub three_d_only: bool,
    /// Valid only on 2D layers.
    #[serde(default)]
    pub two_d_only: bool,
    /// Dimensions shown/edited (2 for 2D position etc.); 0 = all.
    #[serde(default)]
    pub shown_dims: u8,
    /// Can't be keyframed (header-like values, e.g. Mask Mode). Shown without a stopwatch.
    #[serde(default)]
    pub static_only: bool,
}

impl Property {
    pub fn new(uid: Uid, match_id: &str, name: &str, value: Value) -> Property {
        let hold_only = !value.interpolates();
        Property {
            uid,
            match_id: match_id.into(),
            name: name.into(),
            value,
            keys: vec![],
            expr: None,
            ui: ParamUi::Number,
            spatial: false,
            hold_only,
            three_d_only: false,
            two_d_only: false,
            shown_dims: 0,
            static_only: false,
        }
    }
    pub fn with_ui(mut self, ui: ParamUi) -> Property {
        self.ui = ui;
        self
    }
    pub fn spatial(mut self) -> Property {
        self.spatial = true;
        self
    }
    pub fn is_animated(&self) -> bool {
        !self.keys.is_empty()
    }
    pub fn has_expression(&self) -> bool {
        self.expr.as_ref().is_some_and(|e| e.enabled && !e.text.trim().is_empty())
    }
    /// Keyframed value at `t` (no expression).
    pub fn value_at(&self, t: Tick) -> Value {
        if self.keys.is_empty() {
            return self.value.clone();
        }
        // 3D Orientation turns along the shortest arc between its keys, as in After Effects.
        // Routed by `match_id`: "orientation" is only used by the Orientation property of 3D
        // layers, cameras and lights (`build.rs`), always a `Vec3` of degrees. A future property
        // must not reuse that id unless it is a rotation that should slerp the same way.
        if self.match_id == "orientation" && matches!(self.value, Value::Vec3(_)) {
            return effectcraft_keyframe::evaluate_orientation(&self.keys, t).unwrap_or_else(|| self.value.clone());
        }
        effectcraft_keyframe::evaluate(&self.keys, t, self.spatial).unwrap_or_else(|| self.value.clone())
    }
    /// Set the value at `t`: adds/replaces a key when animated, else sets the static value.
    pub fn set_value_at(&mut self, t: Tick, v: Value) {
        if self.keys.is_empty() {
            self.value = v;
        } else {
            let mut k = Keyframe::new(t, v);
            if self.hold_only {
                k = k.hold();
            }
            effectcraft_keyframe::set_key(&mut self.keys, k);
        }
    }
    /// Toggle the stopwatch: on → one key at `t` with the current value; off → keep value at `t`.
    pub fn set_animated(&mut self, on: bool, t: Tick) {
        if on && self.keys.is_empty() {
            let mut k = Keyframe::new(t, self.value.clone());
            if self.hold_only {
                k = k.hold();
            }
            self.keys.push(k);
        } else if !on && !self.keys.is_empty() {
            self.value = self.value_at(t);
            self.keys.clear();
        }
    }
}

/// What a group represents, with its non-animatable header data.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(tag = "kind")]
pub enum GroupKind {
    #[default]
    Plain,
    /// An instance in an indexed list (masks, effects, shape contents, animators): these can be
    /// renamed, reordered, duplicated and deleted.
    Indexed,
    Mask {
        mode: MaskMode,
        inverted: bool,
        color: [u8; 3],
        locked: bool,
        /// Layer ▸ Mask ▸ Motion Blur.
        #[serde(default)]
        motion_blur: MaskMotionBlur,
        /// Layer ▸ Mask ▸ Feather Falloff.
        #[serde(default)]
        feather_falloff: FeatherFalloff,
        /// Layer ▸ Mask and Shape Path ▸ RotoBezier: tangents follow the vertices automatically.
        #[serde(default)]
        roto_bezier: bool,
    },
    Effect {
        /// Effect spec id, e.g. `ec.blur.gaussian`.
        effect: String,
    },
    /// A motion tracker (Motion Trackers ▸ Tracker n) with its Tracker panel settings.
    Tracker { settings: Box<crate::tracking::TrackerSettings> },
    /// The Essential Properties of a precomp layer (master properties): one child per control
    /// of the nested comp's Essential Graphics; `overridden` lists the children (uids) whose
    /// values this instance overrides.
    Essential {
        #[serde(default)]
        overridden: Vec<Uid>,
    },
}

#[derive(Deserialize)]
#[serde(tag = "kind")]
enum BufferedGroupKind {
    Plain,
    Indexed,
    Mask {
        mode: MaskMode,
        inverted: bool,
        color: [u8; 3],
        locked: bool,
        #[serde(default)]
        motion_blur: MaskMotionBlur,
        #[serde(default)]
        feather_falloff: FeatherFalloff,
        #[serde(default)]
        roto_bezier: bool,
    },
    Effect {
        effect: String,
    },
    Tracker {
        settings: Box<crate::tracking::TrackerSettings>,
    },
    Essential {
        #[serde(default)]
        overridden: Vec<Uid>,
    },
}

impl From<BufferedGroupKind> for GroupKind {
    fn from(kind: BufferedGroupKind) -> Self {
        match kind {
            BufferedGroupKind::Plain => GroupKind::Plain,
            BufferedGroupKind::Indexed => GroupKind::Indexed,
            BufferedGroupKind::Mask { mode, inverted, color, locked, motion_blur, feather_falloff, roto_bezier } => {
                GroupKind::Mask { mode, inverted, color, locked, motion_blur, feather_falloff, roto_bezier }
            }
            BufferedGroupKind::Effect { effect } => GroupKind::Effect { effect },
            BufferedGroupKind::Tracker { settings } => GroupKind::Tracker { settings },
            BufferedGroupKind::Essential { overridden } => GroupKind::Essential { overridden },
        }
    }
}

#[derive(Clone, Copy, Deserialize)]
enum GroupKindTag {
    Plain,
    Indexed,
    Mask,
    Effect,
    Tracker,
    Essential,
}

#[derive(Deserialize)]
struct MaskGroupKindFields {
    mode: MaskMode,
    inverted: bool,
    color: [u8; 3],
    locked: bool,
    #[serde(default)]
    motion_blur: MaskMotionBlur,
    #[serde(default)]
    feather_falloff: FeatherFalloff,
    #[serde(default)]
    roto_bezier: bool,
}

#[derive(Deserialize)]
struct EffectGroupKindFields {
    effect: String,
}

#[derive(Deserialize)]
struct TrackerGroupKindFields {
    settings: Box<crate::tracking::TrackerSettings>,
}

#[derive(Deserialize)]
struct EssentialGroupKindFields {
    #[serde(default)]
    overridden: Vec<Uid>,
}

struct GroupKindVisitor;

impl<'de> Visitor<'de> for GroupKindVisitor {
    type Value = GroupKind;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a tagged property-group kind")
    }

    fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
    where
        M: MapAccess<'de>,
    {
        let Some(first) = map.next_key::<TaggedKindField>()? else {
            return Err(de::Error::custom("empty property-group kind"));
        };
        match first {
            TaggedKindField::Kind => {}
            TaggedKindField::Other(name) => {
                let mut fields = serde_json::Map::new();
                fields.insert(name, map.next_value()?);
                while let Some(key) = map.next_key::<String>()? {
                    let value = map.next_value()?;
                    if fields.insert(key.clone(), value).is_some() {
                        return Err(de::Error::custom(format!("duplicate field `{key}`")));
                    }
                }
                return BufferedGroupKind::deserialize(serde_json::Value::Object(fields)).map(Into::into).map_err(de::Error::custom);
            }
        }

        match map.next_value::<GroupKindTag>()? {
            GroupKindTag::Plain => skip_tagged_unit_fields(map).map(|()| GroupKind::Plain),
            GroupKindTag::Indexed => skip_tagged_unit_fields(map).map(|()| GroupKind::Indexed),
            GroupKindTag::Mask => {
                let fields = MaskGroupKindFields::deserialize(MapAccessDeserializer::new(TaggedPayloadFields(map)))?;
                Ok(GroupKind::Mask {
                    mode: fields.mode,
                    inverted: fields.inverted,
                    color: fields.color,
                    locked: fields.locked,
                    motion_blur: fields.motion_blur,
                    feather_falloff: fields.feather_falloff,
                    roto_bezier: fields.roto_bezier,
                })
            }
            GroupKindTag::Effect => EffectGroupKindFields::deserialize(MapAccessDeserializer::new(TaggedPayloadFields(map)))
                .map(|fields| GroupKind::Effect { effect: fields.effect }),
            GroupKindTag::Tracker => TrackerGroupKindFields::deserialize(MapAccessDeserializer::new(TaggedPayloadFields(map)))
                .map(|fields| GroupKind::Tracker { settings: fields.settings }),
            GroupKindTag::Essential => EssentialGroupKindFields::deserialize(MapAccessDeserializer::new(TaggedPayloadFields(map)))
                .map(|fields| GroupKind::Essential { overridden: fields.overridden }),
        }
    }
}

impl<'de> Deserialize<'de> for GroupKind {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(GroupKindVisitor)
    }
}

/// Mask motion blur (sub-frame samples of an animated mask path).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MaskMotionBlur {
    /// Blur when the layer's Motion Blur switch is on.
    #[default]
    SameAsLayer,
    On,
    Off,
}

/// How a mask's feather ramps: Smooth (three box passes, an S-shaped ramp of finite reach, as
/// measured in After Effects) or Linear (one box pass: a straight ramp).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FeatherFalloff {
    #[default]
    Smooth,
    Linear,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MaskMode {
    None,
    #[default]
    Add,
    Subtract,
    Intersect,
    Lighten,
    Darken,
    Difference,
}

impl MaskMode {
    pub const ALL: [MaskMode; 7] =
        [MaskMode::None, MaskMode::Add, MaskMode::Subtract, MaskMode::Intersect, MaskMode::Lighten, MaskMode::Darken, MaskMode::Difference];
    pub fn label(self) -> &'static str {
        match self {
            MaskMode::None => "None",
            MaskMode::Add => "Add",
            MaskMode::Subtract => "Subtract",
            MaskMode::Intersect => "Intersect",
            MaskMode::Lighten => "Lighten",
            MaskMode::Darken => "Darken",
            MaskMode::Difference => "Difference",
        }
    }
    pub fn from_name(s: &str) -> Option<MaskMode> {
        MaskMode::ALL.into_iter().find(|m| m.label().eq_ignore_ascii_case(s))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PropGroup {
    pub uid: Uid,
    #[serde(rename = "match")]
    pub match_id: String,
    pub name: String,
    #[serde(default)]
    pub kind: GroupKind,
    /// fx switch for effects, eye for shape groups, etc.
    #[serde(default = "yes")]
    pub enabled: bool,
    /// An effect's colour label (After Effects 26.5), shown in Effect Controls and the Timeline;
    /// `None` is unlabelled. Absent in older projects.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<effectcraft_color::Label>,
    #[serde(default)]
    pub children: Vec<Node>,
}

fn yes() -> bool {
    true
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "node")]
pub enum Node {
    Prop(Property),
    Group(PropGroup),
}

/// The derive for an internally tagged enum buffers every node into `Content` before it can
/// choose a variant. Project files written by EffectCraft put `node` first, so deserialize the
/// selected struct directly from the remaining map. Keep a buffered fallback for valid files
/// whose object keys are ordered differently.
#[derive(Deserialize)]
#[serde(tag = "node")]
enum BufferedNode {
    Prop(Property),
    Group(PropGroup),
}

struct NodeVisitor;

impl<'de> Visitor<'de> for NodeVisitor {
    type Value = Node;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a tagged property or group node")
    }

    fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<Node, M::Error> {
        let Some(first) = map.next_key::<String>()? else {
            return Err(de::Error::custom("empty project node"));
        };
        if first == "node" {
            let tag = map.next_value::<String>()?;
            return match tag.as_str() {
                "Prop" => Property::deserialize(MapAccessDeserializer::new(map)).map(Node::Prop),
                "Group" => PropGroup::deserialize(MapAccessDeserializer::new(map)).map(Node::Group),
                _ => Err(de::Error::unknown_variant(&tag, &["Prop", "Group"])),
            };
        }

        let mut fields = serde_json::Map::new();
        fields.insert(first, map.next_value()?);
        while let Some(key) = map.next_key::<String>()? {
            fields.insert(key, map.next_value()?);
        }
        match BufferedNode::deserialize(serde_json::Value::Object(fields)).map_err(de::Error::custom)? {
            BufferedNode::Prop(p) => Ok(Node::Prop(p)),
            BufferedNode::Group(g) => Ok(Node::Group(g)),
        }
    }
}

impl<'de> Deserialize<'de> for Node {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Node, D::Error> {
        deserializer.deserialize_map(NodeVisitor)
    }
}

impl Node {
    pub fn uid(&self) -> Uid {
        match self {
            Node::Prop(p) => p.uid,
            Node::Group(g) => g.uid,
        }
    }
    pub fn match_id(&self) -> &str {
        match self {
            Node::Prop(p) => &p.match_id,
            Node::Group(g) => &g.match_id,
        }
    }
    pub fn name(&self) -> &str {
        match self {
            Node::Prop(p) => &p.name,
            Node::Group(g) => &g.name,
        }
    }
    pub fn as_prop(&self) -> Option<&Property> {
        if let Node::Prop(p) = self { Some(p) } else { None }
    }
    pub fn as_group(&self) -> Option<&PropGroup> {
        if let Node::Group(g) = self { Some(g) } else { None }
    }
    pub fn as_group_mut(&mut self) -> Option<&mut PropGroup> {
        if let Node::Group(g) = self { Some(g) } else { None }
    }
}

/// One step of a property path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Seg {
    Uid(Uid),
    /// Match id and 1-based occurrence among siblings with that match id.
    Match(String, usize),
    /// 1-based child index.
    Index(usize),
}

/// Parse `transform/position`, `effects/#1/blurriness`, `effects/@57/blurriness`,
/// `contents/rect#2/size`.
/// `@57` (a whole path that is one uid).
fn bare_uid(path: &str) -> Option<Uid> {
    path.trim().strip_prefix('@')?.parse().ok()
}

pub fn parse_path(s: &str) -> Vec<Seg> {
    s.split(['/', '.'])
        .filter(|x| !x.is_empty())
        .map(|seg| {
            if let Some(u) = seg.strip_prefix('@').and_then(|u| u.parse().ok()) {
                Seg::Uid(u)
            } else if let Some(i) = seg.strip_prefix('#').and_then(|u| u.parse().ok()) {
                Seg::Index(i)
            } else if let Some((m, n)) = seg.rsplit_once('#')
                && let Ok(n) = n.parse()
            {
                Seg::Match(m.to_string(), n)
            } else {
                Seg::Match(seg.to_string(), 1)
            }
        })
        .collect()
}

impl PropGroup {
    /// An instance in an indexed list (effect, mask, shape item, text animator, tracker): these
    /// can be renamed, removed, duplicated and reordered; fixed groups (Transform…) can't.
    pub fn is_instance(&self) -> bool {
        matches!(self.kind, GroupKind::Indexed | GroupKind::Effect { .. } | GroupKind::Mask { .. } | GroupKind::Tracker { .. })
    }

    pub fn new(uid: Uid, match_id: &str, name: &str) -> PropGroup {
        PropGroup { uid, match_id: match_id.into(), name: name.into(), kind: GroupKind::Plain, enabled: true, label: None, children: vec![] }
    }
    pub fn with(mut self, n: impl Into<Node>) -> PropGroup {
        self.children.push(n.into());
        self
    }

    fn child_index(&self, seg: &Seg) -> Option<usize> {
        match seg {
            Seg::Uid(u) => self.children.iter().position(|c| c.uid() == *u),
            Seg::Index(i) => (*i >= 1 && *i <= self.children.len()).then(|| i - 1),
            Seg::Match(m, n) => {
                self.children.iter().enumerate().filter(|(_, c)| c.match_id() == m || c.name().eq_ignore_ascii_case(m)).nth(n.saturating_sub(1)).map(|(i, _)| i)
            }
        }
    }

    pub fn node(&self, path: &str) -> Option<&Node> {
        let segs = parse_path(path);
        let (last, init) = segs.split_last()?;
        let mut g = self;
        for s in init {
            g = g.children.get(g.child_index(s)?)?.as_group()?;
        }
        g.children.get(g.child_index(last)?)
    }
    pub fn node_mut(&mut self, path: &str) -> Option<&mut Node> {
        let segs = parse_path(path);
        let (last, init) = segs.split_last()?;
        let mut g = self;
        for s in init {
            let i = g.child_index(s)?;
            g = g.children.get_mut(i)?.as_group_mut()?;
        }
        let i = g.child_index(last)?;
        g.children.get_mut(i)
    }
    /// A property by path; a bare `@uid` finds the property anywhere below this group.
    pub fn prop(&self, path: &str) -> Option<&Property> {
        if let Some(u) = bare_uid(path) {
            return self.find(u);
        }
        self.node(path)?.as_prop()
    }
    pub fn prop_mut(&mut self, path: &str) -> Option<&mut Property> {
        if let Some(u) = bare_uid(path) {
            return self.find_mut(u);
        }
        match self.node_mut(path)? {
            Node::Prop(p) => Some(p),
            _ => None,
        }
    }
    pub fn group(&self, path: &str) -> Option<&PropGroup> {
        self.node(path)?.as_group()
    }
    pub fn group_mut(&mut self, path: &str) -> Option<&mut PropGroup> {
        self.node_mut(path)?.as_group_mut()
    }
    /// Direct child property by match id.
    pub fn get(&self, match_id: &str) -> Option<&Property> {
        self.children.iter().find_map(|c| c.as_prop().filter(|p| p.match_id == match_id))
    }
    pub fn get_mut(&mut self, match_id: &str) -> Option<&mut Property> {
        self.children.iter_mut().find_map(|c| match c {
            Node::Prop(p) if p.match_id == match_id => Some(p),
            _ => None,
        })
    }
    pub fn sub(&self, match_id: &str) -> Option<&PropGroup> {
        self.children.iter().find_map(|c| c.as_group().filter(|g| g.match_id == match_id))
    }
    pub fn sub_mut(&mut self, match_id: &str) -> Option<&mut PropGroup> {
        self.children.iter_mut().find_map(|c| match c {
            Node::Group(g) if g.match_id == match_id => Some(g),
            _ => None,
        })
    }
    pub fn groups(&self) -> impl Iterator<Item = &PropGroup> {
        self.children.iter().filter_map(Node::as_group)
    }
    pub fn props(&self) -> impl Iterator<Item = &Property> {
        self.children.iter().filter_map(Node::as_prop)
    }

    /// Visit every property depth-first with its uid-path (`@g/@g/@p`).
    pub fn walk<'a>(&'a self, prefix: &str, f: &mut dyn FnMut(&str, &'a Property)) {
        for c in &self.children {
            let p = format!("{prefix}@{}", c.uid());
            match c {
                Node::Prop(pr) => f(&p, pr),
                Node::Group(g) => g.walk(&format!("{p}/"), f),
            }
        }
    }
    pub fn walk_mut(&mut self, f: &mut dyn FnMut(&mut Property)) {
        for c in &mut self.children {
            match c {
                Node::Prop(p) => f(p),
                Node::Group(g) => g.walk_mut(f),
            }
        }
    }
    /// Uid path (`@1/@2/@3`) of the node with `uid`.
    pub fn path_of(&self, uid: Uid) -> Option<String> {
        for c in &self.children {
            if c.uid() == uid {
                return Some(format!("@{uid}"));
            }
            if let Node::Group(g) = c
                && let Some(p) = g.path_of(uid)
            {
                return Some(format!("@{}/{p}", g.uid));
            }
        }
        None
    }
    /// Match-id path (`transform/position`, `effects/ec.blur.gaussian#2/blurriness`) of the node
    /// with `uid`: portable between layers with the same structure (keyframe paste, pick-whip).
    pub fn match_path_of(&self, uid: Uid) -> Option<String> {
        for (i, c) in self.children.iter().enumerate() {
            let n = self.children[..i].iter().filter(|o| o.match_id() == c.match_id()).count() + 1;
            let seg = if n == 1 { c.match_id().to_string() } else { format!("{}#{n}", c.match_id()) };
            if c.uid() == uid {
                return Some(seg);
            }
            if let Node::Group(g) = c
                && let Some(p) = g.match_path_of(uid)
            {
                return Some(format!("{seg}/{p}"));
            }
        }
        None
    }
    /// Chain of nodes from this group's children down to `uid` (inclusive).
    pub fn node_chain(&self, uid: Uid) -> Option<Vec<&Node>> {
        for c in &self.children {
            if c.uid() == uid {
                return Some(vec![c]);
            }
            if let Node::Group(g) = c
                && let Some(mut rest) = g.node_chain(uid)
            {
                rest.insert(0, c);
                return Some(rest);
            }
        }
        None
    }
    /// Human path (`Transform/Position`) of the node with `uid`.
    pub fn name_path_of(&self, uid: Uid) -> Option<String> {
        for c in &self.children {
            if c.uid() == uid {
                return Some(c.name().to_string());
            }
            if let Node::Group(g) = c
                && let Some(p) = g.name_path_of(uid)
            {
                return Some(format!("{}/{p}", g.name));
            }
        }
        None
    }
    /// Find a property anywhere in the tree by uid.
    pub fn find(&self, uid: Uid) -> Option<&Property> {
        for c in &self.children {
            match c {
                Node::Prop(p) if p.uid == uid => return Some(p),
                Node::Group(g) => {
                    if let Some(p) = g.find(uid) {
                        return Some(p);
                    }
                }
                _ => {}
            }
        }
        None
    }
    pub fn find_mut(&mut self, uid: Uid) -> Option<&mut Property> {
        for c in &mut self.children {
            match c {
                Node::Prop(p) if p.uid == uid => return Some(p),
                Node::Group(g) => {
                    if let Some(p) = g.find_mut(uid) {
                        return Some(p);
                    }
                }
                _ => {}
            }
        }
        None
    }
    pub fn find_group(&self, uid: Uid) -> Option<&PropGroup> {
        if self.uid == uid {
            return Some(self);
        }
        self.groups().find_map(|g| g.find_group(uid))
    }
    pub fn find_group_mut(&mut self, uid: Uid) -> Option<&mut PropGroup> {
        if self.uid == uid {
            return Some(self);
        }
        self.children.iter_mut().find_map(|c| match c {
            Node::Group(g) => g.find_group_mut(uid),
            _ => None,
        })
    }
    /// Parent group of the node with `uid`.
    pub fn parent_of(&self, uid: Uid) -> Option<&PropGroup> {
        if self.children.iter().any(|c| c.uid() == uid) {
            return Some(self);
        }
        self.groups().find_map(|g| g.parent_of(uid))
    }
    /// Parent group of the node with `uid`.
    pub fn parent_of_mut(&mut self, uid: Uid) -> Option<&mut PropGroup> {
        if self.children.iter().any(|c| c.uid() == uid) {
            return Some(self);
        }
        self.children.iter_mut().find_map(|c| match c {
            Node::Group(g) => g.parent_of_mut(uid),
            _ => None,
        })
    }
    /// Any property in this subtree animated (keys or expression).
    pub fn any_animated(&self) -> bool {
        self.children.iter().any(|c| match c {
            Node::Prop(p) => p.is_animated() || p.has_expression(),
            Node::Group(g) => g.any_animated(),
        })
    }
    /// Re-assign fresh uids (after duplicating a subtree).
    pub fn reassign_uids(&mut self, next: &mut Uid) {
        *next += 1;
        self.uid = *next;
        for c in &mut self.children {
            match c {
                Node::Prop(p) => {
                    *next += 1;
                    p.uid = *next;
                }
                Node::Group(g) => g.reassign_uids(next),
            }
        }
    }
    pub fn max_uid(&self) -> Uid {
        self.children
            .iter()
            .map(|c| match c {
                Node::Prop(p) => p.uid,
                Node::Group(g) => g.max_uid(),
            })
            .fold(self.uid, Uid::max)
    }
}

impl From<Property> for Node {
    fn from(p: Property) -> Node {
        Node::Prop(p)
    }
}
impl From<PropGroup> for Node {
    fn from(g: PropGroup) -> Node {
        Node::Group(g)
    }
}

#[cfg(test)]
mod node_deserialization_tests {
    use super::Node;

    #[test]
    fn tagged_nodes_accept_tag_first_and_reordered_fields() {
        let tag_first = r#"{"node":"Group","uid":2,"match":"root","name":"Root","children":[{"node":"Prop","uid":1,"match":"value","name":"Value","value":{"t":"Scalar","v":2.0}}]}"#;
        let reordered = r#"{"uid":2,"match":"root","name":"Root","children":[{"uid":1,"match":"value","name":"Value","value":{"t":"Scalar","v":2.0},"node":"Prop"}],"node":"Group"}"#;
        let direct: Node = serde_json::from_str(tag_first).unwrap();
        let fallback: Node = serde_json::from_str(reordered).unwrap();
        assert_eq!(direct, fallback);
    }
}

#[cfg(test)]
mod group_kind_deserialization_tests {
    use super::{FeatherFalloff, GroupKind, MaskMode, MaskMotionBlur};

    #[test]
    fn tagged_group_kinds_accept_tag_first_and_reordered_fields() {
        let samples = [
            GroupKind::Plain,
            GroupKind::Indexed,
            GroupKind::Mask {
                mode: MaskMode::Subtract,
                inverted: true,
                color: [7, 19, 31],
                locked: true,
                motion_blur: MaskMotionBlur::On,
                feather_falloff: FeatherFalloff::Linear,
                roto_bezier: true,
            },
            GroupKind::Effect { effect: "ec.blur.gaussian".into() },
            GroupKind::Tracker { settings: Box::default() },
            GroupKind::Essential { overridden: vec![4, 8, 15] },
        ];
        for sample in &samples {
            let encoded = serde_json::to_string(sample).unwrap();
            let parsed: GroupKind = serde_json::from_str(&encoded).unwrap();
            assert_eq!(&parsed, sample, "tag-first encoding: {encoded}");
        }

        let reordered = r#"{"mode":"Subtract","inverted":true,"color":[7,19,31],"locked":true,"kind":"Mask","motion_blur":"On","feather_falloff":"Linear","roto_bezier":true}"#;
        let parsed: GroupKind = serde_json::from_str(reordered).unwrap();
        assert_eq!(parsed, samples[2]);
        assert!(serde_json::from_str::<GroupKind>(r#"{"kind":"Plain","kind":"Indexed"}"#).is_err());
        assert!(serde_json::from_str::<GroupKind>(r#"{"kind":"Effect","effect":"x","kind":"Plain"}"#).is_err());
    }
}

#[cfg(test)]
mod param_ui_deserialization_tests {
    use super::ParamUi;

    #[test]
    fn tagged_property_uis_accept_tag_first_and_reordered_fields() {
        let samples = [
            ParamUi::Number,
            ParamUi::Slider { min: -3.0, max: 9.0, slider_min: 0.0, slider_max: 6.0, decimals: 2 },
            ParamUi::Percent,
            ParamUi::Angle,
            ParamUi::Pixels,
            ParamUi::Point,
            ParamUi::Point3,
            ParamUi::Color,
            ParamUi::Checkbox,
            ParamUi::Popup { options: vec!["One".into(), "Two".into()] },
            ParamUi::Layer,
            ParamUi::Mask,
            ParamUi::Path,
            ParamUi::Text,
            ParamUi::File { filters: vec!["*.cube".into()] },
            ParamUi::Gradient,
            ParamUi::Hidden,
        ];
        for sample in &samples {
            let encoded = serde_json::to_string(sample).unwrap();
            let parsed: ParamUi = serde_json::from_str(&encoded).unwrap();
            assert_eq!(&parsed, sample, "tag-first encoding: {encoded}");
        }

        let reordered = r#"{"min":-3.0,"max":9.0,"slider_min":0.0,"slider_max":6.0,"decimals":2,"kind":"Slider"}"#;
        let parsed: ParamUi = serde_json::from_str(reordered).unwrap();
        assert_eq!(parsed, samples[1]);
        assert!(serde_json::from_str::<ParamUi>(r#"{"kind":"Number","kind":"Percent"}"#).is_err());
        assert!(serde_json::from_str::<ParamUi>(r#"{"kind":"Popup","options":[],"kind":"Number"}"#).is_err());
    }
}
