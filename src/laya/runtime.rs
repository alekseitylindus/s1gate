//! Candle weight loading and the Laya forward pass.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::num::NonZeroUsize;
use std::path::Path;

use candle_core::safetensors::Load as _;
use candle_core::{D, DType, Device, Tensor};
use candle_nn::ops;

use crate::error::{Error, Result};

use super::SpecialTokens;
use super::config::{AgentConfig, AttentionType, EncoderConfig};
use super::prompt::Prepared;

/// A Candle failure is an inference failure: the Checkpoint, the tensor library and this pass are
/// one execution path to the operator, who acts on the message either way. This lives here, with
/// the only module that talks to Candle, so the error type stays free of it.
impl From<candle_core::Error> for Error {
    fn from(error: candle_core::Error) -> Self {
        Self::Inference {
            message: error.to_string(),
        }
    }
}

/// The value an attention score, or a Marker slot no Option occupies, takes where attention or
/// selection is not allowed. It dominates any score the Checkpoint produces, so it softens and
/// normalizes to exactly zero.
const MASK_FILL: f32 = -1e4;
/// The `LayerNorm` epsilon the decision head was trained with.
const LN_EPS: f32 = 1e-5;

/// One Checkpoint's parameters, materialized as F32: the published weights are 16-bit, and every
/// operation in this module is F32.
#[derive(Debug)]
pub(super) struct Weights {
    values: HashMap<String, Tensor>,
}

impl Weights {
    #[inline]
    fn get(&self, name: &str) -> Result<&Tensor> {
        self.values.get(name).ok_or_else(|| Error::Inference {
            message: format!("missing materialized parameter `{name}`"),
        })
    }
}

/// One reusable buffer holding a layer's parameter names: naming the parameters of a layer costs no
/// allocation each, which matters because the encoder stack looks up dozens per layer.
struct Names {
    buffer: String,
    prefix: usize,
}

impl Names {
    fn new() -> Self {
        Self {
            buffer: String::with_capacity(64),
            prefix: 0,
        }
    }

    /// Name the layer `prefix` addresses, e.g. `encoder.layers.3` or `head.layers.0`.
    fn layer(&mut self, prefix: &str, layer: usize) {
        self.buffer.clear();
        write!(self.buffer, "{prefix}{layer}").expect("a String never fails");
        self.prefix = self.buffer.len();
    }

    /// The current layer's parameter `suffix`, e.g. `.attn.Wqkv.weight`.
    fn parameter<'a>(&mut self, weights: &'a Weights, suffix: &str) -> Result<&'a Tensor> {
        self.buffer.truncate(self.prefix);
        self.buffer.push_str(suffix);
        weights.get(&self.buffer)
    }
}

/// Materialize the parameters of `path`, a Checkpoint's published `model.safetensors`, converted to
/// F32. The Parameter Manifest is validated against the file's header before this runs, so every
/// tensor here is one the Manifest expects.
pub(super) fn load_weights(path: &Path) -> Result<Weights> {
    // SAFETY: `MmapedSafetensors` requires the mapped file to keep the bytes it was opened with.
    // The store is read-only to the runtime (ADR-0003), and `verify` has just re-hashed this file
    // against its Provenance, so it is the Checkpoint the Model Source published.
    let file =
        unsafe { candle_core::safetensors::MmapedSafetensors::new(path) }.map_err(|error| {
            Error::Inference {
                message: format!("loading safetensors `{}`: {error}", path.display()),
            }
        })?;
    let mut values = HashMap::new();
    for (name, view) in file.tensors() {
        let value = view
            .load(&Device::Cpu)
            .and_then(|value| value.to_dtype(DType::F32))
            .map_err(|error| Error::Inference {
                message: format!("converting parameter `{name}` to F32: {error}"),
            })?;
        values.insert(name, value);
    }
    Ok(Weights { values })
}

/// A projection of every leading position at once: the leading dimensions are folded into the
/// rows of a single matrix multiplication, which is what a rank-3 input means here.
#[inline]
fn linear(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    let dims = x.dims();
    let mut projected = dims[..dims.len() - 1].to_vec();
    projected.push(weight.dim(0)?);
    let y = x
        .flatten_to(D::Minus2)?
        .matmul(&weight.t()?)?
        .reshape(projected)?;
    match bias {
        Some(bias) => Ok(y.broadcast_add(bias)?),
        None => Ok(y),
    }
}

/// Rotary position embeddings, in the layout this Checkpoint's `default` `RoPE` uses: the head is
/// split in half and each half is rotated against the other at the frequency `base` sets.
fn rope(x: &Tensor, base: f32) -> Result<Tensor> {
    let (_, _, length, head_width) = x.dims4()?;
    let half = head_width / 2;
    // A half-index's frequency depends only on the index, so it is computed once per head width
    // rather than for every position of every attention call.
    let frequency: Vec<f32> = (0..half)
        .map(|index| base.powf(2.0 * index as f32 / head_width as f32))
        .collect();
    let mut cosine = Vec::with_capacity(length * half);
    let mut sine = Vec::with_capacity(length * half);
    for position in 0..length {
        for frequency in &frequency {
            let angle = position as f32 / frequency;
            cosine.push(angle.cos());
            sine.push(angle.sin());
        }
    }
    let cosine = Tensor::from_vec(cosine, (1, 1, length, half), x.device())?;
    let sine = Tensor::from_vec(sine, (1, 1, length, half), x.device())?;
    let first = x.narrow(D::Minus1, 0, half)?;
    let second = x.narrow(D::Minus1, half, half)?;
    Ok(Tensor::cat(
        &[
            &(first.broadcast_mul(&cosine)? - second.broadcast_mul(&sine)?)?,
            &(second.broadcast_mul(&cosine)? + first.broadcast_mul(&sine)?)?,
        ],
        D::Minus1,
    )?)
}

/// One attention layer's projection: the weight, and the bias it may carry. Both projections a
/// layer applies share this shape.
struct Projection<'a> {
    weight: &'a Tensor,
    bias: Option<&'a Tensor>,
}

/// One attention layer: scaled `QK^T`, the mask added, softmax, times V.
///
/// `rope_base` is the only difference between an encoder layer, which carries positional
/// information, and a decision-head layer, which does not. `mask` is additive, `MASK_FILL` where a
/// key is not allowed, and broadcast over the heads.
fn attention(
    x: &Tensor,
    qkv: Projection<'_>,
    out: Projection<'_>,
    heads: NonZeroUsize,
    rope_base: Option<f32>,
    mask: &Tensor,
) -> Result<Tensor> {
    let (batch, length, hidden) = x.dims3()?;
    let heads = heads.get();
    if !hidden.is_multiple_of(heads) {
        return Err(Error::Inference {
            message: "hidden size must be divisible by the attention head count".to_string(),
        });
    }
    let head_width = hidden / heads;
    let qkv = linear(x, qkv.weight, qkv.bias)?.reshape((batch, length, 3, heads, head_width))?;
    let project = |index: usize| -> Result<Tensor> {
        Ok(qkv
            .narrow(2, index, 1)?
            .squeeze(2)?
            .transpose(1, 2)?
            .contiguous()?)
    };
    let query = project(0)?;
    let key = project(1)?;
    let value = project(2)?;
    let (query, key) = match rope_base {
        Some(base) => (rope(&query, base)?, rope(&key, base)?),
        None => (query, key),
    };
    let scale = f64::from((head_width as f32).powf(-0.5));
    let scores = query
        .matmul(&key.t()?)?
        .affine(scale, 0.0)?
        .broadcast_add(mask)?;
    let probabilities = ops::softmax_last_dim(&scores)?;
    let attended = probabilities
        .matmul(&value)?
        .transpose(1, 2)?
        .contiguous()?
        .reshape((batch, length, hidden))?;
    linear(&attended, out.weight, out.bias)
}

/// The `ModernBERT` encoder: token embeddings, the encoder stack, and the final norm.
fn encoder(
    config: &EncoderConfig,
    weights: &Weights,
    input_ids: &Tensor,
    masks: &Masks,
    zero_bias: &Tensor,
    names: &mut Names,
    heads: NonZeroUsize,
) -> Result<Tensor> {
    let (batch, length) = input_ids.dims2()?;
    let embeddings = weights.get("encoder.embeddings.tok_embeddings.weight")?;
    let mut x = embeddings
        .index_select(&input_ids.flatten_all()?, 0)?
        .reshape((batch, length, config.hidden_size))?;
    x = ops::layer_norm(
        &x,
        weights.get("encoder.embeddings.norm.weight")?,
        zero_bias,
        config.norm_eps,
    )?;
    for layer in 0..config.num_hidden_layers {
        names.layer("encoder.layers.", layer);
        // Layer zero normalizes nothing: the Checkpoint has no attention norm for it.
        let normalized = if layer == 0 {
            x.clone()
        } else {
            ops::layer_norm(
                &x,
                names.parameter(weights, ".attn_norm.weight")?,
                zero_bias,
                config.norm_eps,
            )?
        };
        let kind = config.layer_type(layer)?;
        // Every key of a full-attention layer that carries a token is reachable; a sliding one
        // reaches only its window.
        let mask = if matches!(kind, AttentionType::Full) {
            &masks.keys
        } else {
            &masks.sliding
        };
        let attended = attention(
            &normalized,
            Projection {
                weight: names.parameter(weights, ".attn.Wqkv.weight")?,
                bias: None,
            },
            Projection {
                weight: names.parameter(weights, ".attn.Wo.weight")?,
                bias: None,
            },
            heads,
            Some(config.rope_base(kind)?),
            mask,
        )?;
        x = x.add(&attended)?;
        let normalized = ops::layer_norm(
            &x,
            names.parameter(weights, ".mlp_norm.weight")?,
            zero_bias,
            config.norm_eps,
        )?;
        let gated = linear(
            &normalized,
            names.parameter(weights, ".mlp.Wi.weight")?,
            None,
        )?;
        // GeGLU: the projection holds the activation input and its gate side by side.
        let width = config.intermediate_size;
        let activation = gated.narrow(D::Minus1, 0, width)?.gelu_erf()?;
        let gate = gated.narrow(D::Minus1, width, width)?;
        let activated = activation.mul(&gate)?;
        let projected = linear(
            &activated,
            names.parameter(weights, ".mlp.Wo.weight")?,
            None,
        )?;
        x = x.add(&projected)?;
    }
    Ok(ops::layer_norm(
        &x,
        weights.get("encoder.final_norm.weight")?,
        zero_bias,
        config.norm_eps,
    )?)
}

/// One decision-head layer: self attention without positional encoding, then a `ReLU` feed-forward.
fn head_layer(
    x: &Tensor,
    layer: usize,
    weights: &Weights,
    heads: NonZeroUsize,
    mask: &Tensor,
    names: &mut Names,
) -> Result<Tensor> {
    names.layer("head.layers.", layer);
    let normalized = ops::layer_norm(
        x,
        names.parameter(weights, ".norm1.weight")?,
        names.parameter(weights, ".norm1.bias")?,
        LN_EPS,
    )?;
    let attended = attention(
        &normalized,
        Projection {
            weight: names.parameter(weights, ".self_attn.in_proj_weight")?,
            bias: Some(names.parameter(weights, ".self_attn.in_proj_bias")?),
        },
        Projection {
            weight: names.parameter(weights, ".self_attn.out_proj.weight")?,
            bias: Some(names.parameter(weights, ".self_attn.out_proj.bias")?),
        },
        heads,
        None,
        mask,
    )?;
    let x = x.add(&attended)?;
    let normalized = ops::layer_norm(
        &x,
        names.parameter(weights, ".norm2.weight")?,
        names.parameter(weights, ".norm2.bias")?,
        LN_EPS,
    )?;
    let feed = linear(
        &normalized,
        names.parameter(weights, ".linear1.weight")?,
        Some(names.parameter(weights, ".linear1.bias")?),
    )?;
    let feed = feed.relu()?;
    let feed = linear(
        &feed,
        names.parameter(weights, ".linear2.weight")?,
        Some(names.parameter(weights, ".linear2.bias")?),
    )?;
    Ok(x.add(&feed)?)
}

/// The hidden state of every Question at each of its Marker slots: `indices` holds one flat
/// `row * length + position` per slot, so a Marker slot no Option occupies repeats its Question's
/// first position, which [`score_markers`] then fills in.
fn gather_markers(hidden: &Tensor, indices: &Tensor, marker_slots: usize) -> Result<Tensor> {
    let (batch, length, width) = hidden.dims3()?;
    let rows = batch.checked_mul(length).ok_or_else(|| Error::Inference {
        message: "input batch is too large".to_string(),
    })?;
    Ok(hidden
        .reshape((rows, width))?
        .index_select(indices, 0)?
        .reshape((batch, marker_slots, width))?)
}

/// The score of every Marker slot, with the slots no Option occupies filled in so they sort and
/// soften to nothing.
fn score_markers(states: &Tensor, mask: &Tensor, weights: &Weights) -> Result<Tensor> {
    let score = ops::layer_norm(
        states,
        weights.get("scorer.0.weight")?,
        weights.get("scorer.0.bias")?,
        LN_EPS,
    )?;
    let score = linear(
        &score,
        weights.get("scorer.1.weight")?,
        Some(weights.get("scorer.1.bias")?),
    )?
    .gelu_erf()?;
    let score = linear(
        &score,
        weights.get("scorer.3.weight")?,
        Some(weights.get("scorer.3.bias")?),
    )?
    .squeeze(2)?;
    Ok(mask.where_cond(
        &score,
        &Tensor::full(MASK_FILL, mask.shape(), mask.device())?,
    )?)
}

/// What every layer of one forward pass adds to its scores: `keys` marks the positions that carry
/// a token, which is all a full-attention layer, and the decision head, need; `sliding` narrows a
/// real query to the keys within its window. Both broadcast over the queries and the heads.
struct Masks {
    keys: Tensor,
    sliding: Tensor,
}

impl Masks {
    /// The masks for `batch` Questions padded to `length`, `valid` marking which positions carry a
    /// token, and `window` the sliding attention width.
    fn new(valid: &[bool], batch: usize, length: usize, window: usize) -> Result<Self> {
        let device = Device::Cpu;
        let cells = batch
            .checked_mul(length)
            .and_then(|cells| cells.checked_mul(length))
            .ok_or_else(|| Error::Inference {
                message: "attention mask is too large".to_string(),
            })?;
        let radius = window / 2;
        let mut sliding = vec![0f32; cells];
        // `chunks_exact_mut` needs a non-zero chunk: a batch of empty Questions masks nothing.
        if length > 0 {
            for (row, plane) in sliding.chunks_exact_mut(length * length).enumerate() {
                let valid_row = &valid[row * length..][..length];
                for (query, cell_row) in plane.chunks_exact_mut(length).enumerate() {
                    let valid_query = valid_row[query];
                    for (key, cell) in cell_row.iter_mut().enumerate() {
                        // A padded key is never attended to. A real query of a sliding layer
                        // reaches `radius` positions either side of itself; a padded query reaches
                        // every key its own mask allows, because its output is never read.
                        if !valid_row[key] || (valid_query && query.abs_diff(key) > radius) {
                            *cell = MASK_FILL;
                        }
                    }
                }
            }
        }
        Ok(Self {
            keys: Tensor::from_vec(
                valid
                    .iter()
                    .map(|valid| if *valid { 0f32 } else { MASK_FILL })
                    .collect::<Vec<_>>(),
                (batch, 1, 1, length),
                &device,
            )?,
            sliding: Tensor::from_vec(sliding, (batch, 1, length, length), &device)?,
        })
    }
}

/// The per-Question rows one forward pass returns: the Marker logits, and the Action weights
/// beside them, share this shape, one row per Question of the System One Call.
pub(super) type Scores = Vec<Vec<f32>>;

/// What one forward pass returns: the Marker logits of every Question, and its Action weights.
pub(super) struct Forward {
    pub(super) logits: Scores,
    pub(super) actions: Scores,
}

/// Judge every Question of one System One Call in a single forward pass.
///
/// # Errors
///
/// The padded batch is too large to address, a Question's Marker position or Question Type lies
/// outside the batch, the Checkpoint is missing a materialized parameter, or the pass produces a
/// value that is not finite.
pub(super) fn forward(
    config: &EncoderConfig,
    agent: &AgentConfig,
    weights: &Weights,
    prepared: &[Prepared],
    special: &SpecialTokens,
) -> Result<Forward> {
    let count = prepared.len();
    let length = prepared
        .iter()
        .map(|item| item.sequence.ids.len())
        .max()
        .unwrap_or(0);
    // Every Question's Markers are padded to the widest one's, at least two: a Marker slot no
    // Option fills carries no Option and is masked out below.
    let marker_slots = prepared
        .iter()
        .map(|item| item.sequence.markers.len())
        .max()
        .unwrap_or(0)
        .max(2);
    let cells = count.checked_mul(length).ok_or_else(|| Error::Inference {
        message: "input batch is too large".to_string(),
    })?;
    if u32::try_from(cells).is_err() {
        return Err(Error::Inference {
            message: "input batch is too large".to_string(),
        });
    }
    let slots = count
        .checked_mul(marker_slots)
        .ok_or_else(|| Error::Inference {
            message: "input batch is too large".to_string(),
        })?;
    let mut ids = vec![special.pad; cells];
    let mut valid = vec![false; cells];
    let mut marker_flat = vec![0u32; slots];
    let mut marker_mask = vec![0u8; slots];
    let mut kinds = Vec::with_capacity(count);
    for (row, item) in prepared.iter().enumerate() {
        let offset = row * length;
        let tokens = item.sequence.ids.len();
        ids[offset..offset + tokens].copy_from_slice(&item.sequence.ids);
        valid[offset..offset + tokens].fill(true);
        // The batch fits in u32, so `base + position` addresses a position of this Question.
        let base = u32::try_from(offset).map_err(|_| Error::Inference {
            message: "input batch is too large".to_string(),
        })?;
        for (option, marker) in item.sequence.markers.iter().enumerate() {
            let position = u32::try_from(*marker).map_err(|_| Error::Inference {
                message: "marker position exceeds the batch".to_string(),
            })?;
            marker_flat[row * marker_slots + option] = base + position;
            marker_mask[row * marker_slots + option] = 1;
        }
        kinds.push(
            u32::try_from(item.kind.index()).map_err(|_| Error::Inference {
                message: "question type is outside the type embedding".to_string(),
            })?,
        );
    }
    let device = Device::Cpu;
    let input_ids = Tensor::from_vec(ids, (count, length), &device)?;
    let marker_flat = Tensor::from_vec(marker_flat, (slots,), &device)?;
    let marker_mask = Tensor::from_vec(marker_mask, (count, marker_slots), &device)?;
    let kinds = Tensor::from_vec(kinds, (count,), &device)?;
    let masks = Masks::new(&valid, count, length, config.local_attention)?;
    let zero_bias = Tensor::zeros(config.hidden_size, DType::F32, &device)?;

    // Both head counts are non-zero by construction: `validate_config` rejects a Checkpoint whose
    // dimensions would make either one zero.
    let encoder_heads =
        NonZeroUsize::new(config.num_attention_heads).ok_or_else(|| Error::Inference {
            message: "attention head count must not be zero".to_string(),
        })?;
    let decision_heads =
        NonZeroUsize::new(config.hidden_size / 64).ok_or_else(|| Error::Inference {
            message: "hidden size is too small for the decision head".to_string(),
        })?;
    let mut names = Names::new();

    let hidden = encoder(
        config,
        weights,
        &input_ids,
        &masks,
        &zero_bias,
        &mut names,
        encoder_heads,
    )?;
    let type_embedding = weights
        .get("type_emb.weight")?
        .index_select(&kinds, 0)?
        .unsqueeze(1)?;
    let mut hidden = hidden.broadcast_add(&type_embedding)?;
    for layer in 0..agent.head_layers {
        hidden = head_layer(
            &hidden,
            layer,
            weights,
            decision_heads,
            &masks.keys,
            &mut names,
        )?;
    }

    let states = gather_markers(&hidden, &marker_flat, marker_slots)?;
    let logits = score_markers(&states, &marker_mask, weights)?;
    let probabilities = ops::softmax_last_dim(&logits)?;
    let (sorted, _) = probabilities.sort_last_dim(true)?;
    let width = sorted.dim(1)?;
    let top = sorted.narrow(1, width.saturating_sub(1), 1)?;
    let runner_up = sorted.narrow(1, width.saturating_sub(2), 1)?;
    // The features read the Options a Question actually has, which is what its Markers mark.
    let options = marker_mask.sum_keepdim(1)?.to_dtype(DType::F32)?;
    let entropy = probabilities
        .mul(&probabilities.maximum(1e-9f64)?.log()?)?
        .sum_keepdim(1)?
        .neg()?
        .broadcast_div(&options.maximum(2f64)?.log()?)?;
    let features = Tensor::cat(
        &[&top, &top.sub(&runner_up)?, &entropy, &(options / 255f64)?],
        1,
    )?;
    let pooled = hidden.narrow(1, 0, 1)?.squeeze(1)?;
    let pooled = Tensor::cat(&[&pooled, &features], 1)?;
    let action = linear(
        &pooled,
        weights.get("act_head.0.weight")?,
        Some(weights.get("act_head.0.bias")?),
    )?
    .gelu_erf()?;
    let action = linear(
        &action,
        weights.get("act_head.2.weight")?,
        Some(weights.get("act_head.2.bias")?),
    )?;
    let action = ops::softmax_last_dim(&action)?;

    let logits = logits.to_vec2::<f32>()?;
    let actions = action.to_vec2::<f32>()?;
    if logits
        .iter()
        .flatten()
        .chain(actions.iter().flatten())
        .any(|value| !value.is_finite())
    {
        return Err(Error::Inference {
            message: "model produced non-finite output".to_string(),
        });
    }
    Ok(Forward { logits, actions })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::call::QuestionType;
    use crate::laya::prompt::{Prepared, Responses, Sequence};

    /// A tensor of `shape` from a generator seeded by the shape alone, so a parameter holds the
    /// same numbers in every test that builds it.
    fn tensor(shape: &[usize]) -> Tensor {
        let mut state = shape.iter().fold(0x2545_f491_4f6c_dd1du64, |state, dim| {
            state.wrapping_mul(0x1000_0000_01b3) ^ *dim as u64
        });
        let values = (0..shape.iter().product::<usize>())
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((state >> 33) % 2000) as f32 / 1000.0 - 1.0
            })
            .collect::<Vec<_>>();
        Tensor::from_vec(values, shape, &Device::Cpu).unwrap()
    }

    /// The parameters of a two-layer encoder with one decision-head layer, matching `tiny_config`.
    fn tiny_weights() -> Weights {
        let mut values = HashMap::new();
        let shapes: &[(&str, &[usize])] = &[
            ("encoder.embeddings.tok_embeddings.weight", &[32, 64]),
            ("encoder.embeddings.norm.weight", &[64]),
            ("encoder.final_norm.weight", &[64]),
            ("encoder.layers.0.attn.Wqkv.weight", &[192, 64]),
            ("encoder.layers.0.attn.Wo.weight", &[64, 64]),
            ("encoder.layers.0.mlp.Wi.weight", &[64, 64]),
            ("encoder.layers.0.mlp.Wo.weight", &[64, 32]),
            ("encoder.layers.0.mlp_norm.weight", &[64]),
            ("encoder.layers.1.attn_norm.weight", &[64]),
            ("encoder.layers.1.attn.Wqkv.weight", &[192, 64]),
            ("encoder.layers.1.attn.Wo.weight", &[64, 64]),
            ("encoder.layers.1.mlp.Wi.weight", &[64, 64]),
            ("encoder.layers.1.mlp.Wo.weight", &[64, 32]),
            ("encoder.layers.1.mlp_norm.weight", &[64]),
            ("type_emb.weight", &[3, 64]),
            ("head.layers.0.norm1.weight", &[64]),
            ("head.layers.0.norm1.bias", &[64]),
            ("head.layers.0.norm2.weight", &[64]),
            ("head.layers.0.norm2.bias", &[64]),
            ("head.layers.0.self_attn.in_proj_weight", &[192, 64]),
            ("head.layers.0.self_attn.in_proj_bias", &[192]),
            ("head.layers.0.self_attn.out_proj.weight", &[64, 64]),
            ("head.layers.0.self_attn.out_proj.bias", &[64]),
            ("head.layers.0.linear1.weight", &[256, 64]),
            ("head.layers.0.linear1.bias", &[256]),
            ("head.layers.0.linear2.weight", &[64, 256]),
            ("head.layers.0.linear2.bias", &[64]),
            ("scorer.0.weight", &[64]),
            ("scorer.0.bias", &[64]),
            ("scorer.1.weight", &[64, 64]),
            ("scorer.1.bias", &[64]),
            ("scorer.3.weight", &[1, 64]),
            ("scorer.3.bias", &[1]),
            ("act_head.0.weight", &[256, 68]),
            ("act_head.0.bias", &[256]),
            ("act_head.2.weight", &[1, 256]),
            ("act_head.2.bias", &[1]),
        ];
        for (name, shape) in shapes {
            assert!(values.insert(name.to_string(), tensor(shape)).is_none());
        }
        Weights { values }
    }

    fn tiny_config() -> EncoderConfig {
        EncoderConfig::from_json(
            br#"{
                "vocab_size": 32, "hidden_size": 64, "intermediate_size": 32,
                "num_hidden_layers": 2, "num_attention_heads": 8, "local_attention": 4,
                "max_position_embeddings": 64,
                "layer_types": ["full_attention", "sliding_attention"],
                "rope_parameters": {
                    "full_attention": {"rope_type": "default", "rope_theta": 160000.0},
                    "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0}
                },
                "norm_eps": 0.00001
            }"#,
        )
        .unwrap()
    }

    fn tiny_agent() -> AgentConfig {
        AgentConfig {
            max_len: 64,
            head_max_len: 32,
            head_layers: 1,
            temperature: vec![1.0; 3],
            temperature_by_options: BTreeMap::new(),
        }
    }

    fn tiny_special() -> SpecialTokens {
        SpecialTokens {
            cls: 1,
            sep: 2,
            mask: 3,
            pad: 0,
            mask_text: "[MASK]".to_string(),
        }
    }

    fn question(kind: QuestionType, ids: &[u32], markers: &[usize]) -> Prepared {
        Prepared {
            sequence: Sequence {
                ids: ids.to_vec(),
                markers: markers.to_vec(),
            },
            kind,
            options: markers
                .iter()
                .map(|marker| format!("option {marker}"))
                .collect(),
            responses: Responses::Unnamed,
        }
    }

    /// One Question of six real positions inside eight, window four, so a sliding layer reaches two
    /// positions either side of a query.
    #[test]
    fn full_attention_reaches_every_token_and_sliding_attention_the_window() {
        let valid = [true, true, true, true, true, true, false, false];
        let masks = Masks::new(&valid, 1, 8, 4).unwrap();
        let grid = |mask: &Tensor| mask.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let row = |keys: &[usize]| {
            (0..8)
                .map(|key| if keys.contains(&key) { 0.0 } else { MASK_FILL })
                .collect::<Vec<f32>>()
        };

        // One key mask serves every full-attention layer and the decision head: broadcast over the
        // queries and heads, it reaches every real key and no padded one.
        assert_eq!(grid(&masks.keys), row(&[0, 1, 2, 3, 4, 5]));

        let sliding = grid(&masks.sliding);
        assert_eq!(
            sliding,
            [
                row(&[0, 1, 2]),
                row(&[0, 1, 2, 3]),
                row(&[0, 1, 2, 3, 4]),
                row(&[1, 2, 3, 4, 5]),
                row(&[2, 3, 4, 5]),
                row(&[3, 4, 5]),
                // The padded queries reach every key their mask allows; their own attention
                // output is never read.
                row(&[0, 1, 2, 3, 4, 5]),
                row(&[0, 1, 2, 3, 4, 5]),
            ]
            .concat()
        );
    }

    /// Four head values, one per half of the head, rotated against the other half at the frequency
    /// of the position: a first half of `1, 2` and a second half of `3, 4` at position one.
    #[test]
    fn rope_rotates_the_two_halves_of_a_head_against_each_other() {
        let input = Tensor::from_vec(
            vec![1f32, 2., 3., 4., 5., 6., 7., 8.],
            (1, 1, 2, 4),
            &Device::Cpu,
        )
        .unwrap();
        let rotated = rope(&input, 10_000.0)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        // Position zero is not rotated at all.
        assert_eq!(&rotated[..4], &[1.0, 2.0, 3.0, 4.0]);
        // At position one the head turns by the two frequencies its four values are spread over:
        // `1` and `1/100`. Each half is turned against the other, not element against its
        // neighbour.
        let (first, second) = ((1.0f32).cos(), (0.01f32).cos());
        let (third, fourth) = ((1.0f32).sin(), (0.01f32).sin());
        let expected = [
            5.0 * first - 7.0 * third,
            6.0 * second - 8.0 * fourth,
            7.0 * first + 5.0 * third,
            8.0 * second + 6.0 * fourth,
        ];
        for (actual, expected) in rotated[4..].iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 1e-6,
                "{rotated:?} should end in {expected}"
            );
        }
    }

    #[test]
    fn markers_are_read_at_each_question_own_positions() {
        let hidden = Tensor::from_vec(
            (0..24).map(|value| value as f32).collect(),
            (2, 4, 3),
            &Device::Cpu,
        )
        .unwrap();
        // Two Questions of four positions each: the second starts at flat position four.
        let indices = Tensor::from_vec(vec![1u32, 3, 4, 6], (4,), &Device::Cpu).unwrap();

        let states = gather_markers(&hidden, &indices, 2)
            .unwrap()
            .to_vec3::<f32>()
            .unwrap();

        assert_eq!(
            states,
            vec![
                vec![vec![3., 4., 5.], vec![9., 10., 11.]],
                vec![vec![12., 13., 14.], vec![18., 19., 20.]],
            ]
        );
    }

    /// Padding a Question into a wider batch must not move its Markers' scores, and a Marker slot
    /// no Option occupies must carry no score of its own.
    #[test]
    fn a_question_scores_the_same_however_the_batch_pads_it() {
        let config = tiny_config();
        let agent = tiny_agent();
        let weights = tiny_weights();
        let special = tiny_special();
        let wide = || question(QuestionType::Choice, &[5, 6, 7, 8], &[1, 2]);
        let narrow = || question(QuestionType::Choice, &[9, 10, 11], &[0, 1, 2]);

        let Forward {
            logits: batched, ..
        } = forward(&config, &agent, &weights, &[wide(), narrow()], &special).unwrap();
        let Forward {
            logits: wide_alone, ..
        } = forward(&config, &agent, &weights, &[wide()], &special).unwrap();
        let Forward {
            logits: narrow_alone,
            ..
        } = forward(&config, &agent, &weights, &[narrow()], &special).unwrap();

        assert_eq!(batched.len(), 2);
        assert_eq!(
            batched[0].len(),
            3,
            "the batch is as wide as its widest Question"
        );
        assert_eq!(
            batched[0][2], MASK_FILL,
            "the extra slot belongs to no Option"
        );
        for (value, alone) in batched[0]
            .iter()
            .zip(&wide_alone[0])
            .chain(batched[1].iter().zip(&narrow_alone[0]))
        {
            assert!(
                (value - alone).abs() < 1e-5,
                "{value} is not {alone}: padding changed a score"
            );
        }
    }
}
