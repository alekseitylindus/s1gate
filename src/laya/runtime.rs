//! MLX weight loading and the Laya forward pass.

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::Array;
use mlx_rs::error::Exception;
use mlx_rs::ops::{self, *};

use crate::error::{Error, Result};

use super::SpecialTokens;
use super::config::{AgentConfig, AttentionType, EncoderConfig};
use super::prompt::Prepared;

const MASK_FILL: f32 = -1e4;
const LN_EPS: f32 = 1e-5;

type MlxResult<T> = std::result::Result<T, Exception>;

#[derive(Debug)]
pub(super) struct Weights {
    values: HashMap<String, Array>,
}

impl Weights {
    fn get(&self, name: &str) -> MlxResult<&Array> {
        self.values
            .get(name)
            .ok_or_else(|| Exception::custom(format!("missing materialized parameter `{name}`")))
    }
}

pub(super) fn load_weights(path: &Path) -> Result<Weights> {
    let loaded = Array::load_safetensors(path).map_err(|error| Error::Inference {
        message: format!("loading safetensors `{}`: {error}", path.display()),
    })?;
    let mut values = HashMap::with_capacity(loaded.len());
    for (name, value) in loaded {
        let value = value.as_type::<f32>().map_err(|error| Error::Inference {
            message: format!("converting parameter `{name}` to F32: {error}"),
        })?;
        values.insert(name, value);
    }
    Ok(Weights { values })
}

fn linear(x: &Array, weight: &Array, bias: Option<&Array>) -> MlxResult<Array> {
    let y = x.matmul(weight.t())?;
    match bias {
        Some(bias) => y.add(bias),
        None => Ok(y),
    }
}

fn layer_norm(x: &Array, weight: &Array, bias: Option<&Array>, eps: f32) -> MlxResult<Array> {
    mlx_rs::fast::layer_norm(x, Some(weight), bias, eps)
}

fn attention(
    x: &Array,
    qkv_weight: &Array,
    qkv_bias: Option<&Array>,
    out_weight: &Array,
    out_bias: Option<&Array>,
    heads: usize,
    rope_base: Option<f32>,
    mask: &Array,
) -> MlxResult<Array> {
    let shape = x.shape();
    let batch = shape[0];
    let length = shape[1];
    let hidden = shape[2];
    let heads = i32::try_from(heads)
        .map_err(|_| Exception::custom("attention head count exceeds MLX limits"))?;
    if heads == 0 || hidden % heads != 0 {
        return Err(Exception::custom(
            "hidden size must be divisible by a non-zero attention head count",
        ));
    }
    let head_dim = hidden / heads;
    let qkv = linear(x, qkv_weight, qkv_bias)?.reshape(&[batch, length, 3, heads, head_dim])?;
    let axis_indices = [0i32];
    let take = |index: i32| -> MlxResult<Array> {
        qkv.take_axis(Array::from_slice(&[index], &axis_indices), 2)?
            .squeeze_axes(&[2])?
            .transpose_axes(&[0, 2, 1, 3])
    };
    let mut q = take(0)?;
    let mut k = take(1)?;
    let v = take(2)?;
    if let Some(base) = rope_base {
        q = mlx_rs::fast::rope(&q, head_dim, false, Some(base), 1.0, 0, None::<&Array>)?;
        k = mlx_rs::fast::rope(&k, head_dim, false, Some(base), 1.0, 0, None::<&Array>)?;
    }
    let attended = mlx_rs::fast::scaled_dot_product_attention(
        &q,
        &k,
        &v,
        (head_dim as f32).powf(-0.5),
        mask,
        None::<&Array>,
    )?;
    let attended = attended
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[batch, length, hidden])?;
    linear(&attended, out_weight, out_bias)
}

fn encoder_mask(attention_mask: &Array, window: usize) -> MlxResult<(Array, Array)> {
    let shape = attention_mask.shape();
    if shape.len() != 2 {
        return Err(Exception::from("attention mask must be rank two"));
    }
    let batch = shape[0];
    let length = shape[1];
    let full = attention_mask.reshape(&[batch, 1, 1, length])?;
    let length_usize = usize::try_from(length)
        .map_err(|_| Exception::custom("attention mask has a negative length"))?;
    let cells = length_usize
        .checked_mul(length_usize)
        .ok_or_else(|| Exception::custom("attention mask is too large"))?;
    let radius = window / 2;
    let mut local = vec![false; cells];
    for row in 0..length_usize {
        for col in 0..length_usize {
            local[row * length_usize + col] = row.abs_diff(col) <= radius;
        }
    }
    let local = Array::from_slice(&local, &[1, 1, length, length]);
    let valid_queries = attention_mask
        .logical_not()?
        .reshape(&[batch, 1, length, 1])?;
    let sliding = local.logical_or(&valid_queries)?.logical_and(&full)?;
    Ok((full, sliding))
}

fn encoder(
    config: &EncoderConfig,
    weights: &Weights,
    input_ids: &Array,
    attention_mask: &Array,
) -> MlxResult<Array> {
    let embeddings = weights.get("encoder.embeddings.tok_embeddings.weight")?;
    let mut x = embeddings.take_axis(input_ids, 0)?;
    x = layer_norm(
        &x,
        weights.get("encoder.embeddings.norm.weight")?,
        None,
        config.norm_eps,
    )?;
    let (full_mask, sliding_mask) = encoder_mask(attention_mask, config.local_attention)?;
    for layer in 0..config.num_hidden_layers {
        let prefix = format!("encoder.layers.{layer}");
        let normalized = if layer == 0 {
            x.clone()
        } else {
            layer_norm(
                &x,
                weights.get(&format!("{prefix}.attn_norm.weight"))?,
                None,
                config.norm_eps,
            )?
        };
        let kind = config.layer_type(layer)?;
        let mask = if matches!(kind, AttentionType::Full) {
            &full_mask
        } else {
            &sliding_mask
        };
        let attended = attention(
            &normalized,
            weights.get(&format!("{prefix}.attn.Wqkv.weight"))?,
            None,
            weights.get(&format!("{prefix}.attn.Wo.weight"))?,
            None,
            config.num_attention_heads,
            Some(config.rope_base(kind)?),
            mask,
        )?;
        x = x.add(&attended)?;
        let normalized = layer_norm(
            &x,
            weights.get(&format!("{prefix}.mlp_norm.weight"))?,
            None,
            config.norm_eps,
        )?;
        let gated = linear(
            &normalized,
            weights.get(&format!("{prefix}.mlp.Wi.weight"))?,
            None,
        )?;
        let halves = gated.split_equal(2, Some(-1))?;
        let activated = mlx_rs::nn::gelu(&halves[0])?.multiply(&halves[1])?;
        let projected = linear(
            &activated,
            weights.get(&format!("{prefix}.mlp.Wo.weight"))?,
            None,
        )?;
        x = x.add(&projected)?;
    }
    layer_norm(
        &x,
        weights.get("encoder.final_norm.weight")?,
        None,
        config.norm_eps,
    )
}

fn head_layer(
    x: &Array,
    prefix: &str,
    weights: &Weights,
    heads: usize,
    mask: &Array,
) -> MlxResult<Array> {
    let normalized = layer_norm(
        x,
        weights.get(&format!("{prefix}.norm1.weight"))?,
        Some(weights.get(&format!("{prefix}.norm1.bias"))?),
        LN_EPS,
    )?;
    let attended = attention(
        &normalized,
        weights.get(&format!("{prefix}.self_attn.in_proj_weight"))?,
        Some(weights.get(&format!("{prefix}.self_attn.in_proj_bias"))?),
        weights.get(&format!("{prefix}.self_attn.out_proj.weight"))?,
        Some(weights.get(&format!("{prefix}.self_attn.out_proj.bias"))?),
        heads,
        None,
        mask,
    )?;
    let x = x.add(&attended)?;
    let normalized = layer_norm(
        &x,
        weights.get(&format!("{prefix}.norm2.weight"))?,
        Some(weights.get(&format!("{prefix}.norm2.bias"))?),
        LN_EPS,
    )?;
    let feed = linear(
        &normalized,
        weights.get(&format!("{prefix}.linear1.weight"))?,
        Some(weights.get(&format!("{prefix}.linear1.bias"))?),
    )?;
    let feed = mlx_rs::nn::relu(&feed)?;
    let feed = linear(
        &feed,
        weights.get(&format!("{prefix}.linear2.weight"))?,
        Some(weights.get(&format!("{prefix}.linear2.bias"))?),
    )?;
    x.add(&feed)
}

pub(super) fn forward(
    config: &EncoderConfig,
    agent: &AgentConfig,
    weights: &Weights,
    prepared: &[Prepared],
    special: &SpecialTokens,
) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
    let count_usize = prepared.len();
    let length_usize = prepared
        .iter()
        .map(|item| item.sequence.ids.len())
        .max()
        .unwrap_or(0);
    let markers_usize = prepared
        .iter()
        .map(|item| item.sequence.markers.len())
        .max()
        .unwrap_or(0)
        .max(2);
    let count = i32::try_from(count_usize).map_err(|_| Error::Inference {
        message: "too many Questions".to_string(),
    })?;
    let length = i32::try_from(length_usize).map_err(|_| Error::Inference {
        message: "input is too long".to_string(),
    })?;
    let markers = i32::try_from(markers_usize).map_err(|_| Error::Inference {
        message: "too many Options".to_string(),
    })?;
    let hidden = i32::try_from(config.hidden_size).map_err(|_| Error::Inference {
        message: "hidden size exceeds MLX limits".to_string(),
    })?;
    let token_cells = count_usize
        .checked_mul(length_usize)
        .ok_or_else(|| Error::Inference {
            message: "input batch is too large".to_string(),
        })?;
    let marker_cells = count_usize
        .checked_mul(markers_usize)
        .ok_or_else(|| Error::Inference {
            message: "marker batch is too large".to_string(),
        })?;
    let mut ids = vec![special.pad; token_cells];
    let mut mask = vec![false; token_cells];
    let mut marker_pos = vec![0i32; marker_cells];
    let mut marker_mask = vec![false; marker_cells];
    let mut qtypes = Vec::with_capacity(prepared.len());
    for (row, item) in prepared.iter().enumerate() {
        let base = row * length_usize;
        ids[base..base + item.sequence.ids.len()].copy_from_slice(&item.sequence.ids);
        mask[base..base + item.sequence.ids.len()].fill(true);
        for (index, marker) in item.sequence.markers.iter().enumerate() {
            let position = i32::try_from(*marker).map_err(|_| Error::Inference {
                message: "marker position exceeds MLX limits".to_string(),
            })?;
            marker_pos[row * markers_usize + index] = position;
            marker_mask[row * markers_usize + index] = true;
        }
        qtypes.push(item.kind.index());
    }
    let result = (|| -> MlxResult<(Array, Array)> {
        let input_ids = Array::from_slice(&ids, &[count, length]);
        let attention_mask = Array::from_slice(&mask, &[count, length]);
        let marker_pos = Array::from_slice(&marker_pos, &[count, markers]);
        let marker_mask = Array::from_slice(&marker_mask, &[count, markers]);
        let qtype = Array::from_slice(&qtypes, &[count]);
        let mut hidden = encoder(config, weights, &input_ids, &attention_mask)?;
        let type_emb = weights
            .get("type_emb.weight")?
            .take_axis(&qtype, 0)?
            .expand_dims(1)?;
        hidden = hidden.add(&type_emb)?;
        let head_mask = attention_mask.reshape(&[count, 1, 1, length])?;
        for layer in 0..agent.head_layers {
            hidden = head_layer(
                &hidden,
                &format!("head.layers.{layer}"),
                weights,
                config.hidden_size / 64,
                &head_mask,
            )?;
        }
        let marker_indices = marker_pos
            .expand_dims(2)?
            .broadcast_to(&[count, markers, hidden])?;
        let marker_hidden = hidden.take_along_axis(&marker_indices, 1)?;
        let scorer = layer_norm(
            &marker_hidden,
            weights.get("scorer.0.weight")?,
            Some(weights.get("scorer.0.bias")?),
            LN_EPS,
        )?;
        let scorer = linear(
            &scorer,
            weights.get("scorer.1.weight")?,
            Some(weights.get("scorer.1.bias")?),
        )?;
        let scorer = mlx_rs::nn::gelu(&scorer)?;
        let scorer = linear(
            &scorer,
            weights.get("scorer.3.weight")?,
            Some(weights.get("scorer.3.bias")?),
        )?
        .squeeze_axes(&[2])?;
        let logits = ops::select(&marker_mask, &scorer, &Array::from_f32(MASK_FILL))?;
        let probabilities = ops::softmax_axis(&logits, -1, false)?;
        let sorted = ops::sort_axis(&probabilities, -1)?;
        let top = sorted.take_axis(Array::from_slice(&[markers - 2, markers - 1], &[2]), -1)?;
        let top_one = top
            .take_axis(Array::from_slice(&[1], &[1]), -1)?
            .squeeze_axes(&[1])?;
        let top_two = top
            .take_axis(Array::from_slice(&[0], &[1]), -1)?
            .squeeze_axes(&[1])?;
        let choices = marker_mask
            .count_nonzero(mlx_rs::ops::CountNonzeroOptions {
                axes: mlx_rs::Axes::Axis(-1),
                keep_dims: false,
            })?
            .as_type::<f32>()?;
        let entropy = probabilities
            .multiply(&ops::maximum(&probabilities, &Array::from_f32(1e-9))?.log()?)?
            .negative()?
            .sum_axis(-1, false)?
            / ops::maximum(&choices, &Array::from_f32(2.0))?.log()?;
        let features = ops::stack(
            &[
                &top_one,
                &top_one.subtract(&top_two)?,
                &entropy,
                &(choices / 255.0),
            ],
            -1,
        )?;
        let pooled = hidden
            .take_axis(Array::from_slice(&[0], &[1]), 1)?
            .squeeze_axes(&[1])?;
        let pooled = ops::concatenate(&[&pooled, &features], 1)?;
        let action = linear(
            &pooled,
            weights.get("act_head.0.weight")?,
            Some(weights.get("act_head.0.bias")?),
        )?;
        let action = mlx_rs::nn::gelu(&action)?;
        let action = linear(
            &action,
            weights.get("act_head.2.weight")?,
            Some(weights.get("act_head.2.bias")?),
        )?;
        Ok((logits, ops::softmax_axis(&action, -1, false)?))
    })()
    .map_err(|error| Error::Inference {
        message: error.to_string(),
    })?;
    mlx_rs::transforms::eval([&result.0, &result.1]).map_err(|error| Error::Inference {
        message: error.to_string(),
    })?;
    let logits_slice = result.0.as_slice::<f32>();
    let actions_slice = result.1.as_slice::<f32>();
    if logits_slice
        .iter()
        .chain(actions_slice)
        .any(|value| !value.is_finite())
    {
        return Err(Error::Inference {
            message: "model produced non-finite output".to_string(),
        });
    }
    let logits = logits_slice
        .chunks(markers_usize)
        .map(|row| row.to_vec())
        .collect();
    let actions = actions_slice
        .chunks(agent.act_costs.len() + 1)
        .map(|row| row.to_vec())
        .collect();
    Ok((logits, actions))
}
